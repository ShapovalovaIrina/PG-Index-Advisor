import collections
import json
import logging
import copy
import gym
import random
import numpy as np

from typing import List, Set

from .common import EnvironmentType, IndexType
from .action_manager import MultiColumnIndexActionManager as ActionManager
from .observation_manager import SingleColumnIndexPlanEmbeddingObservationManagerWithCost as ObservationManager
from .reward_manager import CostAndStorageRewardManager as RewardManager
from pg_index_advisor.schema.db_connector import UserPostgresDatabaseConnector as DatabaseConnector
from pg_index_advisor.utils import index_from_column_combination, remove_if_exists
from pg_index_advisor.schema.structures import RealIndex, PotentialIndex
from pg_index_advisor.schema.cost_evaluation import CostEvaluationWithHidingIndex as CostEvaluation
from index_selection_evaluation.selection.utils import b_to_mb
from index_selection_evaluation.selection.workload import Workload


class PGIndexAdvisorEnv(gym.Env):
    action_manager: ActionManager
    observation_manager: ObservationManager
    reward_manager: RewardManager

    workloads: List[Workload]
    current_workload: Workload

    current_created_indexes: Set[PotentialIndex]
    current_indexes_to_delete: Set[PotentialIndex]

    def __init__(self, environment_type=EnvironmentType.TRAINING, config=None):
        logging.debug("__init__() was called")

        super(PGIndexAdvisorEnv, self).__init__()

        self.rnd = random.Random()
        self.rnd.seed(config["random_seed"])
        self.env_id = config["env_id"]
        self.environment_type = environment_type
        self.config = config

        self.number_of_resets = 0
        self.total_number_of_steps = 0

        db_config = config["database"]
        connector = DatabaseConnector(
            db_config["database"],
            db_config["username"],
            db_config["password"],
            db_port=db_config["port"],
            autocommit=True
        )
        self.cost_evaluation = CostEvaluation(connector)

        self.globally_indexable_columns = config["globally_indexable_columns"]
        # In certain cases, workloads are consumed: therefore, we need copy
        self.workloads = copy.copy(config["workloads"])
        self.current_workload_idx = 0
        self.current_costs = 0
        self.similar_workloads = config["similar_workloads"]
        self.max_steps_per_episode = config["max_steps_per_episode"]
        self.valid_actions = np.array([])

        self.action_manager = config["action_manager"]
        self.action_space = self.action_manager.get_action_space()

        self.observation_manager = config["observation_manager"]
        self.observation_space = self.observation_manager.get_observation_space()

        self.reward_manager = config["reward_manager"]

        self._get_initial_observation()

        if self.environment_type != environment_type.TRAINING:
            self.episode_performances = collections.deque(maxlen=len(config["workloads"]))

    def reset(self, seed=None, options=None):
        logging.debug("reset() was called")

        self.number_of_resets += 1
        self.total_number_of_steps += self.steps_taken

        initial_observation = self._get_initial_observation()

        return initial_observation

    def step(self, action_idx):
        logging.info(f"Take action: {self._action_idx_to_str(action_idx)}")

        self._step_asserts(action_idx)

        self.steps_taken += 1

        environment_state = None
        action = self.valid_actions[action_idx]

        assert action in [self.action_manager.ALLOW_TO_CREATE, self.action_manager.ALLOW_TO_DELETE]

        if action == self.action_manager.ALLOW_TO_CREATE:
            environment_state = self._create_index(action_idx)
        if action == self.action_manager.ALLOW_TO_DELETE:
            environment_state = self._delete_index(action_idx)

        current_observation = self.observation_manager.get_observation(environment_state)

        self.valid_actions, is_valid_action_left = \
            self.action_manager.update_valid_actions(
                action_idx,
                self.current_budget,
                self.current_storage_consumption
            )
        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left

        reward = self.reward_manager.calculate_reward(environment_state)

        if episode_done and self.environment_type != EnvironmentType.TRAINING:
            self._report_episode_performance()
            self.current_workload_idx += 1

        info = self._get_info()

        return current_observation, reward, episode_done, info

    def _create_index(self, action_idx):
        old_index_size = 0

        new_index = PotentialIndex(self.globally_indexable_columns[action_idx])
        self.current_created_indexes.add(new_index)

        if not new_index.is_single_column():
            parent_index = PotentialIndex(new_index.columns[:-1])

            old_index_size, index_type = self._get_parent_index_estimated_size(parent_index)

            if index_type == IndexType.VIRTUAL:
                # Remove parent index from current_created_indexes ->
                # virtual index deleted in _get_env_state
                remove_if_exists(self.current_created_indexes, parent_index)

            assert old_index_size > 0, \
                "Parent index size must have been found if not single column index."

        return self._get_env_state_for_created_index(
            new_index=new_index,
            old_index_size=old_index_size
        )

    def _delete_index(self, action_idx):
        action_columns = self.globally_indexable_columns[action_idx]
        deleted_index = None

        for i in self.current_indexes_to_delete:
            if i.columns == action_columns:
                deleted_index = i
                break

        assert deleted_index is not None

        self.current_indexes_to_delete.remove(deleted_index)

        # TODO: hide index

        return self._get_env_state_for_deleted_index(deleted_index=deleted_index)


    def _get_parent_index_estimated_size(self, parent_index):
        index_type = None
        old_index_size = 0

        for current_index in self.current_created_indexes:
            if current_index == parent_index:
                old_index_size = current_index.estimated_size
                index_type = IndexType.VIRTUAL
                break

        if not old_index_size:
            for initial_index in self.action_manager.initial_indexes:
                if initial_index == parent_index:
                    old_index_size = initial_index.estimated_size
                    index_type = IndexType.REAL
                    break

        assert index_type is not None, f"""
        Index {parent_index} is not found in \
        created indexes in env or initial indexes in action manager.
        
        State:
        {json.dumps(self._get_env_state_for_debug(), indent=2)}
        """

        return old_index_size, index_type

    def valid_action_mask(self):
        allowed_actions = [
            self.action_manager.ALLOW_TO_CREATE,
            self.action_manager.ALLOW_TO_DELETE
        ]
        return [action in allowed_actions for action in self.action_manager.valid_actions]

    def _step_asserts(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        assert (
            self.valid_actions[action] == self.action_manager.ALLOW_TO_CREATE or
            self.valid_actions[action] == self.action_manager.ALLOW_TO_DELETE
        ), f"""
        Agent has chosen invalid action: {action} ({self._action_idx_to_str(action)}.
        
        State:
        {json.dumps(self._get_env_state_for_debug(), indent=2)}
        )"""

        if self.valid_actions[action] == self.action_manager.ALLOW_TO_CREATE:
            assert (
                PotentialIndex(self.globally_indexable_columns[action]) not in self.current_created_indexes
            ), f"{PotentialIndex(self.globally_indexable_columns[action])} already in self.current_created_indexes"

        if self.valid_actions[action] == self.action_manager.ALLOW_TO_DELETE:
            assert (
                PotentialIndex(self.globally_indexable_columns[action]) in self.current_indexes_to_delete
            ), f"{PotentialIndex(self.globally_indexable_columns[action])} not in self.current_indexes_to_delete"

    def _get_initial_observation(self):
        self.steps_taken = 0
        
        self.current_created_indexes = set()
        self.current_indexes_to_delete = copy.copy(self.action_manager.initial_indexes)

        self.current_storage_consumption = sum([
            index.estimated_size
            for index
            in self.current_indexes_to_delete
        ])
        
        self.reward_manager.reset()

        if len(self.workloads) == 0:
            self.workloads = copy.copy(self.config["workloads"])

        # TODO: ???
        if self.environment_type == EnvironmentType.TRAINING:
            if self.similar_workloads:
                # 200 is an arbitrary value
                self.current_workload = self.workloads.pop(0 + self.env_id * 200)
            else:
                self.current_workload = self.rnd.choice(self.workloads)
        else:
            self.current_workload = self.workloads[self.current_workload_idx % len(self.workloads)]

        self.current_budget = self.current_workload.budget
        self.previous_cost = None

        self.valid_actions = self.action_manager.get_initial_valid_actions(
            self.current_workload,
            self.current_budget,
            self.current_storage_consumption
        )
        environment_state = self._get_init_env_state()

        episode_state = {
            "budget": self.current_budget,
            "workload": self.current_workload,
            "initial_cost": self.initial_costs
        }
        self.observation_manager.init_episode(episode_state)

        initial_observation = self.observation_manager.get_observation(environment_state)

        return initial_observation

    def _action_idx_to_str(self, action_idx):
        action_str = {
            self.action_manager.ALLOW_TO_CREATE: 'create',
            self.action_manager.ALLOW_TO_DELETE: 'delete',
            self.action_manager.FORBIDDEN_ACTION: 'forbid'
        }
        return f'{action_str[self.valid_actions[action_idx]]} {self.globally_indexable_columns[action_idx]}'

    def _get_env_state_for_created_index(
            self,
            new_index: PotentialIndex,
            old_index_size: int
    ):
        self.cost_evaluation.update_created_indexes(self.current_created_indexes, store_size=True)

        total_costs, plans_per_query, costs_per_query = \
            self.cost_evaluation.calculate_cost_and_plans(
                self.current_workload,
                self.current_created_indexes,
                store_size=True
            )

        self.previous_cost = self.current_costs
        self.previous_storage_consumption = self.current_storage_consumption

        self.current_costs = total_costs
        self.current_storage_consumption += new_index.estimated_size
        self.current_storage_consumption -= old_index_size

        assert new_index.estimated_size >= old_index_size

        new_index_relative_size = max(new_index.estimated_size - old_index_size, 1)

        if self.current_budget:
            assert b_to_mb(self.current_storage_consumption) <= self.current_budget, (
                "Storage consumption exceeds budget: "
                f"{b_to_mb(self.current_storage_consumption)} "
                f" > {self.current_budget}"
            )
                
        environment_state = self._get_env_state_dict(
            plans_per_query,
            costs_per_query,
            new_index_relative_size
        )

        return environment_state

    def _get_env_state_for_deleted_index(
            self,
            deleted_index: PotentialIndex
    ):
        self.cost_evaluation.update_deleted_indexes(self.current_indexes_to_delete)

        total_costs, plans_per_query, costs_per_query = \
            self.cost_evaluation.calculate_cost_and_plans(
                self.current_workload,
                self.current_created_indexes,
                store_size=True
            )

        self.previous_cost = self.current_costs
        self.previous_storage_consumption = self.current_storage_consumption

        self.current_costs = total_costs
        self.current_storage_consumption -= deleted_index.estimated_size

        if self.current_budget:
            assert b_to_mb(self.current_storage_consumption) <= self.current_budget, (
                "Storage consumption exceeds budget: "
                f"{b_to_mb(self.current_storage_consumption)} "
                f" > {self.current_budget}"
            )

        environment_state = self._get_env_state_dict(
            plans_per_query,
            costs_per_query,
            -deleted_index.estimated_size
        )

        return environment_state

    def _get_init_env_state(self):
        self.cost_evaluation.reset_hypopg()

        total_costs, plans_per_query, costs_per_query = \
            self.cost_evaluation.calculate_cost_and_plans(
                self.current_workload,
                self.current_created_indexes,
                store_size=True
            )

        self.current_costs = total_costs
        self.initial_costs = total_costs

        environment_state = self._get_env_state_dict(plans_per_query, costs_per_query)

        return environment_state

    def _get_env_state_dict(self, plans_per_query, costs_per_query, new_index_relative_size=None):
        return {
            "action_status": self.action_manager.current_actions_status,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            "new_index_relative_size": new_index_relative_size,
            "plans_per_query": plans_per_query,
            "costs_per_query": costs_per_query
        }

    def _get_info(self):
        return {"action_mask": self.valid_actions}

    def _get_env_state_for_debug(self):
        return {
            'Valid actions': ', '.join(map(str, self.valid_actions)),
            'Current created indexes': ', '.join(map(str, self.current_created_indexes)),
            'Current undeleted indexes': ', '.join(map(str, self.current_indexes_to_delete)),
            'Initial undeleted indexes': ', '.join(map(str, self.action_manager.initial_indexes))
        }

    def get_cost_eval_cache_info(self):
        return self.cost_evaluation.cost_requests, \
            self.cost_evaluation.cache_hits, \
            self.cost_evaluation.costing_time

    def get_cost_eval_cache(self):
        return self.cost_evaluation.cache

    def _report_episode_performance(self):
        episode_performance = {
            "achieved_cost": self.current_costs / self.initial_costs * 100,
            "memory_consumption": self.current_storage_consumption,
            "available_budget": self.current_budget,
            "evaluated_workload": self.current_workload,
            "indexes": self.current_created_indexes,
        }

        output = (
            f"Evaluated Workload ({self.environment_type}): {self.current_workload}\n    "
            f"Initial cost: {self.initial_costs:,.2f}, now: {self.current_costs:,.2f} "
            f"({episode_performance['achieved_cost']:.2f}). Reward: {self.reward_manager.accumulated_reward}.\n    "
            f"Size: {b_to_mb(self.current_storage_consumption):.2f} with {len(self.current_created_indexes)} indexes:\n    "
            f"{self.current_created_indexes}\n    "
        )
        logging.warning(output)

        self.episode_performances.append(episode_performance)

    def render(self, mode="human"):
        logging.warning("render() was called")
        pass

    def close(self):
        logging.warning("close() was called")

