import collections
import logging
import copy
import gym
import random
import numpy as np

from typing import List

from .common import EnvironmentType
from .action_manager import MultiColumnIndexActionManager as ActionManager
from .observation_manager import SingleColumnIndexPlanEmbeddingObservationManagerWithCost as ObservationManager
from .reward_manager import CostAndStorageRewardManager as RewardManager
from pg_index_advisor.schema.db_connector import UserPostgresDatabaseConnector as DatabaseConnector
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.index import Index as PotentialIndex
from index_selection_evaluation.selection.utils import b_to_mb
from index_selection_evaluation.selection.workload import Workload


class PGIndexAdvisorEnv(gym.Env):
    action_manager: ActionManager
    observation_manager: ObservationManager
    reward_manager: RewardManager
    workloads: List[Workload]
    current_workload: Workload

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
        self.connector = DatabaseConnector(
            db_config["database"],
            db_config["username"],
            db_config["password"],
            db_port=db_config["port"],
            autocommit=True
        )
        self.cost_evaluation = CostEvaluation(self.connector)

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
            self.episode_performance = collections.deque(maxlen=len(config["workloads"]))

    def reset(self, seed=None, options=None):
        logging.debug("reset() was called")

        self.number_of_resets += 1
        self.total_number_of_steps += self.steps_taken

        initial_observation = self._get_initial_observation()
        info = self._get_info()

        return initial_observation, info

    def step(self, action):
        logging.debug(f"Take action: {self._action_idx_to_str(action)}")

        self._step_asserts(action)

        self.steps_taken += 1
        old_index_size = 0

        new_index = PotentialIndex(self.globally_indexable_columns[action])
        self.current_indexes.add(new_index)

        if not new_index.is_single_column():
            parent_index = PotentialIndex(new_index.columns[:-1])

            for index in self.current_indexes:
                if index == parent_index:
                    old_index_size = index.estimates_size

            self.current_indexes.remove(parent_index)

            assert old_index_size > 0, \
                "Parent index size must have been found if not single column index."

        environment_state = self._get_env_state(
            new_index=new_index,
            old_index_size=old_index_size
        )
        current_observation = self.observation_manager.get_observation(environment_state)

        self.valid_actions, is_valid_action_left = \
            self.action_manager.update_valid_actions(
                action,
                self.current_budget,
                self.current_storage_consumption
            )
        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left

        reward = self.reward_manager.calculate_reward(environment_state)

        if episode_done and self.environment_type != EnvironmentType.TRAINING:
            # TODO: report episode performance
            self.current_workload_idx += 1

        info = self._get_info()

        return current_observation, reward, episode_done, False, info

    def valid_action_mask(self):
        return [bool(action) for action in self.valid_actions]

    def _step_asserts(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert (
            self.valid_actions[action] == self.action_manager.ALLOWED_ACTION
        ), f"Agent has chosen invalid action: {action}"
        assert (
            PotentialIndex(self.globally_indexable_columns[action]) not in self.current_indexes
        ), f"{PotentialIndex(self.globally_indexable_columns[action])} already in self.current_indexes"


    def _get_initial_observation(self):
        self.current_indexes = set()
        self.steps_taken = 0
        self.current_storage_consumption = 0
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
            self.current_budget
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
        return self.globally_indexable_columns[action_idx]

    def _get_env_state(
            self,
            new_index: PotentialIndex,
            old_index_size: int
    ):
        total_costs, plans_per_query, costs_per_query = \
            self.cost_evaluation.calculate_cost_and_plans(
                self.current_workload,
                self.current_indexes,
                store_size=True
            )

        self.previous_cost = self.current_costs
        self.previous_storage_consumption = self.current_storage_consumption

        self.current_costs = total_costs
        self.current_storage_consumption += new_index.estimated_size
        self.current_storage_consumption -= old_index_size

        # This assumes that old_index_size is not None if new_index is not None
        assert new_index.estimated_size >= old_index_size

        # TODO: минимальный размер нового индекса равен 1?
        new_index_size = max(new_index.estimated_size - old_index_size, 1)

        if self.current_budget:
            assert b_to_mb(self.current_storage_consumption) <= self.current_budget, (
                "Storage consumption exceeds budget: "
                f"{b_to_mb(self.current_storage_consumption)} "
                f" > {self.current_budget}"
            )
                
        environment_state = self._get_env_state_dict(
            plans_per_query,
            costs_per_query,
            new_index_size
        )

        return environment_state

    def _get_init_env_state(self):
        total_costs, plans_per_query, costs_per_query = \
            self.cost_evaluation.calculate_cost_and_plans(
                self.current_workload,
                self.current_indexes,
                store_size=True
            )

        self.current_costs = total_costs
        self.initial_costs = total_costs

        environment_state = self._get_env_state_dict(plans_per_query, costs_per_query)

        return environment_state

    def _get_env_state_dict(self, plans_per_query, costs_per_query, new_index_size=None):
        return {
            "action_status": self.action_manager.current_actions_status,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            "new_index_size": new_index_size,
            "plans_per_query": plans_per_query,
            "costs_per_query": costs_per_query
        }

    def _get_info(self):
        return {"action_mask": self.valid_actions}

    def render(self, mode="human"):
        logging.warning("render() was called")
        pass

    def close(self):
        logging.warning("close() was called")

