import json

import numpy as np
import copy
import logging

from gym import spaces
from typing import List, Tuple, Set

from index_selection_evaluation.selection.utils import b_to_mb
from pg_index_advisor.schema.structures import RealIndex, PotentialIndex, Column
from pg_index_advisor.utils import add_if_not_exists, remove_if_exists, index_from_column_combination

FORBIDDEN_ACTION = 0
ALLOW_TO_CREATE = 1
ALLOW_TO_DELETE = 2

class MultiColumnIndexActionManager(object):
    """
    Action Manager for multi-column indexes.

    :param indexable_column_combination: List of [[single column indexes], [2-column combinations]...]
    :param indexable_column_combination_flat: List of [single column index, 2-column combination, ...]
    :param action_storage_consumption: List of index storage consumption [8192, 8192, 0, 0 ...]
    :param max_index_width: Max index width
    :param reenable_indexes: Flag for something that IDK
    """

    indexable_column_combinations_flat: List[Tuple[Column]]
    initial_indexes: Set[PotentialIndex]
    initial_combinations: Set[Tuple[Column]]

    combinations_to_create: Set[Tuple[Column]]
    combinations_to_delete: Set[Tuple[Column]]
    
    def __init__(
            self,
            indexable_column_combinations: List[List[Tuple[Column]]],
            indexable_column_combinations_flat: List[Tuple[Column]],
            action_storage_consumption: List[int],
            initial_indexes: List[RealIndex],
            max_index_width: int
    ):
        self.valid_actions = []
        self._remaining_valid_actions = []
        self.current_actions_status = None
        self.combinations_to_create = set()
        self.combinations_to_delete = set()
        self.applied_actions = []

        self.MAX_INDEX_WIDTH = max_index_width
        
        self.FORBIDDEN_ACTION = FORBIDDEN_ACTION
        self.ALLOW_TO_CREATE = ALLOW_TO_CREATE
        self.ALLOW_TO_DELETE = ALLOW_TO_DELETE

        self.indexable_column_combinations = indexable_column_combinations
        self.indexable_column_combinations_flat = indexable_column_combinations_flat
        self.create_action_storage_consumptions = action_storage_consumption

        self.number_of_columns = len(self.indexable_column_combinations[0])
        self.number_of_actions = len(self.indexable_column_combinations_flat)

        # convert from list of tuples to list
        self.indexable_columns = list(map(
            lambda one_column_combination: one_column_combination[0],
            self.indexable_column_combinations[0]
        ))

        self.column_to_idx = {}
        for idx, column in enumerate(self.indexable_columns):
            self.column_to_idx[column] = idx

        """
        {
            "(C user_application.inserted_at,)": 0,
            "(C user_application.updated_at,)": 1,
            ...
            "(C user_application.prepared_at, C user_application.vehicle_id)": 16,
            "(C user_application.channel_version_id, C user_application.previous)": 17,
            ...
        }
        """
        self.column_combination_to_idx = {}
        for idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            cc = str(column_combination)
            self.column_combination_to_idx[cc] = idx

        """
        Dictionary of combination and it's dependent combinations.
        
        One-column index unlocks two-column indexes.
        Two-column index unlocks three-columns indexes.
        And so on.
        
        {
            (C user_application.inserted_at,): [21, 22, 53, 68],
            ...
            (C user_application.inserted_at, C user_application.updated_at): [673, 872, 972],
            ...
        }
        """
        self._init_candidate_dependent_map(max_index_width)

        self._save_initial_indexes_combinations(initial_indexes)
        
    def get_action_space(self):
        return spaces.Discrete(self.number_of_actions)

    def get_initial_valid_actions(self, workload, budget, initial_storage_consumption):
        # 0 for actions not taken yet
        # 1 for single columns index present
        # 0.5 for two-column index present
        # 0.33 for three-column index present
        # ...
        # TODO: учесть индексы, которые созданы
        self.current_actions_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.FORBIDDEN_ACTION for actions in range(self.number_of_actions)]
        self._remaining_valid_actions = []
        # TODO: надо ли добавить существующие индексы в self.combinations_to_create?
        self.combinations_to_create = set()
        self.combinations_to_delete = copy.copy(self.initial_combinations)

        self._valid_actions_based_on_initial_indexes()
        logging.info('finish _valid_actions_based_on_initial_indexes')
        self._valid_actions_based_on_workload(workload)
        logging.info('finish _valid_actions_based_on_workload')
        self._valid_actions_based_on_budget(
            budget,
            current_storage_consumption=initial_storage_consumption
        )
        logging.info('finish _valid_actions_based_on_budget')

        return np.array(self.valid_actions)
    
    def update_valid_actions(self, action_idx, budget, current_storage_consumption):
        self._update_asserts(action_idx)

        action = self.valid_actions[action_idx]
        column_combination = self.indexable_column_combinations_flat[action_idx]
        action_index_width = len(column_combination)

        if action_index_width == 1:
            if action == self.ALLOW_TO_CREATE:
                self.current_actions_status[action_idx] += 1
            if action == self.ALLOW_TO_DELETE:
                self.current_actions_status[action_idx] -= 1

        else:
            combination_to_be_extended = column_combination[:-1]

            if action == self.ALLOW_TO_CREATE:
                assert combination_to_be_extended in self.combinations_to_create \
                       or combination_to_be_extended in self.initial_combinations, f"""
                       Action combination {combination_to_be_extended} doesn't present in 
                       action manager combinations to create 
                       or initial indexes.
                       State:
                       {json.dumps(self._get_state_for_logging(), indent=2)}
                       """

            status_value = 1 / action_index_width

            last_action_back_column = column_combination[-1]
            last_action_back_column_idx = self.column_to_idx[last_action_back_column]

            if action == self.ALLOW_TO_CREATE:
                self.current_actions_status[last_action_back_column_idx] += status_value
                remove_if_exists(self.combinations_to_create, combination_to_be_extended)
            if action == self.ALLOW_TO_DELETE:
                self.current_actions_status[last_action_back_column_idx] -= status_value

        if action == self.ALLOW_TO_CREATE:
            self.combinations_to_create.add(column_combination)
        if action == self.ALLOW_TO_DELETE:
            self.combinations_to_delete.remove(column_combination)

        self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(action_idx)
        self._log_action(action_idx, "forbid", "update_valid_actions")

        self._valid_actions_based_on_last_action(action_idx)
        self._valid_actions_based_on_budget(budget, current_storage_consumption)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

    def _update_asserts(self, action_idx):
        action = self.valid_actions[action_idx]

        assert action in [self.ALLOW_TO_CREATE, self.ALLOW_TO_DELETE]

        if action == self.ALLOW_TO_CREATE:
            combination = self.indexable_column_combinations_flat[action_idx]
            assert combination not in self.combinations_to_create, \
                f"""
                Expected action {combination} not to \
                be in current created combinations.
                State:
                {json.dumps(self._get_state_for_logging(), indent=2)}
                """

        if action == self.ALLOW_TO_DELETE:
            combination = self.indexable_column_combinations_flat[action_idx]
            assert combination in self.combinations_to_delete, \
                f"""
                Expected action {combination} to \
                be in current combinations to delete.
                State:
                {json.dumps(self._get_state_for_logging(), indent=2)}
                """

    # UPDATE VALID ACTIONS BASED ON CRITERION

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        """
        Calls on getting initial action space and updating action space after the action taken.
        """
        if budget is None:
            return

        new_remaining_actions = []
        for action_idx in self._remaining_valid_actions:
            storage_consumption_with_action = None

            action = self.valid_actions[action_idx]

            assert action in [self.ALLOW_TO_CREATE, self.ALLOW_TO_DELETE]

            if action == self.ALLOW_TO_CREATE:
                storage_consumption_with_action = \
                    current_storage_consumption + self.create_action_storage_consumptions[action_idx]

            if action == self.ALLOW_TO_DELETE:
                column_combination = self.indexable_column_combinations_flat[action_idx]
                for initial_index in self.initial_indexes:
                    if initial_index.columns == column_combination:
                        storage_consumption_with_action = \
                            current_storage_consumption - \
                            initial_index.estimated_size

                assert storage_consumption_with_action is not None, \
                    f"Index {column_combination} doesn't present in action manager initial indexes"

            if b_to_mb(storage_consumption_with_action) > budget:
                self._log_action(action_idx, "forbid", "_valid_actions_based_on_budget")
                self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
            else:
                new_remaining_actions.append(action_idx)

        self._remaining_valid_actions = new_remaining_actions

    def _valid_actions_based_on_last_action(self, action_idx, setup=False):
        """
        Update valid actions list based on the last action.

        :param action_idx: last applied column combination, for example: (C user_application.inserted_at,)
        :return: None
        """

        combination = self.indexable_column_combinations_flat[action_idx]
        combination_length = len(combination)

        # Allow dependent combinations
        if combination_length != self.MAX_INDEX_WIDTH:
            for column_combination_idx in self.candidate_dependent_map[combination]:
                column_combination = self.indexable_column_combinations_flat[column_combination_idx]
                possible_extended_column = column_combination[-1]

                # TODO: понаблюдать, все ли окей
                if self.valid_actions[column_combination_idx] == self.ALLOW_TO_DELETE:
                #     self._valid_actions_based_on_last_action(column_combination_idx)
                    continue

                # wl_indexable_columns - columns from workload
                if possible_extended_column not in self.wl_indexable_columns:
                    continue

                # Already applied
                if column_combination in self.combinations_to_create:
                    continue

                self._log_action(
                    column_combination_idx,
                    "allow dependent",
                    "_valid_actions_based_on_last_action"
                )
                add_if_not_exists(self._remaining_valid_actions, column_combination_idx)
                self.valid_actions[column_combination_idx] = self.ALLOW_TO_CREATE

        # Disable invalid combinations
        # TODO: действительно ли это необходимо?
        for column_combination_idx in copy.copy(self._remaining_valid_actions):
            column_combination = self.indexable_column_combinations_flat[column_combination_idx]
            column_combination_length = len(column_combination)

            if self.valid_actions[column_combination_idx] == self.ALLOW_TO_DELETE:
                continue

            if column_combination_length == 1:
                continue

            if column_combination_length != combination_length:
                continue

            if combination[:-1] != column_combination[:-1]:
                continue

            self._log_action(column_combination_idx, "forbid", "_valid_actions_based_on_last_action")
            remove_if_exists(self._remaining_valid_actions, column_combination_idx)
            self.valid_actions[column_combination_idx] = self.FORBIDDEN_ACTION

    def _valid_actions_based_on_workload(self, workload):
        """
        Calls on getting initial action space.
        """
        indexable_columns = workload.indexable_columns(return_sorted=False)
        indexable_columns = indexable_columns & frozenset(self.indexable_columns)
        self.wl_indexable_columns = indexable_columns

        for indexable_column in indexable_columns:
            # only single column indexes
            for column_combination_idx, column_combination in enumerate(
                self.indexable_column_combinations[0]
            ):
                # If index is created by user, so consider is as taken action
                if self.valid_actions[column_combination_idx] == self.ALLOW_TO_DELETE:
                    self._valid_actions_based_on_last_action(column_combination_idx)
                    continue

                if indexable_column == column_combination[0]:
                    self._log_action(column_combination_idx, "allow", "_valid_actions_based_on_workload")
                    add_if_not_exists(self._remaining_valid_actions, column_combination_idx)
                    self.valid_actions[column_combination_idx] = self.ALLOW_TO_CREATE

        assert np.count_nonzero(np.array(self.valid_actions) == self.ALLOW_TO_CREATE) >= len(
            indexable_columns
        ), "Valid actions mismatch indexable columns"
        
    def _valid_actions_based_on_initial_indexes(self):
        """
        Calls on getting initial action space.
        """
        for idx, action, applied_action in \
                zip(range(self.number_of_actions), self.valid_actions, self.applied_actions):
            if applied_action == self.ALLOW_TO_DELETE:
                self.valid_actions[idx] = ALLOW_TO_DELETE
                add_if_not_exists(self._remaining_valid_actions, idx)

    # INITIALIZE HELPERS
        
    def _save_initial_indexes_combinations(self, initial_indexes: List[RealIndex]):
        self.initial_combinations = set()
        self.initial_indexes = set()
        self.applied_actions = [self.FORBIDDEN_ACTION for actions in range(self.number_of_actions)]

        for action_idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            supposed_index = index_from_column_combination(column_combination)

            for index in initial_indexes:
                if not index.is_primary and \
                        index.table == supposed_index.table and \
                        index.columns == supposed_index.columns:
                    assert index.size != 0, f"Index {supposed_index} doesn't exist"

                    new_index = PotentialIndex(
                        column_combination,
                        estimated_size=index.size,
                        hypopg_name=index.name,
                        hypopg_oid=index.oid
                    )

                    self.applied_actions[action_idx] = self.ALLOW_TO_DELETE
                    self.initial_combinations.add(tuple(column_combination))
                    self.initial_indexes.add(new_index)

                    logging.warning(
                        f"Save existing index {new_index} in action manager and "
                        f"mark combination as applied "
                        f"because it is present in initial indexes"
                    )

    def _init_candidate_dependent_map(self, max_index_width):
        self.candidate_dependent_map = {}
        for column_combination in self.indexable_column_combinations_flat:
            if len(column_combination) > max_index_width - 1:
                continue
            self.candidate_dependent_map[column_combination] = []

        for column_combination_idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            if len(column_combination) < 2:
                continue
            dependent_of = column_combination[:-1]
            self.candidate_dependent_map[dependent_of].append(column_combination_idx)

    # LOG HELPERS

    def _log_action(self, column_combination_idx, action, function):
        logging.debug(
            f"{function}: " +
            f"{action} action {self.indexable_column_combinations_flat[column_combination_idx]}"
        )

    def _get_state_for_logging(self):
        return {
            'Current created combinations': ', '.join(map(str, self.combinations_to_create)),
            'Current combinations to delete': ', '.join(map(str, self.combinations_to_delete)),
            'Remaining valid actions': self._remaining_valid_actions
        }

