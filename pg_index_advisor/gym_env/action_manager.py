import numpy as np
import copy
import logging

from gym import spaces

from index_selection_evaluation.selection.utils import b_to_mb


class ActionManager(object):
    def __init__(self, max_index_width, action_storage_consumptions):
        self.valid_actions = None
        self._remaining_valid_actions = None
        self.number_of_actions = None
        self.current_actions_status = None
        self.number_of_columns = 0
        self.current_combinations = set()
        self.indexable_column_combinations_flat = []
        self.column_to_idx = {}

        self.test_variable = None

        self.MAX_INDEX_WIDTH = max_index_width
        self.action_storage_consumptions = action_storage_consumptions

        self.FORBIDDEN_ACTION = -np.inf
        self.ALLOWED_ACTION = 0

    def get_action_space(self):
        return spaces.Discrete(self.number_of_actions)

    def get_initial_valid_actions(self, workload, budget):
        # 0 for actions not taken yet
        # 1 for single columns index present
        # 0.5 for two-column index present
        # 0.33 for three-column index present
        # ...
        self.current_actions_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.FORBIDDEN_ACTION for actions in range(self.number_of_actions)]
        # List of valid actions (combinations) IDs
        self._remaining_valid_actions = []

        self._valid_actions_based_on_workload(workload)
        self._valid_actions_based_on_budget(budget, current_storage_consumption=0)

        self.current_combinations = set()

        return np.array(self.valid_actions)

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        actions_index_width = len(self.indexable_column_combinations_flat[last_action])
        if actions_index_width == 1:
            self.current_actions_status[last_action] += 1
        else:
            combination_to_be_extended = self.indexable_column_combinations_flat[last_action][:-1]
            assert combination_to_be_extended in self.current_combinations

            status_value = 1 / actions_index_width

            last_action_back_column = self.indexable_column_combinations_flat[last_action][-1]
            last_action_back_column_idx = self.column_to_idx[last_action_back_column]
            self.current_actions_status[last_action_back_column_idx] += status_value

            self.current_combinations.remove(combination_to_be_extended)

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action)

        self._valid_actions_based_on_last_action(last_action)
        self._valid_actions_based_on_budget(budget, current_storage_consumption)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        if budget is None:
            return

        new_remaining_actions = []
        for action_idx in self._remaining_valid_actions:
            if b_to_mb(current_storage_consumption + self.action_storage_consumptions[action_idx]) > budget:
                self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
            else:
                new_remaining_actions.append(action_idx)

        self._remaining_valid_actions = new_remaining_actions

    def _valid_actions_based_on_last_action(self, last_action):
        raise NotImplementedError

    def _valid_actions_based_on_workload(self, workload):
        raise NotImplementedError


class MultiColumnIndexActionManager(ActionManager):
    """
    Action Manager for multi-column indexes.

    :param indexable_column_combination: List of [[single column indexes], [2-column combinations]...]
    :param indexable_column_combination_flat: List of [single column index, 2-column combination, ...]
    :param action_storage_consumption: List of index storage consumption [8192, 8192, 0, 0 ...]
    :param max_index_width: Max index width
    :param reenable_indexes: Flag for something that IDK
    """
    def __init__(
            self,
            indexable_column_combinations: list,
            indexable_column_combinations_flat: list,
            action_storage_consumption: list,
            max_index_width: int,
            reenable_indexes: bool
    ):
        ActionManager.__init__(
            self,
            max_index_width=max_index_width,
            action_storage_consumptions=action_storage_consumption
        )

        self.indexable_column_combinations = indexable_column_combinations
        self.indexable_column_combinations_flat = indexable_column_combinations_flat
        self.action_storage_consumptions = action_storage_consumption

        self.number_of_columns = len(self.indexable_column_combinations[0])
        self.number_of_actions = len(self.indexable_column_combinations_flat)

        # convert from list of tuples to list
        self.indexable_columns = list(map(
            lambda one_column_combination: one_column_combination[0],
            self.indexable_column_combinations[0]
        ))

        self.REENABLE_INDEXES = reenable_indexes

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

    def _valid_actions_based_on_last_action(self, last_action):
        """
        Update valid actions list based on the last action.

        :param last_action: last applied column combination, for example: (C user_application.inserted_at,)
        :return: None
        """

        last_combination = self.indexable_column_combinations_flat[last_action]
        last_combination_length = len(last_combination)

        # Allow dependent combinations
        if last_combination_length != self.MAX_INDEX_WIDTH:
            for column_combination_idx in self.candidate_dependent_map[last_combination]:
                column_combination = self.indexable_column_combinations_flat[column_combination_idx]
                possible_extended_column = column_combination[-1]

                # wl_indexable_columns - columns from workload
                if possible_extended_column not in self.wl_indexable_columns:
                    continue
                # Already applied
                if column_combination in self.current_combinations:
                    continue

                self._remaining_valid_actions.append(column_combination_idx)
                self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

        # Disable invalid combinations
        for column_combination_idx in copy.copy(self._remaining_valid_actions):
            column_combination = self.indexable_column_combinations_flat[column_combination_idx]
            column_combination_length = len(column_combination)

            if column_combination_length == 1:
                continue

            if column_combination_length != last_combination_length:
                continue

            if last_combination[:-1] != column_combination[:-1]:
                continue

            if column_combination_idx in self._remaining_valid_actions:
                self._remaining_valid_actions.remove(column_combination_idx)
            self.valid_actions[column_combination_idx] = self.FORBIDDEN_ACTION

        # TODO: какой смысл помечать родительские индексы валидными?
        if self.REENABLE_INDEXES and last_combination_length > 1:
            last_combination_without_extension = last_combination[:-1]

            if len(last_combination_without_extension) > 1:
                # The presence of last_combination_without_extension's parent is a precondition
                last_combination_without_extension_parent = last_combination_without_extension[:-1]
                if last_combination_without_extension_parent not in self.current_combinations:
                    return

            column_combination_idx = self.column_combination_to_idx[str(last_combination_without_extension)]
            self._remaining_valid_actions.append(column_combination_idx)
            self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

            logging.debug(f"REENABLE_INDEXES: {last_combination_without_extension} after {last_combination}")

    def _valid_actions_based_on_workload(self, workload):
        indexable_columns = workload.indexable_columns(return_sorted=False)
        indexable_columns = indexable_columns & frozenset(self.indexable_columns)
        self.wl_indexable_columns = indexable_columns

        for indexable_column in indexable_columns:
            # only single column indexes
            for column_combination_idx, column_combination in enumerate(
                self.indexable_column_combinations[0]
            ):
                if indexable_column == column_combination[0]:
                    self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION
                    self._remaining_valid_actions.append(column_combination_idx)

        assert np.count_nonzero(np.array(self.valid_actions) == self.ALLOWED_ACTION) == len(
            indexable_columns
        ), "Valid actions mismatch indexable columns"


