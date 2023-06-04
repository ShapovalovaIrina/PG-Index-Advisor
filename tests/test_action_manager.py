import unittest
import copy
import logging

from pg_index_advisor.gym_env.action_manager import MultiColumnIndexActionManager, FORBIDDEN_ACTION, ALLOW_TO_DELETE, ALLOW_TO_CREATE
from pg_index_advisor.database.structures import Table, Column, RealIndex, PotentialIndex, Query, Workload


class TestActionManager(unittest.TestCase):

    def setUp(self) -> None:
        self.action_manager = object.__new__(MultiColumnIndexActionManager)
        self.action_manager.FORBIDDEN_ACTION = FORBIDDEN_ACTION
        self.action_manager.ALLOW_TO_CREATE = ALLOW_TO_CREATE
        self.action_manager.ALLOW_TO_DELETE = ALLOW_TO_DELETE

        self.table = Table('user_application')
        self.indexable_column_combinations = [
            [
                (Column('id', table=self.table),),
                (Column('status', table=self.table),),
                (Column('type', table=self.table),)
            ],
            [
                (Column('id', table=self.table), Column('status', table=self.table)),
                (Column('id', table=self.table), Column('type', table=self.table)),
                (Column('status', table=self.table), Column('id', table=self.table)),
                (Column('status', table=self.table), Column('type', table=self.table)),
                (Column('type', table=self.table), Column('id', table=self.table)),
                (Column('type', table=self.table), Column('status', table=self.table)),
            ]
        ]
        self.indexable_column_combinations_flat = [
            item
            for sublist
            in self.indexable_column_combinations
            for item
            in sublist
        ]
        self.action_storage_consumption = [8192 for _ in self.indexable_column_combinations_flat]

        indexes_dict = [
            ('user_application_id_index', 'user_application', ['id']),
            ('user_application_status_version_index', 'user_application', ['id', 'status']),
            ('user_application_status_version_index', 'user_application', ['status', 'version'])
        ]

        self.initial_indexes = [
            RealIndex(table, index_name, 0, columns, 8192, False)
            for index_name, table, columns
            in indexes_dict
        ]


    def test_init_candidate_dependent_map(self):
        max_index_width = 3
        self.action_manager.indexable_column_combinations_flat = [
            (Column('id',     table=self.table),),
            (Column('status', table=self.table),),
            (Column('type',   table=self.table),),
            (Column('id',     table=self.table), Column('status', table=self.table)),
            (Column('id',     table=self.table), Column('type',   table=self.table)),
            (Column('status', table=self.table), Column('id',     table=self.table)),
            (Column('status', table=self.table), Column('type',   table=self.table)),
            (Column('type',   table=self.table), Column('id',     table=self.table)),
            (Column('type',   table=self.table), Column('status', table=self.table)),
            (Column('id',     table=self.table), Column('status', table=self.table), Column('type',   table=self.table)),
            (Column('id',     table=self.table), Column('type',   table=self.table), Column('status', table=self.table)),
            (Column('status', table=self.table), Column('id',     table=self.table), Column('type',   table=self.table)),
            (Column('status', table=self.table), Column('type',   table=self.table), Column('id',     table=self.table)),
            (Column('type',   table=self.table), Column('id',     table=self.table), Column('status', table=self.table)),
            (Column('type',   table=self.table), Column('status', table=self.table), Column('id',     table=self.table))
        ]

        self.action_manager._init_candidate_dependent_map(max_index_width)

        expected_dependent_map = {
            (Column('id',     table=self.table),): [3, 4],
            (Column('status', table=self.table),): [5, 6],
            (Column('type',   table=self.table),): [7, 8],
            (Column('id',     table=self.table), Column('status', table=self.table)): [9],
            (Column('id',     table=self.table), Column('type',   table=self.table)): [10],
            (Column('status', table=self.table), Column('id',     table=self.table)): [11],
            (Column('status', table=self.table), Column('type',   table=self.table)): [12],
            (Column('type',   table=self.table), Column('id',     table=self.table)): [13],
            (Column('type',   table=self.table), Column('status', table=self.table)): [14]
        }

        self.assertEqual(self.action_manager.candidate_dependent_map, expected_dependent_map)

    def test_save_initial_indexes_combinations(self):
        self.action_manager.number_of_actions = 5
        self.action_manager.indexable_column_combinations_flat = self.indexable_column_combinations_flat

        self.action_manager._save_initial_indexes_combinations(self.initial_indexes)

        expected_initial_combinations = [
            (Column('id', table=self.table),),
            (Column('id', table=self.table), Column('status', table=self.table))
        ]
        expected_initial_indexes = [PotentialIndex(columns) for columns in expected_initial_combinations]

        self.assertEqual(self.action_manager.initial_combinations, set(expected_initial_combinations))
        self.assertEqual(self.action_manager.initial_indexes, set(expected_initial_indexes))

    def test_get_initial_valid_actions(self):
        action_manager = self.init_action_manager()

        workload = self.get_workload(query_columns_num=3)
        budget_mb = 10
        storage_consumption_b = 0

        action_manager.get_initial_valid_actions(workload, budget_mb, storage_consumption_b)

        valid_actions = [FORBIDDEN_ACTION for _ in range(action_manager.number_of_actions)]

        valid_actions[0] = ALLOW_TO_DELETE # (id)
        valid_actions[1] = ALLOW_TO_CREATE # (status)
        valid_actions[2] = ALLOW_TO_CREATE # (type)
        valid_actions[3] = ALLOW_TO_DELETE # (id, status)
        valid_actions[4] = ALLOW_TO_CREATE # (id, type)

        self.assertEqual(valid_actions, action_manager.valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 1, 2, 3, 4])


    def test_update_valid_actions_with_create_action(self):
        action_manager = self.init_action_manager()

        workload = self.get_workload(query_columns_num=3)
        budget_mb = 10
        storage_consumption_b = 0

        action_manager.get_initial_valid_actions(workload, budget_mb, storage_consumption_b)

        valid_actions = [FORBIDDEN_ACTION for _ in range(action_manager.number_of_actions)]

        valid_actions[0] = ALLOW_TO_DELETE  # (id)
        valid_actions[1] = ALLOW_TO_CREATE  # (status)
        valid_actions[2] = ALLOW_TO_CREATE  # (type)
        valid_actions[3] = ALLOW_TO_DELETE  # (id, status)
        valid_actions[4] = ALLOW_TO_CREATE  # (id, type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 1, 2, 3, 4])

        action_manager.update_valid_actions(1, budget_mb, storage_consumption_b)

        valid_actions[0] = ALLOW_TO_DELETE  # (id)
        valid_actions[1] = FORBIDDEN_ACTION # (status)
        valid_actions[2] = ALLOW_TO_CREATE  # (type)
        valid_actions[3] = ALLOW_TO_DELETE  # (id, status)
        valid_actions[4] = ALLOW_TO_CREATE  # (id, type)
        valid_actions[5] = ALLOW_TO_CREATE  # (status, is)
        valid_actions[6] = ALLOW_TO_CREATE  # (status, type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 2, 3, 4, 5, 6])

    def test_update_valid_actions_with_delete_action(self):
        action_manager = self.init_action_manager()

        workload = self.get_workload(query_columns_num=3)
        budget_mb = 10
        storage_consumption_b = 0

        action_manager.get_initial_valid_actions(workload, budget_mb, storage_consumption_b)

        valid_actions = [FORBIDDEN_ACTION for _ in range(action_manager.number_of_actions)]

        valid_actions[0] = ALLOW_TO_DELETE  # (id)
        valid_actions[1] = ALLOW_TO_CREATE  # (status)
        valid_actions[2] = ALLOW_TO_CREATE  # (type)
        valid_actions[3] = ALLOW_TO_DELETE  # (id, status)
        valid_actions[4] = ALLOW_TO_CREATE  # (id, type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 1, 2, 3, 4])

        action_manager.update_valid_actions(0, budget_mb, storage_consumption_b)

        valid_actions[0] = FORBIDDEN_ACTION  # (id)
        valid_actions[1] = ALLOW_TO_CREATE # (status)
        valid_actions[2] = ALLOW_TO_CREATE  # (type)
        valid_actions[3] = ALLOW_TO_DELETE  # (id, status)
        valid_actions[4] = ALLOW_TO_CREATE  # (id, type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [1, 2, 3, 4])

    def test_valid_actions_based_on_initial_indexes(self):
        action_manager = self.init_action_manager()
        self.get_initial_valid_actions_preset(action_manager)

        valid_actions = [FORBIDDEN_ACTION for _ in range(action_manager.number_of_actions)]

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(action_manager._remaining_valid_actions, [])

        action_manager._valid_actions_based_on_initial_indexes()

        valid_actions[0] = ALLOW_TO_DELETE # (id)
        valid_actions[3] = ALLOW_TO_DELETE # (id, status)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 3])

    def test_valid_actions_based_on_workload(self):
        action_manager = self.init_action_manager()
        self.get_initial_valid_actions_preset(action_manager)

        valid_actions = [FORBIDDEN_ACTION for _ in range(action_manager.number_of_actions)]

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(action_manager._remaining_valid_actions, [])

        action_manager._valid_actions_based_on_workload(self.get_workload(query_columns_num=2))

        valid_actions[0] = ALLOW_TO_CREATE  # (id)
        valid_actions[2] = ALLOW_TO_CREATE  # (type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 2])

    def test_valid_actions_based_on_budget(self):
        action_manager = self.init_action_manager()
        self.get_initial_valid_actions_preset(action_manager)

        valid_actions = [FORBIDDEN_ACTION for _ in range(action_manager.number_of_actions)]

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(action_manager._remaining_valid_actions, [])

        action_manager._valid_actions_based_on_workload(self.get_workload(query_columns_num=2))

        valid_actions[0] = ALLOW_TO_CREATE  # (id)
        valid_actions[2] = ALLOW_TO_CREATE  # (type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 2])

        action_manager._valid_actions_based_on_budget(
            10,
            current_storage_consumption=1024*1024*10*2
        )

        valid_actions[0] = FORBIDDEN_ACTION  # (id)
        valid_actions[2] = FORBIDDEN_ACTION  # (type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(action_manager._remaining_valid_actions, [])


    def test_valid_actions_based_on_last_action(self):
        action_manager = self.init_action_manager()
        self.get_initial_valid_actions_preset(action_manager)

        valid_actions = [FORBIDDEN_ACTION for _ in range(action_manager.number_of_actions)]

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(action_manager._remaining_valid_actions, [])

        action_manager._valid_actions_based_on_workload(self.get_workload(query_columns_num=3))

        valid_actions[0] = ALLOW_TO_CREATE  # (id)
        valid_actions[1] = ALLOW_TO_CREATE  # (status)
        valid_actions[2] = ALLOW_TO_CREATE  # (type)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 1, 2])

        action_manager._valid_actions_based_on_last_action(2)

        valid_actions[7] = ALLOW_TO_CREATE  # (type, id)
        valid_actions[8] = ALLOW_TO_CREATE  # (type, status)

        self.assertEqual(action_manager.valid_actions, valid_actions)
        self.assertEqual(sorted(action_manager._remaining_valid_actions), [0, 1, 2, 7, 8])

    @staticmethod
    def get_initial_valid_actions_preset(action_manager):
        action_manager.current_actions_status = [
            0
            for action
            in range(action_manager.number_of_columns)
        ]

        action_manager.valid_actions = [
            action_manager.FORBIDDEN_ACTION
            for actions
            in range(action_manager.number_of_actions)
        ]
        action_manager._remaining_valid_actions = []
        action_manager.combinations_to_create = set()
        action_manager.combinations_to_delete = copy.copy(action_manager.initial_combinations)

    def get_workload(self, query_columns_num=2):
        assert query_columns_num in [2, 3]

        if query_columns_num == 2:
            query_sql = 'SELECT v0."id" FROM "vehicle_application" AS v0'
            query_columns = [
                Column('id', table=self.table),
                Column('type', table=self.table)
            ]
        else:
            query_sql = 'SELECT v0."id", v0."status" FROM "vehicle_application" AS v0'
            query_columns = [
                Column('id', table=self.table),
                Column('status', table=self.table),
                Column('type', table=self.table)
            ]

        query = Query(1, query_sql, frequency=4732)
        query.columns = query_columns

        return Workload([query], 1000)

    def init_action_manager(self, max_index_width=2):
        return MultiColumnIndexActionManager(
            self.indexable_column_combinations,
            self.indexable_column_combinations_flat,
            self.action_storage_consumption,
            self.initial_indexes,
            max_index_width
        )


