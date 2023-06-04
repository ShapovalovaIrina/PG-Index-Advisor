import unittest

from pg_index_advisor import utils
from pg_index_advisor.database.structures import Table, Column

from tests.resources.constants import *


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.table = Table('user_application')
        self.columns = [
            Column('id',     table=self.table),
            Column('status', table=self.table),
            Column('type',   table=self.table)
        ]

    def test_create_column_permutation_indexes_with_max_width_2(self):
        max_width = 2

        column_permutations = utils.create_column_permutation_indexes(
            self.columns,
            max_width
        )
        column_permutations = [sorted(permutation) for permutation in column_permutations]

        expected_permutations = [
            [
                (Column('id',     table=self.table), ),
                (Column('status', table=self.table), ),
                (Column('type',   table=self.table), )
            ],
            [
                (Column('id',     table=self.table), Column('status', table=self.table)),
                (Column('id',     table=self.table), Column('type',   table=self.table)),
                (Column('status', table=self.table), Column('id',     table=self.table)),
                (Column('status', table=self.table), Column('type',   table=self.table)),
                (Column('type',   table=self.table), Column('id',     table=self.table)),
                (Column('type',   table=self.table), Column('status', table=self.table))
            ]
        ]
        expected_permutations = [sorted(permutation) for permutation in expected_permutations]

        self.assertEqual(column_permutations, expected_permutations)

    def test_create_column_permutation_indexes_with_max_width_3(self):
        max_width = 3

        column_permutations = utils.create_column_permutation_indexes(
            self.columns,
            max_width
        )
        column_permutations = [sorted(permutation) for permutation in column_permutations]

        expected_permutations = [
            [
                (Column('id',     table=self.table), ),
                (Column('status', table=self.table), ),
                (Column('type',   table=self.table), )
            ],
            [
                (Column('id',     table=self.table), Column('status', table=self.table)),
                (Column('id',     table=self.table), Column('type',   table=self.table)),
                (Column('status', table=self.table), Column('id',     table=self.table)),
                (Column('status', table=self.table), Column('type',   table=self.table)),
                (Column('type',   table=self.table), Column('id',     table=self.table)),
                (Column('type',   table=self.table), Column('status', table=self.table))
            ],
            [
                (Column('id',     table=self.table), Column('status', table=self.table), Column('type',   table=self.table)),
                (Column('id',     table=self.table), Column('type',   table=self.table), Column('status', table=self.table)),
                (Column('status', table=self.table), Column('id',     table=self.table), Column('type',   table=self.table)),
                (Column('status', table=self.table), Column('type',   table=self.table), Column('id',     table=self.table)),
                (Column('type',   table=self.table), Column('id',     table=self.table), Column('status', table=self.table)),
                (Column('type',   table=self.table), Column('status', table=self.table), Column('id',     table=self.table))
            ]
        ]
        expected_permutations = [sorted(permutation) for permutation in expected_permutations]

        self.assertEqual(column_permutations, expected_permutations)

    def test_predict_index_sizes(self):
        columns_combinations = [
            (Column('id', table=self.table),),
            (Column('status', table=self.table),),
            (Column('type', table=self.table),)
        ]
        predicted_sizes = utils.predict_index_sizes(columns_combinations, db_config)

        self.assertEqual(predicted_sizes, [8192, 8192, 8192])
