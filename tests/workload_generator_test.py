import unittest

from pg_index_advisor.schema.structures import View, Query
from pg_index_advisor.workload_generator import WorkloadGenerator
from pg_index_advisor.schema.schema import Table, Column
from tests.resources.constants import *


class WorkloadGeneratorTest(unittest.TestCase):
    queries = [
        [
            'SELECT v0."id", v0."version", v0."type", v0."status", v0."submitted_by", v0."vehicle_id" FROM "vehicle_application" AS v0 WHERE ((v0."vehicle_id" = \'8e31e5551a1a6c715a24d94cda164981cd2f74a71cb33adc034facefcd67f488\') AND (v0."submitted_by" = 2000042467))',
            'SELECT v0."id", v0."version", v0."type", v0."status", v0."submitted_by", v0."vehicle_id" FROM "vehicle_application" AS v0 WHERE ((v0."vehicle_id" = \'24fc70f21c09b163a2eabd8db0872ccc8a79b024a1f95ae6b65b8db98448a433\') AND (v0."submitted_by" = 2000148603))',
            'SELECT v0."id", v0."version", v0."type", v0."status", v0."submitted_by", v0."vehicle_id" FROM "vehicle_application" AS v0 WHERE ((v0."vehicle_id" = \'c88acaed699d71b994277e0931da31fb93322dc16772b411ba0c2a96b0b7337e\') AND (v0."submitted_by" = 2000188607))'
        ],
        [
            'SELECT a0."vehicle_id", a0."company_id", a0."version" FROM "accepted_vehicle_application" AS a0 WHERE (a0."vehicle_id" = \'8e31e5551a1a6c715a24d94cda164981cd2f74a71cb33adc034facefcd67f488\')',
            'SELECT a0."vehicle_id", a0."company_id", a0."version" FROM "accepted_vehicle_application" AS a0 WHERE (a0."vehicle_id" = \'c88acaed699d71b994277e0931da31fb93322dc16772b411ba0c2a96b0b7337e\')',
            'SELECT a0."vehicle_id", a0."company_id", a0."version" FROM "accepted_vehicle_application" AS a0 WHERE (a0."vehicle_id" = \'24fc70f21c09b163a2eabd8db0872ccc8a79b024a1f95ae6b65b8db98448a433\')'
        ],
        [
            'SELECT a0."submitted_by", a0."company_id", a0."version" FROM "accepted_profile_application" AS a0 WHERE (a0."submitted_by" = 2000042467)',
            'SELECT a0."submitted_by", a0."company_id", a0."version" FROM "accepted_profile_application" AS a0 WHERE (a0."submitted_by" = 2000188607)'
        ]
    ]

    def setUp(self):
        random_seed = 1
        table = Table('user_application')
        self.indexable_columns = [
            Column('id', table=table),
            Column('version', table=table),
            Column('previous', table=table),
            Column('submitted_by', table=table),
            Column('submitted_at', table=table),
            Column('status', table=table),
            Column('type', table=table),
            Column('vehicle_id', table=table)
        ]
        not_indexable_columns = [
            Column('additional_details', table=table)
        ]
        views = [
            View('profile_application', profile_application_view),
            View('accepted_profile_application', accepted_profile_application_view),
            View('vehicle_application', vehicle_application_view),
            View('accepted_vehicle_application', accepted_vehicle_application_view),
        ]

        self.workload_generator = object.__new__(WorkloadGenerator)
        self.workload_generator._init_parameters(
            workload_config,
            self.indexable_columns + not_indexable_columns,
            views,
            db_config,
            random_seed,
            1,
            False,
            False
        )
        self.workload_generator.QUERY_PATH = 'resources'

    def test_retrieve_query_texts(self):
        queries = self.workload_generator._retrieve_query_texts()

        self.assertEqual(queries, self.queries, "Queries from workload generator differs from expected")

    def test_correct_query_classes(self):
        self.workload_generator.query_texts = self.workload_generator._retrieve_query_texts()
        number_of_query_classes = self.workload_generator._set_number_of_query_classes()

        self.assertEqual(number_of_query_classes, len(self.queries))

    def test_select_indexable_columns(self):
        self._init_queries()

        indexable_columns = self.workload_generator._select_indexable_columns()

        self.assertEqual(indexable_columns, self.indexable_columns)

    def test_generate_workload(self):
        pass

    def test_generate_random_workload_for_two_classes(self):
        self._init_queries()

        query_classes_per_workload = 2
        unknown_query_probability = None

        workload_tuple = self.workload_generator._generate_random_workload(
            query_classes_per_workload,
            unknown_query_probability
        )

        self.assertEqual(workload_tuple, ((1, 3), (4732, 5118)))

    def test_generate_random_workload_for_max_query_classes(self):
        self._init_queries()

        query_classes_per_workload = len(self.workload_generator.available_query_classes)
        unknown_query_probability = None

        workload_tuple = self.workload_generator._generate_random_workload(
            query_classes_per_workload,
            unknown_query_probability
        )

        self.assertEqual(workload_tuple, ((1, 3, 2), (4732, 5118, 7551)))

    def test_workloads_from_tuples(self):
        self._init_queries()

        query_classes_per_workload = 2
        workload_amount = 2

        workload_tuple = [
            self.workload_generator._generate_random_workload(query_classes_per_workload)
            for i
            in range(workload_amount)
        ]

        workload = self.workload_generator._workloads_from_tuples(workload_tuple)

        workload_queries = [w.queries for w in workload]
        workload_budget = [w.budget for w in workload]
        
        expected_queries = self._get_expected_queries_in_workload()

        self.assertEqual(workload_budget, [None for i in range(workload_amount)])
        for i in range(workload_amount):
            self.assertEqual([q.nr for q in workload_queries[i]], [q.nr for q in expected_queries[i]])
            self.assertEqual([q.text for q in workload_queries[i]], [q.text for q in expected_queries[i]])
            self.assertEqual([q.columns for q in workload_queries[i]], [q.columns for q in expected_queries[i]])
            self.assertEqual([q.frequency for q in workload_queries[i]], [q.frequency for q in expected_queries[i]])



    def _init_queries(self):
        self.workload_generator.query_texts = self.workload_generator._retrieve_query_texts()
        self.workload_generator.number_of_query_classes = \
            self.workload_generator._set_number_of_query_classes()
        self.workload_generator.available_query_classes = \
            set(range(1, self.workload_generator.number_of_query_classes + 1))
        
    def _get_expected_queries_in_workload(self):
        expected_queries = [
            [
                Query(
                    1,
                    'SELECT v0."id", v0."version", v0."type", v0."status", v0."submitted_by", v0."vehicle_id" FROM "vehicle_application" AS v0 WHERE ((v0."vehicle_id" = \'24fc70f21c09b163a2eabd8db0872ccc8a79b024a1f95ae6b65b8db98448a433\') AND (v0."submitted_by" = 2000148603))',
                    frequency=4732
                ),
                Query(
                    3,
                    'SELECT a0."submitted_by", a0."company_id", a0."version" FROM "accepted_profile_application" AS a0 WHERE (a0."submitted_by" = 2000188607)',
                    frequency=5118
                )
            ],
            [
                Query(
                    2,
                    'SELECT a0."vehicle_id", a0."company_id", a0."version" FROM "accepted_vehicle_application" AS a0 WHERE (a0."vehicle_id" = \'c88acaed699d71b994277e0931da31fb93322dc16772b411ba0c2a96b0b7337e\')',
                    frequency=7551
                ),
                Query(
                    1,
                    'SELECT v0."id", v0."version", v0."type", v0."status", v0."submitted_by", v0."vehicle_id" FROM "vehicle_application" AS v0 WHERE ((v0."vehicle_id" = \'c88acaed699d71b994277e0931da31fb93322dc16772b411ba0c2a96b0b7337e\') AND (v0."submitted_by" = 2000188607))',
                    frequency=9504
                )
            ]
        ]

        for sublist in expected_queries:
            for q in sublist:
                self.workload_generator._store_indexable_columns(q)
                
        return expected_queries

if __name__ == '__main__':
    unittest.main()
