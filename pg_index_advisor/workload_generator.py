import random
import logging
import os
import math

import numpy as np

from typing import List, Tuple
from sql_metadata import Parser
from schema.structures import Query, Workload


QUERY_PATH = "query_examples"


class WorkloadGenerator(object):
    def __init__(self,
                 workload_config,
                 columns,
                 views,
                 db_config,
                 random_seed,
                 experiment_id,
                 filter_utilized_columns,
                 logging_mode
                 ):
        # For create view statement differentiation
        self.experiment_id = experiment_id
        self.filter_utilized_columns = filter_utilized_columns
        self.logging_mode = logging_mode

        self.rnd = random.Random()
        self.rnd.seed(random_seed)
        self.np_rnd = np.random.default_rng(seed=random_seed)

        self.schema_columns = columns
        self.db_config = db_config
        self.views = views

        self.number_of_query_classes = self._set_number_of_query_classes()
        self.excluded_query_classes = set()  # Empty set
        self.varying_frequencies = workload_config["varying_frequencies"]

        # self.query_texts is list of lists.
        # Outer list for query classes, inner list for instances of this class.
        self.query_texts = self._retrieve_query_texts()
        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes - self.excluded_query_classes

        self.globally_indexable_columns = self._select_indexable_columns()

        if logging_mode == "verbose":
            print(
                f"Schema columns amount: {len(self.schema_columns)}",
                *self.schema_columns,
                sep='\n'
            )
            print(
                f"Globally indexable columns amount: {len(self.globally_indexable_columns)}",
                *self.globally_indexable_columns,
                sep='\n'
            )

        query_classes_per_workload = workload_config["size"]
        training_instances = workload_config["training_instances"]
        validation_instances = workload_config["validation_testing"]["number_of_workloads"]
        test_instances = workload_config["validation_testing"]["number_of_workloads"]
        self.wl_validation = [None]
        self.wl_testing = [None]

        # TODO: Why training is list, validation and testing is list of lists???
        self.wl_training, self.wl_validation[0], self.wl_testing[0] = \
            self._generate_workloads(
                training_instances,
                validation_instances,
                test_instances,
                query_classes_per_workload
            )

        if self.logging_mode == "verbose":
            logging.info(f"Training workload length {len(self.wl_training)}")
            logging.info(f"Validation workload length {len(self.wl_validation[0])}")
            logging.info(f"Testing workload length {len(self.wl_testing[0])}")

        logging.info("Finished generating workloads.")



    @staticmethod
    def _set_number_of_query_classes():
        files_amount = len([
            entry
            for entry
            in os.listdir(QUERY_PATH)
            if os.path.isfile(os.path.join(QUERY_PATH, entry))
        ])
        return files_amount

    def _retrieve_query_texts(self) -> List[List[str]]:
        query_files = [
            open(f"{QUERY_PATH}/query_{file_number}.txt", "r")
            for file_number in range(1, self.number_of_query_classes + 1)
        ]

        queries = []
        for query_file in query_files:
            # TODO: раньше был выбор только первой строки. Проверить, не сломалось ли что
            file_queries = query_file.readlines()

            queries.append(file_queries)

            query_file.close()

        if self.logging_mode == "verbose":
            print("_retrieve_query_texts:", *queries, sep='\n')
            print()

        assert len(queries) == self.number_of_query_classes

        return queries

    def _select_indexable_columns(self):
        available_query_classes = tuple(self.available_query_classes)
        query_class_frequencies = tuple([1] * len(available_query_classes))

        logging.info(f"Selecting indexable columns on {len(available_query_classes)} query classes.")

        # Returns list with the one item.
        # The item is a list of queries (Query) for each class in available_query_classes
        workload = self._workloads_from_tuples([(available_query_classes, query_class_frequencies)])[0]

        indexable_columns = workload.indexable_columns()

        # TODO: реализовать self._only_utilized_indexes(indexable_columns)???

        selected_columns = []

        global_column_id = 0
        for column in self.schema_columns:
            if column in indexable_columns:
                column.global_column_id = global_column_id
                global_column_id += 1

                selected_columns.append(column)

        return selected_columns

    def _workloads_from_tuples(
            self,
            tuples: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
            unknown_query_probability=None
    ) -> List[Workload]:
        # TODO: Handle unknown queries

        workloads = []

        for query_classes, query_class_frequencies in tuples:
            queries = []

            for query_class, frequency in zip(query_classes, query_class_frequencies):
                query_text = self.rnd.choice(self.query_texts[query_class - 1])

                query = Query(query_class, query_text, frequency=frequency)

                self._store_indexable_columns(query)
                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                queries.append(query)

            workloads.append(Workload(queries))

        return workloads

    def _store_indexable_columns(self, query):
        parser = Parser(query.text)
        query_columns = self._resolve_columns(parser)
        query_columns = set(query_columns)

        for column in self.schema_columns:
            full_column_name = f"{column.table}.{column.name}"
            if full_column_name in query_columns:
                query.columns.append(column)

    def _resolve_columns(self, query_parser):
        query_columns = query_parser.columns

        for view in self.views:
            for table in query_parser.tables:
                if table == view.name:
                    view_parser = Parser(view.definition)
                    query_columns.extend(self._resolve_columns(view_parser))

        return query_columns

    def _generate_workloads(
            self,
            train_instances,
            validation_instances,
            test_instances,
            query_classes_per_workload,
            unknown_query_probability=None
    ):
        required_unique_workloads = train_instances + validation_instances + test_instances

        logging.info(f"Required unique workload tuples: {required_unique_workloads}")

        if not self.varying_frequencies:
            query_classes_factorial = math.factorial(self.number_of_query_classes)
            assert required_unique_workloads <= query_classes_factorial, \
                f"""
                Total workload amount must be less then factorial of the query classes.
                Total workload amount: {required_unique_workloads}.
                Query classes factorial: {self.number_of_query_classes}! = {query_classes_factorial}.
                """

        unique_workload_tuples = set()

        while required_unique_workloads > len(unique_workload_tuples):
            workload_tuple = self._generate_random_workload(
                query_classes_per_workload,
                unknown_query_probability
            )
            unique_workload_tuples.add(workload_tuple)

        validation_workload_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
        unique_workload_tuples = unique_workload_tuples - set(validation_workload_tuples)

        test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)
        unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

        assert len(unique_workload_tuples) == train_instances
        train_workload_tuples = list(unique_workload_tuples)

        assert (
                len(train_workload_tuples) +
                len(test_workload_tuples) +
                len(validation_workload_tuples) == required_unique_workloads
        )

        validation_workloads = self._workloads_from_tuples(validation_workload_tuples, unknown_query_probability)
        test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
        train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)

        return train_workloads, validation_workloads, test_workloads

    def _generate_random_workload(
            self,
            query_classes_per_workload: int,
            unknown_query_probability=None
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        assert query_classes_per_workload <= self.number_of_query_classes, \
            "Cannot generate workload with more queries than query classes"

        if unknown_query_probability is not None:
            raise NotImplementedError("Workload generation with unknown queries is not supported")
        else:
            workload_query_classes = tuple(self.rnd.sample(
                self.available_query_classes, query_classes_per_workload
            ))

        # Create frequencies
        # Used in total cost calculation and observation
        if self.varying_frequencies:
            """
            INFO:root:query_class: 1. frequency: 2087
            INFO:root:query_class: 3. frequency: 2033
            INFO:root:query_class: 1. frequency: 3732
            INFO:root:query_class: 2. frequency: 8988
            INFO:root:query_class: 1. frequency: 1868
            INFO:root:query_class: 3. frequency: 5517
            INFO:root:query_class: 3. frequency: 4293
            INFO:root:query_class: 1. frequency: 2039
            """
            query_class_frequencies = tuple(list(self.np_rnd.integers(1, 10000, query_classes_per_workload)))
        else:
            query_class_frequencies = tuple([
                len(self.query_texts[query_class - 1])
                for query_class
                in workload_query_classes
            ])

        workload_tuple = (workload_query_classes, query_class_frequencies)

        logging.debug(f"Generate workload tuple: {workload_tuple}")

        return workload_tuple


