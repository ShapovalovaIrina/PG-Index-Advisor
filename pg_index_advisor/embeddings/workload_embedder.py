import logging

import gensim

from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.workload import Query

from .boo import BagOfOperators
from pg_index_advisor.schema.db_connector import UserPostgresDatabaseConnector


class WorkloadEmbedder(object):
    def __init__(self, query_texts, representation_size, database_connector, columns=None, retrieve_plans=False):
        self.INDEXES_SIMULATED_IN_PARALLEL = 1000
        self.query_texts = query_texts
        self.representation_size = representation_size
        self.database_connector = database_connector
        self.plans = None
        self.columns = columns

        if retrieve_plans:
            cost_evaluation = CostEvaluation(self.database_connector)
            # [without indexes], [with indexes]
            self.plans = ([], [])
            for query_idx, query_texts_per_query_class in enumerate(query_texts):
                query_text = query_texts_per_query_class[0]
                query = Query(query_idx, query_text)
                plan = self.database_connector.get_plan(query)
                self.plans[0].append(plan)

            for n, n_column_combinations in enumerate(self.columns):
                logging.critical(f"Creating all indexes of width {n+1}.")

                created_indexes = 0
                while created_indexes < len(n_column_combinations):
                    potential_indexes = []
                    for i in range(self.INDEXES_SIMULATED_IN_PARALLEL):
                        potential_index = Index(n_column_combinations[created_indexes])
                        cost_evaluation.what_if.simulate_index(potential_index, True)
                        potential_indexes.append(potential_index)
                        created_indexes += 1
                        if created_indexes == len(n_column_combinations):
                            break

                    for query_idx, query_texts_per_query_class in enumerate(query_texts):
                        query_text = query_texts_per_query_class[0]
                        query = Query(query_idx, query_text)
                        plan = self.database_connector.get_plan(query)
                        self.plans[1].append(plan)

                    for potential_index in potential_indexes:
                        cost_evaluation.what_if.drop_simulated_index(potential_index)

                    logging.critical(f"Finished checking {created_indexes} indexes of width {n+1}.")

        self.database_connector = None

    def get_embeddings(self, workload):
        raise NotImplementedError


class PlanEmbedder(WorkloadEmbedder):
    def __init__(self, query_texts, representation_size, database_connector, columns, without_indexes=False):
        WorkloadEmbedder.__init__(
            self, query_texts, representation_size, database_connector, columns, retrieve_plans=True
        )

        self.plan_embedding_cache = {}

        self.relevant_operators = []
        self.relevant_operators_wo_indexes = []
        self.relevant_operators_with_indexes = []

        logging.info("Init Bag Of Operators")
        self.boo_creator = BagOfOperators()

        # self.plans[0] - plans for queries without indexes
        for plan in self.plans[0]:
            boo = self.boo_creator.boo_from_plan(plan)
            self.relevant_operators.append(boo)
            self.relevant_operators_wo_indexes.append(boo)

        # self.plans[1] - plans for queries with indexes
        if without_indexes is False:
            for plan in self.plans[1]:
                boo = self.boo_creator.boo_from_plan(plan)
                self.relevant_operators.append(boo)
                self.relevant_operators_with_indexes.append(boo)

        # Deleting the plans to avoid costly copying later.
        self.plans = None

        self.dictionary = gensim.corpora.Dictionary(self.relevant_operators)
        logging.info(f"Dictionary has {len(self.dictionary)} entries.")
        self.bow_corpus = [self.dictionary.doc2bow(query) for query in self.relevant_operators]

        logging.info("Create LSI model")
        self._create_model()

        # Deleting the bow_corpus to avoid costly copying later.
        self.bow_corpus = None

    def _create_model(self):
        raise NotImplementedError

    def _infer(self, bow, boo):
        raise NotImplementedError

    def get_embeddings(self, plans):
        embeddings = []

        for plan in plans:
            cache_key = str(plan)
            if cache_key not in self.plan_embedding_cache:
                boo = self.boo_creator.boo_from_plan(plan)
                bow = self.dictionary.doc2bow(boo)

                vector = self._infer(bow, boo)

                self.plan_embedding_cache[cache_key] = vector
            else:
                vector = self.plan_embedding_cache[cache_key]

            embeddings.append(vector)

        return embeddings


class PlanEmbedderLSI(PlanEmbedder):
    def __init__(
            self,
            query_texts,
            representation_size,
            columns,
            database_config,
            without_indexes=False
    ):
        database_connector = UserPostgresDatabaseConnector(
            database_config["database"],
            database_config["username"],
            database_config["password"],
            db_port=database_config["port"],
            autocommit=True
        )
        PlanEmbedder.__init__(
            self,
            query_texts,
            representation_size,
            database_connector,
            columns,
            without_indexes
        )

    def _create_model(self):
        self.lsi_model = gensim.models.LsiModel(
            corpus=self.bow_corpus,
            id2word=self.dictionary,
            num_topics=self.representation_size
        )

        assert (len(self.lsi_model.get_topics()) == self.representation_size), \
            f"Topic-representation_size mismatch: {len(self.lsi_model.get_topics())} " +\
            f"vs {self.representation_size}"

    def _infer(self, bow, boo):
        result = self.lsi_model[bow]

        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector
