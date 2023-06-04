import datetime
import logging

from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from pg_index_advisor.database.what_if_index_manager import WhatIfIndexManager

class CostEvaluationWithHidingIndex(CostEvaluation):
    def __init__(self, db_connector, cost_estimation="whatif"):
        logging.debug("Init cost evaluation")

        self.db_connector = db_connector
        self.cost_estimation = cost_estimation

        logging.info("Cost estimation with " + self.cost_estimation)
        self.what_if = WhatIfIndexManager(db_connector)

        self.current_indexes = set()
        self.current_deleted_indexes = set()

        assert len(self.what_if.all_simulated_indexes()) == len(self.current_indexes)

        self.cost_requests = 0
        self.cache_hits = 0
        # Cache structure:
        # {(query_object, relevant_indexes): cost}
        self.cache = {}

        # Cache structure:
        # {(query_object, relevant_indexes): (cost, plan)}
        self.cache_plans = {}

        self.completed = False
        # It is not necessary to drop hypothetical indexes during __init__().
        # These are only created per connection. Hence, non should be present.

        self.relevant_indexes_cache = {}

        self.costing_time = datetime.timedelta(0)

    def update_created_indexes(self, created_indexes, store_size=False):
        for index in set(created_indexes) - self.current_indexes:
            self._simulate_or_create_index(index, store_size=store_size)

        for index in self.current_indexes - set(created_indexes):
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set(created_indexes)

    def update_deleted_indexes(self, deleted_indexes):
        for index in set(deleted_indexes) - self.current_deleted_indexes:
            self._hide_or_drop_index(index)

        for index in self.current_deleted_indexes - set(deleted_indexes):
            self._unhide_or_create_index(index)

        assert self.current_deleted_indexes == set(deleted_indexes)

    def _hide_or_drop_index(self, index):
        if self.cost_estimation == "whatif":
            self.what_if.hide_index(index)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.drop_index(index)

        self.current_deleted_indexes.add(index)

    def _unhide_or_create_index(self, index):
        if self.cost_estimation == "whatif":
            self.what_if.unhide_index(index)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.create_index(index)

        self.current_deleted_indexes.remove(index)

    def calculate_cost_and_plans(
            self,
            workload,
            created_indexes,
            store_size=False
    ):
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        start_time = datetime.datetime.now()

        total_cost = 0
        plans = []
        costs = []

        for query in workload.queries:
            self.cost_requests += 1
            cost, plan = self._request_cache_plans(query, created_indexes)
            total_cost += cost * query.frequency
            plans.append(plan)
            costs.append(cost)

        end_time = datetime.datetime.now()
        self.costing_time += end_time - start_time

        return total_cost, plans, costs

    def reset_hypopg(self):
        if self.cost_estimation == "whatif":
            self.what_if.drop_all_simulated_indexes()
            self.what_if.drop_all_hidden_indexes()

            self.current_indexes = set()
            self.current_deleted_indexes = set()
