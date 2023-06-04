from index_selection_evaluation.selection.what_if_index_creation import WhatIfIndexCreation
from pg_index_advisor.database.db_connector import UserPostgresDatabaseConnector
from pg_index_advisor.database.structures import PotentialIndex


class WhatIfIndexManager(WhatIfIndexCreation):
    db_connector: UserPostgresDatabaseConnector
    def __init__(self, db_connector: UserPostgresDatabaseConnector):
        WhatIfIndexCreation.__init__(self, db_connector)

        self.hidden_indexes = {}

    def hide_index(self, potential_index: PotentialIndex):
        assert potential_index.hypopg_name is not None
        assert potential_index.hypopg_oid is not None
        assert potential_index.estimated_size is not None

        self.db_connector.hide_index(potential_index.hypopg_name)
        self.hidden_indexes[potential_index.hypopg_oid] = potential_index.hypopg_name

    def unhide_index(self, index: PotentialIndex):
        self.db_connector.unhide_index(index.hypopg_name)
        del self.hidden_indexes[index.hypopg_oid]

    def all_simulated_indexes(self):
        statement = "select * from hypopg_list_indexes"
        indexes = self.db_connector.exec_fetch(statement, one=False)
        return indexes

    def all_hidden_indexes(self):
        statement = "select * from hypopg_hidden_indexes"
        indexes = self.db_connector.exec_fetch(statement, one=False)
        return indexes

    def drop_all_simulated_indexes(self):
        for oid in self.simulated_indexes:
            self.db_connector.drop_simulated_index(oid)
        self.simulated_indexes = {}

    def drop_all_hidden_indexes(self):
        for names in self.hidden_indexes.values():
            self.db_connector.unhide_index(names)
        self.hidden_indexes = {}
