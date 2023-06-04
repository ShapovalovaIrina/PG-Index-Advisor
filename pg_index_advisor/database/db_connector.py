import logging
import psycopg2
import time

from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.database_connector import DatabaseConnector


class UserPostgresDatabaseConnector(PostgresDatabaseConnector):
    def __init__(
            self,
            db_name,
            db_user,
            db_password,
            db_host="localhost",
            db_port=5432,
            autocommit=False
    ):
        DatabaseConnector.__init__(self, db_name, autocommit=autocommit)
        self.db_system = "postgres"
        self._connection = None

        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.create_connection()

        self.hidden_indexes = 0

        self.set_random_seed()

        self.exec_only("SET max_parallel_workers_per_gather = 0;")
        self.exec_only("SET enable_bitmapscan TO off;")

        logging.debug("Postgres connector created: {}".format(db_name))

    def create_connection(self):
        if self._connection:
            self.close()
        self._connection = psycopg2.connect(
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        self._connection.autocommit = self.autocommit
        self._cursor = self._connection.cursor()

    def hide_index(self, index_name):
        self.hidden_indexes += 1

        start_time = time.time()
        self._hide_index(index_name)
        end_time = time.time()
        self.index_simulation_duration += end_time - start_time

    def _hide_index(self, index_name):
        statement = f"SELECT hypopg_hide_index('{index_name}'::regclass);"
        result = self.exec_fetch(statement)

        assert result[0] is True, f"Could not hide read index with name = {index_name}."

    def unhide_index(self, index_name):
        start_time = time.time()
        self._unhide_index(index_name)
        end_time = time.time()
        self.index_simulation_duration += end_time - start_time

    def _unhide_index(self, index_name):
        statement = f"SELECT hypopg_unhide_index('{index_name}'::regclass);"
        result = self.exec_fetch(statement)

        assert result[0] is True, f"Could not unhide read index with name = {index_name}."

    def drop_simulated_index(self, identifier):
        start_time = time.time()
        self._drop_simulated_index(identifier)
        end_time = time.time()
        self.index_simulation_duration += end_time - start_time

    def unhide_all_indexes(self):
        start_time = time.time()
        self._unhide_all_indexes()
        end_time = time.time()
        self.index_simulation_duration += end_time - start_time

    def _unhide_all_indexes(self):
        statement = f"SELECT * FROM hypopg_unhide_all_indexes();"
        result = self.exec_fetch(statement)

        assert result[0] is True, f"Could not unhide all indexes."
