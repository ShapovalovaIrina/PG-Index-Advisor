import logging
import psycopg2

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

        self.set_random_seed()

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
