import logging

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
