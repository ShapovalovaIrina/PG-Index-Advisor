import logging

from .db_connector import UserPostgresDatabaseConnector


class TableNumRowsFilter(object):
    def __init__(self, params, db_config):
        self.threshold = params["threshold"]

        # TODO: conn param or new object?
        self.connector = UserPostgresDatabaseConnector(
            db_config["database"],
            db_config["username"],
            db_config["password"],
            autocommit=True
        )
        self.connector.create_statistics()

    def apply_filter(self, tables, columns):
        output_columns = []

        for table in tables:
            table_name = table.name
            table_num_rows = self.connector.exec_fetch(
                f"SELECT reltuples::bigint AS estimate FROM pg_class where relname='{table_name}'"
            )[0]

            if table_num_rows > self.threshold:
                for column in table.columns:
                    output_columns.append(column)
            else:
                logging.info(
                    f"Skip the table {table.name} because " +
                    f"the number of rows is less than the threshold value."
                )

        logging.warning(f"Reduced columns from {len(columns)} to {len(output_columns)}.")

        return output_columns


class TableNameFilter(object):
    def __init__(self, params, db_config):
        self.allowed_tables = params["allowed_tables"]

        # TODO: conn param or new object?
        self.connector = UserPostgresDatabaseConnector(
            db_config["database"],
            db_config["username"],
            db_config["password"],
            autocommit=True
        )

    def apply_filter(self, tables, columns):
        output_columns = []

        for table in tables:
            if table.name in self.allowed_tables:
                for column in table.columns:
                    output_columns.append(column)

        logging.warning(f"Reduced columns from {len(columns)} to {len(output_columns)}.")

        return output_columns


class IndexConstraintFilter(object):
    def __init__(self, params, db_config):
        self.skip_primary_key = params["skip_primary_key"]

        # TODO: conn param or new object?
        if self.skip_primary_key:
            self.connector = UserPostgresDatabaseConnector(
                db_config["database"],
                db_config["username"],
                db_config["password"],
                autocommit=True
            )

    def apply_filter(self, indexes):
        if not self.skip_primary_key:
            return indexes

        output_indexes = []

        for index in indexes:
            primary_key = self.connector.exec_fetch(f"""
            SELECT conname
            FROM   pg_constraint
            WHERE  connamespace = 'public'::regnamespace
            AND    contype = 'p'
            AND    conname = '{index.name}'
            AND    conrelid = '{index.table}'::regclass;
            """
            )

            if not primary_key:
                output_indexes.append(index)

        logging.warning(f"Reduced indexes from {len(indexes)} to {len(output_indexes)}.")

        return output_indexes