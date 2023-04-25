import unittest

from pg_index_advisor.schema.db_connector import UserPostgresDatabaseConnector
from pg_index_advisor.schema.schema import Schema
from pg_index_advisor.schema.structures import Table, Column


class SchemaTests(unittest.TestCase):
    schema = None

    @classmethod
    def setUpClass(cls) -> None:
        with open('script.sql') as f:
            sql = f.readlines()
        sql = "\n".join(sql)

        cls.generation_connector = UserPostgresDatabaseConnector(
            "advisor_tests",
            "advisor_user",
            "advisor_pass",
            db_port=6543,
            autocommit=False
        )
        cls.generation_connector.exec_only(sql)

    def test_read_tables(self):
        schema = object.__new__(Schema)
        schema.generation_connector = self.generation_connector
        schema.tables = []
        schema.columns = []

        schema._read_tables()

        tables_dict = {
            'user_application': [
                'id',
                'version',
                'previous',
                'submitted_by',
                'submitted_at',
                'status',
                'type',
                'vehicle_id'
            ],
            'profiles': [
                'user_id',
                'username',
                'deleted',
                'contacts'
            ]
        }
        tables = []

        for table_name, columns in tables_dict.items():
            table = Table(table_name)
            for c in columns:
                table.add_column(Column(c))
            tables.append(table)

        assert sorted(schema.tables) == sorted(tables), \
            f"Expect schema tables to be equal {tables}. " \
            f"Actual: {schema.tables}"

    def test_read_columns(self):
        schema = object.__new__(Schema)
        schema.generation_connector = self.generation_connector
        schema.tables = []
        schema.columns = []

        schema._read_tables()
        schema._read_columns_from_tables()

        tables_dict = {
            'user_application': [
                'id',
                'version',
                'previous',
                'submitted_by',
                'submitted_at',
                'status',
                'type',
                'vehicle_id'
            ],
            'profiles': [
                'user_id',
                'username',
                'deleted',
                'contacts'
            ]
        }
        tables = []
        columns = []

        for table_name, columns_names in tables_dict.items():
            table = Table(table_name)
            for c in columns_names:
                table.add_column(Column(c))
            tables.append(table)

        for t in tables:
            for c in t.columns:
                columns.append(c)

        assert sorted(schema.columns) == sorted(columns), \
            f"Expect schema columns to be equal {columns}. " \
            f"Actual: {schema.columns}"


if __name__ == "__main__":
    unittest.main()
