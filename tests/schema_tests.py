import unittest

from pg_index_advisor.schema.db_connector import UserPostgresDatabaseConnector
from pg_index_advisor.schema.schema import Schema
from pg_index_advisor.schema.structures import Table, Column, RealIndex, View


class SchemaTests(unittest.TestCase):
    db_config = {
        'database': 'advisor_tests',
        'username': 'advisor_user',
        'password': 'advisor_pass',
        'port': 6543
    }
    database_tables_dict = {
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
        ],
        'user_application_response': [
            'id',
            'application_id',
            'company_id',
            'reviewer_type',
            'result'
        ]
    }
    database_indexes_dict = [
        ('profiles_pkey', 'profiles', ['user_id']),
        ('profiles_username_index', 'profiles', ['username']),
        ('user_application_pkey', 'user_application', ['id']),
        ('user_application_previous_index', 'user_application', ['previous']),
        ('user_application_submitted_by_version_index', 'user_application', ['submitted_by', 'version']),
        ('user_application_vehicle_id_version_index', 'user_application', ['vehicle_id', 'version']),
        ('user_application_response_pkey', 'user_application_response', ['id']),
        ('user_application_response_application_id_reviewer_type_company_', 'user_application_response', ['application_id', 'reviewer_type', 'company_id']),
    ]
    database_views_dict = [
        'user_application_company_response',
        'profile_application',
        'accepted_profile_application',
        'vehicle_application',
        'accepted_vehicle_application'
    ]
    generation_connector = None

    @classmethod
    def setUpClass(cls):
        with open('script.sql') as f:
            sql = f.readlines()
        sql = "\n".join(sql)

        cls.generation_connector = UserPostgresDatabaseConnector(
            cls.db_config['database'],
            cls.db_config['username'],
            cls.db_config['password'],
            db_port=cls.db_config['port'],
            autocommit=True
        )
        cls.generation_connector.exec_only(sql)
        
    def setUp(self):
        self.schema = object.__new__(Schema)
        self.schema.generation_connector = self.generation_connector
        self.schema.db_config = self.db_config
        self.schema.tables = []
        self.schema.columns = []
        self.schema.indexes = []
        self.schema.views = []

    def test_read_tables(self):
        self.schema._read_tables()

        tables = self.tables_from_dict()

        assert sorted(self.schema.tables) == sorted(tables), \
            f"Expect schema tables to be equal to {tables}. " \
            f"Actual: {self.schema.tables}"

    def test_filter_tables_by_name(self):
        allowed_tables = [
            "user_application",
            "profiles"
        ]

        self.schema._read_tables()
        self.schema._filter_tables({"TableNameFilter": {"allowed_tables": allowed_tables}})

        tables = []
        for t in self.tables_from_dict():
            if t.name in allowed_tables:
                tables.append(t)

        assert sorted(self.schema.tables) == sorted(tables), \
            f"Expect schema tables to be equal to {tables}. " \
            f"Actual: {self.schema.tables}"

    def test_read_columns(self):
        self.schema._read_tables()
        self.schema._read_columns_from_tables()

        columns = []

        for t in self.tables_from_dict():
            for c in t.columns:
                columns.append(c)

        assert sorted(self.schema.columns) == sorted(columns), \
            f"Expect schema columns to be equal to {columns}. " \
            f"Actual: {self.schema.columns}"

    def test_read_indexes(self):
        self.schema._read_existing_indexes()

        indexes = self.indexes_from_dict()

        assert len(indexes) == len(self.schema.indexes), \
            f"Read indexes length is {len(self.schema.indexes)} " \
            f"when {len(indexes)} is expected."

        assert sorted(self.schema.indexes) == sorted(indexes), \
            f"Expect schema indexes to be equal to {indexes}. " \
            f"Actual: {self.schema.indexes}"

    def test_read_views(self):
        self.schema._read_existing_views()

        views = self.views_from_dict()

        assert len(views) == len(self.schema.views), \
            f"Read views length is {len(self.schema.views)} " \
            f"when {len(views)} is expected."

        assert sorted(self.schema.views) == sorted(views), \
            f"Expect schema views to be equal to {views}. " \
            f"Actual: {self.schema.views}"

    def tables_from_dict(self):
        tables = []

        for table_name, columns_names in self.database_tables_dict.items():
            table = Table(table_name)
            for c in columns_names:
                table.add_column(Column(c))
            tables.append(table)

        return tables

    def indexes_from_dict(self):
        indexes = []

        for index_name, table, columns in self.database_indexes_dict:
            index = RealIndex(table, index_name, 0, columns, 0, False)
            indexes.append(index)

        return indexes

    def views_from_dict(self):
        views = []

        for view in self.database_views_dict:
            view = View(view, "")
            views.append(view)

        return views


if __name__ == "__main__":
    unittest.main()
