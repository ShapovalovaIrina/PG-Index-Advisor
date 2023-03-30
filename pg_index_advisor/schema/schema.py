import importlib

from .structures import Index, View, Column, Table
from .db_connector import UserPostgresDatabaseConnector


class Schema(object):
    def __init__(self, database_config, filters=None):
        self.tables = []
        self.columns = []
        self.indexes = []
        self.views = []
        self.db_config = database_config

        if filters is None:
            self.filters = {}

        self.generation_connector = UserPostgresDatabaseConnector(
            database_config["database"],
            database_config["username"],
            database_config["password"],
            autocommit=True
        )

        self._read_tables()
        self._filter_tables(filters.get("table", {}))
        self._read_columns_from_tables()
        self._read_existing_indexes()
        self._filter_indexes(filters.get("index", {}))
        self._read_existing_views()

    def _read_tables(self):
        tables = self.generation_connector.exec_fetch("""
        SELECT columns.table_name, array_to_string(array_agg(column_name), ',')
        FROM information_schema.columns
                 JOIN information_schema.tables on columns.table_name = tables.table_name
        WHERE columns.table_schema = 'public'
          AND tables.table_type = 'BASE TABLE'
        group by columns.table_name;
        """, one=False)

        for (table, columns) in tables:
            columns = columns.split(",")

            table = Table(table)
            for column_name in columns:
                column = Column(column_name)
                table.add_column(column)

            self.tables.append(table)

    def _read_columns_from_tables(self):
        for table in self.tables:
            for column in table.columns:
                self.columns.append(column)

    def _read_existing_indexes(self):
        indexes = self.generation_connector.exec_fetch("""
        SELECT
            t.relname AS table_name,
            i.relname AS index_name,
            array_to_string(array_agg(a.attname), ', ') AS column_names
        FROM
            pg_class t,
            pg_class i,
            pg_index ix,
            pg_attribute a
        WHERE
            t.oid = ix.indrelid
            AND i.oid = ix.indexrelid
            AND a.attrelid = t.oid
            AND a.attnum = ANY(ix.indkey)
            AND t.relkind = 'r'
        GROUP BY
            t.relname,
            i.relname
        ORDER BY 
            t.relname,
            i.relname;        
        """, one=False)

        for (table, index_name, columns) in indexes:
            index = Index(table, index_name, columns)
            self.indexes.append(index)

    def _read_existing_views(self):
        views = self.generation_connector.exec_fetch(f"""
        SELECT viewname, definition FROM pg_views 
        WHERE viewowner = '{self.db_config["username"]}';       
        """, one=False)

        for (viewname, definition) in views:
            view = View(viewname, definition)
            self.views.append(view)

    def _filter_tables(self, table_filters):
        for (filter_name, filter_value) in table_filters.items():
            filter_class = getattr(
                importlib.import_module("pg_index_advisor.schema.filters"),
                filter_name
            )
            filter_instance = filter_class(filter_value, self.db_config)
            self.tables = filter_instance.apply_filter(self.tables)

    def _filter_indexes(self, index_filters):
        for (filter_name, filter_value) in index_filters.items():
            filter_class = getattr(
                importlib.import_module("pg_index_advisor.schema.filters"),
                filter_name
            )
            filter_instance = filter_class(filter_value, self.db_config)
            self.indexes = filter_instance.apply_filter(self.indexes)
