import importlib

from .structures import RealIndex, View, Column, Table
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
            db_port=database_config["port"],
            autocommit=True
        )

        self._read_tables()
        self._filter_tables(filters.get("table", {}))
        self._read_columns_from_tables()
        self._read_existing_indexes()
        # self._filter_indexes(filters.get("index", {}))
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
        SELECT t.relname                                  AS table_name,
               i.relname                                  AS index_name,
               i.oid                                      AS oid,
               pg_get_indexdef(i.oid)                     AS index_definition,
               pg_relation_size(ix.indexrelid)            AS index_size,
               ix.indisprimary                            AS is_primary,
               array_to_string(array_agg(a.attname), ' ') AS column_names,
               ix.indkey                                  AS index_order,
               array_to_string(array_agg(a.attnum), ' ')  AS columns_order
        FROM pg_class t,
             pg_class i,
             pg_index ix,
             pg_attribute a,
             pg_namespace ns
        WHERE t.oid = ix.indrelid
          AND i.oid = ix.indexrelid
          AND a.attrelid = t.oid
          AND a.attnum = ANY (ix.indkey)
          AND t.relnamespace = ns.oid
          AND t.relkind = 'r'
          AND ns.nspname = 'public'
        GROUP BY t.relname,
                 i.relname,
                 i.oid,
                 ix.indexrelid,
                 ix.indisprimary,
                 ix.indkey
        ORDER BY t.relname,
                 i.relname;
        """, one=False)

        # TODO: use index definition for partial indexes
        for (table_name, index_name, oid, definition, size, is_primary, columns, index_columns_order,
             row_columns_order) in indexes:
            columns_numbers = dict(zip(row_columns_order.split(' '), columns.split(' ')))

            columns = []
            for i in index_columns_order.split(' '):
                columns.append(columns_numbers[i])

            index = RealIndex(table_name, index_name, oid, columns, int(size), is_primary)
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
                importlib.import_module("pg_index_advisor.database.filters"),
                filter_name
            )
            filter_instance = filter_class(filter_value, self.db_config)
            self.tables = filter_instance.apply_filter(self.tables)

    def _filter_indexes(self, index_filters):
        for (filter_name, filter_value) in index_filters.items():
            filter_class = getattr(
                importlib.import_module("pg_index_advisor.database.filters"),
                filter_name
            )
            filter_instance = filter_class(filter_value, self.db_config)
            self.indexes = filter_instance.apply_filter(self.indexes)
