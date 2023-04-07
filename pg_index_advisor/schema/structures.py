from typing import List

from index_selection_evaluation.selection.index import Index as PotentialIndex


class Query:
    def __init__(self, query_id, query_text, columns=None):
        self.nr = query_id
        self.text = query_text

        # Indexable columns
        if columns is None:
            self.columns = []
        else:
            self.columns = columns

    def __repr__(self):
        return f"Q{self.nr}"


class Workload:
    queries: List[Query]

    def __init__(self, queries):
        self.queries = queries

    def indexable_columns(self):
        indexable_columns = set()
        for query in self.queries:
            indexable_columns |= set(query.columns)
        return sorted(list(indexable_columns))

    def potential_indexes(self):
        return sorted([PotentialIndex([c]) for c in self.indexable_columns()])


class Column:
    def __init__(self, name):
        self.name = name.lower()
        self.table = None

    def __lt__(self, other):
        return self.name < other.name

    def __repr__(self):
        return f"C {self.table}.{self.name}"

    # We cannot check self.table == other.table here since Table.__eq__()
    # internally checks Column.__eq__. This would lead to endless recursions.
    def __eq__(self, other):
        if not isinstance(other, Column):
            return False

        assert (
            self.table is not None and other.table is not None
        ), "Table objects should not be None for Column.__eq__()"

        return self.table.name == other.table.name and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.table.name))


class Table:
    def __init__(self, name):
        self.name = name.lower()
        self.columns = []

    def add_column(self, column):
        column.table = self
        self.columns.append(column)

    def add_columns(self, columns):
        for column in columns:
            self.add_column(column)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Table):
            return False

        return self.name == other.name and tuple(self.columns) == tuple(other.columns)

    def __hash__(self):
        return hash((self.name, tuple(self.columns)))


class Index:
    def __init__(self, table, name, columns):
        self.table = table.lower()
        self.name = name.lower()
        self.columns = columns.split(",")

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Index):
            return False

        return self.table == other.table \
            and tuple(self.columns) == tuple(other.columns)

    def __hash__(self):
        return hash((self.table, self.name, tuple(self.columns)))


class View(object):
    def __init__(self, name, definition):
        self.name = name
        self.definition = definition.strip(";")

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, View):
            return False

        return self.name == other.name \
            and self.definition == other.definition

    def __hash__(self):
        return hash((self.name, self.definition))

