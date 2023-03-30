import itertools
import json
import logging

from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from schema.db_connector import UserPostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index as PotentialIndex


# Todo: This could be improved by passing index candidates as input
# TODO: see TODO in simulate_index
def predict_index_sizes(column_combinations, db_config):
    connector = UserPostgresDatabaseConnector(
        db_config["database"],
        db_config["username"],
        db_config["password"],
        db_port=db_config["port"],
        autocommit=True
    )
    # connector.drop_indexes()

    cost_evaluation = CostEvaluation(connector)

    predicted_index_sizes = []

    parent_index_size_map = {}

    for column_combination in column_combinations:
        potential_index = PotentialIndex(column_combination)
        cost_evaluation.what_if.simulate_index(potential_index, True)

        full_index_size = potential_index.estimated_size

        index_delta_size = full_index_size
        if len(column_combination) > 1:
            index_delta_size -= parent_index_size_map[column_combination[:-1]]

        predicted_index_sizes.append(index_delta_size)
        cost_evaluation.what_if.drop_simulated_index(potential_index)

        parent_index_size_map[column_combination] = full_index_size

    logging.info("Potential index sizes:")
    for c, size in parent_index_size_map.items():
        logging.info(f"Columns {c}: {size} bytes")

    return predicted_index_sizes


def create_column_permutation_indexes(columns, max_index_width):
    result_column_combinations = []

    table_column_dict = {}
    for column in columns:
        if column.table not in table_column_dict:
            table_column_dict[column.table] = set()
        table_column_dict[column.table].add(column)

    for length in range(1, max_index_width + 1):
        unique = set()
        count = 0
        for key, columns_per_table in table_column_dict.items():
            unique |= set(itertools.permutations(columns_per_table, length))
            count += len(set(itertools.permutations(columns_per_table, length)))
        print(f"{length}-column indexes: {count}")

        result_column_combinations.append(list(unique))

    return result_column_combinations
