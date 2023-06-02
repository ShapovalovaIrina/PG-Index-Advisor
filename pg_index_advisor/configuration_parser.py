import json
import logging


class ConfigurationParser(object):
    def __init__(self, configuration_file):
        self.REQUIRED_CONFIGURATION_OPTIONS_FIRST_LEVEL = [
            "id",
            "description",
            "rl_algorithm",
            "random_seed",
            "max_steps_per_episode",
            "max_index_width",
            "result_path",
            "database",
            "workload",
            "workload_embedder",
            "filter_utilized_columns",
            "logging",
            "budgets",
            "reenable_indexes",
            "validation_frequency",
            "timesteps",
            "query_file"
        ]

        self.REQUIRED_CONFIGURATION_OPTIONS_FURTHER = {
            "rl_algorithm": [
                "algorithm",
                "gamma",
                "policy",
                "model_architecture",
                "args"
            ],
            "database": [
                "database",
                "username",
                "password",
                "hostname",
                "port"
            ],
            "workload": [
                "size",
                "varying_frequencies",
                "training_instances",
                "validation_testing",
                "excluded_query_classes",
                "similar_workloads"
            ],
            "workload_embedder": [
                "representation_size"
            ],
            "budgets": [
                "training",
                "validation_and_testing"
            ]
        }

        with open(configuration_file) as f:
            self.config = json.load(f)

        # Check if configuration options are missing in json file
        self._determine_missing_configuration_options(
            self.REQUIRED_CONFIGURATION_OPTIONS_FIRST_LEVEL, self.config.keys()
        )
        for key, required_options in self.REQUIRED_CONFIGURATION_OPTIONS_FURTHER.items():
            self._determine_missing_configuration_options(required_options, self.config[key].keys())

        # Check if the json file has unknown configuration options
        self._determine_missing_configuration_options(
            self.config.keys(), self.REQUIRED_CONFIGURATION_OPTIONS_FIRST_LEVEL, throw_exception=False
        )

    @staticmethod
    def _determine_missing_configuration_options(expected_options, actual_options, throw_exception=True):
        missing_options = set(expected_options) - set(actual_options)

        if throw_exception:
            assert (missing_options == frozenset()), \
                f"Configuration misses required configuration option: {missing_options}"
        else:
            if len(missing_options) > 0:
                logging.warning(
                    f"The following configuration options are missing or optional: {missing_options}"
                )
