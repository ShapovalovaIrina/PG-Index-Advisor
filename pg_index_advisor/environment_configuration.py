import logging
import random
import os
import pickle
import utils

from configuration_parser import ConfigurationParser
from datetime import datetime

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization


from schema.schema import Schema
from workload_generator import WorkloadGenerator


class EnvironmentConfiguration(object):
    def __init__(self, configuration_file):
        self._init_time()

        cp = ConfigurationParser(configuration_file)
        self.config = cp.config

        self.id = self.config["id"]
        self.model = None

        self.rnd = random.Random()
        self.rnd.seed(self.config["random_seed"])

        self.schema = None
        self.workload_generator = None
        self.number_of_features = None
        self.number_of_actions = None
        self.evaluated_workloads_strs = []
        self.globally_indexable_columns = []
        self.single_column_flat_set = set()
        self.globally_indexable_columns_flat = []
        self.action_storage_consumptions = []

        self.ENVIRONMENT_RESULT_PATH = self.config["result_path"]
        self._create_environment_folder()

    def prepare(self):
        """
        Get filtered DB schema elements:
            - columns
            - indexes
        """
        self.schema = Schema(
            self.config["database"],
            self.config.get("filters")
        )

        self.workload_generator = WorkloadGenerator(
            self.config["workload"],
            columns=self.schema.columns,
            views=self.schema.views,
            random_seed=self.config["random_seed"],
            db_config=self.schema.db_config,
            experiment_id=self.id,
            filter_utilized_columns=self.config["filter_utilized_columns"],
            logging_mode=self.config["logging"]
        )

        self._assign_budgets_to_workloads()
        # TODO: для чего это?
        self._pickle_workloads()

        self.globally_indexable_columns = self.workload_generator.globally_indexable_columns

        """
        [
            [single column indexes], 
            [2-column combinations], 
            [3-column combinations]
            ...
        ]
        """
        self.globally_indexable_columns = utils.create_column_permutation_indexes(
            self.globally_indexable_columns,
            self.config["max_index_width"]
        )

        self.single_column_flat_set = set(map(lambda x: x[0], self.globally_indexable_columns[0]))

        self.globally_indexable_columns_flat = [item for sublist in self.globally_indexable_columns for item in sublist]
        logging.info(f"Feeding {len(self.globally_indexable_columns_flat)} candidates into the environments.")

        # List of index storage consumption [8192, 8192, 0, 0 ...]
        # First - one-column indexes, later - multi-column indexes
        # Consumption multi-column consumption calculated as
        # (size(multi-column) - size(multi-column[:-1]))
        self.action_storage_consumptions = utils.predict_index_sizes(
            self.globally_indexable_columns_flat, self.schema.db_config
        )

        # TODO: workload_embedder

        # TODO: where multi_validation_wl is used?
        self.multi_validation_wl = []
        if len(self.workload_generator.wl_validation) > 1:
            for workloads in self.workload_generator.wl_validation:
                self.multi_validation_wl.extend(
                    self.rnd.sample(workloads, min(7, len(workloads)))
                )



    def _init_time(self):
        self.start_time = datetime.now()

        self.end_time = None
        self.training_start_time = None
        self.training_end_time = None

    def _create_environment_folder(self):
        if os.path.exists(self.ENVIRONMENT_RESULT_PATH):
            assert os.path.isdir(self.ENVIRONMENT_RESULT_PATH), \
                f"Folder for environment results must be a folder, not a file: ./{self.ENVIRONMENT_RESULT_PATH}"
        else:
            os.makedirs(self.ENVIRONMENT_RESULT_PATH)

        self.environment_folder_path = f"{self.ENVIRONMENT_RESULT_PATH}/ID_{self.id}"

        if not os.path.exists(self.environment_folder_path):    
            os.mkdir(self.environment_folder_path)
        else:
            logging.warning(
                f"Experiment folder already exists at: ./{self.environment_folder_path} - "
                "terminating here because we don't want to overwrite anything."
            )

    def _assign_budgets_to_workloads(self):
        # TODO: training budget?

        for workload_list in self.workload_generator.wl_testing:
            for workload in workload_list:
                workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])

        for workload_list in self.workload_generator.wl_validation:
            for workload in workload_list:
                workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])

    def _pickle_workloads(self):
        with open(f"{self.environment_folder_path}/testing_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{self.environment_folder_path}/validation_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)
