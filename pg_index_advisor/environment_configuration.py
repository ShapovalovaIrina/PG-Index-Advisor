import logging
import random
import os

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

