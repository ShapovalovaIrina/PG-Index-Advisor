import logging
import random
import os
import utils
import gym

from configuration_parser import ConfigurationParser
from datetime import datetime
from sb3_contrib.common.wrappers import ActionMasker
from typing import List

from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from schema.schema import Schema
from workload_generator import WorkloadGenerator
from embeddings.workload_embedder import PlanEmbedderLSI
from gym_env.common import EnvironmentType
from gym_env.action_manager import MultiColumnIndexActionManager as ActionManager
from gym_env.observation_manager import SingleColumnIndexPlanEmbeddingObservationManagerWithCost as ObservationManager
from gym_env.reward_manager import CostAndStorageRewardManager as RewardManager
from gym_env.env import PGIndexAdvisorEnv


def mask_fn(env: PGIndexAdvisorEnv) -> List[bool]:
    return env.valid_action_mask()


class IndexAdvisor(object):
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
        self.workload_embedder = None
        self.evaluated_workloads_strs = []
        self.globally_indexable_columns = []
        self.single_column_flat_set = set()
        self.globally_indexable_columns_flat = []
        self.action_storage_consumptions = []

        self.start_time = None
        self.end_time = None
        self.training_start_time = None
        self.training_end_time = None

        self.result_path = self.config["result_path"]
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
        # TODO: filter globally_indexable_columns by existing indexes

        self.single_column_flat_set = set(map(lambda x: x[0], self.globally_indexable_columns[0]))

        self.globally_indexable_columns_flat = [item for sublist in self.globally_indexable_columns for item in sublist]
        logging.info(f"Feeding {len(self.globally_indexable_columns_flat)} candidates into the environments.")

        """
        List of index storage consumption [8192, 8192, 0, 0 ...]
        First - one-column indexes, later - multi-column indexes
        Consumption multi-column consumption calculated as
        (size(multi-column) - size(multi-column[:-1]))
        """
        self.action_storage_consumptions = utils.predict_index_sizes(
            self.globally_indexable_columns_flat,
            self.schema.db_config,
            self.config["logging"]
        )

        self.workload_embedder = PlanEmbedderLSI(
            self.workload_generator.query_texts,
            self.config["workload_embedder"]["representation_size"],
            self.globally_indexable_columns,
            self.schema.db_config
        )

        # TODO: in original paper there is multi_validation_wl, needs to figure out why this is
        assert len(self.workload_generator.wl_validation) == 1, \
            "Expected wl_validation to be one element list"

    def make_env(
            self,
            env_id,
            environment_type=EnvironmentType.TRAINING,
            workloads_in=None
    ):
        def _init():
            action_manager = ActionManager(
                indexable_column_combinations=self.globally_indexable_columns,
                indexable_column_combinations_flat=self.globally_indexable_columns_flat,
                action_storage_consumption=self.action_storage_consumptions,
                max_index_width=self.config["max_index_width"],
                reenable_indexes=self.config["reenable_indexes"]
            )

            if self.number_of_actions is None:
                self.number_of_actions = action_manager.number_of_actions

            observation_manager_config = {
                "workload_embedder": self.workload_embedder,
                "workload_size": self.config["workload"]["size"]
            }
            observation_manager = ObservationManager(
                number_of_actions=action_manager.number_of_columns,
                config=observation_manager_config
            )
            if self.number_of_features is None:
                self.number_of_features = observation_manager.number_of_features

            reward_manager = RewardManager()

            if workloads_in is None:
                workloads = {
                    EnvironmentType.TRAINING: self.workload_generator.wl_training,
                    EnvironmentType.TESTING: self.workload_generator.wl_testing[-1],
                    EnvironmentType.VALIDATION: self.workload_generator.wl_validation[-1]
                }[environment_type]
            else:
                workloads = workloads_in

            env_config = {
                "database": self.schema.db_config,
                "globally_indexable_columns": self.globally_indexable_columns_flat,
                "workloads": workloads,
                "action_manager": action_manager,
                "observation_manager": observation_manager,
                "reward_manager": reward_manager,
                "random_seed": self.config["random_seed"] + env_id,
                "max_steps_per_episode": self.config["max_steps_per_episode"],
                "env_id": env_id,
                "similar_workloads": self.config["workload"]["similar_workloads"]
            }
            env = gym.make(
                "PGIndexAdvisor-v0",
                environment_type=environment_type,
                config=env_config
            )
            env = ActionMasker(env, mask_fn)

            return env

        set_random_seed(self.config["random_seed"])

        return _init

    def set_model(self, model):
        self.model = model

    def _init_time(self):
        self.start_time = datetime.now()

    def start_learning_time(self):
        self.training_start_time = datetime.now()

    def finish_learning_time(self):
        self.training_end_time = datetime.now()

    def _create_environment_folder(self):
        if os.path.exists(self.result_path):
            assert os.path.isdir(self.result_path), \
                f"Folder for environment results must be a folder, not a file: {self.result_path}"
        else:
            os.makedirs(self.result_path)

        self.folder_path = f"{self.result_path}/ID_{self.id}"

        if not os.path.exists(self.folder_path):    
            os.mkdir(self.folder_path)
        else:
            logging.warning(
                f"Experiment folder already exists at: {self.folder_path} - "
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

    def get_callback(
            self,
            env_type: EnvironmentType,
            best_model_save_path=None,
            parallel_environments=1
    ):
        callback_env = VecNormalize(
            DummyVecEnv([self.make_env(0, env_type)]),
            norm_obs=True,
            norm_reward=False,
            gamma=self.config["rl_algorithm"]["gamma"],
            training=False
        )
        callback = MaskableEvalCallback(
            eval_env=callback_env,
            n_eval_episodes=self.config["workload"]["validation_testing"]["number_of_workloads"],
            eval_freq=round(
                self.config["validation_frequency"] / parallel_environments
            ),
            verbose=1,
            deterministic=True,
            best_model_save_path=best_model_save_path
        )

        return callback
