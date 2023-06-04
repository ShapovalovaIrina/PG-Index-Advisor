import copy
import gzip
import json
import logging
import pickle
import random
import os
import subprocess

import gym
import numpy as np

from pg_index_advisor import utils
from pg_index_advisor.configuration_parser import ConfigurationParser
from pg_index_advisor import utils
from datetime import datetime, timedelta
from typing import List, Optional

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.utils import set_random_seed

from pg_index_advisor.database.schema import Schema
from pg_index_advisor.workload_generator import WorkloadGenerator
from pg_index_advisor.gym_env.common import EnvironmentType
from pg_index_advisor.gym_env.action_manager import MultiColumnIndexActionManager as ActionManager
from pg_index_advisor.gym_env.observation_manager import SingleColumnIndexPlanEmbeddingObservationManagerWithCost as ObservationManager
from pg_index_advisor.gym_env.reward_manager import CostAndStorageRewardManager as RewardManager
from pg_index_advisor.gym_env.env import PGIndexAdvisorEnv


def mask_fn(env: PGIndexAdvisorEnv) -> List[bool]:
    return env.valid_action_mask()


class IndexAdvisor(object):
    model: Optional[MaskablePPO]

    def __init__(self, configuration_file):
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

        self._init_time()

    def prepare(self, budget=None):
        """
        Get filtered DB database elements:
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
            query_file=self.config["query_file"],
            logging_mode=self.config["logging"]
        )

        self._assign_budgets_to_workloads(budget)

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
                initial_indexes=self.schema.indexes,
                max_index_width=self.config["max_index_width"]
            )

            if self.number_of_actions is None:
                self.number_of_actions = action_manager.number_of_actions

            observation_manager_config = {
                "workload_embedder": {
                    "query_texts": self.workload_generator.query_texts,
                    "representation_size": self.config["workload_embedder"]["representation_size"],
                    "globally_indexable_columns": copy.copy(self.globally_indexable_columns),
                    "db_config": self.schema.db_config
                },
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
                "similar_workloads": self.config["workload"]["similar_workloads"],
                "initial_indexes": self.schema.indexes
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
                f"Experiment folder already exists at: {self.folder_path}."
            )

    def _assign_budgets_to_workloads(self, budget):
        # TODO: training budget?

        budgets = self.config["budgets"]["validation_and_testing"]

        def get_budget(i):
            wl_budget = copy.copy(budget)

            if wl_budget is None:
                i = min(i, len(budgets))
                wl_budget = budgets[i]

            # if wl_budget is None:
            #     wl_budget = self.rnd.choice(budgets)
            return wl_budget

        for workload_list in self.workload_generator.wl_testing:
            for (i, workload) in enumerate(workload_list):
                workload.budget = get_budget(i)

        for workload_list in self.workload_generator.wl_validation:
            for (i, workload) in enumerate(workload_list):
                workload.budget = get_budget(i)

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

    def finish_learning(self, training_env: VecNormalize, parallel_environments=1):
        self.moving_average_validation_model_at_step = None
        self.best_mean_model_step = None

        self.model.save(f"{self.folder_path}/final_model")
        training_env.save(f"{self.folder_path}/training_env_vec_normalize.pkl")

        self.evaluated_episodes = 0
        for number_of_resets in training_env.get_attr("number_of_resets"):
            self.evaluated_episodes += number_of_resets

        self.total_steps_taken = 0
        for total_number_of_steps in training_env.get_attr("total_number_of_steps"):
            self.total_steps_taken += total_number_of_steps

        self.cache_hits = 0
        self.cost_requests = 0
        self.costing_time = timedelta(0)
        for cache_info in training_env.env_method("get_cost_eval_cache_info"):
            self.cache_hits += cache_info[1]
            self.cost_requests += cache_info[0]
            self.costing_time += cache_info[2]
        self.costing_time /= parallel_environments

        self.cache_hit_ratio = self.cache_hits / self.cost_requests * 100

        if self.config["pickle_cost_estimation_caches"]:
            caches = []
            for cache in training_env.env_method("get_cost_eval_cache"):
                caches.append(cache)
            combined_caches = {}
            for cache in caches:
                combined_caches = {**combined_caches, **cache}
            with gzip.open(f"{self.folder_path}/caches.pickle.gzip", "wb") as handle:
                pickle.dump(combined_caches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def write_report(self, filename=None):
        self.end_time = datetime.now()

        self.model.training = False
        self.model.env.norm_reward = False
        self.model.env.training = False

        # FM - Final Model
        test_fm = self._get_model_performance(self.model, EnvironmentType.TESTING)
        vali_fm = self._get_model_performance(self.model, EnvironmentType.VALIDATION)

        best_reward_model = MaskablePPO.load(f"{self.folder_path}/best_model.zip")
        best_reward_model.training = False

        # BM - Best Model
        test_bm = self._get_model_performance(best_reward_model, EnvironmentType.TESTING)
        vali_bm = self._get_model_performance(best_reward_model, EnvironmentType.VALIDATION)

        # TODO: multi validation

        models = {
            'fm': {
                'test': test_fm,
                'vali': vali_fm
            },
            'bm': {
                'test': test_bm,
                'vali': vali_bm
            }
        }

        if filename is None:
            filename = f"{self.folder_path}/report_ID_{self.id}.txt"
        self._write_report(models, filename)

        logging.critical(
            (
                f"Finished training of ID {self.id}. Report can be found at "
                f"./{filename}"
            )
        )

    def _get_model_performance(self, model, environment_type: EnvironmentType):
        wl_list = {
            EnvironmentType.TESTING: self.workload_generator.wl_testing,
            EnvironmentType.VALIDATION: self.workload_generator.wl_validation
        }[environment_type]

        model_performances = []
        for wl in wl_list:
            env = DummyVecEnv([self.make_env(0, environment_type, wl)])
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=False,
                gamma=self.config["rl_algorithm"]["gamma"],
                training=False,
            )

            if model != self.model:
                model.set_env(self.model.env)

            model_performance = self._evaluate_model(model, env, len(wl))
            model_performances.append(model_performance)

        return model_performances

    @staticmethod
    def _evaluate_model(model, evaluation_env, n_eval_episodes):
        training_env = model.get_vec_normalize_env()
        sync_envs_normalization(training_env, evaluation_env)

        evaluate_policy(model, evaluation_env, n_eval_episodes)

        episode_performances = evaluation_env.get_attr("episode_performances")[0]
        perfs = []
        for perf in episode_performances:
            perfs.append(round(perf["achieved_cost"], 2))

        mean_performance = np.mean(perfs)
        print(f"Mean performance: {mean_performance:.2f} ({perfs})")

        return episode_performances, mean_performance, perfs

    def _write_report(self, models, filename=None):
        probabilities = self.config["workload"]["validation_testing"]["unknown_query_probabilities"]
        probabilities_len = len(probabilities)

        def final_avg(values):
            val = 0
            for res in values:
                val += res[1]
            return val / probabilities_len

        with open(filename, "w") as f:
            f.write(f"##### Report for Experiment with ID: {self.id} #####\n")
            f.write(f"Description: {self.config['description']}\n")
            f.write("\n")

            f.write(f"Start:                         {self.start_time}\n")
            f.write(f"End:                           {self.end_time}\n")
            f.write(f"Duration:                      {self.end_time - self.start_time}\n")
            f.write("\n")
            f.write(f"Start Training:                {self.training_start_time}\n")
            f.write(f"End Training:                  {self.training_end_time}\n")
            f.write(f"Duration Training:             {self.training_end_time - self.training_start_time}\n")
            f.write(f"Git Hash:                      {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}\n")
            f.write(f"Number of features:            {self.number_of_features}\n")
            f.write(f"Number of actions:             {self.number_of_actions}\n")
            f.write("\n")

            # TODO: unknown queries

            for idx, unknown_query_probability in enumerate(probabilities):
                f.write(f"Unknown query probability: {unknown_query_probability}:\n")
                f.write("    Final mean performance test:\n")

                # Final model
                test_fm_perfs, self.performance_test_final_model, self.test_fm_details = models['fm']['test'][idx]
                vali_fm_perfs, self.performance_vali_final_model, self.vali_fm_details = models['fm']['vali'][idx]

                # Best model
                test_bm_perfs, self.performance_test_best_mean_reward_model, self.test_bm_details = models['bm']['test'][idx]
                vali_bm_perfs, self.performance_vali_best_mean_reward_model, self.vali_bm_details = models['bm']['vali'][idx]


                self.test_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(test_fm_perfs)
                self.vali_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(vali_fm_perfs)

                self.test_fm_wl_consumption = self._get_wl_consumption_from_model_perfs(test_fm_perfs)
                self.vali_fm_wl_consumption = self._get_wl_consumption_from_model_perfs(vali_fm_perfs)

                self.test_bm_wl_consumption = self._get_wl_consumption_from_model_perfs(test_bm_perfs)
                self.vali_bm_wl_consumption = self._get_wl_consumption_from_model_perfs(vali_bm_perfs)

                f.write(
                    (
                        "        Final model:                         "
                        f"{self.performance_test_final_model:.2f} ({self.test_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Best mean reward model:              "
                        f"{self.performance_test_best_mean_reward_model:.2f} ({self.test_bm_details})\n"
                    )
                )

                f.write("\n")

                f.write(
                    (
                        "        Final model reward:                  "
                        f"{self.test_fm_wl_consumption}\n"
                    )
                )
                f.write(
                    (
                        "        Best mean reward model reward:       "
                        f"{self.test_bm_wl_consumption}\n"
                    )
                )

                f.write("\n")
                f.write(f"        Budgets:                            "
                        f"{self.test_fm_wl_budgets}\n")

                f.write("\n")
                f.write("    Final mean performance validation:\n")
                f.write(
                    (
                        "        Final model:                         "
                        f"{self.performance_vali_final_model:.2f} ({self.vali_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Best mean reward model:              "
                        f"{self.performance_vali_best_mean_reward_model:.2f} ({self.vali_bm_details})\n"
                    )
                )

                f.write("\n")

                f.write(
                    (
                        "        Final model reward:                  "
                        f"{self.vali_fm_wl_consumption}\n"
                    )
                )
                f.write(
                    (
                        "        Best mean reward model reward:       "
                        f"{self.vali_bm_wl_consumption}\n"
                    )
                )

                f.write("\n")
                f.write(f"        Budgets:                            "
                        f"{self.vali_fm_wl_budgets}\n")

                f.write("\n")
                f.write("\n")

            f.write("Overall Test:\n")

            f.write(
                "        Final model:           " 
                f"{final_avg(models['fm']['test']):.5f}\n"
            )
            f.write(
                "        Best model:            " 
                f"{final_avg(models['bm']['test']):.5f}\n"
            )
            # TODO: multi validation wl
            f.write("\n")

            f.write("Overall Validation:\n")
            f.write(
                "        Final model:           " 
                f"{final_avg(models['fm']['vali']):.5f}\n"
            )
            f.write(
                "        Best model:            " 
                f"{final_avg(models['bm']['vali']):.5f}\n"
            )
            # TODO: multi validation
            f.write("\n")
            f.write("\n")

            f.write(f"Evaluated episodes:            {self.evaluated_episodes}\n")
            f.write(f"Total steps taken:             {self.total_steps_taken}\n")
            f.write(
                (
                    f"CostEval cache hit ratio:      "
                    f"{self.cache_hit_ratio:.2f} ({self.cache_hits} of {self.cost_requests})\n"
                )
            )
            training_time = self.training_end_time - self.training_start_time
            f.write(
                f"Cost eval time (% of total):   {self.costing_time} ({self.costing_time / training_time * 100:.2f}%)\n"
            )

            f.write("\n\n")
            f.write("Used configuration:\n")
            json.dump(self.config, f)
            f.write("\n\n")

    @staticmethod
    def _get_wl_budgets_from_model_perfs(perfs):
        wl_budgets = []
        for perf in perfs:
            assert perf["evaluated_workload"].budget == perf["available_budget"], "Budget mismatch!"
            wl_budgets.append(perf["evaluated_workload"].budget)
        return wl_budgets

    @staticmethod
    def _get_wl_consumption_from_model_perfs(perfs):
        wl_consumption = []
        for perf in perfs:
            wl_consumption.append(f"{perf['reward']:.6f}")
        return wl_consumption

