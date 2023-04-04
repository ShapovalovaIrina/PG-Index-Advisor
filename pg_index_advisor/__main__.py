import logging
import sys
import importlib

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO

from index_advisor import IndexAdvisor

PARALLEL_ENVIRONMENTS = 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    assert len(sys.argv) == 2, "System configuration file must be provided: main.py path_fo_file.json"
    CONFIGURATION_FILE = sys.argv[1]

    '''
    Parse configuration file and setup environment.
    '''
    index_advisor = IndexAdvisor(CONFIGURATION_FILE)

    index_advisor.prepare()

    VectorizedEnv = SubprocVecEnv if PARALLEL_ENVIRONMENTS > 1 else DummyVecEnv

    training_env = VectorizedEnv(
        [index_advisor.make_env(env_id) for env_id in range(PARALLEL_ENVIRONMENTS)]
    )
    training_env = VecNormalize(
        training_env,
        norm_obs=True,
        norm_reward=True,
        gamma=index_advisor.config["rl_algorithm"]["gamma"],
        training=True
    )

