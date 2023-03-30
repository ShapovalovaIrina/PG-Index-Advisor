import logging
import sys
import importlib

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO

from environment_configuration import EnvironmentConfiguration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    assert len(sys.argv) == 2, "System configuration file must be provided: main.py path_fo_file.json"
    CONFIGURATION_FILE = sys.argv[1]

    '''
    Parse configuration file and setup environment.
    '''
    env_config = EnvironmentConfiguration(CONFIGURATION_FILE)

    env_config.prepare()

