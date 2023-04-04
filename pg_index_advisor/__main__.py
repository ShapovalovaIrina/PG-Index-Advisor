import logging
import sys
import importlib
import copy

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

    # TODO: save env state to file (pickle)

    # This is necessary because SB modifies the passed dict.
    model_architecture = copy.copy(index_advisor.config["rl_algorithm"]["model_architecture"])

    model = MaskablePPO(
        policy=index_advisor.config["rl_algorithm"]["policy"],
        env=training_env,
        verbose=2,
        seed=index_advisor.config["random_seed"],
        gamma=index_advisor.config["rl_algorithm"]["gamma"],
        tensorboard_log="tensor_log",
        policy_kwargs=model_architecture,
        **index_advisor.config["rl_algorithm"]["args"]
    )
    logging.info(f"Creating model with NN architecture: {model_architecture}")

    index_advisor.set_model(model)

