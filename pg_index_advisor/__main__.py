import logging
import copy

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from sb3_contrib.ppo_mask import MaskablePPO

import pg_index_advisor.cli_parser
from index_advisor import IndexAdvisor
from gym_env.common import EnvironmentType

PARALLEL_ENVIRONMENTS = 1


def learn(config_file):
    index_advisor = get_index_advisor(config_file)

    training_env = get_env(index_advisor, EnvironmentType.TRAINING)

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

    # Callbacks
    test_callback = index_advisor.get_callback(
        EnvironmentType.TESTING
    )
    validation_callback = index_advisor.get_callback(
        EnvironmentType.VALIDATION,
        best_model_save_path=index_advisor.folder_path
    )

    callbacks = [validation_callback, test_callback]

    index_advisor.start_learning_time()

    model.learn(
        total_timesteps=index_advisor.config["timesteps"],
        callback=callbacks,
        tb_log_name=index_advisor.id
    )

    index_advisor.finish_learning_time()
    index_advisor.finish_learning(training_env)
    index_advisor.write_report()


def predict(config_file):
    index_advisor = get_index_advisor(config_file)

    env = get_env(index_advisor, EnvironmentType.VALIDATION)

    model = MaskablePPO.load(f"{index_advisor.folder_path}/best_model.zip", env=env)

    vec_env = model.get_env()
    obs = vec_env.reset()

    done = False

    while not done:
        action_mask = vec_env.env_method("valid_action_mask")
        action, _states = model.predict(
            obs,
            deterministic=True,
            action_masks=action_mask
        )
        obs, rewards, dones, info = vec_env.step(action)

        columns = index_advisor.globally_indexable_columns_flat[action[0]]
        done = dones[0]

        print(f"""
        Action: {columns}.
        Reward: {rewards[0]}.
        Done: {done}.
        """)


def get_index_advisor(config_file):
    """
    Parse configuration file and setup environment.
    """
    index_advisor = IndexAdvisor(config_file)

    index_advisor.prepare()

    return index_advisor


def get_env(index_advisor, env_type=EnvironmentType.TRAINING):
    if env_type == EnvironmentType.TRAINING:
        norm_reward = True
        training = True
        VectorizedEnv = SubprocVecEnv if PARALLEL_ENVIRONMENTS > 1 else DummyVecEnv
        env = VectorizedEnv([
            index_advisor.make_env(env_id, env_type)
            for env_id
            in range(PARALLEL_ENVIRONMENTS)
        ])
    else:
        norm_reward = False
        training = False
        env = DummyVecEnv([index_advisor.make_env(0, env_type)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=norm_reward,
        gamma=index_advisor.config["rl_algorithm"]["gamma"],
        training=training
    )

    return env


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = pg_index_advisor.cli_parser.CLIParser().create_parser()
    args = parser.parse_args()

    config_file = args.config
    action = args.action

    if action == 'learn':
        learn(config_file)
    elif action == 'predict':
        predict(config_file)

