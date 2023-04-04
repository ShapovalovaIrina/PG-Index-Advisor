from gym.envs.registration import register

register(
    id='PGIndexAdvisor-v0',
    entry_point="gym_env.env:PGIndexAdvisorEnv"
)
