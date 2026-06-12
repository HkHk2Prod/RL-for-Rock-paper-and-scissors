from gymnasium.envs.registration import register

register(
    id="gymnasium_env/RPSWorld-v0",
    entry_point="src.gymnasium_envs.envs:RPSEnv",
)