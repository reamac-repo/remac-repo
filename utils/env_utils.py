def unwrap_env(env):
    """
    unwrap env to get the real env
    """
    from robomimic.envs.wrappers import EnvWrapper
    from robomimic.envs.env_robosuite import EnvRobosuite
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    return env