from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def turn_off_faucet_checker(env):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    assert hasattr(env, "sink")
    handle_state = env.sink.get_handle_state(env=env)        
    water_on = handle_state["water_on"]
    success = not water_on
    
    return success