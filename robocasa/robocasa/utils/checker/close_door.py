from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def close_microwave_door_checker(env):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    assert hasattr(env, "microwave")
    door_state = env.microwave.get_door_state(env=env)
    
    success = True
    for joint_p in door_state.values():
        if joint_p > 0.10:
            success = False
            break
    
    return success


def close_cabinet_door_checker(env):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    assert hasattr(env, "cab")
    door_state = env.cab.get_door_state(env=env)
    
    success = True
    for joint_p in door_state.values():
        if joint_p > 0.10:
            success = False
            break
    
    return success