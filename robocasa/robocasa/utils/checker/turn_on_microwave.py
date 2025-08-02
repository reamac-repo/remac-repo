from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def turn_on_microwave_checker(env):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    assert hasattr(env, "microwave")
    microwave_turned_on = env.microwave.get_state()["turned_on"]
    gripper_button_far = env.microwave.gripper_button_far(env, button="start_button")
    
    success = microwave_turned_on and gripper_button_far

    return success