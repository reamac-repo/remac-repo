from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def turn_off_stove_checker(env):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    assert hasattr(env, "stove")
    knobs_state = env.stove.get_knobs_state(env=env)

    location = "front_right"
    assert location in env.stove.burner_sites.keys() and env.stove.burner_sites[location] is not None
    knob_on = (0.35 <= np.abs(knobs_state[location]) <= 2 * np.pi - 0.35) if location in knobs_state else False
    success = not knob_on

    return success