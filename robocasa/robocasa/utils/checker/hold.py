from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def hold_checker(env, extra_para=None, id=0):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    obj_contact = False
    if extra_para is not None:
        assert extra_para in env.objects.keys()
        obj = env.objects[extra_para]
        obj_contact = env.check_contact(obj, env.robots[id].gripper["right"])
    else:
        for key, obj in env.objects.items():
            obj_contact = env.check_contact(obj, env.robots[id].gripper["right"])
            if obj_contact:
                break
    success = obj_contact
    
    return success


def not_hold_checker(env, id=0):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    obj_contact = False
    for key, obj in env.objects.items():
        obj_contact = env.check_contact(obj, env.robots[id].gripper["right"])
        if obj_contact:
            break
    success = not obj_contact
    
    return success

