from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def inside_microwave_checker(env, extra_para=None):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    assert hasattr(env, "microwave")
    object_keys = env.objects.keys()
    
    if extra_para is not None:
        assert extra_para in env.objects.keys()
        obj = env.objects[extra_para]
        for key in object_keys: # could be multiple containers
            if key.endswith("container"): # container object should end with _container
                container = env.objects[key]
                obj_container_contact = env.check_contact(obj, container)
                container_micro_contact = env.check_contact(container, env.microwave)
                success = obj_container_contact and container_micro_contact
                break
    else:
        for key in object_keys:
            obj = env.objects[key]
            obj_micro_contact = env.check_contact(obj, env.microwave)
            success = obj_micro_contact
            break
    
    return success