from robocasa.environments.kitchen.kitchen import *

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite


def location_checker(env, obs, extra_para, id=0):
    
    if isinstance(env, EnvWrapper):
        env = env.env
    if isinstance(env, EnvRobosuite):
        env = env.env
    
    object_keys = env.objects.keys()
    
    fixtures = list(env.fixtures.values())
    fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
    valid_target_fxtr_classes = [
        cls for cls in fxtr_classes if fxtr_classes.count(cls) == 1 and cls in [
            "CoffeeMachine", "Toaster", "Stove", "Stovetop", "OpenCabinet",
            "Microwave", "Sink", "Hood", "Oven", "Fridge", "Dishwasher",
        ]
    ]
    fixture_keys = [fxtr.lower() for fxtr in valid_target_fxtr_classes]
    
    if extra_para in object_keys: # navigate to the given object
        obj_str = extra_para # obj_str should be one of the names in env._get_obj_cfgs()
        obj = env.objects[obj_str] # rot
        obj.pos = obs[obj_str + '_pos'] # pos
        target_pos, target_ori = env.compute_robot_base_placement_pose(obj)
        
    elif extra_para in fixture_keys: # navigate to the given fixture
        fixture_str = extra_para
        for fxtr in fixtures:
            if type(fxtr).__name__.lower() == fixture_str:
                fixture = fxtr
        target_pos, target_ori = env.compute_robot_base_placement_pose(fixture)
    
    else:
        raise ValueError(f'there is no fixture or object {extra_para} in the environment!')
    
    robot_id = env.sim.model.body_name2id(f"base{id}_base")
    base_pos = np.array(env.sim.data.body_xpos[robot_id])
    pos_check = np.linalg.norm(target_pos[:2] - base_pos[:2]) <= 0.20
    base_ori = T.mat2euler(np.array(env.sim.data.body_xmat[robot_id]).reshape((3, 3)))
    ori_check = np.cos(target_ori[2] - base_ori[2]) >= 0.98

    return pos_check and ori_check

