from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.control_utils as CU

from robomimic.envs.wrappers import EnvWrapper
from robomimic.envs.env_robosuite import EnvRobosuite

from termcolor import colored
from copy import deepcopy


robot0_history = []
robot1_history = [] # TODO: more elegant way to handle this?
robot0_aside = False
robot1_aside = False

class NavigationPlanner:
    def __init__(self, env, obs, extra_para, id=0):
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        self.pid_eef_pos_ctlr = deepcopy(CU.pid_eef_pos_ctlr)
        self.pid_eef_axisangle_ctlr = deepcopy(CU.pid_eef_axisangle_ctlr)
        self.pid_base_pos_ctlr = deepcopy(CU.pid_base_pos_ctlr)
        self.pid_base_ori_ctlr = deepcopy(CU.pid_base_ori_ctlr)
        self.pid_base_height_ctlr = deepcopy(CU.pid_base_height_ctlr)
        
        self.task_stage = 0
        self.id = id

        # used in non-direct navigation task
        self.middle_dist = 0.5

        # used in direct (e.g. navigate frontward) navigation task
        self.direct_dist = 0.8
        self.direct = False

        # used in navigate aside task
        self.aside_dist = 1.35
        self.navigate_aside = False
        
        global robot0_history, robot1_history
        
        # initialize history if history is empty
        if isinstance(env.init_robot_base_pos, list):
            if self.id == 0 and not robot0_history:
                robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos[0]))
                robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos[0]))
            elif self.id == 1 and not robot1_history:
                robot1_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos[1]))
                robot1_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos[1]))
        else:
            assert self.id == 0
            if not robot0_history:
                robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos))
                robot0_history.append(env.compute_robot_base_placement_pose(env.init_robot_base_pos))
        
        # get target fixture randomly
        if extra_para == None:
            
            fixtures = list(env.fixtures.values())
            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls for cls in fxtr_classes if fxtr_classes.count(cls) == 1 and cls in [
                    "CoffeeMachine", "Toaster", "Stove", "Stovetop", "OpenCabinet",
                    "Microwave", "Sink", "Hood", "Oven", "Fridge", "Dishwasher",
                ]
            ]
            while True:
                self.target_fixture = env.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if fxtr_class not in valid_target_fxtr_classes:
                    continue
                break
                
            self.target_pos, self.target_ori = env.compute_robot_base_placement_pose(self.target_fixture)
        
        # navigate back to the second to last position
        elif extra_para == 'back':
            if self.id == 0:
                self.target_pos, self.target_ori = robot0_history[-2]
                robot0_history.append((self.target_pos, self.target_ori))
            elif self.id == 1:
                self.target_pos, self.target_ori = robot1_history[-2]
                robot1_history.append((self.target_pos, self.target_ori))
        
        # "navigate aside", used in two agent environment, equivalent to "navigate backward"
        # TODO: find a more elegant and general way, current code can only be applied in layout_id = 0 or 10
        elif extra_para == 'aside':
            self.direct = True
            base_pos = obs[f'robot{self.id}_base_pos'][:2]
            base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
            self.target_pos = base_pos + np.array([-np.cos(base_ori), -np.sin(base_ori)]) * self.aside_dist
            self.target_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))
            if self.id == 0:
                robot0_history.append((self.target_pos, self.target_ori))
            elif self.id == 1:
                robot1_history.append((self.target_pos, self.target_ori))
            self.navigate_aside = True
        
        # navigate at certain direction directly
        elif extra_para.endswith('ward'):
            self.direct = True

            # "navigate frontward"
            if extra_para == 'frontward':
                base_pos = obs[f'robot{self.id}_base_pos'][:2]
                base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
                self.target_pos = base_pos + np.array([np.cos(base_ori), np.sin(base_ori)]) * self.direct_dist
                self.target_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))
            
            # "navigate backward"
            elif extra_para == 'backward':
                base_pos = obs[f'robot{self.id}_base_pos'][:2]
                base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
                self.target_pos = base_pos + np.array([-np.cos(base_ori), -np.sin(base_ori)]) * self.direct_dist
                self.target_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))
            
            # "navigate leftward"
            elif extra_para == 'rightward':
                base_pos = obs[f'robot{self.id}_base_pos'][:2]
                base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
                self.target_pos = base_pos + np.array([np.sin(base_ori), -np.cos(base_ori)]) * self.direct_dist
                self.target_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))
            
            # "navigate rightward"
            elif extra_para == 'leftward':
                base_pos = obs[f'robot{self.id}_base_pos'][:2]
                base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
                self.target_pos = base_pos + np.array([-np.sin(base_ori), np.cos(base_ori)]) * self.direct_dist
                self.target_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))

            else:
                raise ValueError(f'there is no available direction {extra_para}!')
        
        # navigate to the given fixture or object
        else:
            
            # handle special object or fixture cases

            # "navigate to counter"
            if extra_para == 'counter':
                # find the object on the counter
                for obj_cfg in env.object_cfgs:
                    if 'fixture' in obj_cfg['placement'] and type(obj_cfg['placement']['fixture']).__name__.lower() == "counter":
                        extra_para = obj_cfg['name']
                        break
            
            # "navigate to cabinet"
            elif extra_para == 'cabinet':
                # find the object in the cabinet
                for obj_cfg in env.object_cfgs:
                    if 'fixture' in obj_cfg['placement'] and 'cabinet' in type(obj_cfg['placement']['fixture']).__name__.lower():
                        extra_para = obj_cfg['name']
                        break
            
            # get valid object and fixture keys
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
            
            # navigate to the given object
            if extra_para in object_keys: # navigate to the given object
                obj_str = extra_para # obj_str should be one of the names in env._get_obj_cfgs()
                obj = env.objects[obj_str] # rot
                obj.pos = obs[obj_str + '_pos'] # pos
                self.target_pos, self.target_ori = env.compute_robot_base_placement_pose(obj)
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))
            
            # navigate to the given fixture
            elif extra_para in fixture_keys: # navigate to the given fixture
                fixture_str = extra_para
                for fxtr in fixtures:
                    if type(fxtr).__name__.lower() == fixture_str:
                        fixture = fxtr
                self.target_pos, self.target_ori = env.compute_robot_base_placement_pose(fixture)
                if self.id == 0:
                    robot0_history.append((self.target_pos, self.target_ori))
                elif self.id == 1:
                    robot1_history.append((self.target_pos, self.target_ori))
            
            else:
                raise ValueError(f'there is no fixture or object {extra_para} in the environment!')
        
        # get base position and orientation
        base_pos = obs[f'robot{self.id}_base_pos'][:2]
        base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
        
        # get target position and orientation
        target_pos = self.target_pos[:2]
        target_ori = self.target_ori[2]
        
        # calculate middle point 1 and middle point 2
        self.init_pos = base_pos
        self.middle1_pos = np.array([1.5, -1.5])
        self.middle1_pos[0] = base_pos[0] - self.middle_dist * np.cos(base_ori)
        self.middle1_pos[1] = base_pos[1] - self.middle_dist * np.sin(base_ori)
        self.middle2_pos = np.array([1.5, -1.5])
        self.middle2_pos[0] = target_pos[0] - self.middle_dist * np.cos(target_ori)
        self.middle2_pos[1] = target_pos[1] - self.middle_dist * np.sin(target_ori)
    
    def get_control(self, env=None, obs=None):
        """
        control method designed for navigation task
        """
        
        if isinstance(env, EnvWrapper):
            env = env.env
        if isinstance(env, EnvRobosuite):
            env = env.env
        
        end_control = False
    
        # get base position and orientation
        base_pos = obs[f'robot{self.id}_base_pos'][:2]
        base_ori = T.mat2euler(T.quat2mat(obs[f'robot{self.id}_base_quat']))[2]
        
        # get target position and orientation
        target_pos = self.target_pos[:2]
        target_ori = self.target_ori[2]

        # directly go to stage 3 if self.direct is True
        if self.direct == True:
            self.task_stage = 3
        
        # directly go to stage 1 if robot is aside
        global robot0_aside, robot1_aside
        if self.id == 0 and robot0_aside:
            self.task_stage = 1
            robot0_aside = False
        elif self.id == 1 and robot1_aside:
            self.task_stage = 1
            robot1_aside = False
        
        # move to middle position 1
        if self.task_stage == 0:
            action = self.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=self.middle1_pos)
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, id=self.id)
            if np.linalg.norm(self.middle1_pos - base_pos) <= 0.05:
                self.task_stage += 1
                self.pid_base_pos_ctlr.reset()
        
        # then turn the orientation
        elif self.task_stage == 1:
            delta = target_ori - base_ori 
            delta_adjusted = (delta + np.pi) % (2 * np.pi) - np.pi # normalize delta to [-π, π]
            adjusted_target = base_ori + delta_adjusted # adjust target to ensure PID follows the shortest path
            tz = self.pid_base_ori_ctlr.compute(current_value=base_ori, target_value=adjusted_target)
            action = CU.map_action(tz, base_ori) # 1-dim
            action = CU.create_action(base_ori=action, joint="stable", id=self.id)
            if np.cos(target_ori - base_ori) >= 0.998:
                self.task_stage += 1
                self.pid_base_ori_ctlr.reset()
        
        # then move to middle position 2
        elif self.task_stage == 2:
            action = self.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=self.middle2_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, id=self.id)
            if np.linalg.norm(self.middle2_pos - base_pos) <= 0.05:
                self.task_stage += 1
                self.pid_base_pos_ctlr.reset()
            
        # finally move to target position
        elif self.task_stage == 3:
            action = self.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, id=self.id)
            if np.linalg.norm(target_pos - base_pos) <= 0.05:
                # reset all planner related infomation 
                self.task_stage = 0 
                self.init_pos = None
                self.middle1_pos = None
                self.middle2_pos = None
                self.pid_base_pos_ctlr.reset()
                end_control = True

                if self.navigate_aside:
                    if self.id == 0:
                        robot0_aside = True
                    elif self.id == 1:
                        robot1_aside = True
        
        info = {'end_control': end_control, 'arm_need_reset': False}
        return action, info