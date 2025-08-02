import os
import sys
import openai
import re
from pprint import pprint
from termcolor import colored

from openai import AzureOpenAI
from openai import OpenAI

from PIL import Image
import base64

import robocasa
import robocasa.utils.checker as checker
from robocasa.utils.checker import * # import all checkers

BASE_PATH = os.path.abspath(robocasa.__file__ + '/../../../')
sys.path.append(BASE_PATH)
os.chdir(BASE_PATH)

from agents.client import OPENAI_CLIENT as CLIENT

import utils

class MixAgent:
    """
    A flexible agent to generate plan and command for robot in robocasa environment
    """
    def __init__(self, goal, env_info, view=1, task_name=None, id=0):
        """
        initialize VLMAgent with api_key, base_url and default_headers
        """
        
        # unset all_proxy and ALL_PROXY to avoid proxy issues
        os.environ['all_proxy'] = ""
        os.environ['ALL_PROXY'] = ""
        
        # passing parameters
        self.id = id
        self.goal = goal
        self.env_info = env_info
        self.view = view
        self.task_name = task_name
        self.client = CLIENT
        
        # initialize some histories
        self.history_plan = []
        self.history_execution = []
        self.history_pre_check_results = []
        self.history_post_check_results = []
        
        # checking target classification by condition prefix
        self.no_need_checks = [ # no need to check
            "No specific",
        ]
        self.ground_truth_checks = [ # use ground truth checkers
            "gripper is holding",
            "gripper is holding something",
            "gripper is not holding anything",
            "the stove is turned",
            "the microwave is turned",
        ]
        self.vlm_checks = []
    
    
    def get_initial_plan(self, verbose=False):
        """
        call the self.client to output initial plan based on self.goal and self.env_info
        """

        # few-shot examples
        examples = f"""
Q: 
- Environment infomation:
Available fixtures in environment: toaster, stove, fridge, microwave, coffeemachine, sink, dishwasher, opencabinet
Available objects in environment: vegetable, vegetable_container, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, <open or close> microwave door, <open or close> cabinet door, turn <on or off> faucet, turn on stove, turn on microwave
- Goal: pick up vegetable from counter and place it to the microwave
A: Plan: [navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave]

Q:
- Environment infomation:
Available fixtures in environment: sink, stove, fridge, microwave, opencabinet, dishwasher, coffeemachine, toaster
Available objects in environment: vegetable, vegetable_container, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, <open or close> microwave door, <open or close> cabinet door, turn <on or off> faucet, turn on stove, turn on microwave
- Goal: pick up vegetable from counter and place it to the cabinet
A: Plan: [navigate to cabinet, open cabinet door, navigate to vegetable, pick up vegetable, navigate to cabinet, place to cabinet]

Q:
- Environment infomation:
Available fixtures in environment: sink, dishwasher, toaster, stove, fridge, microwave, opencabinet, coffeemachine
Available objects in environment: pan, vegetable, vegetable_container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, <open or close> microwave door, <open or close> cabinet door, turn <on or off> faucet, turn on stove, turn on microwave
- Goal: heat the vegetable on the stove
A: Plan: [navigate to pan, pick up pan, navigate to stove, place to stove, navigate to vegetable, pick up vegetable, navigate to stove, place to pan, turn on stove]

Q:
- Environment infomation:
Available fixtures in environment: sink, dishwasher, toaster, stove, fridge, microwave, opencabinet, coffeemachine
Available objects in environment: fish, fish_container, bowl
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, <open or close> microwave door, <open or close> cabinet door, turn <on or off> faucet, turn on stove, turn on microwave
- Goal: defrost fish in the sink
A: Plan: [navigate to bowl, pick up bowl, navigate to sink, place to sink, navigate to fish, pick up fish, navigate to sink, place to bowl, turn on faucet, turn off faucet]

Q:
- Environment infomation:
Available fixtures in environment: sink, stove, fridge, microwave, opencabinet, dishwasher, coffeemachine, toaster
Available objects in environment: plate, vegetable, vegetable_container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, <open or close> microwave door, <open or close> cabinet door, turn <on or off> faucet, turn on stove, turn on microwave
- Goal: wash the vegetable and then heat it in the microwave
A: Plan: [navigate to vegetable, pick up vegetable, navigate to sink, place to sink, turn on faucet, turn off faucet, navigate to microwave, open microwave door, navigate to sink, pick up vegetable, navigate to microwave, place to microwave, close microwave door, turn on microwave]

""".strip()
        
        # prompt
        message = f"""
Imagine you are a kitchen robot high-level planner, and you are able to plan long-horizon and multi-stage tasks step by step given current environment and goal info.

The input contains two parts, namely environment information and goal:

1. **Environment infomation:** a string that contains available fixtures, objects, and commands in the environment. 
- **Available fixtures:** fixtures in the environment that you can use in the <fixture> in command.
- **Available objects:** objects in the environment that you can use in the <object> in command.
- **Available commands:** each command is a string that you can use to control the robot, with format as "command <object or fixture>" which needs to be strictly followed.

2. **Goal:** a string that contains the goal of the task, offen include hints on how to achieve the goal

You should first think carefully about the current environment and goal, and then generate a step-by-step plan to achieve the goal.
When planning, you should notice that some actions are dependent on the previous actions, and some actions may have pre-conditions that need to be satisfied before execution

**Reminders:** carefully consider the possible reminders when planning.
- the microwave is initially closed, and need to be opened first
- navigate to the object first before pick, place or manipulate something

**Output:** strictly follow the format: Plan: [<command0>, <command1>, ...]

**Few-Shot Examples:** 

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

Q:
- Environment infomation:
{self.env_info}
- Goal: {self.goal}
A: Plan: [<(new) command0>, <command1>, ...]
""".strip()
        
        content = self.client.get_response(message=message)
        
        plan_str = utils.extract_content(content, filter="A:")
        if verbose:
            print(plan_str)
        plan = utils.extract_content(plan_str, filter="Plan:[]").split(", ")
        
        self.history_plan.append(plan)
        self.history_execution.append(plan[0])
        
        return plan
    
    
    def get_plan(self, verbose=False):
        """
        call the self.client to output new plan and command based on last step and relevant check results
        """

        # few-shot examples
        examples = f"""
Q:
- Previous plan: [navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave] 
- Post-condition checking result: Post-condition is satisfied. 
- Pre-condition checking result: Pre-condition is satisfied. 
A: Plan: [open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave] 

Q: 
- Previous plan: [pick up vegetable, navigate to microwave, place to microwave] 
- Post-condition checking result: Post-condition is not satisfied; gripper is not holding the vegetable; pick up vegetable. 
- Pre-condition checking result: Pre-condition is satisfied. 
A: Plan: [pick up vegetable, navigate to microwave, place to microwave]

Q: 
- Previous plan: [open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave]
- Post-condition checking result: Post-condition is satisfied.
- Pre-condition checking result: Pre-condition is not satisfied; the robot is not in front of the microwave; [navigate to microwave, open microwave door].
A: Plan: [navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave] 

Q: 
- Previous plan: [navigate to microwave, open microwave door, place to microwave, close microwave door] 
- Post-condition checking result: Post-condition is satisfied. 
- Pre-condition checking result: Pre-condition is not satisfied; gripper is holding something; [navigate to counter, place to counter, navigate to microwave, open microwave door]. 
A: Plan: [navigate to counter, place to counter, navigate to microwave, open microwave door, place to microwave, close microwave door]

Q: 
- Previous plan: [open microwave door, place to microwave, close microwave door]
- Post-condition checking result: Post-condition is satisfied.
- Pre-condition checking result: Pre-condition is not satisfied; gripper is not holding anything; [navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave]
A: Plan: [navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave, close microwave door]

Q: 
- Previous plan: [open microwave door, place to microwave, close microwave door] 
- Post-condition checking result: Post-condition is satisfied. 
- Pre-condition checking result: Pre-condition is not satisfied; the microwave door is closed; [navigate to counter, place to counter, navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave] 
A: Plan: [navigate to counter, place to counter, navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave, close microwave door] 

Q: 
- Previous plan: [open microwave door, place to microwave, close microwave door] 
- Post-condition checking result: Post-condition is satisfied. 
- Pre-condition checking result: Pre-condition is not satisfied; gripper is not holding anything; [navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave] 
A: Plan: [navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave, close microwave door]

Q: 
- Previous plan: [open cabinet door, place to cabinet] 
- Post-condition checking result: Post-condition is satisfied. 
- Pre-condition checking result: Pre-condition is not satisfied; gripper is not holding anything; [navigate to vegetable, pick up vegetable, navigate to cabinet, place to cabinet] 
A: Plan: [navigate to vegetable, pick up vegetable, navigate to cabinet, place to cabinet]

Q:
- Previous plan: [place to sink, turn on faucet]
- Post-condition checking result: Post-condition is satisfied.
- Pre-condition checking result: Pre-condition is not satisfied; there is no bowl in the sink; [navigate to counter, place to counter, navigate to bowl, pick up bowl, navigate to sink, place to sink, navigate to fish, pick up fish, navigate to sink, place to bowl].
A: Plan: [navigate to counter, place to counter, navigate to bowl, pick up bowl, navigate to sink, place to sink, navigate to fish, pick up fish, navigate to sink, place to bowl, turn on faucet]

""".strip()
        
        # get message prompt
        message = f"""
Imagine you are a kitchen robot high-level planner, and you are able to plan long-horizon and multi-stage tasks step by step given previous plan and condition checking results.

**Previous plan** are a list of strings, each string is a command. 
- **Current task** is the first command in the previous plan, i.e., plan[0].
- **Next task** is the second command in the previous plan, i.e. plan[1].

The condition checking results include **pre-condition** and **post-condition** checking results. 

**Post-condition checking result** may include three components: 
1. Whether the post-condition for **current task** is satisfied. 
2. Reason why the condition is not satisfied (if not satisfied). 
3. Recommended next step (if not satisfied) to replace the previous plan[0]. 

**Pre-condition checking result** may include three components: 
1. Whether the pre-condition for **next task** is satisfied. 
2. Reason why the condition is not satisfied (if not satisfied). 
3. Recommended next steps (if not satisfied) to replace the previous plan[0] and plan[1].

**Detailed Guidance:**
- If the two conditions are all satisfied, then simply output the rest previous steps for completing remaining tasks, i.e., exclude plan[0] and output plan[1:]. 
- If only the post-condition is not satisfied, output a new plan where plan[0] is the recommended next step (always single step) of the post-condition checking result, and the rest part of the new plan, i.e., plan[1:] are basically consistent with previous plan[1:]. 
- If only the pre-condition is not satisfied, output a new plan where plan[0:n] is the recommended next steps (maybe length n) of the pre-condition checking result, and the rest part of the new plan, i.e., plan[n:] are basically consistent with previous plan[2:].
- If both conditions are not satisfied, then consider to adjust and output the plan according to post-condition checking result first. 

**Output:** strictly follow the format: <Plan: [(new) command0, (new) command1, ...]>

**Few-Shot Examples:** 

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

- Environment infomation:
{self.env_info}
- Goal: {self.goal}

Q: 
- Previous plan: {f"[{', '.join(self.history_plan[-1])}]"}
- Post-condition checking result: {self.history_post_check_results[-1]} 
- Pre-condition checking result: {self.history_pre_check_results[-1]} 
A: Plan: [<(new) command0>, <command1>, ...]
""".strip()
        
        content = self.client.get_response(message=message)

        plan_str = utils.extract_content(content, filter="A:")
        if verbose:
            print(plan_str)
        plan = utils.extract_content(plan_str, filter="Plan:[]").split(", ")
        
        self.history_plan.append(plan)
        self.history_execution.append(plan[0])
        
        return plan
    
    
    def delete_last_plan(self):
        """
        delete last plan and execution in case of invalid action generated
        """
        del self.history_plan[-1]
        del self.history_execution[-1]
    
    
    def check_pre_condition(self, image_path, env, obs, task, verbose=False):
        """
        check whether current observation satisfies the pre-condition
        """
        # get pre-condition and judgement
        if verbose:
            print()
            print(f"Agent {self.id}'s next task: {task}")
        pre_condition = self.get_pre_condition(task)
        if verbose:
            print(f"Checking pre-condition: {pre_condition}")
        pre_judgement = self.flexible_check(image_path, env, obs, pre_condition, verbose=False)
        if verbose:
            print(f"Pre-condition judgement: {pre_judgement}")
        
        # get pre-condition checking result
        examples = f"""
Q:
- Task name: open microwave door.
- Pre-condition: there is a microwave in the image; gripper is not holding anything; the microwave door is closed.
- Pre-condition judgement: False; True; True
A: Pre-condition is not satisfied; the robot is not in front of the microwave; [navigate to microwave, open microwave door].

Q:
- Task name: open microwave door.
- Pre-condition: there is a microwave in the image; gripper is not holding anything; the microwave door is closed.
- Pre-condition judgement: True; False; True
A: Pre-condition is not satisfied; gripper is holding something; [navigate to counter, place to counter, navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave].

Q:
- Task name: pick up vegetable.
- Pre-condition: there is a microwave in the image; gripper is not holding anything; if vegetable is inside the microwave, then the microwave door is open.
- Pre-condition judgement: True; True; False
A: Pre-condition is not satisfied; the microwave door is closed; open microwave door.

Q:
- Task name: place to microwave.
- Pre-condition: there is a microwave in the image; gripper is holding something; microwave door is open.
- Pre-condition judgement: True; True; False
A: Pre-condition is not satisfied; the microwave door is closed; [navigate to counter, place to counter, navigate to microwave, open microwave door, navigate to counter, pick up vegetable, navigate to microwave, place to microwave].

Q:
- Task name: place to microwave.
- Pre-condition: there is a microwave in the image; gripper is holding something; microwave door is open.
- Pre-condition judgement: True; False; True
A: Pre-condition is not satisfied; gripper is not holding anything; [navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave].

Q:
- Task name: place to sink.
- Pre-condition: there is a sink in the image; gripper is holding something; there is a bowl in the sink.
- Pre-condition judgement: True; True; False
A: Pre-condition is not satisfied; there is no bowl in the sink; [navigate to counter, place to counter, navigate to bowl, pick up bowl, navigate to sink, place to sink, navigate to fish, pick up fish, navigate to sink, place to bowl].

Q:
- Task name: place to stove.
- Pre-condition: there is a stove in the image; gripper is holding something; there is a pan on the stove.
- Pre-condition judgement: True; True; False
A: Pre-condition is not satisfied; there is no pan on the stove; [navigate to counter, place to counter, navigate to pan, pick up pan, navigate to stove, place to stove, navigate to vegetable, pick up vegetable, navigate to stove, place to pan].

""".strip()
        
        message = f"""
Imagine you are a pre-condition checking machine.

You are given a task name describing a manipulation or navigation task, with some pre-condition and judgement according to current environment observation.
You should output the pre-condition checking result.

**Pre-Condition:**
- Need to be checked before the next task is processed.
- Otherwise, the next task may not be successfully executed.
- May include three parts:
    1. Observation that robots should receive before executing this task.
    2. Should the robot's gripper be holding something or not.
    3. The state of the fixtures or objects in the environment.

**Pre-Condition Judgement:**
- Several boolean value indicating whether the corresponding pre-condition is satisfied or not.

**Detailed Guidance:**
- If all the pre-conditions are satisfied, then simply output "Pre-condition is satisfied".
- If any of the pre-conditions is not satisfied, first find out which pre-condition is unsatisfied, and then think about recommended next step(s) to meet the conditions.
- You should think carefully and step-by-step about the recommended next step(s) to meet the unsatisfied pre-conditions.
- When replanning, you should notice that some actions are dependent on the previous actions, and some actions may have pre-conditions that need to be satisfied before execution.
- Finally output "Pre-condition is not satisfied; <reason why pre-condition is not satisfied>; <recommended next step>"

**Output:** strictly follow the format: 
Pre-condition is not satisfied; <reason why pre-condition is not satisfied>; <recommended next step(s)>.
or Pre-condition is satisfied.

**Few-Shot Examples:**

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

Q:
- Task name: {task}
- Pre-condition: {pre_condition}
- Pre-condition judgement: {pre_judgement}
A: Pre-condition is not satisfied; <reason why pre-condition is not satisfied>; <recommended next step(s)>.
or Pre-condition is satisfied.
""".strip()
        
        content = self.client.get_response(message=message)
        
        pre_check_result = utils.extract_content(content, filter="A:")
        if verbose:
            print(f"Pre-check result: {pre_check_result}")
            
        self.history_pre_check_results.append(pre_check_result)
        
        return pre_check_result
    
    def check_post_condition(self, image_path, env, obs, task, verbose=False):
        """
        check whether current observation satisfies the post-condition
        """
        # get post-condition and judgement
        if verbose:
            print()
            print(f"Agent {self.id}'s current task: {task}")
        post_condition = self.get_post_condition(task)
        if verbose:
            print(f"Checking post-condition: {post_condition}")
        post_judgement = self.flexible_check(image_path, env, obs, post_condition, verbose=False)
        if verbose:
            print(f"Post-condition judgement: {post_judgement}")
        
        # get post-condition checking result
        examples = f"""
Q:
- Task name: open microwave door.
- Post-condition: there is a microwave in the image; gripper is not holding anything; the microwave door is open.
- Post-condition judgement: True; True; False
A: Post-condition is not satisfied; the microwave door is closed; open microwave door.

Q:
- Task name: turn off fauset.
- Post-condition: there is a sink in the image; gripper is not holding anything; the fauset is off.
- Post-condition judgement: True; True; False
A: Post-condition is not satisfied; the fauset is on; turn off fauset.

Q:
- Task name: pick up vegetable.
- Post-condition: gripper is holding the vegetable.
- Post-condition judgement: False
A: Post-condition is not satisfied; gripper is not holding the vegetable; pick up vegetable.

Q:
- Task name: place to microwave.
- Post-condition: there is a microwave in the image; gripper is not holding anything.
- Post-condition judgement: True; True
A: Post-condition is satisfied.

Q:
- Task name: turn on stove.
- Post-condition: there is a stove in the image; gripper is not holding anything; the stove is on.
- Post-condition judgement: True; True; False
A: Post-condition is not satisfied; the stove is off; turn on stove.

Q:
- Task name: turn on microwave.
- Post-condition: there is a microwave in the image; gripper is not holding anything; the microwave is on.
- Post-condition judgement: True; True; False
A: Post-condition is not satisfied; the microwave is off; turn on microwave.

""".strip()
        
        message = f"""
Imagine you are a post-condition checking machine.

You are given a task name describing a manipulation or navigation task, with some post-condition and judgement according to current environment observation.
You should output the post-condition checking result.

**Post-Condition:**
- Need to be checked after the current task is finished.
- Otherwise, it is unsure whether the current task is successfully executed.
- May include three parts:
    1. Observation that robots should receive after executing this task.
    2. Should the robot's gripper be holding something or not.
    3. The state of the fixtures or objects in the environment.

**Post-Condition Judgement:**
- Several boolean value indicating whether the corresponding post-condition is satisfied or not.

**Detailed Guidance:**
- If all the post-conditions are satisfied, then simply output "Post-condition is satisfied".
- If any of the post-conditions is not satisfied, first find out which post-condition is unsatisfied, and then think about recommended next step to meet the conditions.
- For most of the case, the recommended next step is the repeat of the current task itself.
- Finally output "Post-condition is not satisfied; <reason why post-condition is not satisfied>; <recommended next step>"

**Output:** strictly follow the format: 
Post-condition is not satisfied; <reason why post-condition is not satisfied>; <recommended next step>.
or Post-condition is satisfied.

**Few-Shot Examples:**

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

Q:
- Task name: {task}
- Post-condition: {post_condition}
- Post-condition judgement: {post_judgement}
A: Post-condition is not satisfied; <reason why post-condition is not satisfied>; <recommended next step>.
or Post-condition is satisfied.
""".strip()
        
        content = self.client.get_response(message=message)
        
        post_check_result = utils.extract_content(content, filter='A:')
        if verbose:
            print(f"Post-check result: {post_check_result}")
        
        self.history_post_check_results.append(post_check_result)
        
        return post_check_result

    
    def flexible_check(self, image_path, env, obs, condition, verbose=False):
        """
        Flexible check function that uses VLM or ground truth based on the specified checks.
        
        :param image_path: Path to the image for VLM check.
        :param env: Environment object for ground truth check.
        :param obs: Observation object for ground truth check.
        :param condition: The condition to check.
        :return: List of results for each condition.
        """
        
        no_need_checks = tuple(self.no_need_checks)
        ground_truth_checks = tuple(self.ground_truth_checks)
        vlm_checks = tuple(self.vlm_checks)

        results = {}
        for cond in condition.split('; '):
            if cond.startswith(no_need_checks):
                results[cond] = "True"
            elif cond.startswith(ground_truth_checks):
                results[cond] = self.ground_truth_check(env, obs, cond)
            else:
                # Default to VLM if not specified
                results[cond] = self.vlm_check(image_path, cond)
            
        if verbose:
            pprint(results)
            
        results = "; ".join(results.values())
        return results
    
    
    def vlm_check(self, image_path, condition):
        """
        redirect to one view check or two view check based on the view number
        """
        if self.view == 1:
            assert isinstance(image_path, str), "image_path should be a string for one view check"
            return self.vlm_check_one_view(image_path, condition)
        elif self.view == 2:
            assert isinstance(image_path, list) and len(image_path) == 2, "image_path should be a list for two view check"
            return self.vlm_check_two_view(image_path, condition)
        else:
            raise ValueError("Invalid view number")
    
    
    def vlm_check_one_view(self, image_path, condition):
        """
        check whether the condition is satisfied based on the given image
        """

        message = f""" 
Imaging you are a image checking agent, and you are able to check whether the given condition is satisfied based on the given image.

You are given one image and a condition, and you need to output whether the condition is satisfied (True or False) based on the image.

**Image:** 
- The image is a snapshot of the robot's current **front view** in the kitchen environment.
- The robot is a semi-transparent arm, and the kitchen object or fixtures are visible in the background.
- You need to carefully check the existense or state of fixture or objects in the image.

**Condition:**
- The condition is a sentence that describe the existense or state of the fixtures or objects in the image.
- Encountering with special cases like 'No specific pre-condition' or 'No specific post-condition', you should output True directly.

**Output:** output whether the image satisfied the description of the condition, strictly follow the format: <True or False>

**Your Task:**

Q: Condition: {condition}
A: <True or False>
""".strip()
        
        content = self.client.get_response(message=message, image_path=image_path)
        content = utils.extract_content(content, filter="A:")
        content = utils.extract_content(content, filter="True or False")
        judgement = content.strip()
        
        return judgement
    
    
    def vlm_check_two_view(self, image_path, condition):
        """
        check whether the condition is satisfied based on the given image
        """

        message = f""" 
Imaging you are a image checking agent, and you are able to check whether the given condition is satisfied based on the given images.

You are given two images and a condition, and you need to output whether the condition is satisfied (True or False) based on the images.

**Image 1:** 
- The image is a snapshot of the robot's current **front view** in the kitchen environment.
- The robot is a semi-transparent arm, and the kitchen object or fixtures are visible in the background.
- You need to carefully check the existense or state of fixture or objects in the image.

**Image 2:** 
- The image is a snapshot of the robot's **gripper view** from camera on the robot gripper
- The gripper is visible in the image, and the objects on the counter are visible in the image.
- You need to carefully check the whether gripper is holding something or not.
- You also need to carefully check the existense of objects in the image.

**Condition:**
- The condition is a sentence that describe the existense or state of the fixtures or objects in the image.
- Encountering with special cases like 'No specific pre-condition' or 'No specific post-condition', you should output True directly.

**Output:** output whether the image satisfied the description of the condition, strictly follow the format: <True or False>

**Your Task:**

Q: Condition: {condition}
A: <True or False>
""".strip()

        content = self.client.get_response(message=message, image_path=image_path)
        content = utils.extract_content(content, filter='A:')
        content = utils.extract_content(content, filter="True or False")
        judgement = content.strip()
        
        return judgement
    
    
    def ground_truth_check(self, env, obs, condition):
        """
        check whether condition is satisfied using groung truth information
        """

        examples = f"""

Q: Condition: gripper is holding the vegetable.
A: result = hold_checker(env, 'vegetable', id={self.id})

Q: Condition: gripper is not holding anything.
A: result = not_hold_checker(env, id={self.id})

Q: Condition: gripper is holding something.
A: result = hold_checker(env, id={self.id})

Q: Condition: there is a microwave in the image.
A: result = location_checker(env, obs, 'microwave')

Q: Condition: there is a vegetable in the image.
A: result = location_checker(env, obs, 'vegetable')

Q: Condition: the microwave door is closed.
A: result = close_microwave_door_checker(env)

Q: Condition: the microwave door is open.
A: result = open_microwave_door_checker(env)

Q: Condition: the cabinet door is open.
A: result = open_cabinet_door_checker(env)

Q: Condition: the stove is turned on.
A: result = turn_on_stove_checker(env)

Q: Condition: the microwave is turned on.
A: result = turn_on_microwave_checker(env)

Q: Condition: the stove is turned off.
A: result = turn_off_stove_checker(env)

Q: Condition: the microwave is turned off.
A: result = turn_off_microwave_checker(env)

""".strip()
        
        message = f"""
Imagine you are a code generation machine, and you are able to generate code to check whether given condition is satisfied.

You are given a condition, which is one sentence describing current observation.

**Condition:** the condition is a sentence that describe the current observation.

**Ground Truth Checker Functions:** 
- The following checker functions are available for you to use.
{checker.ALL_CHECKERS}
- Don't create a new function on your own!

**Output:** simple plain text using the format "result = <checker_name(env, obs, ...)>" where checker is the relevant checker function.

**Detailed Guidance:**
- If asked about whether gripper is holding something or not, you should call a hold_checker or not_hold_checker.
- If asked about whether something is in the image indicating relative spatial condition, you should call a locaction_checker.
- If asked about whether fixture state is open or close, you should call a relevant open_checker or close_checker.

**Few-Shot Examples:** 

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

Q: Condition: {condition}
A: result = <checker_name(env, obs, ...)>
""".strip()
        
        content = self.client.get_response(message=message)
        
        content = utils.extract_content(content, filter='A:')
        content = utils.extract_content(content, filter="python")
        content = utils.extract_content(content, filter="result")
        content = utils.extract_content(content, filter="```")
        codes = content.strip()
        
        # execute codes to get results
        local_vars = {"env": env, "obs": obs}
        exec(codes, globals(), local_vars)
        results = "; ".join(str(value) for key, value in local_vars.items() if key.startswith("result"))
        
        return results
    
    
    def get_pre_condition(self, task):
        """
        generate pre-condition for the given task
        """

        place_to_sink_pre_contition = \
            "there is a sink in the image; gripper is holding something."
        # if there is bowl in environment objects, then should place to bowl in sink
        if "bowl" in self.env_info:
            for execution in reversed(self.history_execution):
                # get last pick up to infer what object the agent is holding
                if execution.startswith("pick up"):
                    if execution == "pick up bowl":
                        place_to_sink_pre_contition = \
                            "there is a sink in the image; gripper is holding something."                        
                        break
                    else: # holding other things
                        place_to_sink_pre_contition = \
                            "there is a sink in the image; gripper is holding something; there is a bowl in the sink."
                        break

        place_to_stove_pre_condition = \
            "there is a stove in the image; gripper is holding something."
        # if there is pan in environment objects, then should place to pan in stove
        if "pan" in self.env_info:
            for execution in reversed(self.history_execution):
                # get last pick up to infer what object the agent is holding
                if execution.startswith("pick up"):
                    if execution == "pick up pan":
                        place_to_stove_pre_condition = \
                            "there is a stove in the image; gripper is holding something."                        
                        break
                    else: # holding other things
                        place_to_stove_pre_condition = \
                            "there is a stove in the image; gripper is holding something; there is a pan on the stove."
                        break

        examples = f"""

Q: Task name: open microwave door.
A: there is a microwave in the image; gripper is not holding anything; the microwave door is closed.

Q: Task name: turn off fauset.
A: there is a sink in the image; gripper is not holding anything; the fauset is turned on.

Q: Task name: turn on fauset.
A: there is a sink in the image; gripper is not holding anything; the fauset is turned off.

Q: Task name: pick up vegetable.
A: there is a vegetable in the image; gripper is not holding anything.

Q: Task name: place to microwave.
A: there is a microwave in the image; gripper is holding something; microwave door is open.

Q: Task name: place to cabinet.
A: there is a cabinet in the image; gripper is holding something; cabinet door is open.

Q: Task name: place to counter.
A: there is a counter in the image; gripper is holding something.

Q: Task name: place to stove.
A: {place_to_stove_pre_condition}

Q: Task name: place to sink.
A: {place_to_sink_pre_contition}

Q: Task name: place to pan.
A: there is a pan in the image; gripper is holding something.

Q: Task name: place to bowl.
A: there is a bowl in the image; gripper is holding something.

Q: Task name: turn on stove.
A: there is a stove in the image; gripper is not holding anything; the stove is turned off.

Q: Task name: turn on microwave.
A: there is a microwave in the image; gripper is not holding anything; the microwave is turned off.

Q: Task name: navigate to stove.
A: No specific pre-condition.

Q: Task name: wait.
A: No specific pre-condition.

""".strip()
        
        message = f"""
Imagine you are a pre-condition generation machine.

You are given a task name describing a manipulation or navigation task the robot will then execute, and the end effector of the robot is a gripper.
You are required to generate the **pre-condition** for the given next task. 

**Pre-Condition:**
- Need to be checked before the next task is processed.
- Otherwise, the next task may not be successfully executed.
- May include three parts:
    1. Observation that robots should receive before executing this task.
    2. Should the robot's gripper be holding something or not.
    3. The state of the fixtures or objects in the environment.
- Should be infered based on specific task.

**Detailed Guidance** in most cases:
- First pre-condition sentence is what object or fixture the robot should see in the front camera image considering the next task.
- Second pre-condition sentence is whether the gripper should be holding something or not considering the next task.
- Third pre-condition sentence is the state of the fixtures or objects should be in the environment considering the next task.
- Should be infered based on specific task.

In special cases like 'navigate' or 'wait', you can simply output "No specific pre-condition".

**Output:** strictly follow the format: <sentences seperated by ;>

**Few-Shot Examples:** 

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

Q: Task name: {task}
A: <sentences seperated by ;>
""".strip()
        
        content = self.client.get_response(message=message)
        
        pre_condition = utils.extract_content(content, filter='A:')
        
        return pre_condition
    
    
    def get_post_condition(self, task):
        """
        generate post-condition for the given task
        """

        examples = f"""

Q: Task name: navigate to microwave.
A: there is a microwave in the image.

Q: Task name: open microwave door.
A: there is a microwave in the image; gripper is not holding anything; the microwave door is open.

Q: Task name: turn on fauset.
A: there is a sink in the image; gripper is not holding anything; the fauset is turned on.

Q: Task name: turn off fauset.
A: there is a sink in the image; gripper is not holding anything; the fauset is turned off.

Q: Task name: pick up vegetable.
A: gripper is holding the vegetable.

Q: Task name: place to microwave.
A: there is a microwave in the image; gripper is not holding anything.

Q: Task name: place to container.
A: there is a container in the image; gripper is not holding anything.

Q: Task name: place to cabinet.
A: there is a cabinet in the image; gripper is not holding anything.

Q: Task name: place to sink.
A: there is a cabinet in the image; gripper is not holding anything.

Q: Task name: pick up fish.
A: gripper is holding the fish.

Q: Task name: turn on stove.
A: there is a stove in the image; gripper is not holding anything; the stove is turned on.

Q: Task name: turn on microwave.
A: there is a microwave in the image; gripper is not holding anything; the microwave is turned on.

Q: Task name: wait.
A: No specific post-condition.

""".strip()
        
        message = f"""
Imagine you are a post-condition generation machine.

You are given a task name describing a manipulation or navigation task the robot has already executed, and the end effector of the robot is a gripper.
You are required to generate the **post-condition** for the given current task. 

**Post-Condition:**
- Need to be checked after the current task is finished.
- Otherwise, it is unsure whether the current task is successfully executed.
- May include three parts:
    1. Observation that robots should receive after executing this task.
    2. Should the robot's gripper be holding something or not.
    3. The state of the fixtures or objects in the environment.
- Should be infered based on specific task.

**Detailed Guidance** in most cases:
- First post-condition sentence is what object or fixture the robot should see in the front camera image considering the finish of current task.
- Second pre-condition sentence is whether the gripper should be holding something or not considering the finish of current task.
- Third pre-condition sentence is the state of the fixtures or objects should be in the environment considering the finish of current task.
- Should be infered based on specific task.

In special cases like 'wait', you can simply output "No specific pre-condition".

**Output:** strictly follow the format: <sentences seperated by ;>

**Few-Shot Examples:**

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

Q: Task name: {task}
A: <sentences seperated by ;>
""".strip()
        
        content = self.client.get_response(message=message)
        
        post_condition = utils.extract_content(content, filter='A:')

        return post_condition


if __name__ == "__main__":
    
    # unset all_proxy and ALL_PROXY to avoid proxy issues
    os.environ['all_proxy'] = ""
    os.environ['ALL_PROXY'] = ""
    
    goal = "pick up vegetable from counter and place it to the microwave"
    env_info = f"""
Available fixtures in environment: sink, dishwasher, toaster, stove, fridge, microwave, opencabinet, coffeemachine
Available objects in environment: vegetable, vegetable_container, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, open microwave door, close microwave door, turn on faucet, turn off faucet
""".strip()
    agent = MixAgent(goal, env_info, view=2)
    task = "navigate to microwave"
    image_path = ["record/agent-20250206-012515/task0_agent_frame0.jpg", "record/OpenMicrowavePnP/agent-20250206-012515/task0_agent_frame1.jpg"]
    
    pre_check_result = agent.check_pre_condition(image_path=image_path, env=None, obs=None, task=task, verbose=True)
    post_check_result = agent.check_post_condition(image_path=image_path, env=None, obs=None, task=task, verbose=True)