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
from agents.client import REASON_CLIENT
from agents.mix_agent import MixAgent

import utils
import json


class MixCoAgent(MixAgent):
    """
    A flexible agent to generate plan and command for robot in robocasa environment
    """
    def __init__(self, goal, env_info, view=1, task_name=None, id=0):
        super().__init__(goal, env_info, view, task_name, id)

        self.task_name = task_name.replace("TwoAgent", "") # unify task name
        self.reason_client = REASON_CLIENT # used for get twoagent plan
    
    
    def get_initial_plan(self, verbose=False):
        """
        get initial plan based on memory or random examples
        """

        # get relevant two agent examples if given task name
        task_name = self.task_name
        if task_name is not None:
            examples = self.retrieve_examples(task_name, verbose=verbose)
        else: # use random examples
            examples = f"""
Q:
- Environment infomation:
Available fixtures in environment: sink, stove, fridge, microwave, opencabinet, dishwasher, coffeemachine, toaster
Available objects in environment: container, plate, vegetable
Available commands: wait, reset arm, release gripper, pick up (object), place to (object or fixture), navigate to (object or fixture), navigate back, open microwave door, turn on faucet, turn off faucet
- Goal: wash the vegetable in the sink, then pick and steam it in the microwave
A: Plan:
agent0: [navigate to vegetable, pick up vegetable, navigate to sink, place to sink, turn on faucet, turn off faucet, pick up vegetable, navigate to microwave, close microwave door, press the buttom]
agent1: [navigate to microwave, open microwave door, navigate to fridge, wait, wait, wait, wait, wait, wait, wait]

Q:
- Environment infomation:
Available fixtures in environment: sink, dishwasher, toaster, stove, fridge, microwave, opencabinet, coffeemachine
Available objects in environment: vegetable, vegetable_container, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, open microwave door, close microwave door, turn on faucet, turn off faucet
- Goal: pick up vegetable from counter and place it to the microwave
A: Plan:
agent0: [navigate to microwave, open microwave door, navigate to fridge, wait]
agent1: [navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave]

""".strip()
        
        message = f"""
Imagine you are a cooperative kitchen robot high-level planner, and you are able to plan long-horizon and multi-stage tasks step by step given current environment and goal info for two agents.

The input contains two parts, namely environment information and goal:

1. **Environment infomation:** a string that contains available fixtures, objects, and commands in the environment. 
- **Available fixtures:** fixtures in the environment that you can use in the <fixture> in command.
- **Available objects:** objects in the environment that you can use in the <object> in command.
- **Available commands:** each command is a string that you can use to control the robot, with format as "command <object or fixture>" which needs to be strictly followed.

2. **Goal:** a string that contains the goal of the task, offen include hints on how to achieve the goal

When planning for two agents, please make sure that the two agents are able to cooperate with each other, while considering convenience using current environment infomation.
That is to say, you should coordinate them without conflicts, and the plan length for two agent should be the same and as short as possible.

**Output:** strictly follow the format:
Plan:
agent0: [<action1>, <action2>, ...]
agent1: [<action1>, <action2>, ...]

**Attention:**
1. You can only use the combination of available command and object or fixture to plan for the task.
2. The microwave is initially closed, and need to be opened first.
3. Whenever you need to pick up some object, navigate to the object first.
4. Use 'wait' when one agent should wait for the other agent to complete some task.
5. Make sure the two agents are not in the same location at the same time. If so, let one agent navigate to another place first.

**Few-Shot Examples:**

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

Q:
- Environment infomation:
{self.env_info}
- Goal: {self.goal}
A: Plan:
agent0: [<action1>, <action2>, ...]
agent1: [<action1>, <action2>, ...]
""".strip()
        
        content = self.client.get_response(message=message)
        
        # get structured plan
        plan_str = content
        pattern = r'agent\d+: \[(.*?)\]'
        matches = re.findall(pattern, plan_str)
        
        # create two agent plan dict
        plan_dict = {}
        for i, match in enumerate(matches):
            agent_key = f'agent{i}'
            tasks = [task.strip() for task in match.split(',')]
            plan_dict[agent_key] = tasks
        
        if verbose:
            print()
            print("Cooperative plan:")
            pprint(plan_dict)
        
        return plan_dict
    

    def retrieve_examples(self, task_name, verbose=False):
        """
        retrieve relevant examples from the memory file, prepare for the current two agent task
        """

        # check relevant file
        memory_folder_path = os.path.join(BASE_PATH, "memory")
        twoagent_memory_path = os.path.join(memory_folder_path, task_name + "_twoagent.json")

        # check whether two agent memory file exists
        if os.path.exists(twoagent_memory_path):
            with open(twoagent_memory_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            goal = data[-1]['goal']
            env_info = data[-1]['env_info']
            twoagent_plan_dict = data[-1]['twoagent_plan'] # should be a Dict[List[str]]

        else:
            memory_path = os.path.join(memory_folder_path, task_name + ".json")
            if not os.path.exists(memory_path):
                raise FileNotFoundError(f"Memory file {task_name}.json not found in {memory_folder_path}")
            
            with open(memory_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            has_success = any(item["result"] == "success" for item in data)
            assert has_success

            # get relevant example data, assume env not change much
            env_info_list = [item["env_info"] for item in data if item["result"] == "success"]
            goal_list = [item["goal"] for item in data if item["result"] == "success"]
            useful_plan_list = [item["useful_plan_str"] for item in data if item["result"] == "success"]

            # only get the best single agent useful plan and relevant memory
            best_useful_plan = min(useful_plan_list, key=len)
            idx = useful_plan_list.index(best_useful_plan)
            oneagent_plan = best_useful_plan
            
            goal = goal_list[idx]
            env_info = env_info_list[idx]

            # think carefully about the best two agent plan in the same task, given best single agent plan
            twoagent_plan_dict = self.get_twoagent_plan(goal, oneagent_plan, verbose=verbose)
            self.save_twoagent_plan(twoagent_plan_dict, verbose=verbose)
        
        twoagent_plan = f"""
Plan: 
agent0: [{', '.join(twoagent_plan_dict['agent0'])}]
agent1: [{', '.join(twoagent_plan_dict['agent1'])}]
""".strip()

        if verbose:
            print()
            print("Retrieved twoagent plan:")
            print(twoagent_plan)

        # prepare example for the two agent task
        example = f"""
Q:
- Environment information: 
{env_info}
- Goal: {goal}
A: {twoagent_plan}
""".strip()
        
        return example


    def get_twoagent_plan(self, goal, oneagent_plan, verbose=False):
        """
        get twoagent plan given best oneagent plan using a reason model
        """

        # prompt message
        message = f"""
Imagine you are a multiagent highlevel planner, and you are able to plan long-horizon and multi-stage tasks step by step for two agents.

You are given a goal and a plan for a single agent, and you need to decompose the plan into two parts, so that two agents can work in parallel to achieve the goal.

**Goal:** the goal that single-agent plan attempts to achieve.
{goal}

**Plan:** the plan that is the shortest and most efficient for a single agent to achieve the goal.
{oneagent_plan}

Your task is to seperate the plan into two parts, so that two agents can work in parallel to improve efficiency.

**Attention:**
1、the robot is a single-gripper robot, and it cannot manipulate anything after it pick up something, unless it places the object to a fixture or object
2、navigate to the object or fixture before pick, place or manipulate something
3、add 'wait' at suitable location in the plan list to make two agents' action list have the same length, but don't let two robots wait at the same time
4、pay attention to possible crash as two robots may navigate to the same location
5、don't use other action commands except for 'wait' or 'navigate aside' (to avoid crash)

**Output:**
agent0: [<action1>, <action2>, ...]
agent1: [<action1>, <action2>, ...]
""".strip()

        content = self.reason_client.get_response(message=message, verbose=verbose).strip()

        # create two agent plan dict
        pattern = r'agent\d+: \[(.*?)\]'
        matches = re.findall(pattern, content)
        plan_dict = {}
        for i, match in enumerate(matches):
            agent_key = f'agent{i}'
            tasks = [task.strip() for task in match.split(',')]
            plan_dict[agent_key] = tasks
        
        if verbose:
            print()
            print("Cooperative Plan:")
            pprint(plan_dict)
        
        self.plan_dict = plan_dict
        
        return plan_dict
    

    def save_twoagent_plan(self, twoagent_plan_dict, verbose=False):
        """
        save the two agent plan to the other file
        """
        # prepare data
        task_name = self.task_name
        goal = self.goal
        env_info = self.env_info
        data = {
            "task_name": task_name,
            "goal": goal,
            "env_info": env_info,
            "twoagent_plan": twoagent_plan_dict
        }

        # check relevant file
        memory_folder_path = os.path.join(BASE_PATH, "memory")
        twoagent_memory_path = os.path.join(memory_folder_path, task_name + "_twoagent.json")
        utils.append_to_json(twoagent_memory_path, data)

        if verbose:
            print()
            print(f"New twoagent plan data saved to: {twoagent_memory_path}")


if __name__ == "__main__":
    
    # unset all_proxy and ALL_PROXY to avoid proxy issues
    os.environ['all_proxy'] = ""
    os.environ['ALL_PROXY'] = ""
    
    goal = "defrost fish in the sink"
    env_info = f"""
Available fixtures in environment: sink, stove, fridge, microwave, cabinet, dishwasher, coffeemachine, toaster, counter
Available objects in environment: fish, bowl
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, navigate aside, open microwave door, close microwave door, open cabinet door, close cabinet door, turn on faucet, turn off faucet, turn on stove, turn on microwave
""".strip()
    agent = MixCoAgent(goal, env_info, view=2, task_name="DefrostInBowl")
    initial_plan = agent.get_initial_plan(verbose=True)