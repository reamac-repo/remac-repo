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


class MixReAgent(MixAgent):
    """
    A flexible agent to generate plan and command for robot in robocasa environment
    """
    def __init__(self, goal, env_info, view=1, task_name=None, record_folder_path=None, id=0):
        super().__init__(goal, env_info, view, task_name, id)
        
        self.history_reflection = []
        self.interaction_status = "initialized"
        self.record_folder_path = record_folder_path
        self.reason_client = REASON_CLIENT # used for get initial plan and reflection
    
    
    def get_initial_plan(self, verbose=False):
        """
        call the self.client to output initial plan based on self.goal and self.env_info
        """
        
        reminders = self.retrieve_reminders(task_name=self.task_name)
        examples = self.retrieve_examples(task_name=self.task_name)
        
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

**Reminders:** carefully consider the possible reminders (from previous interactions for the same task) when planning.
{reminders}
- all objects except container are originaly on the counter

**Few-Shot Examples:**

Example 0:
- Environment infomation: 
Available fixtures in environment: sink, microwave
Available objects in environment: vegetable, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>
- Goal: pick up vegetable from counter
- Output: [navigate to vegetable, pick up vegetable]

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

- Environment infomation:
{self.env_info}
- Goal: {self.goal}
- Output: <follow the format above, return a list of actions in the plan>
""".strip()
        
        content = self.reason_client.get_response(message=message, verbose=verbose).strip()
        # content = self.client.get_response(message=message).strip()

        # extract plan using re
        plan_str = utils.extract_content(content, filter="Output:")
        if verbose:
            print()
            print(f"Initial plan: {plan_str}")
        plan = utils.extract_content(plan_str, filter="[]").strip()[1:-1].split(", ")

        self.history_plan.append(plan)
        self.history_execution.append(plan[0])
        
        return plan
    
    
    def reflect_pre_check_result(self, verbose=False):
        """
        reflect on the current plan, and check if it is feasible
        mainly two parts of the situations, namely examine pre-condition or call another agent
        """
        current_plan = self.history_plan[-1]
        if len(current_plan) > 1:
            next_execution = current_plan[1]
        else:
            next_execution = 'wait'
        pre_check_result = self.history_pre_check_results[-1]
        
        if "is satisfied" in pre_check_result:
            return
        
        examples = f"""

Example 1:
- Next execution: open microwave door
- Pre-check result: Pre-condition is not satisfied; the robot is not in front of the microwave; navigate to microwave.
- Output: navigate to the object or fixture before pick, place or manipulate something

Example 2:
- Next execution: open microwave door
- Pre-check result: Pre-condition is not satisfied; gripper is holding something; [navigate to counter, place to counter, navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave].
- Output: the robot is a single-gripper robot, and it cannot manipulate anything after it pick up something, unless it places the object to a fixture or object; navigate to and open microwave door before pick up and placing the object to microwave

Example 3:
- Next execution: place to microwave
- Pre-check result: Pre-condition is not satisfied; the microwave door is closed; [navigate to counter, place to counter, navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to microwave].
- Output: open microwave door before place to microwave; place object to counter before open microwave door if gripper is holding something

Example 4:
- Next execution: place to cabinet
- Pre-check result: Pre-condition is not satisfied; gripper is not holding anything; [navigate to counter, pick up vegetable, navigate to cabinet, place to cabinet].
- Output: pick up object after open microwave door; pick up object before place to microwave

Example 5:
- Next execution: place to sink
- Pre-checkk result: Pre-condition is not satisfied; there is no bowl in the sink; [navigate to counter, place to counter, navigate to bowl, pick up bowl, navigate to sink, place to sink, navigate to fish, pick up fish, navigate to sink, place to bowl].
- Output: need to place a bowl in the sink before place the fish and defrost it; place object to counter before pick up bowl if gripper is holding something

""".strip()
        
        message = f"""
Imaging you are a reflective agent, and you are able to reflect on the current plan and pre-check result to generate a reflection for the next step.

**Post-check result** is a string that contains three parts: 
1. Whether the condition is satisfied.
2. Reason why the condition is not satisfied (if not satisfied).
3. Recommended next step (if not satisfied).

**Output:** only one sentence that contains the reflection for unsuccessful pre-check result, or sentences seperated by ;

**Few-Shot Examples:**

{examples}

Remember to strictly follow the examples' format!!!

**Your Task:**

- Next execution: {next_execution}
- Pre-check result: {pre_check_result}
- Output: <follow the format above, return one sentence or sentences seperated by ;>
""".strip()
        
        content = self.client.get_response(message=message)
        
        # get structured plan
        if content.startswith('Output:'):
            match = re.search(r'Output:\s*(.*)', content)
            sentences = match.group(1)
        else:
            sentences = content
        if verbose:
            print()
            print(f"Reflect pre-condition: {sentences}")
        
        reflection = [sentence.strip() for sentence in sentences.split(';') if sentence.strip()]
        self.history_reflection += reflection # should be a list of strings
        return
    
    
    def reflect_final_result(self, verbose=False): # TODO, final plan is useless, left for handle failed interaction
        """
        reflect on the final result of the interaction, and do three analysis:
        1. check if the goal is achieved
        2. extract final plan
        3. simplify the final plan if possible
        """
        history_plans = self.history_plan
        if self.interaction_status == "success":
            self.final_plan = utils.merge_plans_with_last(history_plans)
        else:
            self.final_plan = utils.merge_plans_without_last(history_plans)
        
        self.final_plan_str = f"[{', '.join(self.final_plan)}]"
        
        if verbose:
            print()
            print(f"Final plan: {self.final_plan}")

        self.get_useful_plan(verbose=verbose)
    

    def get_useful_plan(self, verbose=False):
        """
        get useful plan from execution history
        remove repetitive and useless actions
        """

        # extract execution history string and image path list
        history_execution = f"[{', '.join(self.history_execution)}]"
        goal = self.goal
        
        # prompt message
        message = f"""
Imagine you are a reflective agent with reasoning ability, and you are able to reflect on the execution history to generate a useful plan.

You are given a **goal** and **history execution**, which is a list of actions executed by the **single-gripper-robot** step by step.
You should carefully check the **history execution**, and generate a **useful plan** that contains only necessary actions to achieve the goal.

**Goal:** the goal that historical actions attempt to achieve.

**History Execution:** a list of actions that contains all actions executed by the robot step by step.

**Useful Plan:** a list of actions that contains only necessary actions to achieve the goal.

**Output:** the same format as the input!!! Remember to outputs a list of actions as the **useful plan**.

**Tips:**
1. Perhaps two adjacent actions cancel each other out, such as
- picking first and then placing to the same area,
- navigating to a place and then navigating back to the origin.
These actions can be deleted together
2. An action may be repeated multiple times because it was not executed successfully, but the last time it succeeded. In this case, one action can be retained.

**Attention:**
1. The robot is a single-gripper robot, and it cannot manipulate anything after it picks up something, unless it places the object to a fixture or object!!!
2. While replanning, you should notice that some actions have a order between them to avoid conflicts or errors.
    - always navigate to the object or fixture before pick, place or manipulate something
    - open microwave door before pick up and placing the object to microwave
    - place object to counter before open microwave door if gripper is holding something

**Your Task:**

- Goal: {goal}
- History Execution: {history_execution}
- Output: [<action1>, <action2>, ...]
""".strip()
        

        content = self.reason_client.get_response(message=message, verbose=verbose).strip()
        content = utils.extract_content(content, filter="[]")
        useful_plan_str = content.strip()

        if verbose:
            print()
            print(f"Useful plan: {useful_plan_str}")
        self.useful_plan_str = useful_plan_str.strip()
        self.useful_plan = self.useful_plan_str[1:-1].split(", ")

        return useful_plan_str
    
    
    def save_interaction(self, record_folder_path=None, mode="a"):
        """
        save interaction and analysis to record folder, and append them to memory file
        """
        task_name = self.task_name
        
        # gather useful interaction data as a dict
        interaction_data = dict()
        interaction_data["task"] = self.task_name
        interaction_data["goal"] = self.goal
        interaction_data["env_info"] = self.env_info
        interaction_data["plan"] = self.history_plan
        interaction_data["execution"] = self.history_execution
        interaction_data["pre_check_results"] = self.history_pre_check_results
        interaction_data["post_check_results"] = self.history_post_check_results
        interaction_data["reflection"] = self.history_reflection
        interaction_data["final_plan"] = self.final_plan
        interaction_data["final_plan_str"] = self.final_plan_str
        interaction_data["useful_plan"] = self.useful_plan
        interaction_data["useful_plan_str"] = self.useful_plan_str
        interaction_data["result"] = self.interaction_status
        
        # save interaction to memory folder, with the same task saved in the same file
        memory_folder_path = os.path.join(BASE_PATH, "memory")
        os.makedirs(memory_folder_path, exist_ok=True)
        memory_path = os.path.join(memory_folder_path, f"{task_name}.json")
        if mode == "a":
            utils.append_to_json(memory_path, interaction_data)
        elif mode == "w":
            utils.write_to_json(memory_path, interaction_data)
        else:
            raise ValueError(f"Invalid mode {mode}")
        
        # save interaction to record folder if record path is given
        if record_folder_path is None:
            record_folder_path = self.record_folder_path
        os.makedirs(record_folder_path, exist_ok=True)
        record_path = os.path.join(record_folder_path, f"{task_name}_agent{self.id}.json")
        if mode == "a":
            utils.append_to_json(record_path, interaction_data)
        elif mode == "w":
            utils.write_to_json(memory_path, interaction_data)
        else:
            raise ValueError(f"Invalid mode {mode}")
    

    def load_interaction(self, record_folder_path=None):
        """
        load interaction and analysis from record folder
        """
        task_name = self.task_name
        
        # load interaction from record folder
        if record_folder_path is None:
            record_folder_path = self.record_folder_path
        record_path = os.path.join(record_folder_path, f"{task_name}_agent{self.id}.json")
        if not os.path.exists(record_path):
            raise FileNotFoundError(f"Record file {record_path} not found")
        
        with open(record_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        if isinstance(data, list):
            assert len(data) == 1
            data = data[0]
        
        # load interaction data except for the useful plan
        self.goal = data["goal"]
        self.env_info = data["env_info"]
        self.history_plan = data["plan"]
        self.history_execution = data["execution"]
        self.history_pre_check_results = data["pre_check_results"]
        self.history_post_check_results = data["post_check_results"]
        self.history_reflection = data["reflection"]
        self.final_plan = data["final_plan"]
        self.final_plan_str = data["final_plan_str"]
        self.interaction_status = data["result"]
    

    def reflect_last_interaction(self, record_folder_path=None):
        """
        in case some interaction failed, reflect on the last interaction and regenerate useful plan
        """
        if record_folder_path is None:
            record_folder_path = self.record_folder_path
        self.load_interaction(record_folder_path=record_folder_path)
        self.get_useful_plan(verbose=True)
        self.save_interaction(record_folder_path=record_folder_path, mode="w")
    
    
    def retrieve_reminders(self, task_name, verbose=False):
        """
        retrieve relevant reminders from the memory file, prepare for the current task
        """
        
        # check relevant file
        memory_folder_path = os.path.join(BASE_PATH, "memory")
        memory_path = os.path.join(memory_folder_path, task_name + ".json")
        if not os.path.exists(memory_path):
            return ""
        
        with open(memory_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        # retrieve most relevant reminders
        reflection_list = [item["reflection"] for item in data]
        reminder_list = [item for sublist in reflection_list for item in sublist]
        reminders = "\n".join(f"- {reminder}" for reminder in reminder_list) # markdown format
        
        if verbose:
            print()
            print(f"Reminders: {reminders}")
        
        return reminders
    
    
    def retrieve_examples(self, task_name, verbose=False):
        """
        retrieve relevant examples from the memory file, prepare for the current task
        """
        
        # check relevant file
        memory_folder_path = os.path.join(BASE_PATH, "memory")
        memory_path = os.path.join(memory_folder_path, task_name + ".json")
        if not os.path.exists(memory_path):
            return ""
        
        with open(memory_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        has_success = any(item["result"] == "success" for item in data)
        if has_success:
            env_info_list = [item["env_info"] for item in data if item["result"] == "success"]
            goal_list = [item["goal"] for item in data if item["result"] == "success"]
            useful_plan_list = [item["useful_plan_str"] for item in data if item["result"] == "success"]
        else:
            env_info_list = [item["env_info"] for item in data]
            goal_list = [item["goal"] for item in data]
            useful_plan_list = [item["useful_plan_str"] for item in data]
        
        # prepare examples for few-shot prompting in get_initial_plan
        examples = "Task relevent successful examples: \n" if has_success else "Task relevent **unsuccessful** examples: \n"
        for i, (env_info, goal, useful_plan) in enumerate(zip(env_info_list, goal_list, useful_plan_list)):
            example = f"""
Example {i+1}:
- Environment information: 
{env_info}
- Goal: {goal}
- Output: {useful_plan}
""".rstrip()
            examples += example
        examples = examples.strip()
        
        if verbose:
            print()
            print(f"{examples}")

        return examples


if __name__ == "__main__":
    
    # unset all_proxy and ALL_PROXY to avoid proxy issues
    os.environ['all_proxy'] = ""
    os.environ['ALL_PROXY'] = ""

#     goal = "pick up vegetable from counter and place it to the cabinet"
#     env_info = f"""
# Available fixtures in environment: sink, stove, fridge, microwave, cabinet, dishwasher, coffeemachine, toaster, counter
# Available objects in environment: vegetable, container
# Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, navigate aside, open microwave door, close microwave door, open cabinet door, close cabinet door, turn on faucet, turn off faucet, turn on stove, turn on microwave""".strip()
#     agent = MixReAgent(goal, env_info, view=2, task_name="OpenCabinetPnP", record_folder_path=None)
#     initial_plan = agent.get_initial_plan(verbose=True)
    
#     goal = "pick up vegetable and heat on the stove"
#     env_info = f"""
# Available fixtures in environment: fridge, sink, cabinet, dishwasher, stove, microwave, coffeemachine, toaster, counter
# Available objects in environment: pan, vegetable
# Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, navigate aside, open microwave door, close microwave door, open cabinet door, close cabinet door, turn on faucet, turn off faucet, turn on stove, turn on microwave
# """.strip()
#     agent = MixReAgent(goal, env_info, view=2, task_name="HeatOnStove", record_folder_path=None)
#     initial_plan = agent.get_initial_plan(verbose=True)

#     goal = "defrost fish in the sink"
#     env_info = f"""
# Available fixtures in environment: sink, stove, fridge, microwave, cabinet, dishwasher, coffeemachine, toaster, counter
# Available objects in environment: fish, bowl
# Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, navigate aside, open microwave door, close microwave door, open cabinet door, close cabinet door, turn on faucet, turn off faucet, turn on stove, turn on microwave
# """.strip()
#     agent = MixReAgent(goal, env_info, view=2, task_name="DefrostInBowl", record_folder_path=None)
#     initial_plan = agent.get_initial_plan(verbose=True)

    goal = "wash the vegetable in the sink and then heat it in the microwave"
    env_info = f"""
Available fixtures in environment: sink, dishwasher, stove, fridge, microwave, cabinet, toaster, coffeemachine, counter
Available objects in environment: container, vegetable
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, navigate aside, open microwave door, close microwave door, open cabinet door, close cabinet door, turn on faucet, turn off faucet, turn on stove, turn on microwave
""".strip()
    agent = MixReAgent(goal, env_info, view=2, task_name="WashPnPHeat", record_folder_path=None)
    initial_plan = agent.get_initial_plan(verbose=True)