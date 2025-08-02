import robocasa.utils.planner as planner
import robocasa.utils.checker as checker

import robocasa.utils.control_utils as CU
from collections import OrderedDict

import os
import sys
import robocasa
from pprint import pprint

BASE_PATH = os.path.abspath(robocasa.__file__ + '/../../../')
sys.path.append(BASE_PATH)
os.chdir(BASE_PATH)


def search_config(lang_command, controller_dict):
    """
    extract dict key and extra parameters from lang_command
    """
    find_key = False
    for key in controller_dict.keys():
        if lang_command.startswith(key):
            controller_config = controller_dict[key]
            remaining_str = lang_command.replace(key, "", 1).strip()
            find_key = True
            break
    if not find_key:
        raise ValueError("language command cannot match any controller")
    if remaining_str != "":
        extra_para = remaining_str
    else:
        extra_para = None
    return controller_config, extra_para


controller_dict = OrderedDict() # order of the key is important

controller_dict['wait'] = {
    'type': 'planner',
    'planner': planner.WaitPlanner,
    'usage': 'wait',
}

controller_dict['reset arm'] = {
    'type': 'planner',
    'planner': planner.ResetArmPlanner,
    'usage': None,
}

controller_dict['pick up'] = {
    'type': 'planner',
    'planner': planner.PickUpPlanner,
    'usage': 'pick up <object>',
}

controller_dict['place to'] = {
    'type': 'planner',
    'planner': planner.PlaceToPlanner,
    'usage': 'place to <object or fixture>',
}

controller_dict['navigate to'] = {
    'type': 'planner',
    'planner': planner.NavigationPlanner,
    'usage': 'navigate to <object or fixture>',
}

controller_dict['navigate'] = {
    'type': 'planner',
    'planner': planner.NavigationPlanner,
    'usage': 'navigate aside',
}

controller_dict['open microwave door'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/open_single_door.pth',
    'env_lang': 'open the microwave door', 
    'checker': checker.open_microwave_door_checker,
    'usage': 'open microwave door'
}

controller_dict['close microwave door'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/close_single_door.pth',
    'env_lang': 'close the microwave door',
    'checker': checker.close_microwave_door_checker,
    'usage': 'close microwave door'
}

controller_dict['open cabinet door'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/open_single_door.pth',
    'env_lang': 'open the cabinet door',
    'checker': checker.open_cabinet_door_checker,
    'usage': 'open cabinet door'
}

controller_dict['close cabinet door'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/close_single_door.pth',
    'env_lang': 'close the cabinet door',
    'checker': checker.close_cabinet_door_checker,
    'usage': 'close cabinet door'
}

controller_dict['turn on faucet'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/turn_on_faucet.pth',
    'env_lang': 'turn on the sink faucet',
    'checker': checker.turn_on_faucet_checker,
    'usage': 'turn on faucet'
}

controller_dict['turn off faucet'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/turn_off_faucet.pth',
    'env_lang': 'turn off the sink faucet',
    'checker': checker.turn_off_faucet_checker,
    'usage': 'turn off faucet'
}

controller_dict['turn on stove'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/turn_on_stove.pth',
    'env_lang': 'turn on the front right burner of the stove',
    'checker': checker.turn_on_stove_checker,
    'usage': 'turn on stove'
}

controller_dict['turn on microwave'] = {
    'type': 'policy',
    'ckpt_path': 'checkpoints/turn_on_microwave.pth',
    'env_lang': 'press the start button on the microwave',
    'checker': checker.turn_on_microwave_checker,
    'usage': 'turn on microwave'
}


available_commands = ', '.join(
        [controller_dict[key]['usage'] for key in controller_dict.keys() if controller_dict[key]['usage'] is not None]
    ) # Available commands template


if __name__ == "__main__":
    pprint(controller_dict)
    print(search_config('pick up', controller_dict))
    print(available_commands)