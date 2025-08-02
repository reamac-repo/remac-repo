import os
import json
import glob
import re


def append_to_json(file_path, new_data):
    """
    try to append new data to json file
    """
    # check if file exists or empty
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([new_data], f, indent=4)
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):  # ensure ist
                data = [data]
        except json.JSONDecodeError: # handle empty json file
            data = []

    # append new data
    data.append(new_data)

    # write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_to_json(file_path, new_data):
    """
    write new data to original json file
    """
    # check if file exists or empty
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([new_data], f, indent=4)
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):  # ensure ist
                data = [data]
        except json.JSONDecodeError: # handle empty json file
            data = []
    
    if data:
        data[-1] = new_data
    else:
        data.append(new_data)
    
    # write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def extract_path_list(folder_path, name='*_frame0.jpg'):
    """
    extract image path list from folder_path given certain pattern
    """
    def extract_number(file_path):
        match = re.search(r'task(\d+)_.*\.jpg$', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        return -1
    
    pattern = os.path.join(folder_path, name)
    image_path_list = glob.glob(pattern)
    image_path_list.sort(key=extract_number)
    
    return image_path_list


if __name__ == '__main__':
    folder_path = '/home/maay/robocasa_space/LLM-multiagent-robocasa/record/agent-reflect-20250218-013338-good'
    image_path_list = extract_path_list(folder_path, name='*_frame0.jpg')

    for image_path in image_path_list:
        print(image_path)