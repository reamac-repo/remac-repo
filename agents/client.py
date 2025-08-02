import os
import base64
import requests
import json
from openai import OpenAI
import openai

# unset all_proxy and ALL_PROXY
os.environ['all_proxy'] = ""
os.environ['ALL_PROXY'] = ""


def encode_image(image_path):
    # open the image file in binary reading mode
    with open(image_path, "rb") as image_file:
        # decode the encoded byte string into a UTF-8 string for use in text environments
        return base64.b64encode(image_file.read()).decode("utf-8")


class OllamaClient:
    """
    aims to build standard api for Ollama style agent calling
    """
    def __init__(self, model, url):
        self.url = url
        self.model = model
    
    def get_response(self, message, image_path=None):
        
        # prepare structured data
        if image_path != None:
            base64_image = encode_image(image_path)
            data = {
                "model": self.model,
                "prompt": message,
                "images": [base64_image]
            }
        else:
            data = {
                "model": self.model,
                "prompt": message
            }
        
        # get POST response
        try:
            with requests.post(url, json=data, stream=True) as response:
                if response.status_code == 200:
                    responses = [
                        json.loads(line.decode('utf-8'))['response']
                        for line in response.iter_lines() if line
                        if 'response' in json.loads(line.decode('utf-8'))
                    ]
                    complete_response = ''.join(responses)
                else:
                    print("Fail request:", response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            print("Error in client response:", e)
        return complete_response


class OpenAIClient:
    """
    aims to build standard api for openai style agent calling
    """
    def __init__(self, model, api_key, base_url):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1800,
        )
    
    def get_response(self, message, image_path=None):
        
        # prepare structured data
        if isinstance(image_path, str):
            base64_image = encode_image(image_path)
            image_url = f"data:image/jpg;base64,{base64_image}"
            system_prompt = "You are a speculative visual assistant. "
            message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "auto"
                            }
                        }
                    ]
                }
            ]
        elif isinstance(image_path, list):
            image_urls = [f"data:image/jpg;base64,{encode_image(image_path[i])}" for i in range(len(image_path))]
            system_prompt = "You are a speculative visual assistant. "
            message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": "auto"
                                }
                            } for image_url in image_urls
                        ]
                    ]
                }
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        }
                    ]
                }
            ]
        
        # handle content filter and internal server error
        for t in range(10):
            try:
                # get POST response
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    stream=False
                )
                response = chat_completion.choices[0].message.content
                if response is not None:
                    break
                else:
                    print("Content is None, retry...")
            except openai.InternalServerError:
                    print("502 Bad Gateway, retry...")
                    continue

        return response


class ReasonClient:
    """
    aims to build standard api calling for reasoning agent
    """
    def __init__(self, model, api_key, base_url):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1800,
        )
    
    def get_response(self, message, verbose=False):
        """
        use stream 
        """
        try:
            # initiate streaming requests
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": message}
                ],
                stream=True  # enable streaming output
            )

            full_response = ""
            if verbose:
                print()
                print("--- Start of Response ---")

            # gradually output response content
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    if verbose:
                        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                if chunk.choices[0].delta.content:
                    if verbose:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                    full_response += chunk.choices[0].delta.content
            
            if verbose:
                print()
                print("--- End of Response ---")

            return full_response

        except Exception as e:
            print(f"Request failed: {e}")


### Ollama API client ###




model = "deepseek-r1:70b"

OLLAMA_CLIENT = OllamaClient(
    url=url, 
    model=model
)

### Openai API client ###



OPENAI_CLIENT = OpenAIClient(
    model=model,
    api_key=api_key,
    base_url=base_url,
)

### Reason client ###



REASON_CLIENT = ReasonClient(
    model=model,
    api_key=api_key,
    base_url=base_url,
)


if __name__ == '__main__':
    
    reason_client = REASON_CLIENT
    openai_client = OPENAI_CLIENT

    reminders = """
- open microwave door before pick up something
- navigate and pick up something after open microwave door
""".strip()
    
    examples = """
""".strip()
    
    goal = "pick up vegetable from counter and place it to the microwave"

    env_info = f"""
Available fixtures in environment: sink, dishwasher, toaster, stove, fridge, microwave, opencabinet, coffeemachine
Available objects in environment: vegetable, vegetable_container, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, open microwave door, close microwave door, turn on faucet, turn off faucet
""".strip()
    
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
- the robot is a single-gripper robot, and it cannot manipulate anything after it pick up something, unless it places the object to a fixture or object
- open microwave door before pick up something
- navigate and pick up something after open microwave door

**Few-Shot Examples:**

Example 0:
- Environment infomation: 
Available fixtures in environment: sink, microwave
Available objects in environment: vegetable, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>
- Goal: pick up vegetable from counter
- Output: [navigate to vegetable, pick up vegetable]

Remember to strictly follow the examples' format!!!

**Your Task:**

- Environment infomation:
Available fixtures in environment: sink, dishwasher, toaster, stove, fridge, microwave, opencabinet, coffeemachine
Available objects in environment: vegetable, vegetable_container, container
Available commands: wait, pick up <object>, place to <object or fixture>, navigate to <object or fixture>, open microwave door, close microwave door, turn on faucet, turn off faucet
- Goal: pick up vegetable from counter and place it to the microwave
- Output: <follow the format above, return a list of actions in the plan>
""".strip()
    # response = reason_client.get_response(message=message, verbose=True)
    response = openai_client.get_response(message=message)
    print(response)
