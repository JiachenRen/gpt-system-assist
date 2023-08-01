import openai
import json


def read_api_key():
    with open('openai_api_key.txt', 'r') as file:
        return file.read()


openai.api_key = read_api_key()
model_name = 'gpt-3.5-turbo-0613'
objective = '''
You are running in a sandboxed operation system.
The operating system is unknown, it could be Mac, Windows, Linux, or something else.
You have access to a function called execute_shell_command, which takes a shell command as input and returns the output of the command.
You will determine the operating system through executing command / trial and error.
After succeeding you will await further instructions.
'''
initial_msg = {"role": "user", "content": objective}
messages: list[object] = [initial_msg]
functions = [
        {
            "name": "execute_shell_command",
            "description": "Execute shell command and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "shell command to execute",
                    },
                },
                "required": ["command"],
            },
        }
    ]


# Creates a map that maps function names to functions
def create_function_map(functions):
    function_map = {}
    for function in functions:
        function_map[function["name"]] = globals().get(function["name"])
    return function_map


# Executes provided shell command and returns output
def execute_shell_command(command) -> str:
    import subprocess
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'executed: {command}')
    print(f'result: {result.stdout if result.stdout else result.stderr}')
    return json.dumps({"output": result.stdout, "error": result.stderr})


# Get chat completion response from GPT
def get_chat_completion_response(messages, functions, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    return response


def get_user_input():
    user_input = input("you: ")
    return user_input


def start_conversation_loop():
    available_functions = create_function_map(functions)
    while True:
        finish_reason = run_conversation_step(available_functions)
        if finish_reason == 'function_call':
            continue
        user_input = get_user_input().strip()
        if user_input == "exit":
            break
        elif user_input:
            add_message({"role": "user", "content": user_input})


def add_message(message):
    messages.append(message)
    name = message.get("name")
    name = f'({name})' if name else ''
    content = message.get("content")
    if content:
        print(f'{message["role"]}{name}: {content}')


def run_conversation_step(available_functions):
    response = get_chat_completion_response(messages, functions, model_name)
    choice = response["choices"][0]
    finish_reason = choice["finish_reason"]
    response_message = choice["message"]
    add_message(response_message)

    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        if function_name not in available_functions:
            add_message(
                {
                    "role": "user",
                    "content": {
                        "error": f"Function {function_name} not found"
                    },
                }
            )
            return
        function_to_call = available_functions[function_name]

        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            command=function_args.get("command"),
        )

        add_message(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )

    return finish_reason


start_conversation_loop()
