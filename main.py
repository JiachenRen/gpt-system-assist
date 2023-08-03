import openai
from system_interface import SystemInterface
from task_manager import TaskManager, Task
from context_manager import ContextManager
from chat_completion import chat_completion


def read_api_key():
    with open('openai_api_key.txt', 'r') as file:
        return file.read().strip()


openai.api_key = read_api_key()
model_name = 'gpt-3.5-turbo-16k'

task_manager = TaskManager(root_task=Task(description="""
You are an AI running on a real computer. 
You have full access to the system through provided functions.
Your mission is to complete tasks given by the user.

When the current task/subtask is successfully completed, make sure to call complete_task.
When no tasks remain besides this one, ask user for next task.
""", name="Objective"))

context_manager = ContextManager(task_manager, max_tokens=5000)
system_interface = SystemInterface(task_manager, context_manager)


def start_conversation_loop():
    while True:
        finish_reason = run_conversation_step()
        if finish_reason == 'function_call':
            continue
        user_input = system_interface.get_user_input("user: ").strip()
        if user_input == "exit":
            break


def run_conversation_step():
    response = chat_completion.get_chat_completion_response(
        context_manager.get_current_task_context(),
        SystemInterface.get_functions(),
    )
    choice = response["choices"][0]
    finish_reason = choice["finish_reason"]
    response_message = choice["message"]
    context_manager.add_message(response_message)

    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = response_message["function_call"]["arguments"]
        system_interface.invoke_function(function_name, function_args)

    return finish_reason


start_conversation_loop()
