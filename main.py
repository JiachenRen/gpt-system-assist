import openai
from system_interface import SystemInterface
from context_manager import ContextManager
from completion import completion


def read_api_key():
    with open('openai_api_key.txt', 'r') as file:
        return file.read().strip()


openai.api_key = read_api_key()

context_manager = ContextManager(objective="""
You are a program running on a computer with full access to everything.
You will complete tasks given by user.
execute_shell_command should be able to give you all the information you need.

Follow these steps to complete a task:
1. Gather information.
2. Propose a solution and execute. If failed, try a different solution.
4. Verify success.
5. Upon verification of success, call complete_task.

You will now receive tasks from user.
""", max_tokens=14000)
system_interface = SystemInterface(context_manager)


def start_conversation_loop():
    while True:
        finish_reason = run_conversation_step()
        if finish_reason == 'function_call':
            continue
        user_input = system_interface.listen_for_user_input().strip()
        if user_input == "exit":
            break


def run_conversation_step():
    response = completion.get_chat_completion_response(
        context_manager.get_context(),
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
