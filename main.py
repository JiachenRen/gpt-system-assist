import time

import openai
from system_interface import SystemInterface
from context_manager import ContextManager
from chat_completion_interface import completion
from speech_synthesis import SpeechSynthesizer


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
3. Verify success.

You will now receive tasks from user.
""", max_tokens=14000, model_name=completion.model)
system_interface = SystemInterface(context_manager)
speech_synthesizer = SpeechSynthesizer()
speech_synthesizer.init()
tts_summarize_long_response = False


def start_conversation_loop():
    try:
        while True:
            finish_reason = run_conversation_step()
            try:
                speech_synthesizer.wait_for_completion()
            except KeyboardInterrupt:
                speech_synthesizer.stop_tts()
                try:
                    input("\nAborted TTS. Press enter to continue. ")
                except EOFError:
                    print("Exiting...")
                    time.sleep(0.5)
            if finish_reason == 'function_call' or finish_reason == 'length':
                continue
            system_interface.listen_for_user_input()
    except KeyboardInterrupt:
        system_interface.exit_program()


def run_conversation_step():
    stream = completion.get_chat_completion_response(
        context_manager.get_context(),
        SystemInterface.get_functions(),
    )

    def build_obj(obj, k, v):
        if k in obj:
            if isinstance(v, str):
                obj[k] += v
            else:
                for k2, v2 in v.items():
                    build_obj(obj[k], k2, v2)
        else:
            obj[k] = v

    response_message = {}
    finish_reason = None
    printed_role = False

    def consume_new_content(content):
        nonlocal printed_role
        if not tts_summarize_long_response:
            speech_synthesizer.stream_tts(content)
        role = response_message.get("role")
        if not printed_role and role:
            print("assistant: ", end='')
            printed_role = True
        print(content, end='')

    for chunk in stream:
        choice = chunk["choices"][0]
        finish_reason = choice["finish_reason"]
        delta = choice["delta"]
        for key, val in delta.items():
            build_obj(response_message, key, val)
            if key == "content" and val:
                consume_new_content(val)
    print("")
    content = response_message.get("content")
    if tts_summarize_long_response:
        speech_synthesizer.start_tts(content)
    elif content:
        speech_synthesizer.stream_tts(None)

    context_manager.add_message(response_message, print_message=False)

    # While TTS is playing, we can do some work
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = response_message["function_call"]["arguments"]
        system_interface.invoke_function(function_name, function_args)

    return finish_reason


start_conversation_loop()
