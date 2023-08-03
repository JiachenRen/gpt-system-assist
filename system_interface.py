import json
from transcription import RealTimeTranscription


class SystemInterface:

    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.transcriber = RealTimeTranscription()

    @classmethod
    def get_functions(cls):
        return [
            {
                "name": "execute_shell_command",
                "description": "Use this to execute a shell command and return output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "shell command to execute",
                        },
                    },
                    "required": [
                        "command"
                    ],
                },
            },
            {
                "name": "exit_program",
                "description": "Use this to exit the program",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def exit_program(self):
        exit()

    def listen_for_user_input(self):
        user_input = self.transcriber.get_transcription()
        print("")
        self.context_manager.add_message({
            "role": "user",
            "content": user_input
        })
        return user_input

    def get_user_input(self, prompt: str) -> str:
        user_input = input(prompt)
        self.context_manager.add_message({
            "role": "user",
            "content": user_input
        })
        return user_input

    def __report_err(self, fn_name: str, err: Exception):
        self.context_manager.add_message(
            {
                "role": "function",
                "name": fn_name,
                "content": json.dumps({
                    "error": str(err)
                }),
            }
        )

    def invoke_function(self, function_name: str, args_json_str: str):
        try:
            function_args = json.loads(args_json_str)
        except json.JSONDecodeError as e:
            self.__report_err(function_name, e)
            return

        fn = getattr(self, function_name, None)

        if fn and callable(fn):
            try:
                result = fn(**function_args)
                self.context_manager.add_message(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": result,
                    }
                )
            except Exception as e:
                self.__report_err(function_name, e)
        else:
            self.__report_err(function_name, Exception(f"Function {function_name} does not exist"))

    def execute_shell_command(self, command) -> str:
        """
        Executes provided shell command and returns output
        :param command: shell command to execute
        :return: json string with output and error
        """
        import subprocess
        result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f'executed: {command}')
        print(f'result: {result.stdout if result.stdout else result.stderr}')
        result_obj = {
            "output": result.stdout,
        }
        if result.stderr:
            result_obj["error"] = result.stderr
        elif result.returncode == 0:
            result_obj["status"] = "success"
        return json.dumps(result_obj)
