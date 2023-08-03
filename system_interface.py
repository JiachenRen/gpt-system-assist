import json
from typing import Optional
from task_manager import Task, TaskManager


class SystemInterface:

    def __init__(self, task_manager: TaskManager, context_manager):
        self.task_manager: TaskManager = task_manager
        self.context_manager = context_manager

    @classmethod
    def get_functions(cls):
        return [
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
                    "required": [
                        "command"
                    ],
                },
            },
            {
                "name": "add_tasks",
                "description": "Used to keep context when breaking down a complex task. "
                               "Leverage this function to declare your objective and "
                               "break it down into task / subtasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_descriptions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "list task descriptions to add",
                        },
                        "level": {
                            "type": "string",
                            "description":  "if level is 'sibling', "
                                            "adds tasks in the same level as the current task. "
                                            "if level is 'child', "
                                            "adds tasks of the current task (use for sub-tasking)",
                        }
                    },
                    "required": [
                        "task_descriptions",
                        "level"
                    ],
                },
            },
            {
                "name": "complete_task",
                "description": "Mark the current task/subtask as complete.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

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

        try:
            fn = getattr(self, function_name)
        except Exception as e:
            self.__report_err(function_name, e)
            return

        # Check if the method exists in the class
        if hasattr(self, function_name) and callable(fn):
            result = fn(**function_args)
            self.context_manager.add_message(
                {
                    "role": "function",
                    "name": function_name,
                    "content": result,
                }
            )
        else:
            self.__report_err(function_name, Exception(f"Function {function_name} does not exist"))

    def add_tasks(self, task_descriptions: list[str], level='sibling'):
        """
        :param level: if level is 'sibling', adds a task that is the same level as the current task
                        if level is 'child', adds a task as a child of the current task
        :param task_descriptions: descriptions of the tasks
        """
        if level == 'sibling':
            for desc in task_descriptions:
                self.task_manager.add_sibling_task(Task(desc))
        elif level == 'child':
            self.task_manager.add_subtasks([Task(desc) for desc in task_descriptions])

    def complete_task(self):
        """
        Mark current task as complete and go to the next task
        """
        self.task_manager.complete_current_task()

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
