from task_manager import TaskManager


# Manages conversation context. All message history is stored in messages list.
# Keeps track of token size and keeps context size under the limit.
# Only uses messages pertaining to the current task to construct context.
class ContextManager:
    def __init__(self, task_manager, max_tokens):
        self.task_manager: TaskManager = task_manager
        self.max_tokens = max_tokens

    def add_message(self, message):
        if not message.get("content"):
            return
        self.task_manager.current_task.add_message(message)
        name = message.get("name")
        name = f'({name})' if name else ''
        content = message.get("content")
        if content:
            print(f'{message["role"]}{name}: {content}')

    def _user_message(self, message) -> dict:
        return {
            "role": "user",
            "content": message
        }

    def get_current_task_context(self):
        messages = []
        curr_task = self.task_manager.current_task
        parents = self.task_manager.get_task_parents(curr_task)
        for task in parents:
            if task is curr_task:
                for sibling in self.task_manager.get_completed_sibling_tasks():
                    messages.append(self._user_message(sibling.ctx_desc()))
                messages.append(self._user_message(f"Current task (Task ID {curr_task.task_id}): "))

            messages.append(self._user_message(task.ctx_desc()))

            if task is curr_task and task is self.task_manager.root_task:
                for subtask in task.children:
                    messages.append(self._user_message(subtask.ctx_desc()))

            if not task.is_complete:
                messages.extend(task.messages)
        return messages

