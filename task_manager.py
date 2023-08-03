from typing import Optional
from chat_completion import chat_completion


class Task:
    __task_id_counter: int = 0

    def __init__(self, description: str, name: Optional[str] = None):
        self.name = name
        self.description = description
        self.task_id = Task.__task_id_counter
        self.children = []
        self.parent: Optional[Task] = None
        self.next_sibling: Optional[Task] = None
        self.messages: list[object] = []
        self._summarization: Optional[str] = None
        self.is_complete = False

        Task.__task_id_counter += 1

    def add_child(self, child):
        if len(self.children) > 0:
            self.children[-1].next_sibling = child
        self.children.append(child)
        child.parent = self

    def add_message(self, message):
        self.messages.append(message)

    # Use only when task is complete
    def get_summarization(self):
        if not self._summarization:
            if len(self.children) > 0:
                sums = [{
                    "role": "user",
                    "content": f"""
                    Task ID: {self.task_id}
                    Description: {self.description}
                    """
                }]
                for child in self.children:
                    sums.append({
                        "role": "user",
                        "content": child.ctx_desc()
                    })
                self._summarization = chat_completion.summarize(sums)
            else:
                self._summarization = chat_completion.summarize(self.messages)
        return self._summarization

    def ctx_desc(self):
        """
        Returns a description of the task with context information.
        :return:
        """
        parent_info = f"(subtask of task {self.parent.task_id})" if self.parent else ""
        if not self.is_complete:
            return f"Task {self.task_id} {parent_info} in progress: {self.description}"

        return f"""
         Completed task {self.task_id} {parent_info}: 
        {self.get_summarization()}
        """


class TaskManager:

    # Keeps track of an objective (string) and a list of tasks (list[str])
    def __init__(self, root_task: Task):
        self.root_task = root_task
        # Index of the current task
        self.current_task = root_task

    # Adds a task that is the same level as the current task
    def add_sibling_task(self, task: Task):
        if self.current_task is self.root_task:
            self.add_subtask(task)
        else:
            self.current_task.parent.add_child(task)
            self.update_current_task()

    def add_subtasks(self, tasks: list[Task]):
        for task in tasks:
            self.current_task.add_child(task)
        self.update_current_task()

    # Adds a task as a child of the current task
    def add_subtask(self, task: Task):
        self.current_task.add_child(task)
        self.update_current_task()

    # Returns a list of tasks that are parents of the task
    # If task is None, returns the current task's parents
    def get_task_parents(self, task: Optional[Task]) -> list[Task]:
        task = task if task else self.current_task
        task_hierarchy = []
        while task:
            task_hierarchy.insert(0, task)
            if not task.parent:
                break
            task = task.parent
        return task_hierarchy

    def get_completed_sibling_tasks(self):
        if not self.current_task.parent:
            return []
        return [child for child in self.current_task.parent.children if child.is_complete]

    def complete_current_task(self):
        self._mark_task_complete(self.current_task)
        self.update_current_task()

    def _mark_task_complete(self, task: Task):
        if task is self.root_task:
            # Never mark the root task as complete
            return
        task.is_complete = True
        if task.parent and all([child.is_complete for child in task.parent.children]):
            self._mark_task_complete(task.parent)

    # Go to the next task. Always goes to the deepest incomplete task in the tree.
    def update_current_task(self):
        for child in self.current_task.children:
            if not child.is_complete:
                self.current_task = child
                self.update_current_task()
        if not self.current_task.parent:
            return
        if self.current_task.is_complete:
            self.current_task = self.current_task.parent
            self.update_current_task()




