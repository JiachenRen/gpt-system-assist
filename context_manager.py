import tiktoken


class ContextManager:
    """
    Manages conversation context. All message history is stored in messages list.
    Keeps track of token size and keeps context size under the limit.
    Only uses messages pertaining to the current task to construct context.
    """
    def __init__(self, objective, max_tokens, model_name):
        self.max_tokens = max_tokens
        self.messages = []
        self.archived_messages = []
        self.total_tokens = 0
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.objective_msg = self._user_message(objective)
        self.objective_tokens = self.count_tokens_in_msg(self.objective_msg)

    def count_tokens_in_msg(self, message: dict):
        tokens = 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            tokens += len(self.encoding.encode(value))
            if key == "name":  # If there's a name, the role is omitted
                tokens -= 1  # Every reply is primed with <im_start>assistant
        return tokens

    def add_message(self, message):
        self.messages.append(message)
        self.total_tokens += self.count_tokens_in_msg(message)
        while self.total_tokens + self.objective_tokens > self.max_tokens:
            msg = self.messages.pop(0)
            msg_token_size = self.count_tokens_in_msg(msg)
            self.total_tokens -= msg_token_size
            self.archived_messages.append(msg)
        content = message.get("content")
        name = message.get("name")
        name = f'({name})' if name else ''
        if content:
            print(f'{message["role"]}{name}: {content}')

    def _user_message(self, message) -> dict:
        return {
            "role": "user",
            "content": message
        }

    def _sys_message(self, message) -> dict:
        return {
            "role": "system",
            "content": message
        }

    def get_context(self):
        ctx = [self.objective_msg]
        ctx.extend(self.messages)
        return ctx


