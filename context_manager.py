from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')


# Manages conversation context. All message history is stored in messages list.
# Keeps track of token size and keeps context size under the limit.
# Only uses messages pertaining to the current task to construct context.
class ContextManager:
    def __init__(self, objective, max_tokens):
        self.objective = objective
        self.objective_token_size = len(word_tokenize(objective))
        self.max_tokens = max_tokens
        self.messages = []
        self.archived_messages = []
        self.messages_token_size = 0

    def add_message(self, message):
        self.messages.append(message)
        content = message.get("content")
        self.messages_token_size += len(word_tokenize(content)) if content else 0
        while self.messages_token_size + self.objective_token_size > self.max_tokens:
            msg = self.messages.pop(0)
            msg_token_size = len(word_tokenize(msg.get("content"))) if content else 0
            self.messages_token_size -= msg_token_size
            self.archived_messages.append(msg)
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
        ctx = [self._user_message(self.objective)]
        ctx.extend(self.messages)
        return ctx


