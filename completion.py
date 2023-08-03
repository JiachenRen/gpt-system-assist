import openai


class Completion:

    def __init__(self, model):
        self.model = model

    # Get chat completion response from GPT
    def get_chat_completion_response(self, messages, functions):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=functions,
            temperature=0.5,
            function_call="auto",
        )
        return response

    # Summarize the task and findings from sequence of messages
    def summarize(self, messages):
        messages = [m for m in messages]
        messages.append({
            "role": "user",
            "content": "summarize"
        })
        res = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
        )
        return res["choices"][0]["message"]["content"]


completion = Completion("gpt-3.5-turbo-16k")