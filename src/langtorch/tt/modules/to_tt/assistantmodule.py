import json
import logging
import time

from openai import OpenAI

from .textmodule import TextModule
from langtorch.tensors import TextTensor


class OpenAIAssistantModule(TextModule):
    def __init__(self, assistant_id, model_name="gpt-4-1106-preview", tools=None):
        super().__init__()

        self.client = OpenAI()
        self.assistant_id = assistant_id
        self.model_name = model_name
        self.tools = tools

        self.thread = self.client.beta.threads.create()
        self.id = self.thread.id

    def forward(self, text_tensor: TextTensor, tool_kwargs=dict()):
        assert isinstance(text_tensor, TextTensor) and len(text_tensor.to_list()) == 1
        # Assuming text_tensor contains user messages
        # Create and manage threads and runs for each message
        responses = []
        for user_message in text_tensor.to_list():
            self._add_message(user_message)
            run = self._run()
            run = self._wait_on_run(run)
            response, type = self._get_response(run)
            logging.debug(f"Response: {response}, type: {type}")
            if type == "message":
                responses.append(response)
            elif type == "function":
                # function_name
                tool_name, arguments = response
                logging.info(f"Function {tool_name} was called")
                assert tool_name in self.tools
                tool = self.tools[tool_name]
                response = tool(**arguments, **tool_kwargs)
                if response:
                    responses.append(response)

        return TextTensor(responses[0] if len(responses) == 1 else responses, parse=False) if len(responses) else None

    def _add_message(self, user_message, role="user"):
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id, role=role, content=str(user_message)
        )
        return

    def _run(self):
        return self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
            model=self.model_name,
        )

    def _get_response(self, run):
        if run.status == "requires_action":
            tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            return (name, arguments), "function"
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id, order="asc"
        )
        messages = json.loads(messages.model_dump_json())
        # Filter to get only assistant's responses
        assistant_responses = [
            message['content'][0]['text']['value'] for message in messages['data']
            if message['role'] == 'assistant'
        ]
        return assistant_responses[-1], "message"

    def _wait_on_run(self, run):
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run
