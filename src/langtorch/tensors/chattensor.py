import numpy as np

from .texttensor import TextTensor
from ..texts import Chat


class ChatTensor(TextTensor):
    ttype = Chat

    @classmethod
    def input_formatter(cls, content):
        # Set "user" key if there are no keys in each entry
        formatted_content = []
        for entry in content.flat:
            if entry.keys() == [""]:
                formatted_content.append(entry.set_key("user"))
            else:
                formatted_content.append(entry)
        return np.array(formatted_content, dtype=object).reshape(content.shape)

    @classmethod
    def linter(cls, tensor):
        formatted_content = []
        for text_entry in tensor.content.flat:
            for i, key in enumerate(text_entry.keys()):
                if i == len(text_entry.keys()) - 1 and key == "":
                    key = "assistant" if text_entry.keys()[i - 1] == "user" else "user"
                    text_entry.set_key(text_entry.keys()[:-1] + [key])
                if key not in cls.ttype.allowed_keys:
                    raise ValueError(
                        f"Invalid key '{key}' found. Only {cls.ttype.allowed_keys} keys are allowed. Tensor items: {str(tensor.items())}")
            formatted_content.append(cls.ttype(text_entry))
        tensor.content = np.array(formatted_content, dtype=object).reshape(tensor.shape)

        return tensor

    def __new__(cls, content="", *args, **kwargs):
        tensor = super().__new__(cls, content=content, *args, **kwargs)
        assert hasattr(tensor, "_metadata"), "No metadata after initializing ChatTensor"
        return tensor


def User(content):
    return ChatTensor(content).set_key("user")


def Assistant(content):
    return ChatTensor(content).set_key("assistant")


Human, HumanMessage = User, User
AI, AIMessage = Assistant, Assistant
