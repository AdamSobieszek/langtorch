from .text import Text


class Chat(Text):
    language = 'chat'
    allowed_keys = ["system", "user", "assistant"]

    def __str__(self):
        return "\n".join([f"{k}: {Text(v)}" for k, v in self.items()])


class ChatML(Chat):
    language = 'chatml'
    allowed_keys = ["user", "assistant", "system"]

    def __str__(self):
        formatted = "\n".join(f"<|im_start|>{k}\n{Text(v)}<|im_end|>" for k, v in self.items())
        return formatted if self.keys()[-1] == "assistant" else formatted + "\n<|im_start|>assistant\n"


class HumanMessage:
    def __new__(cls, content):
        from langtorch.tensors import TextTensor, ChatTensor
        if isinstance(content, str):
            return Chat([("user", content)], parse=False)
        elif isinstance(content, TextTensor):
            return ChatTensor(content.set_key("user"), parse=False)
        else:
            raise ValueError("Content must be of type Text or str.")


class AIMessage:
    def __new__(cls, content):
        from langtorch.tensors import TextTensor, ChatTensor
        if isinstance(content, str):
            return Chat([("assistant", content)], parse=False)
        elif isinstance(content, TextTensor):
            return ChatTensor(content.set_key("assistant"), parse=False)
        else:
            raise ValueError("Content must be of type Text or str.")


### MESSAGE ALIASES
User, Assistant = HumanMessage, AIMessage
