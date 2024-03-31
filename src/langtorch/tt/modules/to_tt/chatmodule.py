from .textmodule import TextModule
from langtorch.tensors import ChatTensor


class ChatModule(TextModule):
    tensor_class = ChatTensor
