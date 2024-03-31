from typing import Union, Callable

from .textmodule import TextModule
from langtorch.tensors import CodeTensor


class CodeModule(TextModule):
    def __init__(self,
                 prompt: Union[str, 'TextTensor', Callable] = [""],
                 activation=None,
                 key=None, *args, **kwargs):
        super().__init__(prompt, activation, key, *args, **kwargs)
        if not isinstance(prompt, CodeTensor):
            raise ValueError("Expected a CodeTensor for initialization.")

    def forward(self, input_text_tensor):
        return self._prompt.eval(input_text_tensor)
