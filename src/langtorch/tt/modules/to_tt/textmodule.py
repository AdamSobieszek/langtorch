from typing import List
from typing import Union, Callable

import torch

import langtorch.torch_utils
import langtorch.utils
from langtorch.tensors import TextTensor


class TextModule(torch.nn.Module):
    """
       A `TextModule` is an abstraction designed to facilitate operations on `TextTensor` objects using a
       chain of texts transformations and language model inferences. It inherits from `torch.nn.Module`.

       Attributes:
           prompt (TextTensor): A `TextTensor` containing the prompt template(s) that will be
                                 used to format input texts data.
           activation (torch.nn.Module): A callable module, typically representing a language model
                                         inference call, which serves as the 'activation function' for
                                         the module. This activation function is applied to the
                                         formatted texts to obtain the output.
           key (str): An optional string representing the key that will be automatically assigned to
                      the output `TextTensor`.

       Methods:
           __init__(prompt=[""], activation=None, key=None): A `TextModule` may be created from a prompt template (str / TextTensor),
                                                            or a callable to create a nn.Module wrapper for it.
                                                             You can pass an activation (LLM) module,
                                                             and a key that should be added to entries of the output tensors.
           forward(input: TextTensor) -> TextTensor: Processes the input `TextTensor` by formatting it
                                                     with the module's content and then passing the
                                                     result through the activation function. The output
                                                     is then returned with the specified key.

       Examples:
           >>> activation = Activation()  # This should be an instance of torch.nn.Module.
           >>> text_module = TextModule(TextTensor(["Summarize the following texts: {texts}"]),
                                        activation=activation, key="summary")
           >>> input_text = TextTensor({"texts": "An example input texts."})
           >>> summary = text_module(input_text)
           # summary is a TextTensor with the key "summary" containing the summarized texts.
       """
    input_class = TextTensor
    output_class = TextTensor

    def __init__(self,
                 prompt: Union[str, TextTensor, list] = [""],
                 activation=None,
                 key=None,
                 type_checking=False, *args, **kwargs):
        super(TextModule, self).__init__(*args, **kwargs)

        if not isinstance(prompt, TextTensor):
            self._prompt = TextTensor(prompt)
        else:
            self._prompt = prompt[tuple()]

        if isinstance(activation, str):
            from langtorch import Activation
            self.activation = Activation(activation)
        else:
            self.activation = activation  # An identity function if nothing is passed
        self.target_embedding = None
        if not issubclass(self.output_class, TextTensor) and key is not None:
            raise ValueError(
                f"Could not set key '{key}', Module output inidcated in the output_class attr is not a TextTensor")
        self.key = key
        if type_checking:
            self.register_forward_pre_hook(self.pre_forward_hook)

        self.register_forward_hook(self.post_forward_hook)

    @staticmethod
    def pre_forward_hook(module, input: List[TextTensor]):
        for tensor in input:
            assert isinstance(tensor, module.input_class)

    @staticmethod
    def post_forward_hook(module, input, output):
        if module.key is not None and issubclass(module.output_class, TextTensor):
            if not isinstance(output, TextTensor):
                print(f"Could not set key '{module.key}', Module output is not a TextTensor")
            else:
                output.set_key_(module.key)

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        self._prompt = prompt._prompt if isinstance(prompt, TextModule) else prompt if isinstance(prompt,
                                                                                                  TextTensor) else TextTensor(
            prompt)
        assert self._prompt is not None

    @property
    def content(self):
        return self.parameters()

    @content.setter
    def content(self, content):
        raise NotImplementedError(
            'Setting TextModule.content is ambiguous, as it holds all module parameters.\nUse text_module.prompt = "New prompt" instead.')

    def forward(self, *input) -> TextTensor:
        """
                By default, the TextModule forward formats the input using the prompt template tensors using TextTensor multiplication,
                then if not None applies an activation function, then if not None a key is set for all entries of the output TextTensor.
                Subclass this class to use forward hooks that check the input and output types and set the key.

                Parameters:
                    input (TextTensor): The input `TextTensor` that needs to be processed.

                Returns:
                    TextTensor: The output `TextTensor` with the content processed by the activation function
                                and assigned the specified key.

                Examples:
                    >>> input_text = TextTensor({"texts": "An example input texts."})
                    >>> output = text_module(input_text)
                    # output is a TextTensor with the processed content and key "key_points".
                """

        return self._forward(*input)

    def _forward(self, input) -> TextTensor:
        return self.activation(self._prompt * input) if self.activation else self._prompt * input

    def extra_repr(self):
        # This method is used to provide extra information for the print representation of the module
        # form = lambda name, param: f'{", " if repr else "  "}{name}={param}'
        if self._prompt is not None:
            repr = f'  prompt={self._prompt},\n'
        else:
            repr = ''
        for name, param in self.named_parameters():
            if name not in ['_prompt', 'activation']:
                repr += f'  {name}={param},\n'
        if self.activation:
            repr += f'  activation={self.activation},\n'
        if self.key != None:
            repr += f'    key={self.key}'
        return repr if not repr else repr[:-2] if repr[-2:] == ",\n" else repr

    def __contains__(self, item):
        return item in self.content

    def embed(self, *args, **kwargs):
        for param in self.content:
            param.embed(*args, **kwargs)
