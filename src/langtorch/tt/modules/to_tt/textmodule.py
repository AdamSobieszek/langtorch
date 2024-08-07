from typing import List
from typing import Union, Callable

import torch

import langtorch.torch_utils
import langtorch.utils
from langtorch.tensors import TextTensor
from langtorch.texts import Text


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
                 prompt: Union[str, TextTensor, list] = None,
                 activation=None,
                 key=None,
                 backward_prompt = None,
                 type_checking=False,
                 is_param=True, *args, **kwargs):
        super(TextModule, self).__init__(*args, **kwargs)
        if not isinstance(prompt, TextTensor):
            prompt = (TextTensor(prompt)) if prompt is not None else prompt
        else:
            prompt = prompt.copy()
        if is_param and prompt is not None:
            prompt = torch.nn.Parameter(prompt)
        self._prompt = prompt

        if isinstance(activation, str):
            from langtorch import Activation
            self.activation = Activation(activation)
        else:
            self.activation = activation  # An identity function if nothing is passed
        if backward_prompt and self.activation:
            self.activation.backward_prompt = Text(backward_prompt)

        if not issubclass(self.output_class, TextTensor) and key is not None:
            raise ValueError(
                f"Could not set key '{key}', Module output inidcated in the output_class attr is not a TextTensor")
        self.key = key
        if type_checking:
            self.register_forward_pre_hook(self.pre_forward_hook)
        if self.activation is not None or self.key is not None:
            self.register_forward_hook(self.activation_and_key_hook)

    @staticmethod
    def pre_forward_hook(module, input: List[TextTensor]):
        for tensor in input:
            assert isinstance(tensor, module.input_class)

    @staticmethod
    def activation_and_key_hook(module, input, output):
        if module.activation:
            output = module.activation(output)
        if module.key is not None and issubclass(module.output_class, TextTensor):
            if not isinstance(output, TextTensor):
                print(f"Could not set key '{module.key}', Module output is not a TextTensor")
            else:
                with langtorch.skip_mul_grad():
                    output.add_key_(module.key)
        return output

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        self._prompt = prompt._prompt if isinstance(prompt, TextModule) else prompt if isinstance(prompt,
                                                                                                  TextTensor) else TextTensor(
            prompt)

    @property
    def content(self):
        return self.parameters()

    @content.setter
    def content(self, content):
        raise NotImplementedError(
            'Setting TextModule.content is ambiguous, as it holds all module parameters.\nUse text_module.prompt = "New prompt" instead.')

    def forward(self, *inputs) -> TextTensor:
        """
                By default, the TextModule forward formats the input using the prompt template tensors using TextTensor multiplication,
                Then, forward hooks (if not None) apply the activation function and (if not None) set the key for all entries of the output TextTensor.
                Subclass this class to inherit this functionality.

                Parameters:
                    input (TextTensor): The input `TextTensor` that needs to be processed.

                Returns:
                    TextTensor: The output `TextTensor` with the content processed by the activation function
                                and assigned the specified key.

                Examples:
                    >>> text_module = TextModule(TextTensor("Summarize: {text}", activation="gpt-3.5-turbo", key="summary",
                    >>> input_text = TextTensor({"text": "An example input texts."})
                    >>> output = text_module(input_text)
                    # output is a TextTensor with the summary with a key "summary" processed by gpt-3.5-turbo.
                """
        if self._prompt is None:
            return inputs

        output = tuple(self._prompt * input for input in inputs)
        for o in output:
            print(o.requires_grad)
        if len(output) == 1:
            return output[0]
        else:
            return output

    def _get_name(self):
        try:
            return self.__class__.__name__ + f": {self.input_class.__name__}->{self.output_class.__name__} "
        except:
            return self.__class__.__name__

    def extra_repr(self):
        # This method is used to provide extra information for the print representation of the module
        add_indent = lambda name, param: f'({name}): ' + ("\n" + " " * len(f'({name}):')).join(
            str(param).split("\n")) + ",\n"

        if self._prompt is not None:
            repr = add_indent("prompt", self._prompt)
        else:
            repr = ''
        for name, param in self.named_parameters():
            if name not in ['_prompt', 'activation']:
                repr += add_indent(name, param)
        if self.key != None:
            repr += add_indent("key", self.key)
        return repr if not repr else repr[:-2] if repr[-2:] == ",\n" else repr

    def __contains__(self, item):
        return item in self.content

    def embed(self, *args, **kwargs):
        for param in self.content:
            if hasattr(param, 'embed'):
                param.embed(*args, **kwargs)

    def __or__(self, other):
        def chain(self, other):
            torch.nn.Sequential.__or__ = chain
            if isinstance(self, torch.nn.Sequential):
                # If this is already a Sequential, append other at the end
                chain_module = torch.nn.Sequential(*self, other)
                return chain_module
            else:
                # If not, create a new Sequential with self and other'
                chain_module = torch.nn.Sequential(self, other)
                return  chain_module

        return chain(self, other)
