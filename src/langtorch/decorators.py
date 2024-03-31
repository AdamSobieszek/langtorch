import functools
import inspect

import torch

from . import ctx
from .tensors import TextTensor


def set_defaults_from_ctx(func):
    """Use this decorator to set args with the value "default" to values from the context configuration.

    Example:
        @set_defaults_from_ctx
        def some_function(arg1, arg2, arg3='default'):
            # Function implementation, arg3 will be set to ctx.arg3 if arg3 is 'default'
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)

        module_path = func.__module__  # Full module path
        qual_name = func.__qualname__  # Qualified name including class name for class methods
        relative_path = module_path.replace("langtorch.", "", 1) + '.' + qual_name
        relative_path = relative_path.replace(".__new__", "").replace(".__init__", "")
        path_parts = ["cfg"] + relative_path.split('.')  # Split the path into parts
        if "functional" in path_parts:
            path_parts.remove("functional")
        for name, param in sig.parameters.items():
            path_parts_with_name = path_parts + [name]
            if not "tt" in path_parts:
                if name == "model":
                    path_parts_with_name = ["cfg", "default_model_for_functions"]
            try:
                # Navigate through the nested structure
                current_value = ctx._config
                for part in path_parts_with_name:
                    current_value = getattr(current_value, part)
                if name not in bound_args.arguments:
                    kwargs[name] = current_value
            except AttributeError:
                # Handle cases where the attribute does not exist
                # print(f"AttributeError: {name} not found in {path_parts_with_name}, {path_parts}")
                continue

        return func(*args, **kwargs)

    return wrapper


class TextToText(torch.nn.Module):
    """
    Wrapper that converts a TextTensor input to a TextTensor output using a specified model or function.

    Parameters:
    - model (callable): The model or function to be wrapped.
    - reshape (bool, optional): If True, reshapes the output TextTensor to the shape of the input. Default is True.
    - key (str, optional): Key to set for the output TextTensor. Default is None.
    - **model_kwargs: Keyword arguments for the wrapped model or function.
    """

    def __init__(self, model, reshape=True, key=None, **model_kwargs):
        super(TextToText, self).__init__()
        self.model = model
        self.reshape = reshape
        self.key = key
        self.model_kwargs = model_kwargs

    def forward(self, input_texttensor: TextTensor, **gen_kwargs) -> TextTensor:
        """
        Processes the given TextTensor input and returns a TextTensor output.

        Parameters:
        - input_texttensor (TextTensor): Input data in TextTensor format.
        - **gen_kwargs: Additional keyword arguments for the wrapped model or function.

        Returns:
        - TextTensor: Processed output in TextTensor format.
        """
        assert isinstance(input_texttensor, TextTensor)

        texts = [str(entry) for entry in input_texttensor.flat]
        decoded_texts = self.model(texts, **self.model_kwargs, **gen_kwargs)
        output = TextTensor([str(text) for text in decoded_texts]).reshape(
            input_texttensor.shape if self.reshape else (-1,))

        return output if self.key is None else output.set_key(self.key)


class TextToTensor(torch.nn.Module):
    """
    Wrapper that converts a TextTensor input to a torch.Tensor output using a specified model or function.

    Parameters:
    - model (callable): The model or function to be wrapped.
    - reshape (bool, optional): If True, reshapes the output torch.Tensor to the shape of the input. Default is True.
    - key (str, optional): Key to set for the output torch.Tensor. Default is None.
    - **model_kwargs: Keyword arguments for the wrapped model or function.
    """

    def __init__(self, model, reshape=True, key=None, **model_kwargs):
        super(TextToTensor, self).__init__()
        self.model = model
        self.reshape = reshape
        self.key = key
        self.model_kwargs = model_kwargs

    def forward(self, input_texttensor: TextTensor, **gen_kwargs) -> torch.Tensor:
        """
        Processes the given TextTensor input and returns a torch.Tensor output.

        Parameters:
        - input_texttensor (TextTensor): Input data in TextTensor format.
        - **gen_kwargs: Additional keyword arguments for the wrapped model or function.

        Returns:
        - torch.Tensor: Processed output in torch.Tensor format.
        """
        assert isinstance(input_texttensor, TextTensor)

        texts = [str(entry) for entry in input_texttensor.flat]
        outputs = self.model(texts, **self.model_kwargs, **gen_kwargs)
        tensor_outputs = torch.tensor(outputs).reshape(input_texttensor.shape if self.reshape else (-1,))

        return tensor_outputs
