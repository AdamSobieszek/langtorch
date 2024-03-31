from typing import List, Optional, Union, Tuple

import numpy as np
import torch

import langtorch

TextTensor = langtorch.TextTensor
Text = langtorch.Text


def append_text_labels(input: TextTensor, dim: Union[int, List[int]]) -> TextTensor:
    """Formats the input by adding labels based on the specified dimensions."""
    for d in dim:
        labels = TextTensor([f"\nText {i + 1}:\n" for i in range(input.shape[d])]).reshape(
            [1] * d + [input.shape[d]] + [1] * (len(input.shape) - d - 1))
        input = labels + input
    return input
