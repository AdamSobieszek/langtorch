"""Semantic Algebra Functional interface"""
from typing import List, Optional, Union, Tuple

import numpy as np
import torch

import langtorch
from langtorch.semalg_utils import append_text_labels

TextTensor = langtorch.TextTensor
TextModule = langtorch.TextModule
Text = langtorch.Text
set_defaults_from_ctx = langtorch.decorators.set_defaults_from_ctx


@set_defaults_from_ctx
def mean(input: TextTensor,
         definition: str = "a text that is of similar length to the entries, synthesises content from all entries by combining  the most correct and important information, and is written in a similar style. It should not be a summary, but a prototypical best entry.",
         prompt: str = """Inspect the list of texts below to find the common style and content between these entries.\
Your task is to write a text that shares the most with all the entries, an "average" of these texts. \
By an average text we mean {definition}\
Answer only with the text that would be the average text of these entries:\n""",
         dim: Optional[Union[int, List[int]]] = None,
         keepdim: bool = False, model: str = 'default', T: int = 0, **kwargs) -> TextTensor:
    if len(input.flat) <= 1:
        return input

    assert dim is None or dim < len(input.shape), f"Dimension {dim} is out of range for input of shape {input.shape}"
    if dim is None:
        dim = list(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = [dim]

    dim = [len(input.shape) + d if d < 0 else d for d in dim]
    kwargs['model'], kwargs['T'] = model, T
    for d in sorted(dim)[::-1]:
        if input.shape[d] <= 1:
            continue
        input = append_text_labels(input, [d])
        input = input.sum(d, keepdim)
        input = TextModule(prompt.format(definition=definition), activation=langtorch.Activation(**kwargs))(input)
    return input


@set_defaults_from_ctx
def max(input: TextTensor,
        definition: str = "the text that stands out as the best, most helpful and relevant, while being the most correct.",
        prompt: str = """Review the list of texts below to discern the predominant themes, informational content, and stylistic elements shared among these entries.\
Your task is to identify the text with 'maximum value'. \
By 'maximum', we mean {definition}.\
In your response omit the 'Text no:' label. Respond only by copying word by word the text that is of maximum value from this list:\n""",
        dim: Optional[Union[int, List[int]]] = None,
        keepdim: bool = False, model: str = 'default', T: int = 0, **kwargs) -> TextTensor:
    if len(input.flat) <= 1:
        return input

    assert dim is None or dim < len(input.shape), f"Dimension {dim} is out of range for input of shape {input.shape}"
    if dim is None:
        dim = list(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = [dim]

    kwargs['model'], kwargs['T'] = model, T
    for d in sorted(dim)[::-1]:
        if input.shape[d] <= 1:
            continue
        input = append_text_labels(input, [d])
        input = input.sum(d, keepdim)
        input = TextModule(prompt.format(definition=definition), activation=langtorch.Activation(**kwargs))(input)
    return input


@set_defaults_from_ctx
def min(input: TextTensor,
        definition: str = "the text that stands out as incorrect, least complete or of the worst quality.",
        prompt: str = """Review the list of texts below to discern the predominant themes, informational content, and stylistic elements shared among these entries.\
Your task is to identify the text with 'minimum value'. \
By 'minimum', we mean {definition}.\
In your response omit the 'Text no:' label. Respond only by copying word by word the text that is minimum value from this list:\n""",
        dim: Optional[Union[int, List[int]]] = None,
        keepdim: bool = False, model: str = 'default', T: int = 0, **kwargs) -> TextTensor:
    if len(input.flat) <= 1:
        return input

    assert dim is None or dim < len(input.shape), f"Dimension {dim} is out of range for input of shape {input.shape}"
    if dim is None:
        dim = list(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = [dim]

    kwargs['model'], kwargs['T'] = model, T
    for d in sorted(dim)[::-1]:
        if input.shape[d] <= 1:
            continue
        input = append_text_labels(input, [d])
        input = input.sum(d, keepdim)
        input = TextModule(prompt.format(definition=definition), activation=langtorch.Activation(**kwargs))(input)
    return input
