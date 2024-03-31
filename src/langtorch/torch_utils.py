from typing import Optional, Tuple

import torch
from torch.types import _TensorOrTensors

_OptionalTensor = Optional[torch.Tensor]


def tensor_or_tensors_to_tuple(tensors: Optional[_TensorOrTensors], length: int) -> Tuple[_OptionalTensor, ...]:
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, torch.Tensor):
        return (tensors,)
    return tuple(tensors)
