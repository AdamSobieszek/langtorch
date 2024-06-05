from typing import Any, Sequence, Union

import torch

import langtorch

# Convenience aliases for common composite types that we need
# to talk about in LangTorch

TextTensorOrTensors = Union[langtorch.TextTensor, Sequence[langtorch.TextTensor]]
TextTensorOrText = Union[langtorch.TextTensor, str]


# Storage protocol implemented by ${Type}StorageBase classes, maybe use this interface

class Storage:
    _cdata: int
    device: torch.device
    dtype: torch.dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo) -> 'Storage':
        ...

    def _new_shared(self, int) -> 'Storage':
        ...

    def _write_file(self, f: Any, is_real_file: _bool, save_size: _bool, element_size: int) -> None:
        ...

    def element_size(self) -> int:
        ...

    def is_shared(self) -> bool:
        ...

    def share_memory_(self) -> 'Storage':
        ...

    def nbytes(self) -> int:
        ...

    def cpu(self) -> 'Storage':
        ...

    def data_ptr(self) -> int:
        ...

    def from_file(self, filename: str, shared: bool = False, nbytes: int = 0) -> 'Storage':
        ...

    def _new_with_file(self, f: Any, element_size: int) -> 'Storage':
        ...

    ...
