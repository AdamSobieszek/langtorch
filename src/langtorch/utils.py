from typing import Optional, List, Tuple, Sequence

import numpy as np
import torch
from torch.types import _TensorOrTensors

import langtorch
from .torch_utils import _OptionalTensor


def zeros_like(other, **kwargs):
    size = other.content.size if isinstance(other, langtorch.TextTensor) else other.size
    return langtorch.TextTensor([langtorch.Text()] * size, **kwargs).reshape(other.shape)


def zeros(*shape, **kwargs):
    return langtorch.TextTensor(np.char.array(torch.zeros(*shape), unicode=True).join("").astype(dtype="<U"), **kwargs)


def full(shape, text_entry, **kwargs):
    size = np.prod(shape)
    if not 'parse' in kwargs:
        kwargs['parse'] = False
    return langtorch.TextTensor([text_entry] * int(size), **kwargs).reshape(shape)


def full_like(other, text_entry, **kwargs):
    size = other.content.size if isinstance(other, langtorch.TextTensor) else other.size
    if not 'parse' in kwargs:
        kwargs['parse'] = False
    return langtorch.TextTensor([text_entry] * int(size), **kwargs).reshape(other.shape)


def is_Text(obj):
    return isinstance(obj, str) and hasattr(obj, "content")


def is_str(obj):
    return isinstance(obj, str) and not hasattr(obj, "content")


def is_TextTensor(obj):
    return isinstance(obj, torch.Tensor) and hasattr(obj, "content")


def iter_subarrays(arr, dim):
    # Check that dim is a valid dimension
    if dim < 0: dim = dim + len(arr.shape)
    if dim >= len(arr.shape):
        raise ValueError(f"dim must be between -{len(arr.shape)} and {len(arr.shape) - 1}")

    # Split the array along the dim dimension
    subarrays = np.array_split(arr, arr.shape[dim], axis=dim)

    # Return an iterator over the subarrays
    return iter(subarrays)


def tensor_or_tensors_to_tuple(tensors: Optional[_TensorOrTensors], length: int) -> Tuple[_OptionalTensor, ...]:
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, torch.Tensor):
        return (tensors,)
    return tuple(tensors)


def torch_function_on_metadata(text_func, tensor_func, tensors, *func_args, **func_kwargs):
    metadata = [tensor.metadata for tensor in tensors]
    keys = list(set([key for tensor in tensors for key in tensor.metadata.keys()]))

    new_metadata = {}
    for key in keys:
        values = [tensor.metadata.get(key, None) for tensor in tensors]
        not_nan = [value for value in values if value is not None]
        try:
            metadata_type = type(not_nan[0])
            if 0 < len(not_nan) < len(values):
                values = [value if value is not None else metadata_type(np.full_like(tensor.content, np.nan)) for
                          tensor, value in zip(tensors, values)]
        except:
            continue
        new_metadata[key] = (text_func if metadata_type is np.ndarray else tensor_func)(values, *func_args,
                                                                                        **func_kwargs)
    return new_metadata


def _calculate_shape(output: torch.Tensor, grad: torch.Tensor,
                     is_grads_batched: bool):
    # is_same_size ensures that both tensors are either nested or non nested
    if output.is_nested:
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with Nested Tensor.")
        out_shape = output._nested_tensor_size()
        grad_shape = grad._nested_tensor_size()

        return out_shape, grad_shape

    reg_out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return reg_out_shape, reg_grad_shape


def make_grads(outputs: Sequence[torch.Tensor], grads: Sequence[_OptionalTensor],
               is_grads_batched: bool) -> Tuple[_OptionalTensor, ...]:
    new_grads: List[_OptionalTensor] = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, torch.Tensor):
            first_grad = grad if not is_grads_batched else grad[0]
            if not torch.is_same_size(out, first_grad):
                out_shape, grad_shape = langtorch._calculate_shape(out, first_grad, is_grads_batched)
                if is_grads_batched:
                    raise RuntimeError("If `is_grads_batched=True`, we interpret the first "
                                       "dimension of each grad_output as the batch dimension. "
                                       "The sizes of the remaining dimensions are expected to match "
                                       "the shape of corresponding output, but a mismatch "
                                       "was detected: grad_output["
                                       + str(grads.index(grad)) + "] has a shape of "
                                       + str(grad_shape) + " and output["
                                       + str(outputs.index(out)) + "] has a shape of "
                                       + str(out_shape) + ". "
                                                          "If you only want some tensors in `grad_output` to be considered "
                                                          "batched, consider using vmap.")
                else:
                    raise RuntimeError("Mismatch in shape: grad_output["
                                       + str(grads.index(grad)) + "] has a shape of "
                                       + str(grad_shape) + " and output["
                                       + str(outputs.index(out)) + "] has a shape of "
                                       + str(out_shape) + ".")
            if out.dtype.is_complex != grad.dtype.is_complex:
                raise RuntimeError("For complex Tensors, both grad_output and output"
                                   " are required to have the same dtype."
                                   " Mismatch in dtype: grad_output["
                                   + str(grads.index(grad)) + "] has a dtype of "
                                   + str(grad.dtype) + " and output["
                                   + str(outputs.index(out)) + "] has a dtype of "
                                   + str(out.dtype) + ".")
            new_grads.append(grad)
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
            else:
                new_grads.append(None)
        else:
            raise TypeError("gradients can be either Tensors or None, but got " +
                            type(grad).__name__)
    return tuple(new_grads)


def positive_indices(indices, shape):
    """
    Change negative to positive indices, handling newaxis (None) and slices.

    Parameters:
    - indices: a tuple containing integer indices, slices, and/or None values
    - shape: the shape of the array being indexed

    Returns:
    - A tuple of indices with all negative indices converted to positive and
      newaxis (None) preserved.
    """

    if not isinstance(indices, tuple):
        indices = (indices,)

    result = []
    dim = 0  # dimension counter for the shape

    for index in indices:
        if index is None:
            # Preserve newaxis in the result
            result.append(index)
        elif isinstance(index, slice):
            # Convert negative slice indices to positive
            start = index.start
            stop = index.stop
            if start is not None and start < 0:
                start += shape[dim]
            if stop is not None and stop < 0:
                stop += shape[dim]
            result.append(slice(start, stop, index.step))
            dim += 1  # increment dimension counter for slices
        elif isinstance(index, int):
            # Convert negative integer index to positive
            if index < 0:
                index += shape[dim]
            result.append(index)
            dim += 1  # increment dimension counter for integer indices
        else:
            # Pass through any other types (like Ellipsis)
            result.append(index)

    return tuple(result)
