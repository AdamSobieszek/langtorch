import numpy as np
import torch
from torch.autograd.function import Function

import langtorch


def _numpy_object_pad(input_array, pad, value=None):
    """
    Pad a numpy array with object type dtype.

    :param input_array: Array to be padded, of object type.
    :param pad: Tuple of int pairs to pad on the dimensions (e.g., (1, 1, 2, 2) for 2D).
    :param value: Padding value for all sides.
    :return: Padded numpy array.
    """
    # The pad should be specified for each axis
    if len(pad) % 2 != 0:
        raise ValueError("Pad must be a tuple of even length")

    # Calculate the new shape of the padded array
    new_shape = list(input_array.shape)
    for axis, (before, after) in enumerate(zip(pad[::2], pad[1::2])):
        new_shape[axis] += before + after

    # Create a new array with the new shape and object type, fill with padding value
    padded_array = np.full(new_shape, value, dtype=object)

    # Copy the input array into the padded array at the correct start indices
    indices = tuple(slice(before, before + size) for before, size in zip(pad[::2], input_array.shape))
    padded_array[indices] = input_array

    return padded_array


def _unbroadcast_gradients(input_grad, tensor1, tensor2, gradient_equation=(lambda grad, t1, t2: (grad, grad)), sep=""):

        # Get the shapes of the original tensors
        tensor1_shape = tensor1.shape
        tensor2_shape = tensor2.shape

        # Determine the maximum number of dimensions
        max_dims = max(len(tensor1_shape), len(tensor2_shape))

        # Pad the shapes with ones on the left side to make the number of dimensions equal
        tensor1_shape = (1,) * (max_dims - len(tensor1_shape)) + tensor1_shape
        tensor2_shape = (1,) * (max_dims - len(tensor2_shape)) + tensor2_shape

        # Initialize the gradient tensors with the same shape as the input tensors
        grad1, grad2 = gradient_equation(input_grad, tensor1, tensor2)

        # Iterate over the dimensions in reverse order
        for i in range(max_dims - 1, -1, -1):
            if tensor1_shape[i] == 1 and tensor2_shape[i] != 1:
                grad1 = grad1.sum(dim=i, sep=sep, keepdim=True)
            elif tensor1_shape[i] != 1 and tensor2_shape[i] == 1:
                grad2 = grad2.sum(dim=i, sep=sep, keepdim=True)

        # Remove the extra dimensions of size 1 from the gradient tensors
        grad1 = grad1.view(tensor1.shape)
        grad2 = grad2.view(tensor2.shape)

        assert input_grad.__class__ == grad1.__class__

        return grad1, grad2


class AddTextTensor(Function):
    """TextTensor + TextTensor"""

    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input, other)  # Save tensors for backward pass
        assert isinstance(other, langtorch.TextTensor)
        result = input.__class__(input.content + other.content, parse=False)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        assert isinstance(grad_output, langtorch.TextTensor)
        input, other = ctx.saved_tensors

        input_grad = langtorch.zeros(input.shape) + grad_output
        output_grad = langtorch.zeros(input.shape) + grad_output
        if not isinstance(grad_output, langtorch.TextTensor):
            input_grad = langtorch.zeros_like(input)
            output_grad = langtorch.zeros_like(other)
        return input_grad, output_grad


class MulTextTensor(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        # Store the input for backward computation
        assert isinstance(input1, langtorch.TextTensor) and hasattr(input1, "content"), f"{input1} is not a valid TextTensor"
        if not isinstance(input2, langtorch.TextTensor):
            input2 = langtorch.TextTensor(input2, parse=False)

        # Compute the result using the content attribute's multiplication
        # FIX the use of copy especially at the end could be problematic
        result = input1.__class__(input1.content.copy() * input2.content, parse=False)
        if input1.requires_grad or input2.requires_grad:
            result.requires_grad = True
        ctx.skip_mul_grad = langtorch.is_skip_mul_grad_enabled()
        ctx.save_for_backward(input1.copy(), input2.copy())
        return (result)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        assert isinstance(grad_output, langtorch.TextTensor) and hasattr(grad_output,
                                                                         "content"), "Backward got a TextTensor without .content"
        input1, input2 = ctx.saved_tensors
        f = (lambda grad,t1,t2: (grad,grad)) if langtorch.is_skip_mul_grad_enabled() or ctx.skip_mul_grad else (lambda grad, t1, t2: (grad * t2, grad * t1))
        grad_input1, grad_input2 = _unbroadcast_gradients(grad_output, input1, input2, gradient_equation=f) # What's the best way to justify this order in grad calculation

        return grad_input1, grad_input2


class FormatTextTensor(Function):
    """TextModule.format"""

    @staticmethod
    def forward(ctx, input, args, kwargs):
        ctx.save_for_backward(input, kwargs)
        content = input.content
        output = []
        for t in content.flat:
            output.append(t.format(*args, **kwargs))
        return input.__class__(output).reshape(input.shape)

    @staticmethod
    def backward(ctx, grad_output):
        input, args, kwargs = ctx.saved_tensors
        return grad_output, None, None


from torch.autograd import Function


def create_text_tensor_function(func, backward=None):
    """
    Creates a Torch function that applies the given function to each entry of a TextTensor.
    """

    class TextTensorFunction(Function):
        @staticmethod
        def forward(ctx, input, args=tuple(), kwargs=None):
            ctx.save_for_backward(input)
            if kwargs is None:
                kwargs = {}
            content = input.content
            output = [func(t, *args, **kwargs) for t in content.flat]
            return input.__class__(output).reshape(input.shape)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors

            if backward is not None:
                return grad_output, None, None
            else:
                grad_content = grad_output.content
                new_grad = [backward(t) for t in grad_content.flat]
                return grad_output.__class__(new_grad).reshape(grad_output.shape), None, None

    return TextTensorFunction


def format(text, *args, **kwargs):
    return text.format(*args, **kwargs)


FormatTextTensor = create_text_tensor_function(format)


class DropoutTextTensor(Function):
    @staticmethod
    def forward(ctx, input, p=0.5, training=True, inplace=False):
        import langtorch
        if training:
            # Generate dropout mask based on probability p
            mask = torch.bernoulli(torch.full(input.shape, p)).bool()
            if inplace:
                input[mask] = langtorch.Text()
                return input.clone()
            else:
                output = input.clone()
                output[mask] = langtorch.Text()
                return output
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        # Pass through gradients as dropout does not alter gradients in backward pass
        return grad_output, None, None, None


def dropout(input: 'TextTensor', p: float = 0.5, training: bool = True, inplace: bool = False) -> 'TextTensor':
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))

    return DropoutTextTensor.apply(input, p, training, inplace)


# def _dropout(text, p, *args, **kwargs):
#     return text if np.random.random() <= p else text.__class__()
#
#
# DropoutTextTensor = create_text_tensor_function(_dropout)


class SplitTextTensor(Function):
    @staticmethod
    def forward(ctx, input, on, dim=0):
        shape = input.content.shape
        all_splits = [text.split(on) for text in input.content.reshape(-1)]
        max_len = max([len(s) for s in all_splits])
        all_splits = [s + [torch.nan] * (max_len - len(s)) for s in all_splits]
        output = input.__class__(all_splits, parse=False)
        new_shape = output.shape
        ctx.save_for_backward(input, on, new_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, kwargs, new_shape, dim = ctx.saved_tensors
        grad_input = grad_output.sum(dim=0)
        return grad_input, None, None


class RepeatTextTensor(Function):
    """TextModule.format"""

    @staticmethod
    def forward(ctx, input, shape):
        # old_shape = input.content.shape
        # ctx.save_for_backward(input, torch.tensors(old_shape), torch.tensors(shape))
        #
        # result = super(langtorch.TextTensor, input).repeat(*sizes)
        # # to-do
        # return output
        return input

    @staticmethod
    def backward(ctx, grad_output):
        self, kwargs, new_shape = ctx.saved_tensors
        grad_input = grad_output.sum(axis=0)
        return grad_input,


class StackTextTensor(Function):
    @staticmethod
    def forward(ctx, input):
        shape = input.content.shape

        # Add here
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Add here
        return ...


class JoinTextTensor(Function):
    @staticmethod
    def forward(ctx, input, sep="", dim=None):
        shape = input.content.shape
        ctx.save_for_backward(input, langtorch.TextTensor(sep, parse=False), torch.tensor(shape),
                              torch.tensor(dim if dim is not None else torch.nan))
        # Convert the content to a list and join the elements with `on`
        if len(input.shape) == 0: return input
        if dim is not None and len(input.shape) > dim and input.shape[dim] == 1: return input.squeeze(dim=dim)

        if dim is None:
            result = input.ttype()
            for i, t in enumerate(input.content.flat):
                if i != input.content.size - 1:
                    result += (t + sep)
                else:
                    result += t
            output = input.__class__(result, parse=False)
        else:
            if dim < 0: dim = dim + len(input.shape)
            new_shape = input.shape[:dim] + input.shape[dim + 1:]

            for i, tt in enumerate(langtorch.utils.iter_subarrays(input.content, dim)):
                if i == 0:
                    result = tt.reshape(new_shape)
                else:
                    result = np.array([m + sep + t for m, t in zip(result.flat, tt.flat)], dtype=object).reshape(
                        new_shape)
            output = input.__class__(result, parse=False)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, kwargs, shape, dim = ctx.saved_tensors
        if dim != torch.nan:
            dim = sorted(list([d for d in dim.flatten()]))
        else:
            dim = [*range(len(tuple(shape)) - len(tuple(grad_output.shape)))]
        for d in dim:
            grad_output = grad_output.unsqueeze(d)
        grad_output = grad_output.expand(tuple(shape))
        return grad_output, None, None


class ReshapeTextTensor(Function):
    @staticmethod
    def forward(ctx, input, shape):
        ctx.shape = input.shape

        output = input.__class__(input.content.reshape(*shape), parse=False),
        # metadata = input.metadata
        # a_apply(lambda v: v.reshape(*shape), lambda v: v.reshape(*shape, v.shape[-1]))
        # # add metadata
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.reshape(ctx.shape), None


class PermuteTextTensor(Function):
    @staticmethod
    def forward(ctx, input, axes):
        ctx.save_for_backward(input)
        tensor = input.__class__(np.transpose(input.content, axes), parse=False)
        # add metadata
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output.reshape(input), None


class Pad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pad, mode='constant', value=''):
        # Save for backward
        pad = np.array(pad)
        if pad.shape[-1] != 2 or len(pad.shape) != 2:
            pad = pad.reshape(-1, 2)
        ctx.save_for_backward(input, torch.tensor(pad))
        ctx.mode = mode
        ctx.value = value

        # Pad using numpy
        np_input = input.content
        value = value if isinstance(value, langtorch.texts.Text) else langtorch.texts.Text(str(value))
        padded = np.pad(np_input, pad, mode=mode, constant_values=value)

        # Convert back to tensors without changing the type
        tensor_output = input.__class__(padded)
        return tensor_output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass - return the gradient for the input only, rest are None
        input, pad = ctx.saved_tensors
        mode = ctx.mode  # You may add this to the grad output

        # Reshape pad into a (-1, 2) shape to get pairs
        pad = pad.view(-1, 2)

        # Create a list of slices to remove padding from grad_output
        slices = [slice(None)] * input.ndim
        for i, (p1, p2) in enumerate(pad):
            if p1.item() > 0 or p2.item() > 0:
                slices[-(i + 1)] = slice(p1.item(), None if p2.item() == 0 else -p2.item())

        # Slice the grad_output to remove the padding
        grad_input = grad_output[slices]

        return grad_input, None, None, None


class IndexTextTensor(Function):
    @staticmethod
    def forward(ctx, input, index):
        # Save the index and input for backward pass, note that we're assuming index is a tuple
        ctx.save_for_backward(input, torch.tensor(index))
        # Use the index to slice the input's content
        # Create a new TextTensor from the selected content
        output = input.__class__(metadata=input.getitem_over_metadata(index),
                                 ttype=input.ttype,
                                 embedding_model=input.embedding_model,
                                 tokenizer=input.tokenizer,
                                 tokenizer_kwargs=input.tokenizer_kwargs,
                                 requires_grad=input.requires_grad,
                                 is_gradient=input.is_gradient,
                                 is_param=input.is_param,
                                 parse=False)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input and index
        input, index = ctx.saved_tensors
        # Create a gradient tensor of the same shape as input and initialized to zero
        grad_input = langtorch.zeros_like(input, ttype=grad_output.ttype)
        # Place the grad_output in the appropriate places indicated by index
        grad_input[index] = grad_output
        # Only grad_input is needed to pass back since index doesn't require gradient
        return grad_input, None

# functional versions of modules

# def dropout(input: 'TextTensor', p: float = 0.5, training: bool = True, inplace: bool = False) -> torch.Tensor:
#
#     if p < 0.0 or p > 1.0:
#         raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
#     # return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
#
#     return DropoutTextTensor.apply(input, (p,))


# def conv1d(x: 'TextTensor', weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
#     prompts = []
#     for i in range(co):
#         for j in range(co):
#             prompts.append((i, j))

# How many windows fit with this length and stride
