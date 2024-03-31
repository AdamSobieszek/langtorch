import numpy as np
import torch
from torch.autograd.function import Function

import langtorch


def numpy_object_pad(input_array, pad, value=None):
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


class AddTextTensor(Function):
    """TextTensor + TextTensor"""

    @staticmethod
    def forward(ctx, input, other):
        # print("IJUIJU",input, other, type(input), type(other))
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
        assert isinstance(input1, langtorch.TextTensor) and hasattr(input1, "content"), input1
        assert isinstance(input2, langtorch.TextTensor) and hasattr(input2, "content"), input2
        ctx.save_for_backward(input1, input2)

        # Compute the result using the content attribute's multiplication
        result = input1.__class__(input1.content * input2.content, parse=False)
        return (result)

    @staticmethod
    def backward(ctx, grad_output):
        assert isinstance(grad_output, langtorch.TextTensor) and hasattr(grad_output, "content")
        input1, input2 = ctx.saved_tensors
        # print(grad_output[0].item().items()[0][1])
        grad_input1 = grad_output.__class__(langtorch.zeros(input2.shape) * grad_output * input2, parse=False)
        grad_input2 = grad_output.__class__(langtorch.zeros(input1.shape) * grad_output * input1, parse=False)
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
        grad_input = grad_output.sum(axis=0)
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
    def forward(ctx, input, on="", dim=None):
        shape = input.content.shape
        ctx.save_for_backward(input, langtorch.TextTensor(on, parse=False), torch.tensor(shape),
                              torch.tensor(dim if dim is not None else []))
        # Convert the content to a list and join the elements with `on`
        if len(input.shape) == 0: return input
        if dim is not None and len(input.shape) > dim and input.shape[dim] == 1: return input.squeeze(dim=dim)

        if dim is None:
            result = input.ttype()
            for i, t in enumerate(input.content.flat):
                if i != input.content.size - 1:
                    result += (t + on)
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
                    result = np.array([m + on + t for m, t in zip(result.flat, tt.flat)], dtype=object).reshape(
                        new_shape)
            output = input.__class__(result, parse=False)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, kwargs, shape, dim = ctx.saved_tensors
        if not dim:
            grad_input = grad_output.expand(tuple(shape))
        else:
            grad_input = grad_output.expand(tuple(shape))  # ?
        ## write backward
        return grad_input, None, None


class ReshapeTextTensor(Function):
    @staticmethod
    def forward(ctx, input, shape):
        ctx.shape = input.shape
        output = input.__class__(input.content.reshape(*shape), parse=False),
        # metadata = input.metadax
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


empty_embedding = [0.00247, -0.01825, 0.01688, -0.0122, -0.01214, 0.02237, -0.01426, -0.01932, 0.01229, -0.01559,
                   0.02513, 0.01051, -0.00389, -0.00914, 0.00606, 0.00873, 0.02602, -0.01121, 0.01427, -0.03011,
                   -0.00582, 0.01886, -0.0198, -0.013, -0.0044, 0.00925, 0.00721, -0.02493, 0.02735, -0.01461, 0.01004,
                   -0.00446, -0.01759, -0.01903, -0.01866, -0.01644, -0.00724, -0.01684, 0.0123, 0.00889, 0.02815,
                   0.02104, -0.00669, -0.0026, -0.01936, -0.00054, -0.00964, -0.0059, -0.02805, -0.00477, 8e-05,
                   0.01672, -0.00677, -0.003, -0.01728, -0.0112, 0.01161, -0.00485, -0.00256, 0.00548, -0.00404,
                   0.00459, -0.01295, 0.00035, -0.02161, -0.00537, -0.00034, -0.00184, -0.0003, -0.01238, 0.04547,
                   0.02988, 0.01211, -0.03066, 0.02955, -0.01754, 0.0009, -0.00265, -0.01153, -0.0012, 0.01651,
                   -0.02346, -0.02273, 0.00836, 0.01135, 0.00399, 0.00104, 0.01864, -0.00367, -0.01864, 0.01214,
                   0.02461, -0.00918, 0.0167, 0.00809, 0.02016, -0.01378, 0.02515, -0.01577, -0.02382, 0.00978, 0.00283,
                   -0.02052, 0.00566, -0.0395, -0.01268, 0.01798, -0.01209, 0.01745, 0.00105, -0.01573, 0.02334,
                   -0.01663, -0.03017, 0.02558, -0.00369, 0.01656, -0.02189, -0.03001, -0.01183, 0.01316, 0.01048,
                   0.0116, -0.01472, 0.01276, -0.00021, -0.02657, -0.00746, 0.00371, -0.00797, 0.048, 0.04981, 0.0336,
                   0.00438, -0.01737, 0.00907, -0.01875, 0.02851, -0.00636, -0.04319, 0.00711, 0.02374, -0.00833,
                   -0.01222, -0.00849, 0.01044, 0.02691, 0.00026, 0.01073, -0.0134, 0.0104, -0.00528, 0.0062, -0.01956,
                   0.00636, 0.00788, -0.02136, -0.00081, -0.01295, 0.00037, 0.01837, -0.00381, 0.02668, -0.01092,
                   0.00104, 0.01287, 0.02342, 0.0267, 0.00448, -0.0081, 0.02202, 0.03275, -0.02991, 0.0046, -0.00729,
                   0.0246, 0.00396, 0.00215, -0.03332, -0.00944, -0.00884, 0.01081, 0.02937, 0.02624, -0.00377, 0.00732,
                   0.02317, -0.00234, 0.00996, -0.02215, -0.00918, 0.02657, 0.02536, 0.00334, -0.68161, -0.02903,
                   -0.00806, -0.03084, 0.00995, 0.01355, 0.02675, 0.01397, -0.02138, -0.00089, -0.00739, 0.00543,
                   0.01908, -0.01499, -0.00721, 0.00858, -0.0107, -0.00916, 0.01316, 0.00715, -0.00479, -0.00248,
                   -0.03187, -0.00106, 0.02298, 0.00083, 2e-05, -0.012, 0.0199, -0.00057, -0.02257, 0.0195, -0.00097,
                   -0.0229, 0.06426, 0.00326, -0.00101, 0.03376, 0.0107, 0.02916, -0.00084, -0.01265, 0.03593, -0.00192,
                   -0.01517, -0.00612, 0.01444, 0.01109, 0.00212, 0.00346, 0.01609, 0.00559, -0.00287, 0.00661,
                   -0.00799, -0.01413, 0.01856, -0.02002, -0.00285, 0.00428, -0.00797, 0.00701, -0.01776, -0.02567,
                   -0.02533, 0.02493, -0.02526, -0.00461, -0.00178, 0.01279, -0.02921, 0.01797, -0.0233, -0.00759,
                   -0.00029, 0.01224, 0.01853, 0.01011, -0.00964, 0.00098, -0.00392, -0.03001, -0.02245, 0.00203,
                   0.02102, -0.01488, -0.00774, 0.00654, 0.0085, -0.00696, 0.02414, 0.03128, -0.01665, -0.01847,
                   -0.0185, 0.01311, 0.00129, 0.01402, 0.00032, -0.01437, -0.01041, -0.01528, 0.01716, -0.01182,
                   -0.00635, 0.0178, -0.01945, 0.00811, 0.04785, -0.01751, -0.01042, -0.01963, 0.00845, -0.00588,
                   0.00597, -0.02431, -0.00592, -0.00024, 0.02173, -0.0163, 0.02272, -0.00721, 0.01108, -0.02558,
                   0.01507, 0.01045, 0.00174, -0.02691, 0.00152, -0.00772, -0.01271, -0.00984, 0.04263, 0.00124,
                   0.02155, 0.00812, 0.04079, -0.0096, 0.01314, -0.02818, -0.01692, -0.00428, -0.00681, 0.01248, -0.014,
                   -0.03017, -0.01145, -0.02046, -0.0005, 0.00897, 0.00288, 0.00139, -0.0083, 0.02828, 0.01591,
                   -0.00408, -0.02012, -0.00351, -0.02063, -0.01622, 0.01225, 0.02838, -0.02496, -0.00247, -0.01727,
                   0.00647, 0.03159, 0.01258, -0.00879, -0.02169, 0.00834, 0.00659, 0.01248, -0.00602, 0.01548, 0.01716,
                   -0.02158, -0.01139, 0.00283, -0.00089, 0.01913, -0.01187, -0.03311, 0.00076, 0.02704, 0.00341,
                   0.02588, 0.028, -0.00558, 0.00201, -0.00371, 0.01374, -0.00217, 0.01953, -0.00639, 0.01241, 0.00511,
                   0.01092, 0.01763, 0.02283, 0.03125, 0.00316, 0.01999, 0.00719, 0.01591, -0.02107, 0.01789, -0.02451,
                   0.03045, 0.02722, -0.00191, -0.03694, 0.00522, 0.00989, 0.0129, 0.02423, -0.00539, -0.00752,
                   -0.01044, 0.0067, 0.00206, -0.02015, 0.00593, -0.00579, 0.00108, 0.01473, 0.00213, 0.01989, -0.00291,
                   -0.02043, -0.02722, -4e-05, -0.00564, 0.02836, -0.01651, 0.01064, 0.00783, -0.02162, 0.01173,
                   0.00948, -0.01011, 0.02996, 0.03572, 0.00468, 0.00752, 0.01082, 0.03381, -0.00504, -0.00796,
                   -0.01418, -0.00594, 0.01314, -0.00933, 0.0144, 0.00322, -0.02719, 0.00951, 0.01676, 0.03223, 0.03068,
                   0.00824, 0.0204, -0.00188, -0.00109, 0.00383, -0.00428, 0.01019, 0.00237, -0.01051, -0.02323,
                   0.00883, -0.0119, 0.00225, -0.01454, 0.00915, -0.03234, 0.00547, 0.00551, 0.00656, -0.00462,
                   -0.00774, -0.03195, 0.02523, 0.00965, -0.00367, -0.01252, -0.0034, -0.0118, 0.01888, 0.03947,
                   0.02244, 0.00247, -0.00645, -0.00335, 0.00678, 0.00012, 0.03019, -0.00933, 0.00154, 0.00339, 0.00448,
                   0.00756, -0.01101, 0.00231, 0.00649, 0.02208, 0.0011, -0.0209, 0.00179, 0.00163, 0.01406, -0.00731,
                   0.00137, -0.00197, 0.00804, 0.00986, -0.01201, -0.01449, 0.01123, 0.001, -0.02584, -0.00271,
                   -0.01484, 0.0025, 0.03965, 0.0282, 0.00681, 0.01966, 0.00281, 0.01297, -0.00168, -0.00596, 0.01515,
                   -0.00846, -0.00525, -0.00771, 0.01612, -0.0164, 0.01696, -0.00614, 0.01253, -0.02947, 0.00998,
                   -0.00588, 0.00108, -0.01071, 0.01479, -0.00285, 0.00845, 0.01532, 0.00944, 0.03024, 0.02391,
                   -0.02317, -0.00399, 0.00752, 0.01614, 0.02174, -0.00683, 0.00558, 0.00597, 0.00997, 0.03198,
                   -0.01511, 0.03035, 0.01882, 0.01691, -0.0183, 0.03901, -0.03136, -0.00749, 0.01113, 0.01786,
                   -0.01874, 0.01663, 0.00513, -0.01979, -0.01393, -0.01812, 0.02643, 0.0126, -0.01299, -0.00435,
                   -0.01307, -0.00589, -0.01588, 0.00555, -0.03203, -0.01392, -0.02249, -0.03161, 0.0149, -0.00275,
                   0.00496, -0.00471, -0.03211, 0.01026, 0.01081, 0.02771, 0.01258, 0.0118, -0.00478, 0.01121, 0.03164,
                   0.00155, -0.03815, 0.00529, -0.00841, -0.00143, -0.00342, -0.0, 0.01816, -0.00658, 0.00254, 0.01626,
                   -0.02093, 0.01524, 0.00887, 0.0082, -0.00612, 0.00834, 0.00823, -0.00761, 0.00088, 0.01454, -0.02029,
                   -0.00387, -0.00345, 0.01453, 0.0029, 0.01022, 0.03544, -0.03632, -0.01157, 0.00345, -0.00848,
                   0.00512, -0.00716, 0.0107, 0.01197, 0.00666, 0.01729, 0.00425, 0.00455, 0.00336, -0.0167, 0.03528,
                   0.00708, -0.001, -0.00381, -0.0024, -0.05341, -0.02166, -0.00543, 0.00508, 0.00913, 0.01229,
                   -0.01667, -0.05025, -0.01345, -0.00371, 0.02487, -0.00609, -0.01364, 0.00312, 0.01557, -0.00529,
                   -0.00417, -0.00294, -0.03782, 0.00831, -0.00117, 0.0115, 0.01232, -0.00289, -0.01006, -0.02675,
                   -0.01694, -0.01262, -0.03694, 0.00259, -0.00571, 0.03076, 0.01861, 0.0077, -0.00075, -0.00539,
                   -0.00165, -0.02686, 0.01122, -0.00302, -0.01671, -0.01413, 0.01893, 0.03265, 0.00556, 0.0081,
                   0.00383, -0.00154, 0.00796, -0.02435, -0.01389, -0.02019, -0.01228, -0.017, -0.00298, -0.00873,
                   0.00247, -0.00604, 0.00624, 0.01463, 0.00489, 0.0289, -0.00092, 0.03154, -0.01347, 0.02126, -0.0049,
                   -0.03609, -0.00653, -0.01199, -0.04131, -0.01786, 0.029, -0.01068, 0.00974, -0.00973, 0.00903,
                   -0.00979, 0.02542, -0.02617, -0.02109, -0.00511, -0.01023, -0.03014, -0.02507, -0.01308, -0.01621,
                   -0.00089, 0.00922, -0.00249, 0.0015, 0.00257, -0.03565, 0.0191, 0.00063, 0.02297, 0.00432, 0.04433,
                   0.01967, -0.02277, -0.01476, 0.01799, -0.01411, -0.00704, 0.01071, -0.00543, -0.0101, -0.00994,
                   0.0217, 0.00704, -0.0111, -0.03221, 0.02686, 0.01693, -0.02109, -0.02476, -0.00859, -0.03805,
                   -0.00149, -0.02376, 0.00936, -0.00943, -0.00703, -0.00065, 0.03262, -0.02924, 0.03226, 0.00838,
                   -0.00794, -0.01233, 0.01614, -0.00848, 0.01759, 0.0124, 0.03014, -0.01883, 0.02389, 0.00897, 0.01072,
                   -0.02311, -0.003, -0.00075, 0.01751, -0.03583, 0.01319, 0.01039, -0.0017, 0.01439, -0.00241,
                   -0.01497, -0.0166, -0.01742, -0.0042, 0.02071, 0.00873, -0.0202, -0.00757, -0.02686, -0.02102,
                   -0.00436, 0.00932, 0.01174, 0.01643, 0.00097, -0.02113, -0.00596, 0.00374, -0.00374, 0.02442,
                   0.01776, 0.02874, -0.01197, 0.01504, 0.00978, -0.00122, -0.04319, -0.00614, -0.01626, -0.04296,
                   0.02009, -0.00114, -0.01594, -0.01905, 0.00502, 0.01316, -0.00628, -0.00816, 0.02329, -0.00135,
                   0.00311, -0.02607, -0.01441, 0.00404, -0.02784, 0.01013, 0.02162, -0.03797, -0.00171, 0.00904,
                   0.01024, 0.01475, -0.02571, 0.00775, 0.01119, 0.01682, -0.00623, -0.02694, -0.00756, -0.00825,
                   -0.00929, 0.00356, 0.00387, -0.00352, 0.02077, 0.03177, -0.0067, 0.02286, -0.01331, -0.01258,
                   0.00173, 0.00886, -0.01495, -0.01988, -0.00449, -0.00202, 0.03345, -0.00936, -0.0297, 0.00374,
                   -0.01422, 0.00358, -0.00519, 0.02357, 0.01154, -0.00909, -0.01139, 0.02206, -0.02124, 0.02209,
                   -0.00289, -0.02281, 0.00619, -0.00674, 0.00426, 0.01164, -0.03549, -0.0156, 0.01824, 0.00195,
                   0.00909, 0.02144, -0.01006, -0.01654, -0.02258, -0.00944, -0.00021, 0.01497, -0.00783, -0.02474,
                   0.0005, 0.03614, 0.0081, -0.01171, -0.01022, 0.00657, -0.00278, -0.00473, -0.0088, -0.00931, 0.00438,
                   -0.00876, 0.00668, 0.01488, -0.01918, 0.01223, -0.01689, -0.008, -0.019, 0.0008, -0.0387, -0.00664,
                   0.01879, -0.02355, -0.01467, 0.00952, 0.00198, -0.02124, 0.01713, -0.0079, -0.02201, 0.00263,
                   0.01643, 0.00131, 0.00134, 0.00015, 0.0106, 0.0023, -0.01666, -0.01822, 0.00206, -0.00644, 0.02449,
                   0.00244, 0.00384, -0.01047, -0.00482, 0.00486, 0.00019, -0.01371, 0.20618, 0.00774, 0.0013, 0.03433,
                   -0.003, 0.01062, 0.01326, -0.01143, -0.02562, 0.00535, -0.00697, 0.0114, -0.02043, -0.01272, 0.00884,
                   -0.02156, -0.01495, -0.01329, -0.03068, -0.03035, -0.00536, -0.00764, -0.02722, -0.01233, 0.01143,
                   -0.00248, 0.00139, 0.00757, 0.01841, -0.00145, -0.01632, 0.00544, 0.0053, 0.00142, 0.00827, -0.00687,
                   0.0104, -0.01228, 0.0178, 0.01369, -0.005, 0.0075, 0.01373, -0.00524, 0.00713, 0.0094, -0.03004,
                   -0.00887, -0.00483, 0.00987, -0.03288, 0.01756, 0.02367, 0.02748, -0.00319, 0.00618, 0.02993,
                   0.00933, -0.01464, -0.0021, -0.01723, 0.02122, -0.01123, 0.0117, -0.01034, -0.00111, -0.02339,
                   0.00856, -0.00348, -0.0189, 0.00455, -0.01895, -0.00819, 0.00354, -0.03229, -0.02585, 0.00426,
                   0.01307, 0.0161, 0.04456, -0.00843, -0.03539, -0.00834, -0.00903, 0.00181, -0.0388, 0.02409,
                   -0.01223, -0.00894, -0.00351, -0.005, -0.02771, -0.01647, -0.00059, 0.0069, -0.00723, 0.01473,
                   0.00415, -0.00513, 0.00066, -0.01694, 0.03262, 0.02085, 0.00311, -0.01618, 0.01775, 0.00773, 0.01824,
                   -0.00293, -0.02612, -0.02115, -0.02109, 0.01442, 0.01373, 0.01234, 0.00528, -0.00621, -0.01588,
                   0.00766, 0.00087, -0.003, -0.01601, 0.0261, -0.0072, 0.00629, -0.02255, -0.0109, -0.0007, 0.00402,
                   -0.02867, 0.01331, -0.00643, 0.01794, 0.00358, -0.01109, -0.00644, 0.02043, -0.00606, -0.00979,
                   -0.00534, -0.00602, -0.00169, 0.0153, 0.0056, 0.00934, -0.00952, 0.01604, -0.02249, -0.02709,
                   -0.00041, -0.03218, -0.0028, 0.0086, -0.00026, 0.0178, -0.00351, -0.04087, -0.03844, -0.02854,
                   -0.01393, -0.01366, -0.00878, 0.04144, -0.01064, -0.0134, -0.01174, -0.16668, 0.01692, 0.00912,
                   0.00343, 0.00885, -0.01517, 0.01653, -0.01124, -0.00814, 0.01617, 0.0125, -0.01486, -0.04844,
                   -0.01217, 0.02324, 0.0049, -0.01528, 0.01791, 0.02021, 0.02696, 0.04167, -0.02458, 0.00818, -0.02986,
                   0.00176, 0.01975, -0.00979, 0.00825, -0.00342, -0.01365, -0.00753, 0.0108, 0.01298, 0.00772, -0.002,
                   0.00835, -0.02113, -0.0132, 0.00869, 0.00612, 0.02332, 0.02028, -0.00336, 0.00069, -0.00262, 0.03565,
                   0.01103, -0.00161, 0.01565, -0.00821, 0.00734, -0.01875, 0.00209, -0.00308, 0.00384, -0.00296,
                   -0.00797, 0.00679, 0.01649, -0.02006, 0.00386, 0.00573, 0.00646, -0.00638, 0.00677, -0.0086,
                   -0.01066, -0.00331, -0.03717, 0.00582, -0.00761, -0.00952, 0.0039, -0.01835, 0.0247, 0.00667,
                   0.01207, -0.00907, 0.02572, -0.01029, -0.01488, 0.03738, -0.00205, -0.00263, 0.00548, 0.01216,
                   -0.00577, -0.01815, 0.00734, -0.00919, 0.0095, -0.00655, -0.01131, -0.01073, 0.0109, 0.03454,
                   0.01839, 0.01086, -0.00407, 0.00376, 0.02187, -0.01988, -0.01922, 0.02329, 0.03805, -0.00045,
                   0.00074, 0.02496, 0.03552, -0.01366, -0.03141, -0.01812, 0.01519, 0.01888, 0.00637, 0.0036, -0.00606,
                   -0.01414, 0.03306, -0.01529, 0.03006, -0.01366, -0.01742, 0.00881, -0.00991, -0.01877, -0.09911,
                   -0.02807, 0.00404, 0.03311, -0.00058, 0.01256, 0.00496, 0.00347, -0.01216, 0.01851, -0.00699,
                   -0.02893, -0.00838, -0.02173, 0.03247, 0.00051, -0.00794, -0.03908, -0.01424, 0.0273, 0.00984,
                   -0.00366, 0.00242, -2e-05, -0.01285, -0.00386, -0.02991, 0.01079, 0.0239, -0.00517, -0.0121, 0.00233,
                   0.03128, -0.03288, -0.02218, 0.00067, -0.03327, -0.00718, 0.00779, -0.02257, 0.00799, 0.01356,
                   0.01001, -0.01874, 0.00404, -0.0201, -0.0149, 0.02133, -0.00192, -0.00653, -0.02069, -0.00967,
                   -0.04689, -0.01813, 0.0225, 0.00554, 0.00546, 0.01076, -0.01154, 0.00208, 0.00314, 0.00662, 0.00483,
                   0.01458, -0.0009, -0.01684, -0.00961, -0.02261, 0.00594, -0.03192, -0.01095, -0.01214, -0.02288,
                   0.01026, -0.02392, 0.02392, -0.03469, -0.01781, 0.00942, -0.00795, -0.00964, -0.00797, 0.02483,
                   -0.0178, 0.0288, 0.01051, 0.00243, 0.00071, 0.01194, -0.00851, 0.0083, 0.03588, 0.03516, -0.01574,
                   -0.013, -0.01298, -0.00316, -0.02632, -0.00423, 0.01971, -0.01928, -0.01092, -0.07124, 0.01393,
                   -0.01649, 0.00148, 0.01188, -0.00513, 0.01808, -0.0062, -0.02378, 0.01467, -0.02851, 0.01144,
                   0.00158, -0.00592, -0.02538, -0.018, 0.0115, 0.00265, 0.00248, 0.01159, 0.00398, 0.00761, 0.00863,
                   0.01198, -0.01371, 0.0054, -0.00103, 0.0092, -0.02484, -0.02719, 0.00232, -0.03128, -0.01212,
                   0.02973, 0.00546, -0.03257, 0.004, 0.01437, -0.00172, -0.00636, -0.02122, -0.03637, -0.00722,
                   -0.0032, 0.00876, 0.00256, -0.01481, 0.00682, 0.00262, -0.00364, 0.01369, 0.01733, 0.01072, 0.00111,
                   -0.01271, -0.01729, 0.03565, 0.00096, -0.01495, -0.02213, 0.00609, -0.00323, 0.02037, -0.01386,
                   0.02908, 0.0105, -0.00983, 0.00891, 0.01382, -0.02965, -0.03221, 0.01167, -0.00044, 0.00341, 0.01879,
                   0.02527, -0.01682, 0.01387, -0.01262, 0.04371, -0.00183, -0.0106, -0.03689, 0.02426, 0.01922,
                   0.02766, 0.00106, 0.00253, -0.00451, 0.02554, -0.03115, 0.01799, 0.00228, 0.0033, 0.0145, 0.01927,
                   0.00404, -0.01488, 0.02334, 0.0296, 0.01384, 0.00389, 0.01824, -0.02025, -0.01213, 0.00407, -0.02544,
                   -0.02652, -0.00442, 0.00892, 0.02983, -0.01157, -0.00649, 0.02395, -0.02668, 0.00316, 0.00518,
                   -0.02427, -0.02295, 0.01516, 0.02325, 0.0109, 0.01002, 0.00016, 0.01055, 0.00515, 0.01014, -0.02924,
                   0.02156, 0.00861, 0.01747, -0.01272, -0.0095, -0.03322, -0.02019, -0.00013, 0.00404, 0.00233,
                   -0.00728, 0.1003, 0.03058, -0.0113, 0.01857, -0.01613, 0.01648, -0.01988, 0.00936, -0.02387,
                   -0.02706, 0.008, -0.01763, 0.00442, -0.01115, -0.00521, -0.00417, -0.01271, 0.01872, -0.0144,
                   -0.0075, 0.02763, -0.00612, 0.01595, 0.0032, -0.01102, -2e-05, 0.03355, 0.00587, -0.00473, -0.01289,
                   0.01168, 0.01249, -0.03198, -0.00263, 0.01968, -0.0412, 0.00315, -0.00703, 0.00543, 0.01661, 0.00487,
                   0.00082, -0.02929, -0.02784, 0.02562, 0.00352, -0.00873, -0.01983, 0.00504]

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
