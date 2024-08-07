import torch
from torch.utils._contextlib import _DecoratorContextManager
import functools
from langtorch import ctx

def is_skip_mul_grad_enabled():
    return ctx._SKIP_MUL_GRAD

class skip_mul_grad(_DecoratorContextManager):
    """Context-manager that performs operations in forward pass but skips them in backward."""

    def __init__(self):
        super().__init__()

    def __enter__(self):
        self.prev = ctx._SKIP_MUL_GRAD
        ctx._SKIP_MUL_GRAD = True

    def __exit__(self, exc_type, exc_value, traceback):
        ctx._SKIP_MUL_GRAD = self.prev

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

# Example of how to use is_identity_grad_enabled() in a custom Function
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        result = input * 2  # Example operation
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if is_skip_mul_grad_enabled():
            return grad_output
        else:
            input, = ctx.saved_tensors
            return grad_output * 2  # Normal gradient computation
