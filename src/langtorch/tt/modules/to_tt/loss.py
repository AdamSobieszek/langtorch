# Work in progress, not fully functional
import torch
from torch.nn.modules.loss import _Reduction

from langtorch import ctx
from .activation import Activation
from .textmodule import TextModule
from langtorch.tensors import TextTensor

session = ctx


class _TextLoss(TextModule):
    reduction: str = 'sum'

    def __init__(self, prompt: TextTensor, activation=None, key="loss", reduction: str = 'sum') -> None:
        super(_TextLoss, self).__init__(prompt=prompt, activation=activation, key=key)
        self.reduction = reduction
        self.register_forward_hook(self.reduction_hook)

    @staticmethod
    def reduction_hook(module, input, loss):
        if module.reduction == 'none':
            return loss
        elif module.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {loss.reduction}")


class CompareAnswersLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: TextTensor, target: TextTensor, prompt: TextTensor):
        ctx.save_for_backward(input, target)
        assert input.shape == target.shape, f"Input and target must have the same shape. Got {input.shape} and {target.shape} instead."
        loss_query = prompt * input.add_key_("input") * target.add_key_('target')
        loss = Activation(session.default_model_for_functions)(loss_query)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        grad_input = grad_target = None

        if ctx.needs_input_grad[0]:
            # Compute gradient for input
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            # Compute gradient for target
            grad_target = -grad_input

        # The gradients for non-tensors arguments must be None.
        return grad_input, grad_target, None


class TextLoss(_TextLoss):
    key: str = "loss"


class BinaryTextLoss(_TextLoss):
    def __init__(self, prompt: TextTensor, activation=None, key="loss", reduction: str = 'none'):
        super(BinaryTextLoss, self).__init__(prompt=prompt, activation=activation, key=key, reduction=reduction)

    def forward(self, input: TextTensor, target: TextTensor):
        loss = super().forward(TextTensor(input).add_key_("input") + TextTensor(target).add_key_("target"))
        return loss

