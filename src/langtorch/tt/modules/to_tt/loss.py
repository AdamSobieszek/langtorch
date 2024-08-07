# Work in progress, not fully functional
import torch
from torch.nn.modules.loss import _Reduction

import langtorch
from langtorch import ctx
from .activation import Activation
from .textmodule import TextModule
from langtorch.tensors import TextTensor

session = ctx


class _TextLoss(TextModule):
    reduction: str = 'sum'

    def __init__(self, prompt: TextTensor, activation=None, backward_prompt=None, key="loss", reduction: str = 'sum', **kwargs) -> None:
        super(_TextLoss, self).__init__(prompt=prompt, activation=activation, key=key, backward_prompt=backward_prompt, is_param=False, **kwargs)
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


# class CompareAnswersLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input: TextTensor, target: TextTensor, prompt: TextTensor):
#         ctx.save_for_backward(input, target)
#         assert input.shape == target.shape, f"Input and target must have the same shape. Got {input.shape} and {target.shape} instead."
#         loss_query = prompt * input.add_key_("input") * target.add_key_('target')
#         loss = Activation(session.default_model_for_functions)(loss_query)
#         return loss
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, target = ctx.saved_tensors
#         grad_input = grad_target = None
#
#         if ctx.needs_input_grad[0]:
#             # Compute gradient for input
#             grad_input = grad_output
#
#         if ctx.needs_input_grad[1]:
#             # Compute gradient for target
#             grad_target = -grad_input
#
#         # The gradients for non-tensors arguments must be None.
#         return grad_input, grad_target, None


class TextLoss(_TextLoss):
    key: str = "loss"

# class DropoutCorrectAnswer(TextModule):
#     def forward(self):
#         super().forward
"""You will give feedback to a variable with the following role: 
<INPUT> </INPUT>

<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>
We are interested in giving feedback to the response from the language model for this conversation. Specifically, give feedback to the following span of text:
<REQUIRES_GRAD>  </REQUIRES_GRAD>
Given the above history, describe how the response from the language model could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent."""
class BinaryTextLoss(_TextLoss):
    def __init__(self, prompt: TextTensor = "Is the provided answer correct?\nAnswer: {input}\nGround truth answer: {target}", backward_prompt = "{input}\nProvide a short description how the answers differ that could help revise the answer.", activation=None, key="loss", reduction: str = 'none', **kwargs):
        super(BinaryTextLoss, self).__init__(prompt=prompt, activation=activation, key=key, reduction=reduction, backward_prompt=backward_prompt)
        assert int(self._prompt.content.size) == 1, f"Loss prompt should be a single entry TextTensor to assign losses to inputs 1-to-1.\nPrompt: {self._prompt}"
        assert ("input" in self._prompt.item().values()) and  ("target" in self._prompt.item().values()), "Loss prompt should have {input} and {target} entries to compare"

        if self.activation:
            if self.activation.system_message is not None and not "yes or no" in str(self.activation.system_message).lower():
                print("BinaryTextLoss needs only Yes or No outputs, changing the activation system message to request that")
            self.activation.system_message = "Answer only Yes or No"
            self.register_forward_hook(self.to_01_hook)
            self.activation.max_tokens = 1
            self.activation.gradient_mask = None
            self.activation.temperature = 0
            self.activation.cache = True

    def forward(self, input: TextTensor, target: TextTensor):
        # Don't call LLM for perfect matches
        self.activation.forward_mask = TextTensor(input)==TextTensor(target)
        with langtorch.skip_mul_grad():
            loss = super().forward(TextTensor(input).add_key("input") + TextTensor(target).add_key("target"))
        return loss

    @staticmethod
    def to_01_hook(module, input, output):
        answers = [str(m).lower()[:3] for m in output.flat]
        yes_no_dict = {
            'yes': 1, 'y': 1, 'tru': 1, '1': 1, 'nan':1,
            'no': 0, 'n': 0, 'fal': 0, '0': 0
        }
        if any(ans not in yes_no_dict for ans in answers):
            print(f"Error! Invalid answer in Binary Loss: {answers} \nExpected: Yes or No")

        module.activation.gradient_mask = torch.Tensor([not bool(yes_no_dict.get(ans, 0)) for ans in answers]).reshape(
            output.shape) != True


