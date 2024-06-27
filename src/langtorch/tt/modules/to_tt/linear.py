import torch

from langtorch.tensors import TextTensor
from .textmodule import TextModule


class Linear(TextModule):
    r"""Applies a matmul to the incoming TextTensor: :math:`y = A @ x.T + b`


    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        prompt: the TextTensor to format of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the "bias", Text that is added of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`


    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: TextTensor

    def forward(self, input) -> TextTensor:
        return self._prompt @ input
