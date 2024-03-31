import torch.nn
from torch import Tensor

from langtorch.tt.modules.to_tt.textmodule import TextModule
from langtorch.tensors import TextTensor

__all__ = ['CosineSimilarity']  # 'PairwiseDistance',


class CosineSimilarity(TextModule):
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same number of dimensions as x1, matching x1 size at dimension `dim`,
              and broadcastable with x1 at other dimensions.
        - Output: :math:`(\ast_1, \ast_2)`
    Examples::
        >>> cos = tt.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(text1, text2)
    """
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = -1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, t1: TextTensor, t2: TextTensor) -> Tensor:
        return torch.cosine_similarity(t1, t2, self.dim, self.eps)
