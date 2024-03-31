import numpy as np
import torch

from langtorch.api.call import get_embedding
from langtorch.tensors import TextTensor


class EmbeddingModule(torch.nn.Module):
    input_class = TextTensor
    output_class = torch.Tensor

    def __init__(self, model="text-embedding-3-small"):
        super(EmbeddingModule, self).__init__()
        self.model = model

    def forward(self, x: TextTensor) -> torch.Tensor:
        return x.embed(self.model).embedding
