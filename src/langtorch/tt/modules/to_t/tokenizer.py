import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langtorch import TextTensor
from .. import TextModule


class Tokenizer(torch.nn.Module):
    input_class = TextTensor
    output_class = torch.Tensor

    def __init__(self, tokenizer, tokenizer_kwargs=dict()):
        super(Tokenizer, self).__init__()
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        tokenizer_default_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        # Update tokenizer_kwargs with default values if the value isn't overriden
        tokenizer_kwargs = {**tokenizer_default_kwargs, **tokenizer_kwargs}
        self.tokenizer_kwargs = tokenizer_kwargs

    def forward(self, input_texttensor: TextTensor, skip_special_tokens=True, **gen_kwargs) -> TextTensor:
        assert isinstance(input_texttensor, TextTensor)
        # Convert TextTensor to a list of strings
        texts = [str(entry) for entry in input_texttensor.flat]

        # Tokenize and generate responses
        inputs = self.tokenizer(texts, **self.tokenizer_kwargs).to(self.model.device)

        return output if self.key is None else output.set_key(self.key)
