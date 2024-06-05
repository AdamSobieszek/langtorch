import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langtorch import TextTensor


class TransformersCausalLM(torch.nn.Module):
    def __init__(self, model, tokenizer=None, key="assistant", tokenizer_kwargs=dict(), model_kwargs=dict(), **kwargs):
        super(TransformersCausalLM, self).__init__()
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            if tokenizer is None:
                tokenizer = model
        else:
            self.model = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.key = key
        tokenizer_default_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        # Update tokenizer_kwargs with default values if the value isn't overriden
        tokenizer_kwargs = {**tokenizer_default_kwargs, **tokenizer_kwargs}
        self.tokenizer_kwargs = tokenizer_kwargs
        self.model_kwargs = model_kwargs

    def forward(self, input_texttensor: TextTensor, skip_special_tokens=True, **gen_kwargs) -> TextTensor:
        input_texttensor = TextTensor(input_texttensor, tokenizer = self.tokenizer, tokenizer_kwargs=self.tokenizer_kwargs)

        # Tokenize and generate responses
        inputs = input_texttensor.tokenize().to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.model_kwargs, **gen_kwargs)

        # Decode responses and convert to TextTensor
        decoded_texts = [self.tokenizer.decode(output, skip_special_tokens=skip_special_tokens) for output in outputs]
        output = TextTensor([str(text) for text in decoded_texts], parse=False).reshape(input_texttensor.shape)

        return output if self.key is None else output.set_key(self.key)
