
# <img src="langtorch_w_background.png" width="60" height="60" alt="LangTorch Logo" style="vertical-align: middle;"> LangTorch

[![Release Notes](https://img.shields.io/github/release/AdamSobieszek/langtorch)](https://github.com/AdamSobieszek/langtorch/releases)
[![Downloads](https://static.pepy.tech/badge/AdamSobieszek/langtorch)](https://pepy.tech/project/langtorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/AdamSobieszek.svg?style=social&label=Follow%20%40AdamSobieszek)](https://twitter.com/AdamSobieszek)
[![GitHub star chart](https://img.shields.io/github/stars/AdamSobieszek/langtorch?style=social)](https://star-history.com/#AdamSobieszek/langtorch)

[//]: # ([![]&#40;https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat&#41;]&#40;https://discord.gg/6adMQxSpJS&#41;)


LangTorch is a Python package designed to simplify the development of LLM applications by leveraging familiar PyTorch concepts.

## Installation

```bash
pip install langtorch
```

## Overview

LangTorch provides a structured approach to LLM applications, offering:

- **TextTensors**: A unified way to handle prompt templates, completion dictionaries, and chat histories.
- **TextModules**: Building blocks, derived from torch.nn.Module, specifically tailored for text operations and LLM calls both locally and via an API.
- other things that are also better than langchain
## Examples

### TextTensors

Creating and manipulating textual data as tensors:

```python
template = TextTensor([["Explain {theory} in terms of {framework}"],  
                       ["Argue how {framework} can prove {theory}"]])  

result = template * TextTensor({"theory": "active inference", "framework": "thermodynamics" })

print(result)
# Outputs: [[Explain active inference in terms of thermodynamics]
#           [Argue how thermodynamics can prove active inference]]
```

### TextModules

Building sequences of operations on text data:

```python
chain = torch.nn.Sequential(
    TextModule("Calculate this equation: {}"),
    langtorch.methods.CoT,
    GPT4
    TextModule("Is this reasoning correct? {}", activation = GPT4)
)

output = chain(TextTensor(["170*32 =", "4*20 =", "123*45/10 =", "2**10*5 ="]))
```

### Cosine Similarities

Compute similarities between entries:

```python
from langtorch.tt import CosineSimilarity

cos = CosineSimilarity()
similarities = cos(TextTensor([["Yes"], ["No"]]), TextTensor(["1", "0", "Noo", "Yees"]))
```

## Contribute

Your feedback and contributions are valued. Feel free to check out our [contribution guidelines](#).

## License

LangTorch is MIT licensed. See the [LICENSE](#) file for details.
