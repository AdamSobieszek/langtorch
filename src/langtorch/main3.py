

from datasets import load_dataset
import sys



import langtorch
from langtorch import TextTensor, String, Text, Activation
from langtorch import Markdown, XML
from langtorch.tt.functional import dropout
from langtorch import Session, ctx
import numpy as np
from langtorch import TextModule
from langtorch import Chat, ChatML

from langtorch import TextTensor, TextModule, OpenAI, TextLoss, BinaryTextLoss
import torch
from torch.utils.data import DataLoader, TensorDataset
from langtorch import TextOptimizer, TextualGradientDescent
import re
import os
import logging
import pandas as pd
import time

for i in os.listdir("D:/langtorchdocs/site/static")[:4]:
    print(",".join([m for m in os.listdir(f"D:/langtorchdocs/site/static/{i}") if ".png" in m]))

quit()

dataset = load_dataset("hellaswag", split={"train": "train[:1000]", "validation": "validation"})

# Load the dataset without downloading
dataset = load_dataset("hellaswag", trust_remote_code=True, streaming=True)

# Print one entry without downloading everything
print("Sample entry:")
for entry in dataset["train"].take(1):
    print(entry)
    break

# Print info about the dataset
print("\nDataset info:")
print(dataset)

# Specify local path for saving
local_path = "./wikipedia_sample.json"
os.makedirs(local_path, exist_ok=True)
import logging
logging.basicConfig(level=logging.DEBUG                                     )
# Download a maximum of 100 entries
print("\nDownloading up to 100 entries...")
map = {"ctx":"kontekst", "endings":"kontynuacje"}
subset = [dict(m) for m in dataset["train"].shuffle(seed=42).take(250)]
hella1k = [str({map[k]:v for k,v in m.items() if k in ['ctx', 'endings']}) for m in subset]

# Configure logging at the root level of logging
logging.basicConfig(level=logging.INFO)

# Specifically, for PyTorch, set the logging level
logger = logging.getLogger('torch')
logger.setLevel(logging.DEBUG)

T = torch.Tensor
TT = TextTensor

t1 = TT(hella1k, parse=False)
t1 = TT({"system":"Jesteś tłumaczem z angielskiego na polski. Przetłumacz strings napisane po angielsku i odpowiedz w formacie JSON"})+TT("{*:user}")*(
    TT("""Przetłumacz ten json. Składa się on z kontekstu i czterech możliwych kontynuacji. Upewnij się, że tłumaczenie każdej kontynuacji jest spójne z kontekstem.
Bądź tak precyzyjny, jak to tylko możliwe. Zachowaj te same klucze w json odpowiedzi:

{*}""")*t1)

model = Activation("gpt-4-turbo",T=0.)

os.environ["FIREWORKS_API_ KEY"] = "etCNLaffQOypeVjTyqA1gk75A9Wcd1VNq86jTiOn0Lt49uQq"


# res = ([(m[0]) for m in torch.load(r"D:\langtorch\src\langtorch\hellaswag.pt")['responses'].values()] )

dataset = TensorDataset(t1[:250])

train_loader = DataLoader(dataset, batch_size=25)#, shuffle=True)
with Session(path = "hellaswag.yaml"):
    res = []
    for i, t in enumerate(train_loader):
        print(25*i)
        res.extend([str(m) for m in model(t[0]).flat])
final = []
for i, (data, pred) in enumerate(zip(subset, res)):

    try:
        pred = eval(pred)
    except:
        continue
    mmap = {v:k for k,v in map.items()}
    if not all(k in pred for k in mmap):
        continue
    pred = {mmap[k]:v for k,v in pred.items()}
    final.append({**data}|pred)

print(len(final), final)
ctx.a = final
import json
print(final)
with open("hellaswagpl_250.json", "w") as f:
    json.dump({"train":final}, f)

quit()

