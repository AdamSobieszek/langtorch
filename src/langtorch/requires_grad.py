import sys

import langtorch.semantic_algebra

sys.path.append("..")


import langtorch
from langtorch import TextTensor, String, Text, Activation
from langtorch import Markdown, XML, BackwardText
from langtorch.tt.functional import dropout
from langtorch import Session, ctx
from langtorch.autograd.textmask import TextMask
import numpy as np
from langtorch import TextModule
from langtorch import Chat, ChatML

from langtorch import TextTensor, TextModule, OpenAI, TextLoss, BinaryTextLoss
import torch

from langtorch import TextOptimizer, TextualGradientDescent
import re
import logging
import pandas as pd
import time

# Configure logging at the root level of logging
logging.basicConfig(level=logging.INFO)

# Specifically, for PyTorch, set the logging level
logger = logging.getLogger('torch')
logger.setLevel(logging.DEBUG)

T = torch.Tensor
TT = TextTensor

ctx.a = TT("a")

df = pd.read_csv("conf/bbh.csv", index_col=0)
train,val,test = TextTensor.from_df(df[df.split=="train"]),TextTensor.from_df(df[df.split=="val"]),TextTensor.from_df(df[df.split=="test"])
from torch.utils.data import DataLoader, TensorDataset

d = (lambda t: TensorDataset(t[:,0],t[:,1]))
train,val,test = d(train),d(val),d(test)

train_loader = DataLoader(train, batch_size=4)#, shuffle=True)

STARTING_SYSTEM_PROMPT = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
system_prompt = TextTensor({"system": STARTING_SYSTEM_PROMPT},
                            requires_grad=True)
loss_fn = BinaryTextLoss(activation=Activation("gpt-3.5-turbo"))
# optimizer = TextualGradientDescent([system_prompt])

# role_description = "structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task"
task = TextModule(system_prompt)
task = TextModule("Hey what")

import sys
sys.path.append("../src")
from langtorch import Text


# Test basic substitution
left = Text([("key", "Hello, {name}!")]).requires_grad_()
right = Text([("name", "Alice")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
result.grad_mask = [(0,0), (0,2)]
print(result.grad_mask, "span", BackwardText(result))

# Test multiple substitutions
left = Text([("key", "Hello, {name}! You are {age} years old.")])
right = Text([("name", "Bob"), ("age", "")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

print()
quit()
# Test nested substitution
left = Text([("", "Hello, {name}!"), ("info", "You are {age} years old.")])
right = Text([("name", "Charlie"), ("age", ["30"," or ", "40"])])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

# Test positional arguments
left = Text([("", "Hello, {}!"), ("", "Your age is {}.")])
right = Text([("", "David"), ("", "35")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

# Test nested positional arguments
left = Text([("", "Hello, {}!"), ("info", "Your age is {}.")])
right = Text([("", "Henry"), ("info", [("", "45")])])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

# Test duplicate right entries positional substitution
left = Text([("", "Hello, {}!"), ("", "You are {} years old."), ("", "You were born in {}.")])
right = Text([("", "Robert"), ("", "90")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

# Test wildcard
left = Text([("", "Hello, {name}!"), ("key", "*")])
right = Text([("name", "Eve"), ("greeting", "Good morning!")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

# Test key-value swap
left = Text([("greeting", "Hello"), ("name", "world")])
right = Text([("Hello", "greeting"), ("world", "name")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

# Test append unmatched items
left = Text([("key", "Hello, {name}!")])
right = Text([("name", "Frank"), ("age", "40")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)
print((1,)<(2,1))

# Test empty left text
left = Text([])
right = Text([("greeting", "Hello"), ("name", "Grace")])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

# Test empty right text
left = Text([("", "Hello, {name}!")])
right = Text([])
result = left * right
print("Left:", left.items(), "Right:", right.items(), "Result:", result.items())
print(result.grad_mask)

for steps, (input, target) in enumerate(train_loader):
    input = TextTensor("{b}{c:b}")
    answers = task(input).detach().requires_grad_()
    quit()