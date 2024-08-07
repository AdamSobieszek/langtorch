from tqdm import tqdm

import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


def eval_dataset(test_set, eval_fn, model, max_samples: int = None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):

            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


set_seed(args.seed)
llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
llm_api_test = tg.get_engine(engine_name=args.test_engine)
tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api_eval)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()

train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
print(STARTING_SYSTEM_PROMPT)

# Testing the 0-shot performance of the evaluation engine
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                            requires_grad=True,
                            role_description="system prompt to the language model")
model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

if not args.do_not_run_larger_model:
    reference = np.mean(eval_dataset(test_set, eval_fn, model_evaluation))

system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
model = tg.BlackboxLLM(llm_api_test, system_prompt)

optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

results = {"test_acc": [], "prompt": [], "validation_acc": []}
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())

for epoch in range(args.max_epochs):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        optimizer.zero_grad()
        losses = []
        for (x, y) in zip(batch_x, batch_y):
            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
            response = model(x)
            try:
                eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            except:
                eval_output_variable = eval_fn([x, y, response])
            losses.append(eval_output_variable)
        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()
        if args.run_validation:
            run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        print("sys prompt: ", system_prompt)
        test_acc = eval_dataset(test_set, eval_fn, model)
        results["test_acc"].append(test_acc)
        results["prompt"].append(system_prompt.get_value())
        if steps == 3:
            break

# Also dump the final results
import json

with open(f"./figures/results_{args.task}_{args.test_engine}.json", "w") as f:
    json.dump(results, f)


import sys

import langtorch.semantic_algebra

sys.path.append("..")


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
task = TextModule(system_prompt, activation=Activation("gpt-3.5-turbo", T=0))
optimizer = TextOptimizer(task.parameters())

with Session('test.yaml'):
        for steps, (input, target) in enumerate(train_loader):
            if steps == 0:
                continue
            answers = task(input).detach().requires_grad_()
        loss = loss_fn(answers, target)
        print(loss)

        loss.backward()

        # optimizer.step()
        # optimizer.zero_grad()
        print("grad:", answers.grad, task._prompt.grad)
        quit()
        # print(f"Batch {i}:")
        # print("Inputs:", inputs)
        # print("Predicted:", outputs)
        # print("Targets:", targets)
        # print("----")
