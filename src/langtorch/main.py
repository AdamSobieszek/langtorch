import sys

import langtorch.semantic_algebra

sys.path.append("..")
import langtorch
from langtorch import TextTensor, String, Text
from langtorch.tt.functional import dropout
from langtorch import Session
import numpy as np
import torch
from langtorch import TextModule
from langtorch import Chat, ChatML


class ChatML(Chat):
    language = 'chatml'
    allowed_keys = ["user", "assistant", "system"]

    def __str__(self):
        formatted = "\n".join(f"<|im_start|>{k}\n{Text(v)}<|im_end|>" for k, v in self.items())
        return formatted if self.keys()[-1] == "assistant" else formatted + "\n<|im_start|>assistant\n"


chat_template = Chat(
    ("system", "Hello!"),
    ("assistant", "Hi! How can I help you today?"),
    ("user", "Tell me about {thing}."),
    ("system", "Hello!"),
    ("assistant", "Hi! How can I help you today?"),
    ("user", "Tell me about {thing}.")
)
# Use iloc to make sure you format the right message
input = [1,1,1,1,1,1,1,1,1,1,1]
for i, m in enumerate([chat_template]):
    if "system" in m.keys():
        _ = m.loc["system"].values()[-1]
        print("sys",_)
        m = m.loc[["user", "assistant"]]
    if np.all([key in ["user", "assistant"] for key in m.keys()]):
        input[i] = [(k, str(Text(v, parse=False))) for k,v in m.items()]
        print('>', input[i] )
    elif "user" not in m.keys() or "assistant" not in m.keys():
        # NOT STRICT
        input[i] = [("user", str(m))]
    else:
        raise ValueError(
            f"Invalid input to OpenAI Chat. Ambigous input: some but not all items in TextTensor entries have keys 'user' or 'assistant'. Change the input into a valid chat, or the input was a single user message remove these keys or use entry.add_key_('user'): \nentry.items()=={m.items()}")

quit()
# In this case there are no other "thing" values so we could've also used
chat_template *= {"thing": "the weather"}
from langtorch import OpenAI

ensemble_llm = OpenAI("gpt-3.5-turbo",
                      system_message="You are a rewriting bot that answers only with the revised text",
                      T=1.1,  # High temperature to sample diverse completions
                      n=5)  # 5 completions for each entry
paragraphs = Text.from_file("./conf/paper.md").loc["Para"].to_tensor()[:15]

rewrite = TextModule(["Compress all information from the paragraph into an entity-dense telegraphic summary: "],
                     activation=ensemble_llm)
ensemble_summaries = rewrite(paragraphs)
print(ensemble_summaries.shape)
# Outputs (15, 5)
combined_summaries = langtorch.mean(ensemble_summaries, dim=-1)
print(combined_summaries.shape)
# Outputs: (15)

# We can even average again across paragraphs!
summary = langtorch.mean(combined_summaries, dim=-1)
print(summary)
# Outputs:
# Outputs (15)

quit()

import torch

tensor1 = TextTensor([["Yes"],
                      ["No"]])
tensor2 = TextTensor(["Yeah", "Nope", "Yup", "Non"])

torch.cosine_similarity(tensor1, tensor2)
print()

quit()


def modularize(func):
    def forward(tt):
        print(tt)
        assert isinstance(tt, TextTensor)
        result = np.array(func(tt.content.tolist()), dtype=object)
        if len(result) == tt.content.size:
            tt.content = result.reshape(tt.content.shape)
        elif (len(result) / tt.content.size) % 1 < 0.00001:
            tt.content = result.reshape(list(tt.shape) + [len(result) // tt.size])
        else:
            tt.content = result
        return tt

    result = torch.nn.Module()
    result.forward = forward
    return result


def translation_tool(eng_text: str, dutch_text: str, french_text: str) -> str:
    """
    Translates the input text from English to Dutch and French.
    """
    import yaml
    print(yaml.dump({"eng_text": eng_text, "dutch_text": dutch_text, "french_text": french_text}))


tool_activation = Activation(tools=[translation_tool])
# print(tool_activation(TextTensor("translate this text to all 3 languages, use the tool provided:\n'vertaal deze tekst'")))


x = TextTensor([1, 2, 3, 4, 5, 6, 7, 8, 9], requires_grad=True).reshape(3, 3)
print(langtorch.semantic_algebra.mean(x, dim=0), langtorch.semantic_algebra.mean(x, dim=1))
x2 = x + "\nnumber"
print(x)
print("Tensor stack\n", torch.stack([x, x]))
print("Dropout", langtorch.tt.functional.dropout(x))

act = Activation()
# print(modularize(lambda xx: [f"_{x}_" for x in xx])(TextTensor([1,2,3,4,5,6,7,8,9]).reshape(3,3)))
chats = TextTensor([["a", "b", "c"], ["d", "e", "f"]], key="user")

from langtorch.tensors.chattensor import AIMessage

chats = chats + AIMessage("g")
# emb =chats.embed()
# print(torch.cosine_similarity(chats, chats))
# print(torch.cosine_similarity())


chain = torch.nn.Sequential(
    TextModule("Calculate this equation:\n"),
    langtorch.methods.CoT,
    Activation(),
    TextModule("Is this reasoning correct?\n"),
    Activation(T=0.)
)

# p = Text(("greeting","Hello, world!")).add_key_("prompt")
# # Example usage:
# input = (User([f"Is {word} positive?" for word in ["love", "chair", "non-negative"]]) * Assistant(
#     ["Yes", "No", "Yes"])).requires_grad_()
# target = TextTensor(['Yes', "No", "No"]).requires_grad_()  # Dummy tensors-like object with .content attribute
# print(torch.mean(input, dim = 0))


# print(torch.broadcast_tensors(input, target.unsqueeze(0)))

# loss_fn = TextLoss(prompt="")  # Create the loss function instance
# loss = loss_fn(input, target)  # Compute the loss
# loss.backward()  # Backpropagate the loss


from torch.utils.data import DataLoader, TensorDataset

# Define the dataset
input_data = (User([f"Is {word} positive?" for word in ["love", "chair", "non-negative"]]) * Assistant(
    ["Yes", "No", "Yes"])).requires_grad_()
target_data = TextTensor(['Yes', "No", "No"]).requires_grad_()

# Wrap the data in a TensorDataset and then create a DataLoader
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=1)  # Adjust the batch size as needed

# Define your TextModule
text_module = TextModule("{*}")  # Initialize your TextModule here

# Loop over the DataLoader
for i, (inputs, targets) in enumerate(dataloader):
    # Pass the batch through the TextModule
    outputs = text_module(inputs)

    # Compare outputs to targets and save the results
    # Here, you can define your own comparison logic
    # For example, you can save the results in a list or write them to a file
    print(f"Batch {i}:")
    print("Inputs:", inputs)
    print("Predicted:", outputs)
    print("Targets:", targets)
    print("----")
