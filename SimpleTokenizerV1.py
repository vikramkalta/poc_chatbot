from test import vocab
import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


# tokenizer = SimpleTokenizerV1(vocab)
# text = """
# "It's the last he painted, you know,"
# Mrs. Gisburn said with pardonable price.
# """
# ids = tokenizer.encode(text)
# # print(ids)
# decoded_text = tokenizer.decode(ids)
# print(decoded_text)

# tokenizer = SimpleTokenizerV1(vocab)
# text = "Hello, do you like tea. Is this-- a test?"
# x = tokenizer.encode(text)
# print(x)

tokenizer = SimpleTokenizerV1(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|> ".join((text1, text2))
# x = tokenizer.encode(text)
# print(text)

x = tokenizer.encode(text)
# print('x', x)
# y = tokenizer.decode(x)
# print('y', y)

import importlib
import tiktoken

# print("tiktoken version:", importlib.metadata.version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# strings = tokenizer.decode(integers)
# print(strings)

with open("the-verdict.txt", "r") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

enc_sample = enc_text[50:]
# print(enc_sample)

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
# print(f"x: {x}")
# print(f"y:    {y}")

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]

#     print(context, "---->", desired)

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    # print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

import torch

print("PyTorch version:", torch.__version__)
