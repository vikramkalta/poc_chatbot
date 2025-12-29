from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

import os
import requests

if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

# The book originally used the following code below
# However, urllib uses older protocol settings that
# can cause problems for some readers using a VPN.
# The `requests` version above is more robust
# in that regard.

# import sys
# print(sys.__version__)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print("Total number of character: ", len(raw_text))
# print("First 99 characters: ", raw_text[:99])

import re

# text = "Hello, world. This, is a test."
# result = re.split(r'(\s)', text)
# print(result)

# result = re.split(r'([,.]|\s)', text)
# print(result)

# text = "Hello, world. Is this-- a test?"
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# result = [item.strip() for item in result if item.strip()]
# print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
# print("Vocabulary size: ", vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

# for i, item in enumerate(vocab.items()):
#     # print(item)
#     if i >= 50:
#         break

# all_tokens = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_words)}

print(len(vocab.items()))

