import os
import json
import requests
from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch


# Download and load dataset
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
data = download_and_load_file(file_path, url)
print(f"Number of entries: {len(data)}")


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f'\n\n### Instruction:\n{entry["instruction"]}'
    )
    input_text = f'\n\n### Input:\n{entry["input"]}' if entry["input"] else ""
    return instruction_text + input_text


def preprocess(entry):
    prompt = format_input(entry)
    response = f'\n\n### Response:\n{entry["output"]}'
    return {"text": prompt + response}


dataset = [preprocess(e) for e in data]

dataset = Dataset.from_list(dataset)
train_size = int(0.85 * len(dataset))
test_size = int(0.10 * len(dataset))
val_size = len(dataset) - train_size - test_size

dataset = dataset.train_test_split(
    train_size=train_size, test_size=test_size + val_size, seed=42
)
val_test = dataset["test"].train_test_split(
    train_size=val_size, test_size=test_size, seed=42
)
train_dataset = dataset["train"]
val_dataset = val_test["train"]
test_dataset = val_test["test"]

print(f"Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_dataset)}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

max_length = 1024


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


train_tokenized = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
val_tokenized = val_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
test_tokenized = test_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./gpt2-355m-instruct",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
)

trainer.train()

model.save_pretrained("./gpt2-355m-instruct")
tokenizer.save_pretrained("./gpt2-355m-instruct")

print("Training complete. Model saved to ./gpt2-355m-instruct")
