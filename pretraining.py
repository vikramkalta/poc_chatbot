# from importlib.metadata import version
# from tracemalloc import start

# pkgs = ["matplotlib",
# "numpy",
# "tiktoken",
# "torch",
# "tensorflow" # For OpenAI's pretrained weights
# ]

# for pkg in pkgs:
#   print(f"{pkg}: {version(pkg)}")
      

from cProfile import label
import torch
import tiktoken
from torch.cpu import is_available
from GPTDatasetV1 import create_dataloader_v1
from gpt_2 import GPTModel, generate_text_simple
# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference

def text_to_token_ids(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
  return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
  flat = token_ids.squeeze(0) # remove batch dimension
  return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
  model=model,
  idx=text_to_token_ids(start_context, tokenizer),
  max_new_tokens=10,
  context_size=GPT_CONFIG_124M["context_length"]
)

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

with torch.no_grad():
  logits = model(inputs)

probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
# print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)

# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# print('probas', probas)
text_idx = 0
# print('targets[text_idx]', targets[text_idx])
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
# print("Text 1:", target_probas_1)

text_idx = 1
# print('targets[text_idx]', targets[text_idx])
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
# print("Text 2:", target_probas_2)

# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print("Log probabilities:", log_probas)

# Calculate the average probability for each token
avg_log_probas = torch.mean(log_probas)
# print("Average log probabilities:", avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
# print("Negative average log probabilities:", neg_avg_log_probas)

# Logits have shape (batch_size, num_tokens, vocab_size)
# print("Logits shape:", logits.shape)

# Targets have shape (batch_size, num_tokens)
# print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()

# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print("Loss:", loss)

perplexity = torch.exp(loss)
# print("Perplexity:", perplexity)


import os
import requests

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
  response = requests.get(url, timeout=30)
  response.raise_for_status()
  text_data = response.text
  with open(file_path, "w", encoding="utf-8") as file:
      file.write(text_data)
else:
  with open(file_path, "r", encoding="utf-8") as file:
      text_data = file.read()

# First 99 characters
print(text_data[:99])
# Last 99 characters
print(text_data[-99:])

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

# print("Characters:", total_characters)
# print("Tokens:", total_tokens)

# Train/validation ratio
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
  train_data,
  batch_size=2,
  max_length=GPT_CONFIG_124M["context_length"],
  stride=GPT_CONFIG_124M["context_length"],
  drop_last=True,
  shuffle=True,
  num_workers=0
)

val_loader = create_dataloader_v1(
  val_data,
  batch_size=2,
  max_length=GPT_CONFIG_124M["context_length"],
  stride=GPT_CONFIG_124M["context_length"],
  drop_last=True,
  shuffle=False,
  num_workers=0
)

# Sanity check

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
  print("Not enough tokens for the training loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
  print("Not enough tokens for the validation loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "decrease the `training_ratio`")

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

# print("Training tokens:", train_tokens)
# print("Validation tokens:", val_tokens)
# print("All tokens:", train_tokens + val_tokens)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
    
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
      # Reduce the number of batches to match the total number of batches in the data loader
      # if num_batches exceeds the number of batches in the data loader
      num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
      if i < num_batches:
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
      else:
        break
    return total_loss / num_batches

if torch.cuda.is_available():
  device = torch.device("cuda")
elif torch.backends.mps.is_available():
  # Use PyTorch 2.9 or newer for stable mps results
  major, minor = map(int, torch.__version__.split('.')[:2])
  if (major, minor) >= (2, 9):
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
else:
  device = torch.device("cpu")

print(f"Using {device} device.")


model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
  train_loss = calc_loss_loader(train_loader, model, device)
  val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                      eval_freq, eval_iter, start_context, tokenizer):
  # Initialize lists to track losses and tokens seen
  train_losses, val_losses, track_tokens_seen = [], [], []
  tokens_seen, global_step = 0, -1

  # Main training loop
  for epoch in range(num_epochs):
    model.train() # Set model to training mode

    for input_batch, target_batch in train_loader:
      optimizer.zero_grad() # Reset loss gradients from previous batch iteration
      loss = calc_loss_batch(input_batch, target_batch, model, device)
      loss.backward() # Calculate loss gradients
      optimizer.step() # Update model weights using loss gradients
      tokens_seen += input_batch.numel()
      global_step += 1

      # Optional evaluation step
      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(
          model, train_loader, val_loader, device, eval_iter
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
              f"Train loss: {train_loss:.4f}, "
              f"Val loss: {val_loss:.4f}")
      
    # Print a sample text after each epoch
    generate_and_print_sample(
      model, tokenizer, device, start_context
    )
  
  return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
  model.eval()
  with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
  model.train()
  return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer).to(device)
  with torch.no_grad():
    token_ids = generate_text_simple(
      model=model, idx=encoded,
      max_new_tokens=50, context_size=context_size
    )
  decoded_text = token_ids_to_text(token_ids, tokenizer)
  print(decoded_text.replace("\n", " ")) # Compact print format
  model.train()

# Note:
# Uncomment the following code to calculate the execution time
import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 1
# train_losses, val_losses, tokens_seen = train_model_simple(
#   model, train_loader, val_loader, optimizer, device,
#   num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#   start_context="Every effort moves you", tokenizer=tokenizer
# )

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes")
  
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
  fig, ax1 = plt.subplots(figsize=(5, 3))
  
  # Plot training and validation loss against epochs
  ax1.plot(epochs_seen, train_losses, label="Train loss")
  ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Loss")
  ax1.legend(loc="upper right")
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # only show integer labels on x-axis

  # Create a second x-axis for tokens seen
  ax2 = ax1.twiny() # Create a second x-axis that shares the same y-axis
  ax2.plot(tokens_seen, train_losses, alpha=0) # Invisible plot for aligning ticks
  ax2.set_xlabel("Tokens seen")

  fig.tight_layout() # Adjust layout to make room
  plt.savefig("loss-plot.pdf")
  plt.show()

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# NEW: use CPU here as inference is cheap with
# this model and to ensure readers get same results in the
# remaining sections of this book
inference_device = torch.device("cpu")

# model.to(inference_device)
# model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
  model=model,
  idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
  max_new_tokens=25,
  context_size=GPT_CONFIG_124M["context_length"]
)

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

vocab = {
  "closer": 0,
  "every": 1,
  "effort": 2, 
  "forward": 3,
  "inches": 4,
  "moves": 5, 
  "pizza": 6,
  "toward": 7,
  "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
  [4.51, 0.89, -1.9, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()

# The next generated token is then as follows:
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
  torch.manual_seed(123)
  sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
  sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
  for i, freq in enumerate(sampled_ids):
    print(f"{freq} x {inverse_vocab[i]}")
    
print_sampled_tokens(probas)

def softmax_with_temperature(logits, temperature):
  scaled_logits = logits / temperature
  return torch.softmax(scaled_logits, dim=0)

# Temperature values
temperatures = [1,0.1,5] # Original, higher confidence, and lower confidence

# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, t) for t in temperatures]

# Plotting
x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
  rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f"Temp={T}")  

# ax.set_ylabel('Probability')
# ax.set_xlabel('Token')
# ax.set_title('Scaled Probabilities for Next Token')
# ax.legend()

# plt.tight_layout()
# plt.savefig("temp-plot.pdf")
# plt.show()

# print_sampled_tokens(scaled_probas[1])
# print_sampled_tokens(scaled_probas[2])

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)

# print("Top logits:", top_logits)
# print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")), 
    other=next_token_logits
)

print(new_logits)

def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):

  # For-loop is the same as before: Get logits, and only focus on last time step
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)
    logits = logits[:, -1, :]
    
    # New: Filter logits with top_k sampling
    if top_k is not None:
      # Keep only top_k values
      top_logits, _ = torch.topk(logits, top_k)
      min_val = top_logits[:, -1]
      logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

    # New: Apply temp scaling
    if temperature > 0.0:
      logits = logits / temperature

      # New (not in book): numerical stability tip to get equivalent results on map device
      # subtract rowwise max before softmax
      logits = logits - logits.max(dim=-1, keepdim=True).values

      # Apply softmax to get probabilities
      probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

      # Sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

    # Otherwise same as before: get idx of the vocab entry with the highest logits value
    else:
      idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

    if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
      break

    # Same as before: append sampled index to the running sequence
    idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
  
  return idx


torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# NEW
torch.save(model.state_dict(), "model.pth")

model = GPTModel(GPT_CONFIG_124M)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Use PyTorch 2.9 or newer for stable mps results
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
else:
    device = torch.device("cpu")

# print("Device:", device)

model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

torch.save({
  "model_state_dict": model.state_dict(),
  "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")

checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
