import argparse
import json
import os
import re
import torch
import tiktoken

from gpt_2 import GPTModel
from pretrained_weight import generate, text_to_token_ids, token_ids_to_text


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")
    return torch.device("cpu")


def format_input(entry_instruction: str, entry_input: str = "") -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry_instruction}"
    )
    input_text = f"\n\n### Input:\n{entry_input}" if entry_input else ""
    return instruction_text + input_text


BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def infer(
    weights_path: str,
    model_choice: str,
    instruction: str,
    input_text_field: str,
    max_new_tokens: int,
    return_response: bool = False,
):
    device = select_device()
    tokenizer = tiktoken.get_encoding("gpt2")

    config = dict(BASE_CONFIG)
    config.update(MODEL_CONFIGS[model_choice])

    model = GPTModel(config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    prompt = format_input(instruction, input_text_field)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(prompt) :].replace("### Response:", "").strip()
    if return_response:
        return response_text
    print("Prompt:\n" + prompt)
    print("\nModel response:\n>> " + response_text)


def infer_from_dataset(
    weights_path: str, model_choice: str, dataset_path: str, k: int, max_new_tokens: int
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    device = select_device()
    tokenizer = tiktoken.get_encoding("gpt2")

    config = dict(BASE_CONFIG)
    config.update(MODEL_CONFIGS[model_choice])

    model = GPTModel(config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for i, entry in enumerate(data[:k]):
        prompt = format_input(entry.get("instruction", ""), entry.get("input", ""))
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=max_new_tokens,
            context_size=config["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(prompt) :].replace("### Response:", "").strip()
        )
        print("-------------------------------------")
        print(f"Example {i+1}")
        print(prompt)
        print(f"\nCorrect response:\n>> {entry.get('output', '')}")
        print(f"\nModel response:\n>> {response_text}")


def default_weights_name(model_choice: str) -> str:
    # Matches chapter7.py naming: remove spaces and parentheses
    base = re.sub(r"[ ()]", "", model_choice)
    return f"{base}-sft_borga.pth"


def main():
    parser = argparse.ArgumentParser(description="Test inference for SFT GPT-2 model")
    parser.add_argument(
        "--model_choice",
        type=str,
        # default="gpt2-small (124M)",
        default="gpt2-medium (355M)",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size to instantiate (must match weights)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to fine-tuned weights .pth file. Defaults to <model_choice>-sft.pth naming.",
    )
    # parser.add_argument("--instruction", type=str, default="Explain the moon phases in simple terms.")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Provide guidance for the following user query.",
    )
    # parser.add_argument("--instruction", type=str, default="")
    # parser.add_argument("--instruction", type=str, default="Rewrite the sentence using a simile.")
    # parser.add_argument("--input", dest="input_field", type=str, default="Uploading a document for verification")
    parser.add_argument(
        "--input",
        dest="input_field",
        type=str,
        default="my transfer is still showing pending",
    )
    # parser.add_argument("--input", dest="input_field", type=str, default="The car is very fast.")
    # parser.add_argument("--input", dest="input_field", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to instruction-data.json to sample from",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of dataset examples to run if --dataset is provided",
    )

    args = parser.parse_args()

    weights_path = args.weights or default_weights_name(args.model_choice)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if args.dataset:
        infer_from_dataset(
            weights_path, args.model_choice, args.dataset, args.k, args.max_new_tokens
        )
    else:
        infer(
            weights_path,
            args.model_choice,
            args.instruction,
            args.input_field,
            args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
