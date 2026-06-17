import argparse
import json
import math
import sys
from pathlib import Path
from typing import Tuple

from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.append(".")

from src.evaluation.templates import CLASSIFIER_PROMPT_TEMPLATE


def apply_chat_template(tokenizer, messages: list[list[dict]]) -> list[str]:
    if tokenizer.chat_template:
        return [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages
        ]
    # Fallback: concatenate message contents as plain strings
    return ["\n".join(msg["content"] for msg in msgs) for msgs in messages]


def generate_predictions(
    llm,
    tokenizer,
    data,
    max_new_tokens=1024,
    batch_size: int = 2000,
    max_input_tokens: int = 31000,
):
    predictions = []
    num_batches = math.ceil(len(data) / batch_size)
    for i in tqdm(range(num_batches), desc="Processing"):
        prompts = data[i * batch_size : (i + 1) * batch_size]
        
        token_ids_list = tokenizer(prompts, add_special_tokens=False).input_ids
        truncated_ids = [
            (ids[: max_input_tokens // 2] + ids[-max_input_tokens // 2 :]) 
            if len(ids) > max_input_tokens else ids
            for ids in token_ids_list
        ]
        prompts = tokenizer.batch_decode(truncated_ids, skip_special_tokens=False)

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        decoded_output = [output.outputs[0].text.strip() for output in outputs]
        predictions += decoded_output
    return predictions


def _parse_label(value) -> float:
    """Convert a label value to float, handling non-numeric strings."""
    try:
        return float(value)
    except (ValueError, TypeError):
        # Handle common string labels
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("harmful", "unsafe", "yes", "true"):
                return 1.0
            if lower in ("safe", "harmless", "no", "false"):
                return 0.0
        return 0.0


def load_thinking_data(
    dataset_path: Path,
    responses_path: Path,
) -> Tuple[
    list[list[dict]],  # cot_messages
    list[list[dict]],  # response_messages
    list[float],  # labels
]:
    """Load dataset and responses once, returning CoT messages, response messages, and labels."""
    dataset = load_from_disk(str(dataset_path), keep_in_memory=True)

    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    # Ensure dataset and responses have same length
    if len(dataset) != len(responses):
        print(
            f"Warning: Dataset length ({len(dataset)}) != Responses length ({len(responses)}). Using smaller length."
        )
        min_len = min(len(dataset), len(responses))
        dataset = dataset.select(range(min_len))
        responses = responses[:min_len]

    cot_messages_list = []
    response_messages_list = []
    labels_list = []

    for i in range(len(dataset)):
        example = dataset[i]
        full_response = responses[i]

        # User prompt extraction
        user_content = ""
        if "prompt" in example:
            user_content = example["prompt"]
        elif "instruction" in example:
            user_content = example["instruction"]
        elif "messages" in example:
            for msg in example["messages"]:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break

        if not user_content:
            user_content = "N/A"

        # Response splitting — use maxsplit=1 to avoid truncating on multiple </think> tags
        parts = full_response.split("</think>", 1)
        if len(parts) > 1:
            cot_part = parts[0] + "</think>"
            response_part = parts[1]
        else:
            # Fallback if no </think> tag found
            cot_part = ""
            response_part = full_response

        # Construct messages for classifier (both split types)
        cot_messages_list.append(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": cot_part},
            ]
        )
        response_messages_list.append(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_part},
            ]
        )

        # Try to find ground truth labels
        label = 0.0
        if "harmfulness" in example:
            label = _parse_label(example["harmfulness"])
        elif "label" in example:
            label = _parse_label(example["label"])
        elif "prompt_harmfulness" in example:
            label = _parse_label(example["prompt_harmfulness"])
        labels_list.append(label)

    return cot_messages_list, response_messages_list, labels_list


def main(
    base_model: str,
    dataset_path: Path,
    responses_path: Path,
    save_folder: Path,
    batch_size: int = 2000,
    max_new_tokens: int = 1024,
):
    save_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {base_model}")
    llm = LLM(
        model=base_model,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CLASSIFIER_PROMPT_TEMPLATE

    # Load data once for both CoT and Response evaluation
    print("Loading data...")
    messages_cot, messages_response, labels = load_thinking_data(
        dataset_path=dataset_path, responses_path=responses_path
    )

    # Evaluate CoT
    print("Preparing CoT prompts...")
    prompts_cot = apply_chat_template(tokenizer, messages_cot)

    print("Evaluating CoT...")
    preds_cot = generate_predictions(
        llm, tokenizer, prompts_cot, max_new_tokens=max_new_tokens, batch_size=batch_size
    )

    with open(save_folder / "preds_cot.json", "w", encoding="utf-8") as f:
        json.dump(
            {"preds": preds_cot, "labels": labels},
            f,
            ensure_ascii=True,
            indent=True,
        )

    # Evaluate Response
    print("Preparing Response prompts...")
    prompts_response = apply_chat_template(tokenizer, messages_response)

    print("Evaluating Response...")
    preds_response = generate_predictions(
        llm, tokenizer, prompts_response, max_new_tokens=max_new_tokens, batch_size=batch_size
    )

    with open(save_folder / "preds_response.json", "w", encoding="utf-8") as f:
        json.dump(
            {"preds": preds_response, "labels": labels},
            f,
            ensure_ascii=True,
            indent=True,
        )

    print(f"Results saved to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate thinking model safety.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-Guard-3-8B",
        help="Base model to use (classifier).",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the dataset containing prompts.",
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        required=True,
        help="Path to the json file containing model responses.",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("thinking_model_safety_preds"),
        help="Folder to save predictions.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2000, help="Batch size used for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Max new tokens"
    )

    args = parser.parse_args()

    main(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        responses_path=args.responses_path,
        save_folder=args.save_folder,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
