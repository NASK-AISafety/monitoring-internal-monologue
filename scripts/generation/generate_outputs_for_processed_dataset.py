import json
import sys
from argparse import ArgumentParser

sys.path.append(".")
import math
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def apply_chat_template(tokenizer, messages, enable_thinking: bool = False):
    if tokenizer.chat_template:
        return [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
            for messages in messages
        ]
    return messages


def generate_predictions(
    llm,
    tokenizer,
    data,
    max_new_tokens=200,
    batch_size: int = 1,
):
    predictions = []
    num_batches = math.ceil(len(data) / batch_size)
    for i in tqdm(range(num_batches), desc="Processing"):
        prompts = data[i * batch_size : (i + 1) * batch_size]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            min_p=0.0,
        )
        outputs = llm.generate(prompts, sampling_params)
        decoded_output = [output.outputs[0].text.strip() for output in outputs]
        predictions += decoded_output
    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Path to the base model",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="data/processed_benchmarks/harmfulness/prompt/wildguardtest",
        help="Path to dataset saved to disk",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Whether to add generation prompt when applying chat template",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="responses",
        help="Path to folder at which responses will be saved",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default="responses_wg_test.json",
        help="Path to save responses",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequences. If None, no truncation will be applied.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--system_prompt_file",
        type=Path,
        default=None,
        help="Path to a file containing the system prompt.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Whether to enable 'thinking' in the chat template.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    args = parser.parse_args()
    args.save_folder.mkdir(parents=True, exist_ok=True)

    # Initialize vLLM
    if args.system_prompt_file is not None:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = None
    # Set up tokenizer chat template
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # Load dataset using Hugging Face's load_from_disk
    dataset = load_from_disk(args.dataset_path)
    if "messages" not in dataset.column_names:
        messages = [[{"role": "user", "content": x}] for x in dataset["prompt"]]

    else:
        messages = dataset["messages"]
    if system_prompt is not None:
        messages = [
            [{"role": "system", "content": system_prompt}] + x for x in messages
        ]
    applied_chat_template = apply_chat_template(tokenizer, messages, enable_thinking=args.enable_thinking)
    llm = LLM(
        model=args.base_model,
        max_model_len=16384,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    # Generate predictions
    responses = generate_predictions(
        llm,
        tokenizer,
        applied_chat_template,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # Save responses
    with open(args.save_folder / args.output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=True, indent=4)

    print(f"Predictions saved to {args.output_file}")
