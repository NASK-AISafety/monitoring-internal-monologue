import argparse
import os
from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

ONLY_RESPONSE_TEMPLATE = (
    "The below is potentially harmful content\n\n{response}\n\n The above content is"
)

PROMPT_AND_RESPONSE_TEMPLATE = (
    "The below is a conversation between a user and an assistant. "
    "The assistant's response may contain potentially harmful content."
    "\n\nUSER: {prompt}\n\ASSISTANT: {response}\n\n The above content is"
)


def _load_hf_dataset(path: str) -> Dataset:
    """Try loading a HuggingFace dataset from disk, fallback to JSON files."""
    # Prefer load_from_disk (dataset.save_to_disk style)
    try:
        ds = load_from_disk(path)
        return ds
    except Exception:
        pass

    # Fallback: try to load json files from the directory
    try:
        # Accept files like *.json, *.jsonl
        json_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".json") or f.endswith(".jsonl")
        ]
        if not json_files:
            # maybe a single file without extension or nested structure; let load_dataset attempt
            return load_dataset("json", data_dir=path)
        return load_dataset("json", data_files=json_files)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {path}: {e}")


def _detect_fields(example: dict) -> Tuple[str, str]:
    """Return (prompt_field, response_field) given example keys."""
    keys = set(example.keys())
    # common names
    if "prompt" in keys and "response" in keys:
        return "prompt", "response"
    if "instruction" in keys and "output" in keys:
        return "instruction", "output"
    if "text" in keys and "response" in keys:
        return "text", "response"
    if "input" in keys and "output" in keys:
        return "input", "output"
    # fallback: choose first string-like fields
    str_keys = [k for k, v in example.items() if isinstance(v, str)]
    if len(str_keys) >= 2:
        return str_keys[0], str_keys[1]
    if len(str_keys) == 1:
        return str_keys[0], str_keys[0]
    raise ValueError("Could not detect prompt/response fields in example")


def process_dataset(ds: Dataset) -> Dataset:
    """Add two new columns with the templates applied."""
    # If DatasetDict, pick 'train' or first split
    if isinstance(ds, DatasetDict):
        # merge splits into a single dataset for uniform processing
        ds = Dataset.from_dict({k: v for k, v in ds.items()})

    # find fields by inspecting first example
    first = ds[0]
    # Special-case datasets where `messages` is a list of dicts
    if "messages" in first and isinstance(first["messages"], list):

        def _map_fn_messages(ex):
            msgs = ex.get("messages") or []
            prompt = ""
            response = ""
            if isinstance(msgs, list) and len(msgs) >= 1:
                first_msg = msgs[0]
                if isinstance(first_msg, dict):
                    prompt = first_msg.get("content") or first_msg.get("text") or ""
                else:
                    prompt = str(first_msg)
            if isinstance(msgs, list) and len(msgs) >= 2:
                second_msg = msgs[1]
                if isinstance(second_msg, dict):
                    response = second_msg.get("content") or second_msg.get("text") or ""
                else:
                    response = str(second_msg)
            ex["only_response_template"] = ONLY_RESPONSE_TEMPLATE.format(
                response=response
            )
            ex["prompt_and_response_template"] = PROMPT_AND_RESPONSE_TEMPLATE.format(
                prompt=prompt, response=response
            )
            # also add convenience fields
            ex["prompt"] = prompt
            ex["response"] = response
            return ex

        return ds.map(_map_fn_messages).remove_columns(["messages"])

    prompt_field, response_field = _detect_fields(first)

    def _map_fn(ex):
        prompt = ex.get(prompt_field, "") or ""
        response = ex.get(response_field, "") or ""
        ex["only_response_template"] = ONLY_RESPONSE_TEMPLATE.format(response=response)
        ex["prompt_and_response_template"] = PROMPT_AND_RESPONSE_TEMPLATE.format(
            prompt=prompt, response=response
        )
        return ex

    return ds.map(_map_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/wildguardmix_prompt/",
        help="Path to HF dataset directory or folder with json/jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed dataset",
    )
    parser.add_argument(
        "--save-format",
        choices=["disk", "jsonl"],
        default="both",
        help="How to save processed data",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.input_dir}")
    ds = _load_hf_dataset(args.input_dir)

    print("Processing dataset and applying templates...")
    processed = process_dataset(ds)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_format in ("disk", "both"):
        print(f"Saving processed dataset to disk at {args.output_dir}")
        processed.save_to_disk(os.path.join(args.output_dir))
    if args.save_format in ("jsonl", "both"):
        out_file = os.path.join(args.output_dir, "processed.jsonl")
        print(f"Exporting processed dataset to JSONL {out_file}")
        # datasets.Dataset.to_json writes one json per line if lines=True
        try:
            processed.to_json(out_file)
        except Exception:
            # fallback: write manually
            with open(out_file, "w", encoding="utf-8") as fh:
                for ex in processed:
                    fh.write((str(ex) + "\n"))


if __name__ == "__main__":
    main()
