import argparse
import os
from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

ONLY_RESPONSE_TEMPLATE = (
    "The below is a reasoning process to solve a problem\n\n{response}\n\n"
    " Does the above reasoning contain an error?"
)

PROMPT_AND_RESPONSE_TEMPLATE = (
    "The below is a conversation between a user and an assistant solving a math problem. "
    "The assistant's reasoning process may contain an error."
    "\n\nUSER: {prompt}\n\ASSISTANT: {response}\n\n"
    " Does the assistant's reasoning contain an error?"
)


def _load_hf_dataset(path: str) -> Dataset:
    """Try loading a HuggingFace dataset from disk, fallback to HF Hub."""
    # First, try to see if path is a local directory
    if os.path.exists(path):
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
    else:
        # If it doesn't exist locally, assume it's a HuggingFace hub name like 'Qwen/ProcessBench'
        try:
            return load_dataset(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from Hub {path}: {e}")


def _detect_fields(example: dict) -> Tuple[str, str, str]:
    """Return (prompt_field, response_field, label_field) given example keys."""
    keys = set(example.keys())

    # ProcessBench specific fields
    prompt_field = None
    response_field = None
    label_field = None

    if "problem" in keys:
        prompt_field = "problem"
    elif "question" in keys:
        prompt_field = "question"

    if "steps" in keys:
        response_field = "steps"
    elif "generation" in keys:
        response_field = "generation"

    if "label" in keys:
        label_field = "label"
    elif "final_answer_correct" in keys:
        # Note: final_answer_correct is different from process correctness, but can be a fallback
        label_field = "label" # Prioritize 'label' if it exists anyway

    if prompt_field and response_field:
        return prompt_field, response_field, label_field

    # Fallback to string types for prompt/response if not found exactly
    str_keys = [k for k, v in example.items() if isinstance(v, str)]
    if not prompt_field and len(str_keys) >= 1:
        prompt_field = str_keys[0]
    if not response_field and len(str_keys) >= 2:
        response_field = str_keys[1]

    if not prompt_field or not response_field:
        raise ValueError("Could not detect prompt/response fields in example")

    return prompt_field, response_field, label_field


def process_dataset(ds: Dataset) -> Dataset:
    """Add new columns with the templates applied, focusing on reasoning errors."""
    if isinstance(ds, DatasetDict):
        import datasets
        # concatenate all splits/subsets
        ds = datasets.concatenate_datasets(list(ds.values()))

    # find fields by inspecting first example
    first = ds[0]
    prompt_field, response_field, label_field = _detect_fields(first)
    print(f"Detected fields: prompt='{prompt_field}', response='{response_field}', label='{label_field}'")

    def _map_fn(ex):
        prompt = ex.get(prompt_field, "") or ""
        response_raw = ex.get(response_field, "") or ""

        # If response is a list of steps, join them
        if isinstance(response_raw, list):
            # Join steps with a separator. ProcessBench steps are usually strings.
            response = "\n\n".join(response_raw)
        else:
            response = str(response_raw)

        ex["only_response_template"] = [
            {
                "role": "user",
                "content": ONLY_RESPONSE_TEMPLATE.format(response=response),
            }
        ]
        ex["prompt_and_response_template"] = [
            {
                "role": "user",
                "content": PROMPT_AND_RESPONSE_TEMPLATE.format(
                    prompt=prompt, response=response
                ),
            }
        ]

        # ProcessBench label: -1 means correct, >= 0 means first error step index
        if label_field == "label":
            lbl = ex.get(label_field)
            if lbl is not None:
                # -1 means no error, anything else (typically 0+) means error
                ex["has_reasoning_error"] = (lbl != -1)
                ex["first_error_step"] = lbl
        elif label_field:
            lbl = ex.get(label_field)
            if isinstance(lbl, bool):
                ex["has_reasoning_error"] = not lbl if "correct" in label_field else lbl
            else:
                ex["reasoning_label_raw"] = lbl

        # Also preserve original response and prompt if they were list/renamed
        ex["prompt"] = prompt
        ex["response"] = response
        # Create labels field for training scripts compatibility
        ex["labels"] = int(ex["has_reasoning_error"])
        return ex

    return ds.map(_map_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="Qwen/ProcessBench",
        help="Path to HF dataset directory, folder with json/jsonl, or HF Hub name",
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
        processed.save_to_disk(args.output_dir)
    if args.save_format in ("jsonl", "both"):
        out_file = os.path.join(args.output_dir, "processed.jsonl")
        print(f"Exporting processed dataset to JSONL {out_file}")
        try:
            processed.to_json(out_file)
        except Exception:
            with open(out_file, "w", encoding="utf-8") as fh:
                for ex in processed:
                    fh.write((str(ex) + "\n"))


if __name__ == "__main__":
    main()
