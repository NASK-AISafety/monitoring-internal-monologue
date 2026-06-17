import json
from argparse import ArgumentParser
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from src.evaluation.templates import BASE_CHAT_TEMPLATE, CLASSIFIER_PROMPT_TEMPLATE
from src.inference.hidden_states import generate_hidden_states
from src.utils import load_model_for_pred


def _load_dataset_split(dataset_path: Path):
    dataset = load_from_disk(dataset_path, keep_in_memory=True)
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        return next(iter(dataset.values()))
    return dataset


def _build_preprocessed_dataloader(
    dataset_path: Path,
    tokenizer,
    batch_size: int,
    add_generation_prompt: bool,
    use_prompt_and_response_template: bool,
    max_length: int | None,
    is_messages_dataset: bool = False,
    messages_column: str = "messages",
):
    dataset = _load_dataset_split(dataset_path)

    label_field = None
    if "labels" in dataset.column_names:
        label_field = "labels"
    elif "label" in dataset.column_names:
        label_field = "label"

    if is_messages_dataset:
        if messages_column not in dataset.column_names:
            raise ValueError(
                f"Expected '{messages_column}' in dataset columns, found: {dataset.column_names}"
            )

        def process_dataset(example):
            messages = example[messages_column]
            outputs = {
                "input_ids": tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    truncation=max_length is not None,
                    max_length=max_length,
                )
            }
            if label_field is not None:
                outputs["labels"] = example[label_field]
            else:
                outputs["labels"] = 0.0
            return outputs
    else:
        template_column = (
            "prompt_and_response_template"
            if use_prompt_and_response_template
            else "only_response_template"
        )
        if template_column not in dataset.column_names:
            raise ValueError(
                f"Expected '{template_column}' in dataset columns, found: {dataset.column_names}"
            )

        def process_dataset(example):
            text = example.get(template_column, "") or ""
            messages = [{"role": "user", "content": text}]
            outputs = {
                "input_ids": tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    truncation=max_length is not None,
                    max_length=max_length,
                )
            }
            if label_field is not None:
                outputs["labels"] = example[label_field]
            else:
                outputs["labels"] = 0.0
            return outputs

    dataset = dataset.map(process_dataset).remove_columns(
        [x for x in dataset.column_names if x != "labels"]
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Path to the base model",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None, help="Path to the adapter"
    )
    parser.add_argument(
        "--adapter_weight",
        type=float,
        default=None,
        help="Weight with which adapter will be added into base model weights.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="data/cot_experiments/wildguardmix_train_template/",
        help="Path to wildguardmix dataset saved to disk",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Whether to add generation prompt when applying chat template",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="cot_template_wildguard_hidden_states",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--remove_files_if_exists",
        type=bool,
        default=True,
        help="Wheter to remove parquet files if they exists",
    )
    parser.add_argument(
        "--use_prompt_and_response_template",
        action="store_true",
        help="Use prompt+response format when available",
    )
    parser.add_argument(
        "--is_messages_dataset",
        action="store_true",
        help="Whether the dataset is already in a messages format (Scenario B/C)",
    )
    parser.add_argument(
        "--messages_column",
        type=str,
        default="messages",
        help="Name of the column containing messages if is_messages_dataset is true",
    )
    parser.add_argument(
        "--calculate_mean_hidden_states",
        action="store_true",
        help="Whether to calculate mean hidden states for each example",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequences. If None, no truncation will be applied.",
    )
    parser.add_argument(
        "--layer_names_to_extract_file",
        type=Path,
        default=None,
        help="Path to file containing layer names "
        "to extract as dict with boolean value wheter output should be captured",
    )
    args = parser.parse_args()
    args.save_folder.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_for_pred(
        args.base_model,
        args.adapter_path,
        attn_implementation=None,
    )
    if args.layer_names_to_extract_file is not None:
        with open(args.layer_names_to_extract_file, "r") as f:
            layer_names_to_extract = json.load(f)
    else:
        layer_names_to_extract = None
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.chat_template is None:
        if args.base_model == "allenai/wildguard":
            tokenizer.chat_template = CLASSIFIER_PROMPT_TEMPLATE
        else:
            tokenizer.chat_template = BASE_CHAT_TEMPLATE
    # Always use the preprocessed template columns created by preprocessing
    dataloader = _build_preprocessed_dataloader(
        args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        add_generation_prompt=args.add_generation_prompt,
        use_prompt_and_response_template=args.use_prompt_and_response_template,
        max_length=args.max_length,
        is_messages_dataset=args.is_messages_dataset,
        messages_column=args.messages_column,
    )
    generate_hidden_states(
        model=model,
        dataloader=dataloader,
        save_folder=args.save_folder,
        remove_files_if_exists=args.remove_files_if_exists,
        calculate_mean_hidden_states=args.calculate_mean_hidden_states,
        layer_names_to_extract=layer_names_to_extract,
    )
