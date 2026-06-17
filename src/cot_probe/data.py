from typing import Optional

from datasets import Dataset, DatasetDict, load_from_disk
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class CotProbeData(LightningDataModule):
    """Data module for training the CoT MIL probe.

    Loads the WildGuardMix dataset, tokenizes prompt+response,
    and tracks response_length for each example so that the trainer
    can create a response-only mask for max-pooling.

    If the loaded dataset is a single Dataset (not a DatasetDict with
    train/test splits), it is automatically split into train and test sets.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_path: str = "data/wildjailbreak",
        batch_size: int = 2,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        add_generation_prompt: bool = False,
        test_size: float = 0.1,
        seed: int = 42,
        messages_only: bool = False,
        messages_column: str = "messages",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt
        self.test_size = test_size
        self.seed = seed
        self.messages_only = messages_only
        self.messages_column = messages_column

        self.train_dataset = None
        self.val_dataset = None
        self.collator = None

    def setup(self, stage: str):
        dataset = load_from_disk(self.dataset_path, keep_in_memory=True)

        # Handle single Dataset vs DatasetDict
        if isinstance(dataset, Dataset):
            print(
                f"Dataset is a single split with {len(dataset)} examples. "
                f"Splitting into train/test with test_size={self.test_size}."
            )
            splits = dataset.train_test_split(test_size=self.test_size, seed=self.seed)
            train_split = splits["train"]
            test_split = splits["test"]
        elif isinstance(dataset, DatasetDict):
            train_split = dataset["train"]
            test_split = dataset.get("test", dataset.get("validation"))
            if test_split is None:
                raise ValueError(
                    f"DatasetDict has keys {list(dataset.keys())} but no 'test' or 'validation' split."
                )
        else:
            raise TypeError(f"Unexpected dataset type: {type(dataset)}")

        tokenizer = self.tokenizer

        # Fix for DeepSeek-R1-Llama: the default chat template removes everything before </think>
        # in the assistant message. We remove this logic to preserve the CoT tokens for training.
        if tokenizer.chat_template and "{% if '</think>' in content %}" in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.replace(
                "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
                ""
            )

        max_length = self.max_length
        add_generation_prompt = self.add_generation_prompt

        def process_dataset(example):
            messages = example[self.messages_column]
            # Full prompt + response
            full_enc = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                truncation=max_length is not None,
                max_length=max_length,
            )
            if self.messages_only:
                if isinstance(full_enc, list):
                    return {
                        "input_ids": full_enc,
                        "labels": example["labels"],
                    }

                return {
                    **full_enc,
                    "labels": example["labels"],
                }

            # Prompt only (to compute response length)
            prompt_ids = tokenizer.apply_chat_template(
                messages[:1],
                add_generation_prompt=True,
                truncation=max_length is not None,
                max_length=max_length,
            )
            if isinstance(full_enc, list):
                return {
                    "input_ids": full_enc,
                    "labels": example["labels"],
                    "response_length": len(full_enc) - len(prompt_ids),
                }

            return {
                **full_enc,
                "labels": example["labels"],
                "response_length": len(full_enc["input_ids"]) - len(prompt_ids),
            }

        cols_to_remove = [c for c in train_split.column_names if c not in ("labels",)]
        train_split = train_split.map(process_dataset).remove_columns(cols_to_remove)
        test_split = test_split.map(process_dataset).remove_columns(cols_to_remove)

        # Filter out examples where response_length <= 0 (e.g., due to max_length truncation)
        if not self.messages_only:
            orig_train_len = len(train_split)
            train_split = train_split.filter(lambda x: x["response_length"] > 0)
            if len(train_split) < orig_train_len:
                print(f"Dropped {orig_train_len - len(train_split)} train examples due to response_length <= 0.")

            orig_val_len = len(test_split)
            test_split = test_split.filter(lambda x: x["response_length"] > 0)
            if len(test_split) < orig_val_len:
                print(f"Dropped {orig_val_len - len(test_split)} validation examples due to response_length <= 0.")

        self.train_dataset = train_split
        self.val_dataset = test_split
        self.collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )
