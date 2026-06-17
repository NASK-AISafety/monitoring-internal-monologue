from datasets import load_dataset


def load_wildguard_for_prompt_classification(split: str = "train"):
    config = "wildguardtrain" if split == "train" else "wildguardtest"
    ds_split = "train" if split == "train" else "test"
    dataset = load_dataset(
        "allenai/wildguardmix", config, keep_in_memory=True
    )[ds_split]

    columns_to_drop = [
        c for c in dataset.column_names
        if c not in ("prompt", "response")
    ]

    def map_dataset(example):
        example["messages"] = [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ]
        example["labels"] = float(example["prompt_harm_label"] == "harmful")
        example["category"] = example["subcategory"]
        example["is_adversarial"] = example["adversarial"]
        example["prompt_harmfulness"] = example["prompt_harm_label"] == "harmful"
        return example

    dataset = dataset.map(
        map_dataset,
        remove_columns=columns_to_drop,
    )
    return dataset


def load_wildguard_for_response_classification(split: str = "train"):
    config = "wildguardtrain" if split == "train" else "wildguardtest"
    ds_split = "train" if split == "train" else "test"
    dataset = load_dataset(
        "allenai/wildguardmix", config, keep_in_memory=True
    )[ds_split]
    dataset = dataset.filter(
        lambda example: (example["response_harm_label"] is not None)
        and (example["response"] is not None)
        and (example["response"].strip() != "")
    )

    columns_to_drop = [
        c for c in dataset.column_names
        if c not in ("prompt", "response")
    ]

    def map_dataset(example):
        example["messages"] = [
            {
                "role": "user",
                "content": example["prompt"],
            },
            {
                "role": "assistant",
                "content": example["response"],
            },
        ]
        example["labels"] = float(example["response_harm_label"] == "harmful")
        example["category"] = example["subcategory"]
        example["response_harmfulness"] = example["response_harm_label"] == "harmful"
        example["prompt_harmfulness"] = example["prompt_harm_label"] == "harmful"
        return example

    dataset = dataset.map(
        map_dataset,
        remove_columns=columns_to_drop,
    )
    return dataset

def load_wildguardtest_prompt(include_response=True) -> list[dict]:
    df = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    df = df.to_pandas()
    # drop if prompt_harm_label is None
    df = df.dropna(subset=["prompt_harm_label"])
    # setup prompt_type based on "adversarial"
    df["prompt_type"] = df.apply(
        lambda row: "adversarial" if row["adversarial"] else "vanilla", axis=1
    )
    df["gt_label"] = df["prompt_harm_label"]

    return df




def load_wildguardtest_response():
    df = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    df = df.to_pandas()
    # drop if prompt_harm_label is None
    df = df.dropna(subset=["response_harm_label"])
    # setup prompt_type based on "adversarial"
    df["prompt_type"] = df.apply(
        lambda row: "adversarial" if row["adversarial"] else "vanilla", axis=1
    )
    df["gt_label"] = df["response_harm_label"]
    df["prompt_harm_category"] = df["subcategory"]

    return df


def load_aegis_for_prompt_classification(split: str = "train"):
    dataset = load_dataset(
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", keep_in_memory=True
    )[split]

    columns_to_drop = [
        c for c in dataset.column_names
        if c not in ("prompt", "response")
    ]

    def map_dataset(example):
        example["messages"] = [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ]
        example["labels"] = float(example["prompt_label"] != "Safe")
        example["category"] = example["violated_categories"] or ""
        example["prompt_harmfulness"] = example["prompt_label"] != "Safe"
        return example

    dataset = dataset.map(
        map_dataset,
        remove_columns=columns_to_drop,
    )
    return dataset


def load_aegis_for_response_classification(split: str = "train"):
    dataset = load_dataset(
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", keep_in_memory=True
    )[split]
    dataset = dataset.filter(
        lambda example: (example["response_label"] is not None)
        and (example["response"] is not None)
        and (example["response"].strip() != "")
    )

    columns_to_drop = [
        c for c in dataset.column_names
        if c not in ("prompt", "response")
    ]

    def map_dataset(example):
        example["messages"] = [
            {
                "role": "user",
                "content": example["prompt"],
            },
            {
                "role": "assistant",
                "content": example["response"],
            },
        ]
        example["labels"] = float(example["response_label"] != "Safe")
        example["category"] = example["violated_categories"] or ""
        example["response_harmfulness"] = example["response_label"] != "Safe"
        example["prompt_harmfulness"] = example["prompt_label"] != "Safe"
        return example

    dataset = dataset.map(
        map_dataset,
        remove_columns=columns_to_drop,
    )
    return dataset