from pathlib import Path
from typing import Optional, Union

import torch
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_for_pred(
    base_model: str,
    adapter_path: Optional[str] = None,
    merge_and_unload: bool = True,
    device_map: Optional[str] = "auto",
    use_fast_tokenizer: bool = True,
    adapter_weight: Optional[float] = None,
    attn_implementation: str = "flash_attention_2",
    pad_token: str = "<unk>",
    fp8_inference: bool = False,
):
    model_dtype = "auto" if fp8_inference else torch.bfloat16
    quantization_config = (
        BitsAndBytesConfig(
            load_in_8bit=True,
        )
        if fp8_inference
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        dtype=model_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        truncation_side="right",
        trust_remote_code=True,
    )
    if (
        tokenizer.chat_template
        and "{% if '</think>' in content %}" in tokenizer.chat_template
    ):
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
            "",
        )
    if adapter_path is not None:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.pad_token_id = len(tokenizer) - 1
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModelForCausalLM.from_pretrained(
            model,
            model_id=adapter_path,
            device_map=device_map,
            dtype=model_dtype,
        )
        if adapter_weight is not None:
            model.add_weighted_adapter(
                adapters=[model.active_adapter],
                weights=[adapter_weight],
                combination_type="linear",
                adapter_name="scaled",
            )
            model.set_adapter("scaled")
        if merge_and_unload:
            model = model.merge_and_unload()
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.pad_token_id = len(tokenizer) - 1
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer.pad_token = pad_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    return model, tokenizer


def load_predictions(path: str | Path) -> list[dict]:
    """
    Loads probe predictions from either a .json or .parquet file.
    If the file with the given extension does not exist, it tries the other one.
    """
    from pathlib import Path
    path = Path(path)
    
    # List of candidate paths to try
    candidates = [path]
    if path.suffix == ".json":
        candidates.append(path.with_suffix(".parquet"))
    elif path.suffix == ".parquet":
        candidates.append(path.with_suffix(".json"))
    else:
        # If no extension or unknown extension, try both
        candidates.append(path.with_suffix(".parquet"))
        candidates.append(path.with_suffix(".json"))

    for cand in candidates:
        if cand.exists():
            if cand.suffix == ".parquet":
                import pandas as pd
                import numpy as np
                return pd.read_parquet(cand).replace({np.nan: None}).to_dict(orient="records")
            else:
                import json
                with open(cand, "r", encoding="utf-8") as f:
                    # In some files, the JSON is a dict with a "preds" key, 
                    # but for probe results it is usually a list of dicts.
                    data = json.load(f)
                    if isinstance(data, dict) and "preds" in data:
                        return data["preds"]
                    return data
                    
    raise FileNotFoundError(f"Could not find {path} or its alternatives.")
