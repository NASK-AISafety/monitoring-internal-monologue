import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm

sys.path.append(".")

from src.evaluation.templates import BASE_CHAT_TEMPLATE
from src.prototype import get_gda_pred, get_nmc_pred
from src.utils import load_model_for_pred

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_label(value) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
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
    max_samples: Optional[int] = None,
):
    dataset = load_from_disk(str(dataset_path), keep_in_memory=True)
    if "train" in getattr(dataset, "keys", lambda: [])():
        dataset = dataset["train"]

    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    if len(dataset) != len(responses):
        min_len = min(len(dataset), len(responses))
        print(
            f"Warning: Dataset length ({len(dataset)}) != Responses length ({len(responses)}). Using {min_len}."
        )
        dataset = dataset.select(range(min_len))
        responses = responses[:min_len]

    if max_samples is not None:
        max_len = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_len))
        responses = responses[:max_len]

    outputs = []
    for i in range(len(dataset)):
        example = dataset[i]
        full_response = responses[i]

        user_content = ""
        if "prompt" in example:
            user_content = example["prompt"]
        elif "instruction" in example:
            user_content = example["instruction"]
        elif "messages" in example:
            for msg in example["messages"]:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
        if not user_content:
            user_content = "N/A"

        parts = full_response.split("</think>", 1)
        if len(parts) > 1:
            cot_part = parts[0] + "</think>"
            response_part = parts[1]
        else:
            cot_part = ""
            response_part = full_response

        label = 0.0
        if "harmfulness" in example:
            label = _parse_label(example["harmfulness"])
        elif "label" in example:
            label = _parse_label(example["label"])
        elif "prompt_harmfulness" in example:
            label = _parse_label(example["prompt_harmfulness"])

        outputs.append(
            {
                "prompt": user_content,
                "cot": cot_part,
                "response": response_part,
                "label": label,
            }
        )

    return outputs


def _parse_feature_name(feature_name: str) -> tuple[int, int]:
    token_match = re.match(
        r"layer_(.+?)_token_(\d+)_hidden_states\.parquet$",
        feature_name,
    )
    if token_match:
        layer_key = token_match.group(1)
        if not layer_key.isdigit():
            raise ValueError(
                f"Feature '{feature_name}' uses non-numeric layer key '{layer_key}'. "
                "This evaluator supports numeric layer indices."
            )
        return int(layer_key), int(token_match.group(2))

    mean_match = re.match(r"layer_(.+?)_mean_hidden_states\.parquet$", feature_name)
    if mean_match:
        layer_key = mean_match.group(1)
        if not layer_key.isdigit():
            raise ValueError(
                f"Feature '{feature_name}' uses non-numeric layer key '{layer_key}'. "
                "This evaluator supports numeric layer indices."
            )
        return int(layer_key), -1

    raise ValueError(f"Could not parse layer/token from feature name: {feature_name}")


def _predict_feature_batch(
    hidden_state_vecs: torch.Tensor,
    feature_name: str,
    probe_artifact: dict,
) -> np.ndarray:
    use_nmc = probe_artifact["use_nmc"]
    use_distance = probe_artifact["use_distance"]
    params = probe_artifact["layer_params"][feature_name]

    if use_nmc:
        means = params["means"]
        preds = get_nmc_pred(
            hidden_state_vecs,
            means,
            return_probs=not use_distance,
            return_logits=use_distance,
        )
    else:
        W = params["W"]
        b = params["b"]
        preds = get_gda_pred(
            hidden_state_vecs,
            W=W,
            b=b,
            return_probs=not use_distance,
            return_logits=use_distance,
        )

    if use_distance:
        return (preds[:, 1] - preds[:, 0]).detach().to(torch.float32).cpu().numpy()
    return preds[:, 1].detach().to(torch.float32).cpu().numpy()


def main(
    base_model: str,
    probe_path: Path,
    dataset_path: Path,
    responses_path: Path,
    save_folder: Path,
    adapter_path: Optional[str] = None,
    adapter_weight: Optional[float] = None,
    max_samples: Optional[int] = None,
    max_length: Optional[int] = None,
):
    save_folder.mkdir(parents=True, exist_ok=True)

    with open(probe_path, "rb") as f:
        probe_artifact = pickle.load(f)

    feature_names = probe_artifact["feature_names"]
    parsed_features = {name: _parse_feature_name(name) for name in feature_names}
    # Transform W and B to device and dtype of the model for faster prediction
    for layer_name, layer_params in probe_artifact["layer_params"].items():
        if "W" in layer_params and layer_params["W"] is not None:
            layer_params["W"] = torch.tensor(
                layer_params["W"], device=DEVICE, dtype=torch.bfloat16
            )
        if "b" in layer_params and layer_params["b"] is not None:
            layer_params["b"] = torch.tensor(
                layer_params["b"], device=DEVICE, dtype=torch.bfloat16
            )
        if "means" in layer_params and layer_params["means"] is not None:
            layer_params["means"] = torch.tensor(
                layer_params["means"], device=DEVICE, dtype=torch.bfloat16
            )
    print(f"Loading model: {base_model}")
    model, tokenizer = load_model_for_pred(
        base_model,
        adapter_path=adapter_path,
        adapter_weight=adapter_weight,
        merge_and_unload=True,
        attn_implementation=None,
    )
    model.eval()
    if tokenizer.chat_template is None:
        tokenizer.chat_template = BASE_CHAT_TEMPLATE

    print("Loading prompts and CoTs...")
    examples = load_thinking_data(
        dataset_path=dataset_path,
        responses_path=responses_path,
        max_samples=max_samples,
    )

    all_results_cumsum = []
    all_results_unsummed = []
    for sample_id, sample in enumerate(tqdm(examples, desc="Scoring CoT tokens")):
        user_message = [{"role": "user", "content": sample["prompt"]}]
        prefix_ids = tokenizer.apply_chat_template(
            user_message,
            tokenize=True,
            add_generation_prompt=True,
        )
        cot_ids = tokenizer(sample["cot"], add_special_tokens=False)["input_ids"]

        if len(cot_ids) == 0:
            entry = {
                "sample_id": sample_id,
                "prompt": sample["prompt"],
                "label": sample["label"],
                "cot": sample["cot"],
                "num_cot_tokens": 0,
                "token_predictions": [],
            }
            all_results_cumsum.append(entry)
            all_results_unsummed.append(entry)
            continue

        input_ids = prefix_ids + cot_ids
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]

        prefix_len = len(prefix_ids)
        shift = 0
        if max_length is not None and len(prefix_ids) + len(cot_ids) > max_length:
            shift = len(prefix_ids) + len(cot_ids) - max_length
            prefix_len = max(prefix_len - shift, 0)

        model_input_ids = torch.tensor(input_ids, device=model.device).unsqueeze(0)
        attention_mask = torch.ones_like(model_input_ids, device=model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states

        num_cot_tokens = max(0, len(input_ids) - prefix_len)
        cot_positions = torch.arange(
            prefix_len,
            prefix_len + num_cot_tokens,
            device=model.device,
            dtype=torch.long,
        )

        per_layer_features_cumsum = {}
        per_layer_features_unsummed = {}
        for feature_name in feature_names:
            layer_id, token_offset = parsed_features[feature_name]
            if layer_id >= len(hidden_states):
                raise ValueError(
                    f"Requested layer index {layer_id} from feature '{feature_name}', "
                    f"but model returned only {len(hidden_states)} hidden-state tensors."
                )

            layer_hs = hidden_states[layer_id][0]
            if token_offset >= 0:
                feature_positions = torch.clamp(cot_positions - token_offset, min=0)
                vecs_unsummed = layer_hs.index_select(0, feature_positions)
                vecs_cumsum = vecs_unsummed
            else:
                # unsummed: token-wise vectors for each CoT token
                vecs_unsummed = layer_hs.index_select(0, cot_positions)

                # cumsum: cumulative mean over CoT tokens
                cot_layer_hs = layer_hs[prefix_len : prefix_len + num_cot_tokens, :]
                cumulative_sum = torch.cumsum(cot_layer_hs, dim=0)
                denominator = torch.arange(
                    1,
                    num_cot_tokens + 1,
                    device=model.device,
                    dtype=cumulative_sum.dtype,
                ).unsqueeze(1)
                vecs_cumsum = cumulative_sum / denominator

            per_layer_features_cumsum[feature_name] = _predict_feature_batch(
                vecs_cumsum,
                feature_name,
                probe_artifact,
            )
            per_layer_features_unsummed[feature_name] = _predict_feature_batch(
                vecs_unsummed,
                feature_name,
                probe_artifact,
            )

        meta_df_cumsum = pd.DataFrame(per_layer_features_cumsum, columns=feature_names)
        harmful_probs_cumsum = probe_artifact["meta_model"].predict_proba(meta_df_cumsum)[:, 1]

        meta_df_unsummed = pd.DataFrame(per_layer_features_unsummed, columns=feature_names)
        harmful_probs_unsummed = probe_artifact["meta_model"].predict_proba(meta_df_unsummed)[:, 1]

        cot_token_ids = [
            int(x) for x in input_ids[prefix_len : prefix_len + num_cot_tokens]
        ]
        cot_token_texts = tokenizer.batch_decode(
            [[token_id] for token_id in cot_token_ids],
            skip_special_tokens=False,
        )

        token_predictions_cumsum = [
            {
                "token_idx": token_idx,
                "token_id": token_id,
                "token_text": token_text,
                "harmful_prob": float(harmful_prob),
            }
            for token_idx, token_id, token_text, harmful_prob in zip(
                range(num_cot_tokens),
                cot_token_ids,
                cot_token_texts,
                harmful_probs_cumsum,
            )
        ]
        token_predictions_unsummed = [
            {
                "token_idx": token_idx,
                "token_id": token_id,
                "token_text": token_text,
                "harmful_prob": float(harmful_prob),
            }
            for token_idx, token_id, token_text, harmful_prob in zip(
                range(num_cot_tokens),
                cot_token_ids,
                cot_token_texts,
                harmful_probs_unsummed,
            )
        ]

        all_results_cumsum.append(
            {
                "sample_id": sample_id,
                "prompt": sample["prompt"],
                "label": sample["label"],
                "cot": sample["cot"],
                "num_cot_tokens": len(token_predictions_cumsum),
                "token_predictions": token_predictions_cumsum,
            }
        )
        all_results_unsummed.append(
            {
                "sample_id": sample_id,
                "prompt": sample["prompt"],
                "label": sample["label"],
                "cot": sample["cot"],
                "num_cot_tokens": len(token_predictions_unsummed),
                "token_predictions": token_predictions_unsummed,
            }
        )

    cumsum_path = save_folder / "cot_token_probe_preds_cumsum.parquet"
    unsummed_path = save_folder / "cot_token_probe_preds_unsummed.parquet"
    
    pd.DataFrame(all_results_cumsum).to_parquet(cumsum_path)
    pd.DataFrame(all_results_unsummed).to_parquet(unsummed_path)

    print(
        f"Saved token-level probe predictions to: {cumsum_path} and {unsummed_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CoT tokens with a saved multilayer probe."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Model of interest used to extract CoT token activations.",
    )
    parser.add_argument(
        "--probe_path",
        type=Path,
        required=True,
        help="Path to saved multilayer_probe.pkl.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to wildguardtest prompts dataset (HF datasets format).",
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        required=True,
        help="Path to responses json from model of interest (includes CoT).",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("results/cot_eval/probe_eval"),
        help="Folder to save token-level probe predictions.",
    )
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--adapter_weight", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional max sequence length before forward pass (left truncation).",
    )
    args = parser.parse_args()

    main(
        base_model=args.base_model,
        probe_path=args.probe_path,
        dataset_path=args.dataset_path,
        responses_path=args.responses_path,
        save_folder=args.save_folder,
        adapter_path=args.adapter_path,
        adapter_weight=args.adapter_weight,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )
