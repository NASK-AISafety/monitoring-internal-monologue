import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm

sys.path.append(".")

from src.cot_probe.trainer import CotProbeModel
from src.evaluation.templates import BASE_CHAT_TEMPLATE
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
            f"Warning: Dataset length ({len(dataset)}) != Responses length "
            f"({len(responses)}). Using {min_len}."
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


def load_lm_eval_data(
    samples_paths: list[Path],
    max_samples: Optional[int] = None,
):
    """Load thinking data from lm_eval JSONL sample files.

    Each JSONL line is expected to have:
      - doc: dict with 'question' (gsm8k) or 'problem' (minerva_math)
      - resps: list[list[str]] — full model response including <think> tags
      - exact_match: float (1.0 = correct, 0.0 = incorrect)
    """
    all_samples = []  # list of (task_name, raw_sample_dict)
    for path in samples_paths:
        path = Path(path)
        # Derive task name from filename: "samples_minerva_math_algebra_2026-…jsonl"
        # → "minerva_math_algebra"
        stem = path.stem  # e.g. samples_minerva_math_algebra_2026-04-15T20-49-32.366256
        # Strip leading "samples_" and trailing "_<timestamp>" (ISO-like date suffix)
        task_name = re.sub(r"^samples_", "", stem)
        task_name = re.sub(r"_\d{4}-\d{2}-\d{2}T[\d\-\.]+$", "", task_name)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append((task_name, json.loads(line)))

    if max_samples is not None:
        all_samples = all_samples[:max_samples]

    outputs = []
    for task_name, sample in all_samples:
        doc = sample.get("doc", {})

        # Extract prompt: gsm8k uses 'question', minerva_math uses 'problem'
        user_content = (
            doc.get("question")
            or doc.get("problem")
            or doc.get("input")
            or "N/A"
        )

        # Extract full response from lm_eval resps field
        full_response = ""
        resps = sample.get("resps", [[]])
        if resps and resps[0]:
            full_response = resps[0][0]

        # Split into CoT and response parts
        parts = full_response.split("</think>", 1)
        if len(parts) > 1:
            cot_part = parts[0] + "</think>"
            response_part = parts[1]
        else:
            cot_part = ""
            response_part = full_response

        # Label: prefer math_verify (minerva_math) over exact_match (gsm8k)
        if "math_verify" in sample:
            label = float(sample["math_verify"])
        else:
            label = float(sample.get("exact_match", 0.0))

        outputs.append(
            {
                "prompt": user_content,
                "cot": cot_part,
                "response": response_part,
                "label": label,
                "task": task_name,   # e.g. "gsm8k", "minerva_math_algebra"
            }
        )

    return outputs


def main(
    base_model: str,
    probe_checkpoint: Path,
    dataset_path: Path | None,
    responses_path: Path | None,
    save_folder: Path,
    layer_indices: list[int] | None = None,
    adapter_path: Optional[str] = None,
    adapter_weight: Optional[float] = None,
    max_samples: Optional[int] = None,
    max_length: Optional[int] = None,
    pool_over_prompt: bool = False,
    lm_eval_samples_paths: list[Path] | None = None,
):
    save_folder.mkdir(parents=True, exist_ok=True)

    # Load the MIL probe (no base model needed)
    print(f"Loading MIL probe from: {probe_checkpoint}")
    probe_model = CotProbeModel.load_probe(str(probe_checkpoint), map_location=DEVICE)
    probe_model = probe_model.to(DEVICE).to(torch.bfloat16)
    num_probe_layers = len(probe_model.probe.layer_probes)
    print(f"Probe has {num_probe_layers} layer probes")

    # Resolve layer_indices: CLI override > checkpoint hparams > all layers
    if layer_indices is None:
        checkpoint = torch.load(
            str(probe_checkpoint), map_location="cpu", weights_only=False
        )
        hparams = checkpoint.get("hyper_parameters", {})
        if "layer_indices" in hparams and hparams["layer_indices"] is not None:
            layer_indices = hparams["layer_indices"]
            print(f"Using layer_indices from checkpoint: {layer_indices}")
        else:
            print("No layer_indices in checkpoint, hooking all layers")

    # Load the base model
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

    # Attach hooks (same as during training)
    layer_names, captured_states, hooks = CotProbeModel.attach_hooks(
        model, layer_indices=layer_indices
    )
    print(
        f"Attached hooks on {len(layer_names)} layers: "
        f"{[n.split('.')[-1] for n in layer_names]}"
    )

    if len(layer_names) != num_probe_layers:
        raise ValueError(
            f"Number of hooked layers ({len(layer_names)}) does not match "
            f"probe's expected number of layers ({num_probe_layers}). "
            f"Check --layer_indices."
        )

    # Load data
    print("Loading prompts and CoTs...")
    if lm_eval_samples_paths is not None:
        examples = load_lm_eval_data(
            samples_paths=lm_eval_samples_paths,
            max_samples=max_samples,
        )
    else:
        examples = load_thinking_data(
            dataset_path=dataset_path,
            responses_path=responses_path,
            max_samples=max_samples,
        )

    all_results_cummax = []
    all_results_pertoken = []

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
            if pool_over_prompt:
                entry["prompt_token_predictions"] = []
            all_results_cummax.append(entry)
            all_results_pertoken.append(entry.copy())
            continue

        input_ids = prefix_ids + cot_ids
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]

        prefix_len = len(prefix_ids)
        if max_length is not None and len(prefix_ids) + len(cot_ids) > max_length:
            shift = len(prefix_ids) + len(cot_ids) - max_length
            prefix_len = max(prefix_len - shift, 0)

        # Forward pass through the model — hooks capture hidden states
        model_input_ids = torch.tensor(input_ids, device=model.device).unsqueeze(0)
        attention_mask = torch.ones_like(model_input_ids, device=model.device)

        captured_states.clear()
        with torch.no_grad():
            model(input_ids=model_input_ids, attention_mask=attention_mask)

        # Collect hidden states from hooks: list of (1, seq, d_model)
        # Move them to DEVICE in case the base model is spread across multiple GPUs
        hidden_states_per_layer = [captured_states[name].to(DEVICE) for name in layer_names]

        num_cot_tokens = max(0, len(input_ids) - prefix_len)
        cot_start = prefix_len
        cot_end = prefix_len + num_cot_tokens

        # --- Per-token scores (no max-pooling) ---
        # Flatten to (num_cot_tokens, d_model) — each token scored independently
        pertoken_hidden = [
            hs[:, cot_start:cot_end, :].reshape(-1, hs.size(-1))
            for hs in hidden_states_per_layer
        ]
        with torch.no_grad():
            logits, _ = probe_model.predict_from_hidden_states(pertoken_hidden)
        pertoken_probs = torch.sigmoid(logits).cpu().tolist()

        # --- Cumulative max-pool scores (using cummax inside probe) ---
        # Single forward pass with use_cummax=True: O(n) via torch.cummax
        cummax_start = 0 if pool_over_prompt else cot_start
        cummax_len = cot_end - cummax_start
        cummax_hidden = [
            hs[:, cummax_start:cot_end, :] for hs in hidden_states_per_layer
        ]
        cummax_mask = torch.ones(1, cummax_len, device=DEVICE, dtype=torch.float)
        with torch.no_grad():
            logits, _ = probe_model.predict_from_hidden_states(
                cummax_hidden, attention_mask=cummax_mask, use_cummax=True
            )
        cummax_probs_all = torch.sigmoid(logits[0]).cpu().tolist()
        cummax_probs = cummax_probs_all[-num_cot_tokens:] if num_cot_tokens > 0 else []

        prompt_token_preds_cummax = []
        if pool_over_prompt:
            prompt_probs = cummax_probs_all[:len(cummax_probs_all)-num_cot_tokens]
            prompt_token_ids = [int(x) for x in input_ids[:cot_start]]
            prompt_token_texts = tokenizer.batch_decode(
                [[tid] for tid in prompt_token_ids],
                skip_special_tokens=False,
            )
            prompt_token_preds_cummax = [
                {
                    "token_idx": idx,
                    "token_id": tid,
                    "token_text": txt,
                    "harmful_prob": prob,
                }
                for idx, tid, txt, prob in zip(
                    range(len(prompt_token_ids)),
                    prompt_token_ids,
                    prompt_token_texts,
                    prompt_probs,
                )
            ]

        # Build token info
        cot_token_ids = [int(x) for x in input_ids[cot_start:cot_end]]
        cot_token_texts = tokenizer.batch_decode(
            [[tid] for tid in cot_token_ids],
            skip_special_tokens=False,
        )

        token_preds_cummax = [
            {
                "token_idx": idx,
                "token_id": tid,
                "token_text": txt,
                "harmful_prob": prob,
            }
            for idx, tid, txt, prob in zip(
                range(num_cot_tokens),
                cot_token_ids,
                cot_token_texts,
                cummax_probs,
            )
        ]
        token_preds_pertoken = [
            {
                "token_idx": idx,
                "token_id": tid,
                "token_text": txt,
                "harmful_prob": prob,
            }
            for idx, tid, txt, prob in zip(
                range(num_cot_tokens),
                cot_token_ids,
                cot_token_texts,
                pertoken_probs,
            )
        ]

        entry_cummax = {
            "sample_id": sample_id,
            "prompt": sample["prompt"],
            "label": sample["label"],
            "task": sample.get("task"),
            "cot": sample["cot"],
            "num_cot_tokens": num_cot_tokens,
            "token_predictions": token_preds_cummax,
        }
        if pool_over_prompt:
            entry_cummax["prompt_token_predictions"] = prompt_token_preds_cummax
        all_results_cummax.append(entry_cummax)

        entry_pertoken = {
            "sample_id": sample_id,
            "prompt": sample["prompt"],
            "label": sample["label"],
            "task": sample.get("task"),
            "cot": sample["cot"],
            "num_cot_tokens": num_cot_tokens,
            "token_predictions": token_preds_pertoken,
        }
        all_results_pertoken.append(entry_pertoken)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save results
    cummax_path = save_folder / "cot_mil_probe_preds_cummax.parquet"
    pertoken_path = save_folder / "cot_mil_probe_preds_pertoken.parquet"

    pd.DataFrame(all_results_cummax).to_parquet(cummax_path)
    pd.DataFrame(all_results_pertoken).to_parquet(pertoken_path)

    print(f"Saved cumulative max-pool predictions to: {cummax_path}")
    print(f"Saved per-token predictions to: {pertoken_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CoT tokens with a trained MIL probe."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model used to extract hidden states.",
    )
    parser.add_argument(
        "--probe_checkpoint",
        type=Path,
        required=True,
        help="Path to the MIL probe Lightning .ckpt file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Path to prompts dataset (HF datasets format). "
        "Required unless --lm_eval_samples_paths is provided.",
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        default=None,
        help="Path to responses JSON (includes CoT with </think> tags). "
        "Required unless --lm_eval_samples_paths is provided.",
    )
    parser.add_argument(
        "--lm_eval_samples_paths",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to lm_eval JSONL sample files. "
        "Alternative to --dataset_path/--responses_path.",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("results/cot_eval/mil_probe_eval"),
        help="Folder to save token-level probe predictions.",
    )
    parser.add_argument(
        "--layer_indices",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to hook (overrides checkpoint hparams). "
        "If not provided, auto-detected from checkpoint or hooks all layers.",
    )
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--adapter_weight", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max sequence length (left truncation if exceeded).",
    )
    parser.add_argument(
        "--pool_over_prompt",
        action="store_true",
        help="Whether to perform cumulative max pooling over the prompt tokens as well.",
    )
    args = parser.parse_args()

    if args.lm_eval_samples_paths is None:
        if args.dataset_path is None or args.responses_path is None:
            parser.error(
                "--dataset_path and --responses_path are required "
                "unless --lm_eval_samples_paths is provided."
            )

    main(
        base_model=args.base_model,
        probe_checkpoint=args.probe_checkpoint,
        dataset_path=args.dataset_path,
        responses_path=args.responses_path,
        save_folder=args.save_folder,
        layer_indices=args.layer_indices,
        adapter_path=args.adapter_path,
        adapter_weight=args.adapter_weight,
        max_samples=args.max_samples,
        max_length=args.max_length,
        pool_over_prompt=args.pool_over_prompt,
        lm_eval_samples_paths=args.lm_eval_samples_paths,
    )
