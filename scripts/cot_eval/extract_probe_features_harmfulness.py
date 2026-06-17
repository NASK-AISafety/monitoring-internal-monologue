import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import load_predictions

from probe_utils import process_sample


def parse_harmfulness(pred_str: str) -> int:
    """
    Parses the prediction string to determine if the response is harmful.
    Returns 1 if harmful, 0 if safe, -1 if unknown.
    Based on summarize_mil_probe_pred.py
    """
    if not isinstance(pred_str, str):
        return -1
    lower_pred = pred_str.lower()
    if "harmful response: yes" in lower_pred:
        return 1
    elif "harmful response: no" in lower_pred:
        return 0
    # Fallback for simple "yes"/"no"
    if lower_pred.strip() == "yes":
        return 1
    if lower_pred.strip() == "no":
        return 0
    return -1


def detect_model(file_path: Path) -> str:
    """Extracts model name from the directory structure: .../mil_probe_eval/<model>/<task>/..."""
    return file_path.parent.parent.name


def detect_dataset(task_name: str, wg_results_dir: Path) -> str:
    """Finds which dataset in wg_results_dir is a prefix of the task_name."""
    if not wg_results_dir.exists():
        return None
    
    # Sort folders by length descending to find the longest matching prefix
    datasets = sorted([d.name for d in wg_results_dir.iterdir() if d.is_dir()], key=len, reverse=True)
    for ds in datasets:
        if task_name.startswith(ds):
            return ds
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract signal processing features and harmfulness labels."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/harmfulness/mil_probe_eval"),
        help="Base directory containing probe evaluations.",
    )
    parser.add_argument(
        "--wg-results-dir",
        type=Path,
        default=Path("results/harmfulness/eval_wildguard"),
        help="Directory containing WildGuard outputs (nested by dataset/model).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/harmfulness/mil_probe_eval/signal_features.csv"),
        help="Output CSV for features.",
    )
    parser.add_argument(
        "--cot-percentage",
        type=float,
        default=100.0,
        help="Percentage of CoT tokens to use (default 100.0)",
    )
    parser.add_argument(
        "--max-cot-tokens",
        type=int,
        default=-1,
        help="Maximum number of CoT tokens to use (default -1 for unlimited)",
    )
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Find all cummax files in _pool_over_prompt directories
    json_files = list(args.base_dir.rglob("*_pool_over_prompt/cot_mil_probe_preds_cummax.json"))
    parquet_files = list(args.base_dir.rglob("*_pool_over_prompt/cot_mil_probe_preds_cummax.parquet"))
    
    # De-duplicate: prefer .parquet if both exist
    files_dict = {}
    for f in json_files + parquet_files:
        base = f.with_suffix("")
        if base not in files_dict or f.suffix == ".parquet":
            files_dict[base] = f
    
    all_files = sorted(list(files_dict.values()))

    if not all_files:
        print(f"No cummax.json/parquet files found in {args.base_dir}")
        return

    all_features = []

    # Cache for WildGuard labels by model
    wg_cache = {}

    for file_path in tqdm(all_files, desc="Processing probe files"):
        experiment_id = file_path.parent.name
        model_name = detect_model(file_path)
        task_name = experiment_id.replace("_pool_over_prompt", "")
        dataset_name = detect_dataset(task_name, args.wg_results_dir)
        
        cot_labels = []
        resp_labels = []

        if model_name and dataset_name:
            cache_key = (model_name, dataset_name)
            if cache_key not in wg_cache:
                wg_model_dir = args.wg_results_dir / dataset_name / model_name
                try:
                    cot_data = load_predictions(wg_model_dir / "preds_cot.json")
                    cot_labels = [parse_harmfulness(p) for p in cot_data]
                    resp_data = load_predictions(wg_model_dir / "preds_response.json")
                    resp_labels = [parse_harmfulness(p) for p in resp_data]
                    wg_cache[cache_key] = (cot_labels, resp_labels)
                except Exception as e:
                    print(f"Warning: Could not load WG labels for {model_name} / {dataset_name}: {e}")
                    wg_cache[cache_key] = (None, None)
            
            cot_labels, resp_labels = wg_cache[cache_key]
        else:
            if not model_name:
                print(f"Warning: Could not detect model for {file_path}")
            if not dataset_name:
                print(f"Warning: Could not detect dataset for task {task_name} at {file_path}")

        data = load_predictions(file_path)

        for i, sample in enumerate(data):
            cot_preds = sample.get("token_predictions", [])
            keep_len = len(cot_preds)

            if args.cot_percentage < 100.0:
                keep_len = max(1, int(len(cot_preds) * (args.cot_percentage / 100.0)))
            
            if args.max_cot_tokens > 0:
                keep_len = min(keep_len, args.max_cot_tokens)

            if keep_len < len(cot_preds):
                sample["token_predictions"] = cot_preds[:keep_len]
                sample["num_cot_tokens"] = keep_len

            feats = process_sample(sample)
            feats["model"] = model_name or "unknown"
            feats["experiment_id"] = experiment_id
            feats["task"] = task_name
            
            # Incorporate WG labels if available
            feats["cot_label"] = cot_labels[i] if cot_labels and i < len(cot_labels) else -1
            feats["response_label"] = resp_labels[i] if resp_labels and i < len(resp_labels) else -1
            
            all_features.append(feats)

    # DataFrame creation
    df = pd.DataFrame(all_features)
    
    # Rename 'label' to 'prompt_label' for clarity
    df = df.rename(columns={"label": "prompt_label"})
    
    metadata_cols = ["model", "experiment_id", "sample_id", "task", "prompt_label", "cot_label", "response_label", "num_cot_tokens"]
    other_cols = sorted([c for c in df.columns if c not in metadata_cols])
    df = df[metadata_cols + other_cols]

    df.to_csv(args.output_csv, index=False)
    print(f"\nExtracted {len(df)} samples total.")
    print(f"Saved signal processing features to: {args.output_csv}")


if __name__ == "__main__":
    main()
