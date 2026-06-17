import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import load_predictions

from probe_utils import process_sample


def main():
    parser = argparse.ArgumentParser(
        description="Extract signal processing features from cummax files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/math/mil_probe_eval"),
        help="Base directory containing model evaluations.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/math/mil_probe_eval/signal_features.csv"),
        help="Output CSV file for the features.",
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
        print(
            f"No *_pool_over_prompt/cot_mil_probe_preds_cummax.json/parquet files found in {args.base_dir}"
        )
        return

    all_features = []

    for file_path in tqdm(all_files, desc="Processing files"):
        parent_dir_name = file_path.parent.name
        model_name = file_path.parent.parent.name

        data = load_predictions(file_path)

        for sample in data:
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
            feats["model"] = model_name
            feats["experiment_id"] = parent_dir_name
            all_features.append(feats)

    # Convert to dataframe and save
    df = pd.DataFrame(all_features)
    
    # Metadata columns ordering
    metadata_cols = ["model", "experiment_id", "sample_id", "task", "label", "num_cot_tokens"]
    other_cols = sorted([c for c in df.columns if c not in metadata_cols])
    df = df[metadata_cols + other_cols]

    df.to_csv(args.output_csv, index=False)
    print(f"\nExtracted {len(df)} samples total.")
    print(f"Saved signal processing features to: {args.output_csv}")


if __name__ == "__main__":
    main()
