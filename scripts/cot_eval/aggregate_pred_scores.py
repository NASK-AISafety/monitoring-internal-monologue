import argparse
import json
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple metrics.json files into a single CSV"
    )
    parser.add_argument(
        "--metrics-files", nargs="+", required=True, help="List of metrics.json files"
    )
    parser.add_argument(
        "--output-csv", type=str, required=True, help="Path to save the aggregated CSV"
    )
    return parser.parse_args()


def flatten_metrics(metrics, prefix=""):
    """Recursively flatten a nested dictionary into a single level."""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat.update(flatten_metrics(v, prefix + k + "_"))
        else:
            flat[prefix + k] = v
    return flat


def main():
    args = parse_args()
    all_data = []

    for path_str in args.metrics_files:
        p = Path(path_str)
        if not p.exists():
            print(f"Warning: File not found: {path_str}")
            continue

        with open(p, "r") as f:
            metrics = json.load(f)

        flat_metrics = flatten_metrics(metrics)

        # Use the parent directory name as the experiment identifier
        flat_metrics["experiment"] = p.parent.name
        # Use the grandparent directory name as the model identifier
        flat_metrics["model"] = p.parent.parent.name

        all_data.append(flat_metrics)

    if all_data:
        df = pd.DataFrame(all_data)
        # Move 'experiment' and 'model' columns to the front
        cols = ["model", "experiment"] + [c for c in df.columns if c not in ["model", "experiment"]]
        df = df[cols]

        df.to_csv(args.output_csv, index=False)
        print(
            f"Aggregated metrics for {len(all_data)} experiments saved to {args.output_csv}"
        )
        print(df.to_markdown(index=False))
    else:
        print("No valid metrics files were found to aggregate.")


if __name__ == "__main__":
    main()
