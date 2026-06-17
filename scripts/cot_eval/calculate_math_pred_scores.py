import argparse
import json
import pandas as pd
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score,
)
from stats_utils import bootstrap_metrics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to math_probe_summary.csv"
    )
    parser.add_argument(
        "--output-json", type=str, required=True, help="Path to save metrics in JSON"
    )
    parser.add_argument(
        "--threshold-f1",
        type=float,
        default=0.5,
        help="Threshold for F1 score",
    )
    parser.add_argument(
        "--threshold-ba",
        type=float,
        default=0.5,
        help="Threshold for balanced accuracy",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data = pd.read_csv(args.data_path)
    # Drop rows without labels or predictions
    data = data.dropna(subset=["correctness", "cummax_last_token_prob"])

    # We want to predict incorrectness, so y_true is 1 when Incorrect
    y_true = (data["correctness"] == "Incorrect").astype(int)
    y_pred = data["cummax_last_token_prob"]

    y_pred = y_pred.fillna(0.0)

    # Calculate metrics, wrapping in conditional or try-except for edge cases
    # (e.g. only 1 class present)
    if y_true.nunique() > 1:
        metrics_dict = {
            "F1 Score": lambda y_t, y_p: f1_score(y_t, y_p > args.threshold_f1),
            "Balanced Accuracy": lambda y_t, y_p: balanced_accuracy_score(y_t, y_p > args.threshold_ba),
            "AUROC": lambda y_t, y_p: roc_auc_score(y_t, y_p),
            "Average Precision": lambda y_t, y_p: average_precision_score(y_t, y_p),
        }
        results = bootstrap_metrics(y_true.values, y_pred.values, metrics_dict)
    else:
        print("Warning: Only one class present in y_true. Setting metrics to None.")
        results = {
            "F1 Score": 0.0, "F1 Score_ci_lower": 0.0, "F1 Score_ci_upper": 0.0, "F1 Score_std": 0.0,
            "Balanced Accuracy": 0.0, "Balanced Accuracy_ci_lower": 0.0, "Balanced Accuracy_ci_upper": 0.0, "Balanced Accuracy_std": 0.0,
            "AUROC": None, "AUROC_ci_lower": None, "AUROC_ci_upper": None, "AUROC_std": None,
            "Average Precision": None, "Average Precision_ci_lower": None, "Average Precision_ci_upper": None, "Average Precision_std": None,
        }


    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Metrics saved to {args.output_json}")
    if args.verbose:
        print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
