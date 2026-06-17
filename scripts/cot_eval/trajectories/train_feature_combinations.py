import argparse
import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from probe_utils import FEATURE_GROUPS
from stats_utils import bootstrap_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train classifiers on feature combinations."
    )
    parser.add_argument("--model", type=str, required=True, help="Model of interest")
    parser.add_argument(
        "--experiment_id", type=str, required=True, help="Experiment ID of interest"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["harmfulness", "math"],
        required=True,
        help="Experiment mode",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/feature_combination_performance.csv",
        help="Output path for aggregated results",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["lr", "xgb", "rf"],
        default="lr",
        help="Classifier to use for combinations",
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default=None,
        help="Optional path to features CSV file",
    )
    return parser.parse_args()


def load_data(args):
    if args.features_csv:
        data_path = args.features_csv
    elif args.mode == "harmfulness":
        data_path = "results/harmfulness/mil_probe_eval/signal_features.csv"
    else:
        data_path = "results/math/mil_probe_eval/signal_features.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)
    df_target = data[
        (data.model == args.model) & (data.experiment_id == args.experiment_id)
    ].copy()

    if len(df_target) == 0:
        raise ValueError(
            f"No samples found for model {args.model} and experiment {args.experiment_id}"
        )

    if args.mode == "harmfulness":
        # drop rows where labels are unknown (-1)
        df_target = df_target[
            (df_target.response_label != -1) & (df_target.cot_label != -1)
        ]
        y_binary = (df_target["response_label"] == 1).astype(int).values
    else:
        df_target["label"] = 1 - df_target["label"]
        y_binary = df_target["label"].values

    return df_target, y_binary


def train_and_eval(X, y, classifier_type, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if classifier_type == "lr":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=10000,
                        solver="saga",
                        penalty="l1",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    elif classifier_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=42, n_jobs=-1
        )
    else:
        num_neg = np.sum(y == 0)
        num_pos = np.sum(y == 1)
        scale_pos_weight = float(num_neg) / num_pos if num_pos > 0 else 1.0
        clf = XGBClassifier(
            random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight
        )

    y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    metrics = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }

    return bootstrap_metrics(y, y_proba, metrics)


def main():
    args = parse_args()
    logger.info(
        f"Training combinations for {args.model} - {args.experiment_id} ({args.mode})"
    )

    df, y = load_data(args)

    group_names = list(FEATURE_GROUPS.keys())
    results = []

    # Generate all combinations (from size 1 to 6)
    for r in range(1, len(group_names) + 1):
        for combo in itertools.combinations(group_names, r):
            combo_name = "+".join(combo)
            logger.info(f"Evaluating combination: {combo_name}")

            # Extract features for this combination
            selected_features = []
            for g in combo:
                selected_features.extend(FEATURE_GROUPS[g])

            # Check if all features exist in df
            missing = [f for f in selected_features if f not in df.columns]
            if missing:
                logger.warning(f"Missing features in CSV: {missing}. Skipping them.")
                selected_features = [f for f in selected_features if f in df.columns]

            if not selected_features:
                logger.warning(f"No features found for combo {combo_name}. Skipping.")
                continue

            X = df[selected_features].values

            metrics = train_and_eval(X, y, args.classifier, args.n_splits)

            res_row = {
                "model": args.model,
                "experiment_id": args.experiment_id,
                "mode": args.mode,
                "classifier": args.classifier,
                "combination": combo_name,
                "num_groups": r,
                "num_features": len(selected_features),
            }
            res_row.update(metrics)
            results.append(res_row)

    df_results = pd.DataFrame(results)

    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if not os.path.exists(args.output_csv):
        df_results.to_csv(args.output_csv, index=False)
    else:
        df_results.to_csv(args.output_csv, mode="a", header=False, index=False)

    logger.info(f"Saved {len(results)} combination results to {args.output_csv}")


if __name__ == "__main__":
    main()
