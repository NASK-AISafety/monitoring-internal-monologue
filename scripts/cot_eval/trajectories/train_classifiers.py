import argparse
import logging
import os

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import sys
from pathlib import Path

# Add parent directory to path to import stats_utils
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from stats_utils import bootstrap_metrics



def parse_args():
    parser = argparse.ArgumentParser(description="Train classifiers on probe features.")
    parser.add_argument("--model", type=str, required=True, help="Model of interest")
    parser.add_argument(
        "--experiment_id",
        type=str,
        nargs="+",
        required=True,
        help="Experiment ID(s) of interest",
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
        default="results/classifier_performance_aggregation.csv",
        help="Output path for aggregated results",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help="Number of iterations for RandomizedSearchCV",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument(
        "--random_search",
        action="store_true",
        help="Enable random search for hyperparameter optimization",
    )
    parser.add_argument(
        "--four_class",
        action="store_true",
        help="Enable 4-class classification (S/S, H/H, H/S, S/H) in harmfulness mode. Default is binary on response_label.",
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default=None,
        help="Optional path to features CSV file. If not provided, uses defaults based on mode.",
    )
    parser.add_argument(
        "--skip_xgboost",
        action="store_true",
        help="Skip XGBoost training to save time",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save out-of-fold predictions to a CSV file",
    )
    return parser.parse_args()



def load_and_preprocess(args):
    if args.features_csv:
        data_path = args.features_csv
        if args.mode == "harmfulness":
            metadata_cols = [
                "model",
                "experiment_id",
                "sample_id",
                "task",
                "prompt_label",
                "cot_label",
                "response_label",
                "num_cot_tokens",
            ]
        else:
            metadata_cols = [
                "model",
                "experiment_id",
                "sample_id",
                "task",
                "label",
                "num_cot_tokens",
                "response_label",
            ]
    elif args.mode == "harmfulness":
        data_path = "results/harmfulness/mil_probe_eval/signal_features.csv"
        metadata_cols = [
            "model",
            "experiment_id",
            "sample_id",
            "task",
            "prompt_label",
            "cot_label",
            "response_label",
            "num_cot_tokens",
        ]
    else:
        data_path = "results/math/mil_probe_eval/signal_features.csv"
        metadata_cols = [
            "model",
            "experiment_id",
            "sample_id",
            "task",
            "label",
            "num_cot_tokens",
            "response_label",
        ]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)
    
    # Drop samples with no CoT tokens (reasoning trajectory is empty)
    initial_len = len(data)
    data = data[data.num_cot_tokens > 0].copy()
    if len(data) < initial_len:
        logger.info(f"Dropped {initial_len - len(data)} samples with 0 CoT tokens.")

    # Convert experiment_id to list if it's a single string (from argparse nargs="+")
    exp_ids = (
        args.experiment_id
        if isinstance(args.experiment_id, list)
        else [args.experiment_id]
    )

    if len(exp_ids) == 1:
        # Filter for model and experiment
        df_target = data[
            (data.model == args.model) & (data.experiment_id == exp_ids[0])
        ].copy()

        if len(df_target) == 0:
            raise ValueError(
                f"No samples found for model {args.model} and experiment {exp_ids[0]}"
            )

        if args.mode == "harmfulness":
            # drop rows where labels are unknown (-1)
            df_target = df_target[
                (df_target.response_label != -1) & (df_target.cot_label != -1)
            ]
            y_binary = (df_target["response_label"] == 1).astype(int).values
            if args.four_class:
                conditions = [
                    (df_target["cot_label"] == 0) & (df_target["response_label"] == 0),
                    (df_target["cot_label"] == 1) & (df_target["response_label"] == 1),
                    (df_target["cot_label"] == 1) & (df_target["response_label"] == 0),
                    (df_target["cot_label"] == 0) & (df_target["response_label"] == 1),
                ]
                y_train = np.select(conditions, [0, 1, 2, 3], default=-1)
                num_classes = 4
            else:
                y_train = y_binary
                num_classes = 2
        else:
            df_target["label"] = 1 - df_target["label"]
            y_binary = df_target["label"].values
            y_train = y_binary
            num_classes = 2

        X = df_target.drop(columns=metadata_cols)
        baseline_prob = df_target["cot_last"].values
        metadata_df = df_target[metadata_cols].copy()
    else:
        # Merging multiple experiments
        df_merged = None
        best_auc = -1.0
        baseline_prob = None

        for i, exp in enumerate(exp_ids):
            temp_df = data[
                (data.experiment_id == exp) & (data.model == args.model)
            ].copy()
            if len(temp_df) == 0:
                logger.warning(
                    f"No samples found for model {args.model} and experiment {exp}. Skipping."
                )
                continue

            if args.mode == "harmfulness":
                temp_df = temp_df[
                    (temp_df.response_label != -1) & (temp_df.cot_label != -1)
                ]
                temp_df = temp_df[temp_df.prompt_label == 1]
                temp_y = (temp_df["response_label"] == 1).astype(int).values
            else:
                temp_df["label"] = 1 - temp_df["label"]
                temp_y = temp_df["label"].values

            # Calculate baseline AUC for this specific experiment to find the best one
            current_auc = roc_auc_score(temp_y, temp_df["cot_last"])
            if current_auc > best_auc:
                best_auc = current_auc
                # We need to keep track of sample_id to align baseline_prob later
                best_baseline_df = temp_df[["sample_id", "cot_last"]].copy()

            # Rename features
            feat_cols = [c for c in temp_df.columns if c not in metadata_cols]
            rename_dict = {c: f"{c}_exp{i+1}" for c in feat_cols}
            temp_df = temp_df.rename(columns=rename_dict)

            if df_merged is None:
                df_merged = temp_df
            else:
                cols_to_join = list(rename_dict.values()) + ["sample_id"]
                df_merged = pd.merge(df_merged, temp_df[cols_to_join], on="sample_id")

        if df_merged is None:
            raise ValueError(
                f"No samples found for model {args.model} across experiments {exp_ids}"
            )

        # Extract y and target classes from the merged dataframe
        if args.mode == "harmfulness":
            y_binary = (df_merged["response_label"] == 1).astype(int).values
            if args.four_class:
                conditions = [
                    (df_merged["cot_label"] == 0) & (df_merged["response_label"] == 0),
                    (df_merged["cot_label"] == 1) & (df_merged["response_label"] == 1),
                    (df_merged["cot_label"] == 1) & (df_merged["response_label"] == 0),
                    (df_merged["cot_label"] == 0) & (df_merged["response_label"] == 1),
                ]
                y_train = np.select(conditions, [0, 1, 2, 3], default=-1)
                num_classes = 4
            else:
                y_train = y_binary
                num_classes = 2
        else:
            y_binary = df_merged["label"].values
            y_train = y_binary
            num_classes = 2

        # Extract X (drop metadata columns, handling possible duplicates from merge)
        X = df_merged.drop(
            columns=[c for c in df_merged.columns if any(m == c for m in metadata_cols)]
        )

        # Align baseline_prob from the best experiment
        baseline_df_aligned = pd.merge(
            df_merged[["sample_id"]], best_baseline_df, on="sample_id", how="left"
        )
        baseline_prob = baseline_df_aligned["cot_last"].values
        
        # Metadata for merged: we only keep what's common or from first
        metadata_df = df_merged[[c for m in metadata_cols for c in df_merged.columns if m == c]].copy()

    return X, y_train, y_binary, baseline_prob, num_classes, metadata_df



def evaluate_binarized(y_binary, y_pred_proba, num_classes):
    """
    Convert multi-class probabilities to binary response probabilities if needed.
    """
    if num_classes == 4:
        # P(Harmful Response) = P(H/H) + P(S/H) = P(1) + P(3)
        y_proba_bin = y_pred_proba[:, 1] + y_pred_proba[:, 3]
    else:
        y_proba_bin = y_pred_proba[:, 1]

    metrics = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }
    
    results = bootstrap_metrics(y_binary, y_proba_bin, metrics)
    return results, y_proba_bin



def train_and_eval(X, y_train, y_binary, baseline_prob, num_classes, args):
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    results = {}
    predictions = {}

    # Baseline performance
    logger.info("Calculating Baseline Performance...")
    baseline_metrics = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }
    baseline_res = bootstrap_metrics(y_binary, baseline_prob, baseline_metrics)
    for k, v in baseline_res.items():
        results[f"baseline_{k}"] = v
    predictions["baseline"] = baseline_prob

    # 1. Logistic Regression
    # ... (rest of model setup stays same)
    if not args.random_search:
        clf_lr = Pipeline(
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
    else:
        base_lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        class_weight="balanced", max_iter=10000, solver="saga"
                    ),
                ),
            ]
        )
        lr_param_dist = {
            "lr__penalty": ["l1", "l2"],
            "lr__C": [1e-4, 1e-3, 1e-2, 1, 10, 1e2],
        }
        clf_lr = RandomizedSearchCV(
            base_lr,
            lr_param_dist,
            n_iter=args.n_iter,
            cv=3,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=42,
        )

    logger.info("Training Logistic Regression...")
    y_proba_lr = cross_val_predict(clf_lr, X, y_train, cv=cv, method="predict_proba")
    lr_res, y_proba_lr_bin = evaluate_binarized(y_binary, y_proba_lr, num_classes)
    for k, v in lr_res.items():
        results[f"lr_{k}"] = v
    predictions["lr"] = y_proba_lr_bin

    # 2. Random Forest
    if not args.random_search:
        clf_rf = RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=42, n_jobs=-1
        )
    else:
        rf_param_dist = {
            "n_estimators": [10, 50, 100, 200, 500, 1000],
            "max_depth": [None, 10, 20],
            "min_samples_split": randint(2, 10),
            "max_features": ["sqrt", "log2", None],
        }
        base_rf = RandomForestClassifier(class_weight="balanced", random_state=42)
        clf_rf = RandomizedSearchCV(
            base_rf,
            rf_param_dist,
            n_iter=args.n_iter,
            cv=3,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=42,
        )

    logger.info("Training Random Forest...")
    y_proba_rf = cross_val_predict(clf_rf, X, y_train, cv=cv, method="predict_proba")
    rf_res, y_proba_rf_bin = evaluate_binarized(y_binary, y_proba_rf, num_classes)
    for k, v in rf_res.items():
        results[f"rf_{k}"] = v
    predictions["rf"] = y_proba_rf_bin

    # 3. XGBoost
    if not args.skip_xgboost:
        xgb_base_params = {"random_state": 42}
        if num_classes > 2:
            xgb_base_params["objective"] = "multi:softprob"
            xgb_base_params["num_class"] = num_classes

        # For binary XGBoost, we might want scale_pos_weight if not 4-class
        if num_classes == 2:
            num_neg = np.sum(y_train == 0)
            num_pos = np.sum(y_train == 1)
            if num_pos > 0:
                xgb_base_params["scale_pos_weight"] = float(num_neg) / num_pos

        if not args.random_search:
            xgb_base_params["n_jobs"] = -1
            clf_xgb = XGBClassifier(**xgb_base_params)
        else:
            xgb_param_dist = {
                "n_estimators": [10, 50, 100, 200, 500, 1000],
                "learning_rate": [0.01, 0.03, 0.1, 0.3],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }
            base_xgb = XGBClassifier(**xgb_base_params)
            clf_xgb = RandomizedSearchCV(
                base_xgb,
                xgb_param_dist,
                n_iter=args.n_iter,
                cv=3,
                scoring="neg_log_loss",
                n_jobs=-1,
                random_state=42,
            )

        logger.info("Training XGBoost...")
        y_proba_xgb = cross_val_predict(clf_xgb, X, y_train, cv=cv, method="predict_proba")
        xgb_res, y_proba_xgb_bin = evaluate_binarized(y_binary, y_proba_xgb, num_classes)
        for k, v in xgb_res.items():
            results[f"xgb_{k}"] = v
        predictions["xgb"] = y_proba_xgb_bin
    else:
        logger.info("Skipping XGBoost training as requested.")
        for m in ["auc", "ap", "f1"]:
            for suffix in ["", "_ci_lower", "_ci_upper", "_std"]:
                results[f"xgb_{m}{suffix}"] = None

    return results, predictions



def main():
    args = parse_args()
    exp_str = (
        "+".join(args.experiment_id)
        if isinstance(args.experiment_id, list)
        else args.experiment_id
    )
    logger.info(f"Processing Model: {args.model}, Experiment: {exp_str} ({args.mode})")

    X, y_train, y_binary, baseline_prob, num_classes, metadata_df = load_and_preprocess(args)
    logger.info(f"Loaded {len(X)} samples. Target classes: {num_classes}")

    metrics, predictions = train_and_eval(X, y_train, y_binary, baseline_prob, num_classes, args)

    # Create row for CSV
    row = {
        "model": args.model,
        "experiment_id": exp_str,
        "mode": args.mode,
        "num_samples": len(X),
    }
    row.update(metrics)

    # Save/Append to CSV
    df_res = pd.DataFrame([row])
    if not os.path.exists(args.output_csv):
        df_res.to_csv(args.output_csv, index=False)
    else:
        # Load existing to avoid duplicate columns if header changed, but here we just append
        df_res.to_csv(args.output_csv, mode="a", header=False, index=False)

    logger.info(f"Results saved to {args.output_csv}")

    # Save predictions if requested
    if args.save_predictions:
        output_dir = Path(args.output_csv).parent / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pred_df = metadata_df.copy()
        pred_df["y_true"] = y_binary
        for clf_name, y_proba in predictions.items():
            pred_df[f"{clf_name}_proba"] = y_proba
        
        pred_file = output_dir / f"{args.model}_{exp_str.replace('+', '_')}_{args.mode}_preds.csv"
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved to {pred_file}")



if __name__ == "__main__":
    main()
