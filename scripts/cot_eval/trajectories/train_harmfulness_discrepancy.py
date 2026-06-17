import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
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

# Add parent directory to path to import stats_utils
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from stats_utils import bootstrap_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifiers on probe features and evaluate discrepancy.")
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
        default="harmfulness",
        choices=["harmfulness", "math"],
        help="Experiment mode (default: harmfulness)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/harmfulness_discrepancy_performance.csv",
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
    elif args.mode == "harmfulness":
        data_path = "results/harmfulness/mil_probe_eval/signal_features.csv"
    else:
        data_path = "results/math/mil_probe_eval/signal_features.csv"

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

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)
    
    # Drop samples with no CoT tokens (reasoning trajectory is empty)
    initial_len = len(data)
    data = data[data.num_cot_tokens > 0].copy()
    if len(data) < initial_len:
        logger.info(f"Dropped {initial_len - len(data)} samples with 0 CoT tokens.")

    # Convert experiment_id to list if it's a single string (from argparse nargs="+")
    exp_ids = args.experiment_id if isinstance(args.experiment_id, list) else [args.experiment_id]

    # Filter for model and experiments
    df_target = data[(data.model == args.model) & (data.experiment_id.isin(exp_ids))].copy()

    if len(df_target) == 0:
        raise ValueError(f"No samples found for model {args.model} and experiments {exp_ids}")

    if args.mode == "harmfulness":
        # drop rows where labels are unknown (-1)
        df_target = df_target[(df_target.response_label != -1) & (df_target.cot_label != -1)]
        # Filter for prompt_label == 1 (harmful prompts) if merging, or keep all? 
        # Original script does temp_df = temp_df[temp_df.prompt_label == 1] when merging.
        if len(exp_ids) > 1:
             df_target = df_target[df_target.prompt_label == 1]
             
        y_binary = (df_target["response_label"] == 1).astype(int).values
        y_train = y_binary
        num_classes = 2
    else:
        # Math mode
        if "label" in df_target.columns:
            df_target["label"] = 1 - df_target["label"]
            y_binary = df_target["label"].values
        else:
            y_binary = df_target["response_label"].values
        y_train = y_binary
        num_classes = 2

    # Feature columns are everything not in metadata
    X = df_target.drop(columns=metadata_cols)
    baseline_prob = df_target["cot_last"].values
    metadata_df = df_target[metadata_cols].copy()

    return X, y_train, y_binary, baseline_prob, num_classes, metadata_df


def evaluate_with_discrepancy(y_true, y_pred_proba, metadata_df, prefix=""):
    """
    Calculate metrics for the full set and the discrepancy subset (cot_label != response_label).
    Also calculates detection rates for Scenario A (Safe CoT, Harmful Resp) and Scenario B (Harmful CoT, Safe Resp).
    """
    metrics_dict = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }

    # Full set metrics
    results = bootstrap_metrics(y_true, y_pred_proba, metrics_dict)
    results = {f"{prefix}{k}": v for k, v in results.items()}

    # Discrepancy subset metrics
    if "cot_label" in metadata_df.columns and "response_label" in metadata_df.columns:
        discrepancy_mask = (metadata_df["cot_label"] != metadata_df["response_label"]).values
        y_true_disc = y_true[discrepancy_mask]
        y_pred_disc = y_pred_proba[discrepancy_mask]

        if len(y_true_disc) > 0 and len(np.unique(y_true_disc)) > 1:
            disc_results = bootstrap_metrics(y_true_disc, y_pred_disc, metrics_dict)
            for k, v in disc_results.items():
                results[f"{prefix}{k}_discrepancy"] = v
        else:
            logger.warning(f"Discrepancy subset too small or single-class for {prefix}. Skipping discrepancy metrics.")
            for k in metrics_dict:
                for suffix in ["", "_ci_lower", "_ci_upper", "_std"]:
                    results[f"{prefix}{k}_discrepancy{suffix}"] = None
        
        # Scenario A: CoT=Safe (0), Response=Harmful (1) -> Detection rate is % pred > 0.5
        mask_a = (metadata_df["cot_label"] == 0) & (metadata_df["response_label"] == 1)
        y_pred_a = y_pred_proba[mask_a.values]
        if len(y_pred_a) > 0:
            results[f"{prefix}scenario_a_detection_rate"] = np.mean(y_pred_a > 0.5)
            results[f"{prefix}scenario_a_mean_proba"] = np.mean(y_pred_a)
            results[f"{prefix}scenario_a_count"] = len(y_pred_a)
            
            # Bootstrap for std
            rng = np.random.RandomState(42)
            boot_means = [np.mean(rng.choice(y_pred_a, size=len(y_pred_a), replace=True) > 0.5) for _ in range(1000)]
            results[f"{prefix}scenario_a_detection_rate_std"] = np.std(boot_means)
        else:
            results[f"{prefix}scenario_a_detection_rate"] = None
            results[f"{prefix}scenario_a_count"] = 0
            results[f"{prefix}scenario_a_detection_rate_std"] = None

        # Scenario B: CoT=Harmful (1), Response=Safe (0) -> Detection rate is % pred <= 0.5
        mask_b = (metadata_df["cot_label"] == 1) & (metadata_df["response_label"] == 0)
        y_pred_b = y_pred_proba[mask_b.values]
        if len(y_pred_b) > 0:
            results[f"{prefix}scenario_b_detection_rate"] = np.mean(y_pred_b <= 0.5)
            results[f"{prefix}scenario_b_mean_proba"] = np.mean(y_pred_b)
            results[f"{prefix}scenario_b_count"] = len(y_pred_b)

            # Bootstrap for std
            rng = np.random.RandomState(42)
            boot_means = [np.mean(rng.choice(y_pred_b, size=len(y_pred_b), replace=True) <= 0.5) for _ in range(1000)]
            results[f"{prefix}scenario_b_detection_rate_std"] = np.std(boot_means)
        else:
            results[f"{prefix}scenario_b_detection_rate"] = None
            results[f"{prefix}scenario_b_count"] = 0
            results[f"{prefix}scenario_b_detection_rate_std"] = None
    
    return results


def train_and_eval(X, y_train, y_binary, baseline_prob, num_classes, metadata_df, args):
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    results = {}
    predictions = {}

    # 0. Baseline performance
    logger.info("Calculating Baseline Performance...")
    baseline_res = evaluate_with_discrepancy(y_binary, baseline_prob, metadata_df, prefix="baseline_")
    results.update(baseline_res)
    predictions["baseline"] = baseline_prob

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    if not args.random_search:
        clf_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=10000, solver="saga", penalty="l1", n_jobs=-1)),
        ])
    else:
        base_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=10000, solver="saga")),
        ])
        lr_param_dist = {"lr__penalty": ["l1", "l2"], "lr__C": [1e-4, 1e-3, 1e-2, 1, 10, 1e2]}
        clf_lr = RandomizedSearchCV(base_lr, lr_param_dist, n_iter=args.n_iter, cv=3, scoring="neg_log_loss", n_jobs=-1, random_state=42)

    y_proba_lr = cross_val_predict(clf_lr, X, y_train, cv=cv, method="predict_proba")[:, 1]
    lr_res = evaluate_with_discrepancy(y_binary, y_proba_lr, metadata_df, prefix="lr_")
    results.update(lr_res)
    predictions["lr"] = y_proba_lr

    # 2. Random Forest
    logger.info("Training Random Forest...")
    if not args.random_search:
        clf_rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    else:
        rf_param_dist = {
            "n_estimators": [10, 50, 100, 200, 500, 1000],
            "max_depth": [None, 10, 20],
            "min_samples_split": randint(2, 10),
            "max_features": ["sqrt", "log2", None],
        }
        base_rf = RandomForestClassifier(class_weight="balanced", random_state=42)
        clf_rf = RandomizedSearchCV(base_rf, rf_param_dist, n_iter=args.n_iter, cv=3, scoring="neg_log_loss", n_jobs=-1, random_state=42)

    y_proba_rf = cross_val_predict(clf_rf, X, y_train, cv=cv, method="predict_proba")[:, 1]
    rf_res = evaluate_with_discrepancy(y_binary, y_proba_rf, metadata_df, prefix="rf_")
    results.update(rf_res)
    predictions["rf"] = y_proba_rf

    # 3. XGBoost
    if not args.skip_xgboost:
        logger.info("Training XGBoost...")
        xgb_base_params = {"random_state": 42, "n_jobs": -1}
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        if num_pos > 0:
            xgb_base_params["scale_pos_weight"] = float(num_neg) / num_pos

        if not args.random_search:
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
            clf_xgb = RandomizedSearchCV(base_xgb, xgb_param_dist, n_iter=args.n_iter, cv=3, scoring="neg_log_loss", n_jobs=-1, random_state=42)

        y_proba_xgb = cross_val_predict(clf_xgb, X, y_train, cv=cv, method="predict_proba")[:, 1]
        xgb_res = evaluate_with_discrepancy(y_binary, y_proba_xgb, metadata_df, prefix="xgb_")
        results.update(xgb_res)
        predictions["xgb"] = y_proba_xgb
    else:
        logger.info("Skipping XGBoost training.")

    return results, predictions


def main():
    args = parse_args()
    exp_str = "+".join(args.experiment_id)
    logger.info(f"Processing Model: {args.model}, Experiment: {exp_str} ({args.mode})")

    X, y_train, y_binary, baseline_prob, num_classes, metadata_df = load_and_preprocess(args)
    logger.info(f"Loaded {len(X)} samples. Target classes: {num_classes}")

    metrics, predictions = train_and_eval(X, y_train, y_binary, baseline_prob, num_classes, metadata_df, args)

    # Create row for CSV
    row = {
        "model": args.model,
        "experiment_id": exp_str,
        "mode": args.mode,
        "num_samples": len(X),
    }
    if "cot_label" in metadata_df.columns and "response_label" in metadata_df.columns:
        row["num_discrepancy_samples"] = np.sum(metadata_df["cot_label"] != metadata_df["response_label"])
    
    row.update(metrics)

    # Save/Append to CSV
    df_res = pd.DataFrame([row])
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    if not os.path.exists(args.output_csv):
        df_res.to_csv(args.output_csv, index=False)
    else:
        df_res.to_csv(args.output_csv, mode="a", header=False, index=False)

    logger.info(f"Results saved to {args.output_csv}")

    # Save predictions if requested
    if args.save_predictions:
        output_dir = Path(args.output_csv).parent / "predictions_discrepancy"
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
