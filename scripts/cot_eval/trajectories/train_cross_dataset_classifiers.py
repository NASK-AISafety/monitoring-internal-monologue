import argparse
import logging
import os
import hashlib
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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures
from xgboost import XGBClassifier
from scipy.stats import rankdata

# Add parent directory to path to import stats_utils
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from stats_utils import bootstrap_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="Train classifiers on one dataset and evaluate on another.")
    parser.add_argument("--model", type=str, required=True, help="Model of interest")
    parser.add_argument(
        "--experiment_id",
        nargs="+",
        help="Experiment IDs to load from signal_features.csv. If multiple, they are combined.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["harmfulness", "math"],
        help="Experiment mode (optional if --scenario is used)",
    )
    parser.add_argument(
        "--train_task_prefix",
        type=str,
        help="Task prefix for training data (optional if --scenario is used)",
    )
    parser.add_argument(
        "--eval_task_prefix",
        type=str,
        help="Task prefix for evaluation data (optional if --scenario is used)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["wildguard_to_aegis", "gsm8k_to_minerva"],
        help="Predefined scenario for prefixes",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/cross_dataset_classifier_performance.csv",
        help="Output path for aggregated results",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help="Number of iterations for RandomizedSearchCV",
    )
    parser.add_argument(
        "--random_search",
        action="store_true",
        help="Enable random search for hyperparameter optimization",
    )
    parser.add_argument(
        "--four_class",
        action="store_true",
        help="Enable 4-class classification (S/S, H/H, H/S, S/H) in harmfulness mode.",
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default=None,
        help="Optional path to features CSV file.",
    )
    parser.add_argument(
        "--skip_xgboost",
        action="store_true",
        help="Skip XGBoost training",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to a CSV file",
    )
    parser.add_argument(
        "--align_distributions",
        action="store_true",
        help="Use unsupervised domain adaptation by scaling eval set with its own mean/var.",
    )
    parser.add_argument(
        "--drop_non_robust",
        action="store_true",
        help="Drop features that are known to be highly dataset-specific (e.g. length-related).",
    )
    parser.add_argument(
        "--use_robust_scaler",
        action="store_true",
        help="Use RobustScaler instead of StandardScaler.",
    )
    parser.add_argument(
        "--use_quantile_transformer",
        action="store_true",
        help="Use QuantileTransformer to map features to a normal distribution (very robust to shift).",
    )
    parser.add_argument(
        "--use_poly",
        action="store_true",
        help="Use degree-2 polynomial features on key reasoning features.",
    )
    parser.add_argument(
        "--stacking",
        action="store_true",
        help="Train classifier ONLY on dynamics (excludes direct probe summaries) and stack with baseline.",
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
    exp_ids = args.experiment_id if isinstance(args.experiment_id, list) else [args.experiment_id]

    # Filter for model and experiment(s)
    df_filtered = data[(data.model == args.model) & (data.experiment_id.isin(exp_ids))].copy()

    # Drop samples with negative num_cot_tokens (usually error indicators)
    n = len(df_filtered)
    df_filtered = df_filtered[df_filtered.num_cot_tokens >= 0]
    logger.info(f"Dropped {n - len(df_filtered)} samples with negative num_cot_tokens (out of {n} total)")

    if len(df_filtered) == 0:
        raise ValueError(f"No samples found for model {args.model} and experiments {exp_ids} (after filtering num_cot_tokens)")

    if args.mode == "harmfulness":
        # drop rows where labels are unknown (-1)
        df_filtered = df_filtered[
            (df_filtered.response_label != -1) & (df_filtered.cot_label != -1)
        ]

        df_filtered["target"] = (df_filtered["response_label"] == 1).astype(int)

        if args.four_class:
            conditions = [
                (df_filtered["cot_label"] == 0) & (df_filtered["response_label"] == 0),
                (df_filtered["cot_label"] == 1) & (df_filtered["response_label"] == 1),
                (df_filtered["cot_label"] == 1) & (df_filtered["response_label"] == 0),
                (df_filtered["cot_label"] == 0) & (df_filtered["response_label"] == 1),
            ]
            df_filtered["y_train"] = np.select(conditions, [0, 1, 2, 3], default=-1)
            num_classes = 4
        else:
            df_filtered["y_train"] = df_filtered["target"]
            num_classes = 2
    else:
        # In math mode, the original label is usually "is_correct", but signal_features might have it inverted
        # The original script does: df_target["label"] = 1 - df_target["label"]
        df_filtered["label"] = 1 - df_filtered["label"]
        df_filtered["target"] = df_filtered["label"]
        df_filtered["y_train"] = df_filtered["target"]
        num_classes = 2

    # Split into train and eval based on task prefix
    df_train = df_filtered[df_filtered.task.str.startswith(args.train_task_prefix)].copy()
    df_eval = df_filtered[df_filtered.task.str.startswith(args.eval_task_prefix)].copy()

    if len(df_train) == 0:
        raise ValueError(f"No samples found for training task prefix: {args.train_task_prefix}")
    if len(df_eval) == 0:
        raise ValueError(f"No samples found for evaluation task prefix: {args.eval_task_prefix}")

    logger.info(f"Train samples: {len(df_train)}, Eval samples: {len(df_eval)}")

    X_train = df_train.drop(columns=metadata_cols + ["target", "y_train"], errors="ignore")
    y_train = df_train["y_train"].values

    X_eval = df_eval.drop(columns=metadata_cols + ["target", "y_train"], errors="ignore")
    y_eval_binary = df_eval["target"].values

    baseline_prob_eval = df_eval["cot_last"].values
    metadata_eval = df_eval[metadata_cols].copy()

    if args.drop_non_robust:
        non_robust = [
            "cot_to_prompt_length_ratio",
            "num_cot_tokens",
            "cot_autocorr_lag1",
            "boundary_volatility",
            "terminal_derivative_smoothed",
            "cot_max_consecutive_70",
            "boundary_max_spike",
            "surge_speed_first_5_percent",
            "boundary_max_dip",
            "cot_mean_crossing_rate",
        ]
        logger.info(f"Dropping non-robust features: {non_robust}")
        X_train = X_train.drop(columns=non_robust, errors="ignore")
        X_eval = X_eval.drop(columns=non_robust, errors="ignore")

    if args.stacking:
        # Exclude direct probe value summaries
        direct_probe_features = [
            "cot_last", "cot_mean", "cot_max", "cot_min", "cot_median", "cot_rms",
            "prompt_last", "prompt_mean", "prompt_max", "prompt_min", "prompt_median", "prompt_rms",
            "cot_first", "prompt_first", "cot_tertile_1_mean", "cot_tertile_2_mean", "cot_tertile_3_mean"
        ]
        logger.info(f"Stacking mode: Excluding direct probe features: {direct_probe_features}")
        X_train = X_train.drop(columns=direct_probe_features, errors="ignore")
        X_eval = X_eval.drop(columns=direct_probe_features, errors="ignore")

    return X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval, num_classes, metadata_eval



def evaluate_binarized(y_binary, y_pred_proba, num_classes):
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



def train_and_eval(X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval, num_classes, args):
    results = {}
    predictions = {}

    # Baseline performance on eval set
    logger.info("Calculating Baseline Performance on Eval Set...")
    baseline_metrics = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }
    baseline_res = bootstrap_metrics(y_eval_binary, baseline_prob_eval, baseline_metrics)
    for k, v in baseline_res.items():
        results[f"baseline_{k}"] = v
    predictions["baseline"] = baseline_prob_eval

    if args.use_quantile_transformer:
        scaler_cls = lambda: QuantileTransformer(output_distribution="normal", random_state=42)
    elif args.use_robust_scaler:
        scaler_cls = RobustScaler
    else:
        scaler_cls = StandardScaler

    # 1. Logistic Regression
    lr_steps = [("scaler", scaler_cls())]
    if args.use_poly:
        lr_steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True)))

    if not args.random_search:
        lr_steps.append(("lr", LogisticRegression(class_weight="balanced", max_iter=10000, solver="saga", penalty="l2", C=0.01, n_jobs=-1))) # Even stronger L2
        clf_lr = Pipeline(lr_steps)
    else:
        base_lr = Pipeline(lr_steps + [("lr", LogisticRegression(class_weight="balanced", max_iter=10000, solver="saga"))])
        lr_param_dist = {"lr__penalty": ["l1", "l2"], "lr__C": [1e-4, 1e-3, 0.01, 0.1, 1, 10]}
        clf_lr = RandomizedSearchCV(base_lr, lr_param_dist, n_iter=args.n_iter, cv=3, scoring="neg_log_loss", n_jobs=-1, random_state=42)

    logger.info("Training Logistic Regression...")
    clf_lr.fit(X_train, y_train)

    if args.align_distributions:
        logger.info("Aligning distributions by re-scaling X_eval with its own stats...")
        # We manually scale X_eval using its own scaler to match the N(0,1) assumption of the trained model
        X_eval_proc = scaler_cls().fit_transform(X_eval)
        # We need to bypass the pipeline's scaler for prediction
        y_proba_lr = clf_lr.named_steps["lr"].predict_proba(X_eval_proc)
    else:
        y_proba_lr = clf_lr.predict_proba(X_eval)

    lr_res, y_proba_lr_bin = evaluate_binarized(y_eval_binary, y_proba_lr, num_classes)
    for k, v in lr_res.items():
        results[f"lr_{k}"] = v
    predictions["lr"] = y_proba_lr_bin

    # 2. Random Forest
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

    logger.info("Training Random Forest...")
    clf_rf.fit(X_train, y_train)

    if args.align_distributions:
        X_eval_rf = X_eval # RF doesn't need scaling, but let's keep it consistent
        y_proba_rf = clf_rf.predict_proba(X_eval_rf)
    else:
        y_proba_rf = clf_rf.predict_proba(X_eval)

    rf_res, y_proba_rf_bin = evaluate_binarized(y_eval_binary, y_proba_rf, num_classes)
    for k, v in rf_res.items():
        results[f"rf_{k}"] = v
    predictions["rf"] = y_proba_rf_bin

    # 3. XGBoost
    if not args.skip_xgboost:
        xgb_base_params = {"random_state": 42}
        if num_classes > 2:
            xgb_base_params["objective"] = "multi:softprob"
            xgb_base_params["num_class"] = num_classes
        elif num_classes == 2:
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
            clf_xgb = RandomizedSearchCV(base_xgb, xgb_param_dist, n_iter=args.n_iter, cv=3, scoring="neg_log_loss", n_jobs=-1, random_state=42)

        logger.info("Training XGBoost...")
        clf_xgb.fit(X_train, y_train)

        if args.align_distributions:
            X_eval_xgb = X_eval # XGBoost is also tree-based
            y_proba_xgb = clf_xgb.predict_proba(X_eval_xgb)
        else:
            y_proba_xgb = clf_xgb.predict_proba(X_eval)

        xgb_res, y_proba_xgb_bin = evaluate_binarized(y_eval_binary, y_proba_xgb, num_classes)
        for k, v in xgb_res.items():
            results[f"xgb_{k}"] = v
        predictions["xgb"] = y_proba_xgb_bin
    else:
        logger.info("Skipping XGBoost training.")

    # 4. Ensemble (Rank-based Weighted Ensemble of best classifier and baseline)
    # In stacking mode, this is 0.5 * baseline + 0.5 * TrajectoryClassifer
    best_clf_prob = y_proba_rf_bin if rf_res["auc"] > lr_res["auc"] else y_proba_lr_bin

    # Standardize to ranks to handle different scales
    rank_clf = rankdata(best_clf_prob)
    rank_base = rankdata(baseline_prob_eval)

    # Simple averaging of ranks (stacking trick)
    y_proba_ensemble = (rank_clf + rank_base) / (2.0 * len(rank_clf))

    ensemble_res = bootstrap_metrics(y_eval_binary, y_proba_ensemble, baseline_metrics)
    for k, v in ensemble_res.items():
        results[f"ensemble_{k}"] = v
    predictions["ensemble"] = y_proba_ensemble

    return results, predictions



def main():
    args = parse_args()

    if args.scenario == "wildguard_to_aegis":
        args.train_task_prefix = "wildguardtest"
        args.eval_task_prefix = "aegis"
        args.mode = "harmfulness"
        if not args.experiment_id:
            args.experiment_id = [
                "wildguardtest_prompt_and_response_pool_over_prompt",
                "aegis_test_prompt_and_response_pool_over_prompt"
            ]
    elif args.scenario == "gsm8k_to_minerva":
        args.train_task_prefix = "gsm8k"
        args.eval_task_prefix = "minerva"
        args.mode = "math"
        if not args.experiment_id:
            args.experiment_id = [
                "gsm8k_prompt_and_response_pool_over_prompt",
                "minerva_math_prompt_and_response_pool_over_prompt"
            ]

    if not args.mode or not args.train_task_prefix or not args.eval_task_prefix:
        logger.error("Must provide either --scenario or all of (--mode, --train_task_prefix, --eval_task_prefix)")
        sys.exit(1)

    exp_str = "+".join(args.experiment_id)
    logger.info(f"Cross-Dataset Eval | Model: {args.model}, Experiment: {exp_str} ({args.mode})")
    logger.info(f"Train: {args.train_task_prefix} -> Eval: {args.eval_task_prefix}")

    X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval, num_classes, metadata_eval = load_and_preprocess(args)

    metrics, predictions = train_and_eval(X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval, num_classes, args)

    # Create row for CSV
    row = {
        "model": args.model,
        "experiment_id": exp_str,
        "mode": args.mode,
        "train_task_prefix": args.train_task_prefix,
        "eval_task_prefix": args.eval_task_prefix,
        "num_train_samples": len(X_train),
        "num_eval_samples": len(X_eval),
    }
    row.update(metrics)

    # Save/Append to CSV
    df_res = pd.DataFrame([row])
    if not os.path.exists(args.output_csv):
        df_res.to_csv(args.output_csv, index=False)
    else:
        # Load existing to handle potential column differences
        existing_df = pd.read_csv(args.output_csv)
        df_res = pd.concat([existing_df, df_res], ignore_index=True)
        df_res.to_csv(args.output_csv, index=False)

    logger.info(f"Results saved to {args.output_csv}")

    if args.save_predictions:
        output_dir = Path(args.output_csv).parent / "cross_dataset_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_df = metadata_eval.copy()
        pred_df["y_true"] = y_eval_binary
        for clf_name, y_proba in predictions.items():
            pred_df[f"{clf_name}_proba"] = y_proba

        # Use a hash of exp_str to avoid "File name too long" errors
        exp_hash = hashlib.md5(exp_str.encode()).hexdigest()[:8]
        scenario_tag = args.scenario if args.scenario else f"{args.train_task_prefix}_to_{args.eval_task_prefix}"
        pred_file = output_dir / f"{args.model}_{scenario_tag}_{exp_hash}_preds.csv"
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    main()
