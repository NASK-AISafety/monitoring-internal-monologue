import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import rankdata, randint

# Add parent directory to path for stats_utils
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from stats_utils import bootstrap_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MINERVA_CATEGORIES = [
    "algebra",
    "counting_and_prob",
    "geometry",
    "intermediate_algebra",
    "num_theory",
    "prealgebra",
    "precalc"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Simplified Leave-one-out cross-category training on Minerva.")
    parser.add_argument("--model", type=str, required=True, help="Model of interest")
    parser.add_argument("--eval_category", type=str, choices=MINERVA_CATEGORIES, required=True, help="Category to hold out for evaluation")
    parser.add_argument("--output_csv", type=str, default="results/minerva_loo_performance.csv", help="Output results file")
    parser.add_argument("--features_csv", type=str, default="results/math/mil_probe_eval/signal_features.csv", help="Path to signal_features.csv")
    parser.add_argument("--stacking", action="store_true", help="Excludes direct probe summaries from features")
    parser.add_argument("--drop_non_robust", action="store_true", help="Drop length-related features")
    parser.add_argument("--skip_xgboost", action="store_true", help="Skip XGBoost training")
    return parser.parse_args()

def load_and_preprocess(args):
    data = pd.read_csv(args.features_csv)

    # Drop samples with no CoT tokens (reasoning trajectory is empty)
    data = data[data.num_cot_tokens > 0].copy()

    # Use standard experiment IDs
    exp_ids = [
        "minerva_math_prompt_and_response_pool_over_prompt",
        # "minerva_math_response_only_pool_over_prompt"
    ]
    df_filtered = data[(data.model == args.model) & (data.experiment_id.isin(exp_ids))].copy()

    # Flip labels (Math mode: 1 is incorrect)
    df_filtered["label"] = 1 - df_filtered["label"]
    df_filtered["target"] = df_filtered["label"]
    df_filtered["y_train"] = df_filtered["target"]

    eval_task = f"minerva_math_{args.eval_category}"
    train_categories = [c for c in MINERVA_CATEGORIES if c != args.eval_category]
    train_tasks_prefixes = [f"minerva_math_{c}" for c in train_categories]

    df_eval = df_filtered[df_filtered.task == eval_task].copy()
    df_train = df_filtered[df_filtered.task.str.startswith(tuple(train_tasks_prefixes))].copy()

    if len(df_eval) == 0:
        logger.warning(f"No samples found for category {args.eval_category}. Skipping.")
        return None

    logger.info(f"LOO Category: {args.eval_category}")
    logger.info(f"Train samples: {len(df_train)}, Eval samples: {len(df_eval)}")

    metadata_cols = ["model", "experiment_id", "sample_id", "task", "label", "num_cot_tokens", "response_label", "target", "y_train"]
    X_train = df_train.drop(columns=metadata_cols, errors="ignore")
    y_train = df_train["y_train"].values
    X_eval = df_eval.drop(columns=metadata_cols, errors="ignore")
    y_eval_binary = df_eval["target"].values
    baseline_prob_eval = df_eval["cot_last"].values

    if args.drop_non_robust:
        non_robust = ["cot_to_prompt_length_ratio", "num_cot_tokens", "cot_autocorr_lag1", "boundary_volatility", "terminal_derivative_smoothed", "cot_max_consecutive_70", "boundary_max_spike", "surge_speed_first_5_percent", "boundary_max_dip", "cot_mean_crossing_rate"]
        X_train = X_train.drop(columns=non_robust, errors="ignore")
        X_eval = X_eval.drop(columns=non_robust, errors="ignore")

    if args.stacking:
        direct_probe_features = ["cot_last", "cot_mean", "cot_max", "cot_min", "cot_median", "cot_rms", "prompt_last", "prompt_mean", "prompt_max", "prompt_min", "prompt_median", "prompt_rms", "cot_first", "prompt_first", "cot_tertile_1_mean", "cot_tertile_2_mean", "cot_tertile_3_mean"]
        X_train = X_train.drop(columns=direct_probe_features, errors="ignore")
        X_eval = X_eval.drop(columns=direct_probe_features, errors="ignore")

    return X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval

def evaluate_binarized(y_binary, y_pred_proba):
    y_proba_bin = y_pred_proba[:, 1]
    metrics_dict = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }
    results = bootstrap_metrics(y_binary, y_proba_bin, metrics_dict)
    return results, y_proba_bin

def train_and_eval(X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval, args):
    results = {}

    # Baseline
    metrics_dict = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }
    baseline_res = bootstrap_metrics(y_eval_binary, baseline_prob_eval, metrics_dict)
    for k, v in baseline_res.items(): results[f"baseline_{k}"] = v

    # 1. Logistic Regression (No Scaling)
    logger.info("Training Logistic Regression...")
    clf_lr = LogisticRegression(class_weight="balanced", max_iter=10000, solver="lbfgs", C=0.01) # LBFGS is robust without scaling for small feature sets
    clf_lr.fit(X_train, y_train)
    y_proba_lr = clf_lr.predict_proba(X_eval)
    lr_res, y_proba_lr_bin = evaluate_binarized(y_eval_binary, y_proba_lr)
    for k, v in lr_res.items(): results[f"lr_{k}"] = v

    # 2. Random Forest
    logger.info("Training Random Forest...")
    clf_rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    y_proba_rf = clf_rf.predict_proba(X_eval)
    rf_res, y_proba_rf_bin = evaluate_binarized(y_eval_binary, y_proba_rf)
    for k, v in rf_res.items(): results[f"rf_{k}"] = v

    # 3. XGBoost
    if not args.skip_xgboost:
        logger.info("Training XGBoost...")
        clf_xgb = XGBClassifier(random_state=42, n_jobs=-1)
        clf_xgb.fit(X_train, y_train)
        y_proba_xgb = clf_xgb.predict_proba(X_eval)
        xgb_res, y_proba_xgb_bin = evaluate_binarized(y_eval_binary, y_proba_xgb)
        for k, v in xgb_res.items(): results[f"xgb_{k}"] = v
    else:
        xgb_res = {"auc": 0}

    # Ensemble (Rank-based)
    logger.info("Calculating Ensemble...")
    # Best classifier from RF and LR
    best_clf_prob = y_proba_rf_bin if rf_res["auc"] > lr_res["auc"] else y_proba_lr_bin
    rank_clf = rankdata(best_clf_prob)
    rank_base = rankdata(baseline_prob_eval)
    y_proba_ensemble = (rank_clf + rank_base) / (2.0 * len(rank_clf))
    ensemble_res = bootstrap_metrics(y_eval_binary, y_proba_ensemble, metrics_dict)
    for k, v in ensemble_res.items(): results[f"ensemble_{k}"] = v

    return results

def main():
    args = parse_args()
    processed = load_and_preprocess(args)
    if processed is None: return

    X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval = processed
    metrics = train_and_eval(X_train, y_train, X_eval, y_eval_binary, baseline_prob_eval, args)

    row = {
        "model": args.model,
        "eval_category": args.eval_category,
        "num_train_samples": len(X_train),
        "num_eval_samples": len(X_eval),
    }
    row.update(metrics)

    df_res = pd.DataFrame([row])
    if not os.path.exists(args.output_csv):
        df_res.to_csv(args.output_csv, index=False)
    else:
        existing_df = pd.read_csv(args.output_csv)
        df_res = pd.concat([existing_df, df_res], ignore_index=True)
        df_res.to_csv(args.output_csv, index=False)

    logger.info(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
