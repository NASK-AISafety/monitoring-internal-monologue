import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent.absolute()))

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate SHAP feature importances using RF.")
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
        default="results/shap_feature_importances.csv",
        help="Output path for aggregated results",
    )
    parser.add_argument(
        "--four_class",
        action="store_true",
        help="Enable 4-class classification in harmfulness mode.",
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default=None,
        help="Optional path to features CSV file.",
    )
    return parser.parse_args()


def load_and_preprocess(args):
    if args.features_csv:
        data_path = args.features_csv
        if args.mode == "harmfulness":
            metadata_cols = [
                "model", "experiment_id", "sample_id", "task",
                "prompt_label", "cot_label", "response_label", "num_cot_tokens",
            ]
        else:
            metadata_cols = [
                "model", "experiment_id", "sample_id", "task",
                "label", "num_cot_tokens", "response_label",
            ]
    elif args.mode == "harmfulness":
        data_path = "results/harmfulness/mil_probe_eval/signal_features.csv"
        metadata_cols = [
            "model", "experiment_id", "sample_id", "task",
            "prompt_label", "cot_label", "response_label", "num_cot_tokens",
        ]
    else:
        data_path = "results/math/mil_probe_eval/signal_features.csv"
        metadata_cols = [
            "model", "experiment_id", "sample_id", "task",
            "label", "num_cot_tokens", "response_label",
        ]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)

    exp_ids = args.experiment_id if isinstance(args.experiment_id, list) else [args.experiment_id]

    if len(exp_ids) == 1:
        df_target = data[(data.model == args.model) & (data.experiment_id == exp_ids[0])].copy()

        if len(df_target) == 0:
            raise ValueError(f"No samples found for model {args.model} and experiment {exp_ids[0]}")

        if args.mode == "harmfulness":
            df_target = df_target[(df_target.response_label != -1) & (df_target.cot_label != -1)]
            y_binary = (df_target["response_label"] == 1).astype(int).values
            if args.four_class:
                conditions = [
                    (df_target["cot_label"] == 0) & (df_target["response_label"] == 0),
                    (df_target["cot_label"] == 1) & (df_target["response_label"] == 1),
                    (df_target["cot_label"] == 1) & (df_target["response_label"] == 0),
                    (df_target["cot_label"] == 0) & (df_target["response_label"] == 1),
                ]
                y_train = np.select(conditions, [0, 1, 2, 3], default=-1)
            else:
                y_train = y_binary
        else:
            df_target["label"] = 1 - df_target["label"]
            y_train = df_target["label"].values

        X = df_target.drop(columns=metadata_cols, errors="ignore")
    else:
        df_merged = None
        for i, exp in enumerate(exp_ids):
            temp_df = data[(data.experiment_id == exp) & (data.model == args.model)].copy()
            if len(temp_df) == 0:
                continue

            if args.mode == "harmfulness":
                temp_df = temp_df[(temp_df.response_label != -1) & (temp_df.cot_label != -1)]
                temp_df = temp_df[temp_df.prompt_label == 1]
            else:
                temp_df["label"] = 1 - temp_df["label"]

            feat_cols = [c for c in temp_df.columns if c not in metadata_cols]
            rename_dict = {c: f"{c}_exp{i+1}" for c in feat_cols}
            temp_df = temp_df.rename(columns=rename_dict)

            if df_merged is None:
                df_merged = temp_df
            else:
                cols_to_join = list(rename_dict.values()) + ["sample_id"]
                df_merged = pd.merge(df_merged, temp_df[cols_to_join], on="sample_id")

        if df_merged is None:
            raise ValueError(f"No samples found for model {args.model} across experiments {exp_ids}")

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
            else:
                y_train = y_binary
        else:
            y_train = df_merged["label"].values

        X = df_merged.drop(columns=[c for c in df_merged.columns if any(m == c for m in metadata_cols)], errors="ignore")

    return X, y_train


def main():
    args = parse_args()
    exp_str = "+".join(args.experiment_id) if isinstance(args.experiment_id, list) else args.experiment_id
    logger.info(f"Processing Model: {args.model}, Experiment: {exp_str} ({args.mode})")

    try:
        X, y_train = load_and_preprocess(args)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    logger.info(f"Loaded {len(X)} samples. Training Random Forest on all data...")

    clf_rf = RandomForestClassifier(
        n_estimators=1000, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf_rf.fit(X, y_train)

    logger.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(clf_rf)
    
    # SHAP values calculation
    shap_values = explainer.shap_values(X)
    
    # For RandomForest (which often outputs probabilities), shap_values might be a list
    # corresponding to each class. For binary classification, we usually care about the positive class.
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values_to_use = shap_values[1] # Use SHAP values for class 1
        elif len(shap_values) == 4:
            # If 4-class, we can sum the SHAP values for the "harmful response" classes if needed
            # but standard is to just report feature importance across all (average magnitude)
            shap_values_to_use = np.mean(np.abs(shap_values), axis=0) # average across classes
        else:
            shap_values_to_use = shap_values[0] # Fallback
    else:
        # shap >= 0.40 sometimes returns a 3D array or an Explanation object
        if len(shap_values.shape) == 3:
            shap_values_to_use = shap_values[:, :, 1]
        else:
            shap_values_to_use = shap_values

    # Calculate mean absolute SHAP values (feature importance) and directional mean
    if isinstance(shap_values_to_use, np.ndarray):
        mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)
        mean_shap = shap_values_to_use.mean(axis=0)
    else:
        # if shap_values_to_use is an Explanation object
        mean_abs_shap = np.abs(shap_values_to_use.values).mean(axis=0)
        mean_shap = shap_values_to_use.values.mean(axis=0)
        
    feature_names = X.columns.tolist()
    
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap
    })
    
    df_importance["model"] = args.model
    df_importance["experiment_id"] = exp_str
    df_importance["mode"] = args.mode
    
    # Reorder columns
    df_importance = df_importance[["model", "experiment_id", "mode", "feature", "mean_abs_shap", "mean_shap"]]
    
    # Sort by importance
    df_importance = df_importance.sort_values(by="mean_abs_shap", ascending=False)
    
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    
    if not os.path.exists(args.output_csv):
        df_importance.to_csv(args.output_csv, index=False)
    else:
        df_importance.to_csv(args.output_csv, mode="a", header=False, index=False)

    logger.info(f"SHAP importances saved to {args.output_csv}")

if __name__ == "__main__":
    main()
