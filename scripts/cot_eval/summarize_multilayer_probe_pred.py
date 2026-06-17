import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.utils import load_predictions
from stats_utils import bootstrap_metrics

sns.set_style("whitegrid")

DEFAULT_DATASET_PATH = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/data/processed_benchmarks/harmfulness/prompt/wildguardtest"
DEFAULT_WG_RESULTS_FOLDER = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/wildguard/deepseek_r1_distill_llama_8b/"
DEFAULT_PERTOKEN_RESULTS = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/probe_multilayer/deepseek_r1_distill_llama_8b_wg_test/cot_token_probe_preds_unsummed.json"
DEFAULT_CUMMAX_RESULTS = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/probe_multilayer/deepseek_r1_distill_llama_8b_wg_test/cot_token_probe_preds_cumsum.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Multilayer probe CoT predictions.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the HF dataset saved on disk.",
    )
    parser.add_argument(
        "--wg-results-folder",
        type=Path,
        default=DEFAULT_WG_RESULTS_FOLDER,
        help="Folder containing WildGuard preds_cot.json and preds_response.json.",
    )
    parser.add_argument(
        "--pertoken-results",
        type=Path,
        default=DEFAULT_PERTOKEN_RESULTS,
        help="Path to the per-token probe predictions JSON.",
    )
    parser.add_argument(
        "--cummax-results",
        type=Path,
        default=DEFAULT_CUMMAX_RESULTS,
        help="Path to the cumulative/mean probe predictions JSON.",
    )
    parser.add_argument(
        "--save-folder",
        type=Path,
        default=Path("plotting/outputs/multilayer_probe"),
        help="Folder to save output plots.",
    )
    return parser.parse_args()

def parse_harmfulness(pred_str: str) -> int:
    """
    Parses the prediction string to determine if the response is harmful.
    Returns 1 if harmful, 0 if safe, -1 if unknown.
    """
    if not isinstance(pred_str, str):
        return -1
    lower_pred = pred_str.lower()
    if "harmful response: yes" in lower_pred:
        return 1
    elif "harmful response: no" in lower_pred:
        return 0
    return -1

def get_label_str(label_int: int) -> str:
    if label_int == 1:
        return "Harmful"
    elif label_int == 0:
        return "Safe"
    return "Unknown"

def extract_probe_stats(probe_data: list[dict]) -> dict:
    """Extract mean and last-token harmful probs from probe predictions."""
    means = []
    lasts = []
    for sample in probe_data:
        token_predictions = sample.get("token_predictions", [])
        token_probs = [
            float(t.get("harmful_prob", 0.0))
            for t in token_predictions
            if isinstance(t, dict)
        ]
        mean_pred = sum(token_probs) / len(token_probs) if token_probs else None
        last_pred = (
            token_predictions[-1].get("harmful_prob", None)
            if len(token_predictions) > 0
            else None
        )
        means.append(mean_pred)
        lasts.append(last_pred)
    return {"mean": means, "last": lasts}

def main(args: argparse.Namespace):
    args.save_folder.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    with open(args.wg_results_folder / "preds_cot.json", "r", encoding="utf-8") as f:
        cot_data = json.load(f)
        cot_preds = cot_data["preds"]

    with open(
        args.wg_results_folder / "preds_response.json", "r", encoding="utf-8"
    ) as f:
        response_data = json.load(f)
        response_preds = response_data["preds"]

    if len(cot_preds) != len(response_preds):
        print(
            f"Warning: Number of predictions mismatch "
            f"(CoT: {len(cot_preds)}, Response: {len(response_preds)})"
        )
        min_len = min(len(cot_preds), len(response_preds))
        cot_preds = cot_preds[:min_len]
        response_preds = response_preds[:min_len]

    cummax_data = load_predictions(args.cummax_results)
    pertoken_data = load_predictions(args.pertoken_results)

    cummax_stats = extract_probe_stats(cummax_data)
    pertoken_stats = extract_probe_stats(pertoken_data)

    data = {
        "prompt_harmfulness": [],
        "cot_harmfulness": [],
        "response_harmfulness": [],
        "cummax_mean_harmful_prob": [],
        "cummax_last_token_harmful_prob": [],
        "pertoken_mean_harmful_prob": [],
        "pertoken_last_token_harmful_prob": [],
    }
    for i in tqdm(range(len(cummax_data))):
        cot_pred = parse_harmfulness(cot_preds[i] if i < len(cot_preds) else None)
        response_pred = parse_harmfulness(
            response_preds[i] if i < len(response_preds) else None
        )
        prompt_harmfulness = dataset[i].get("prompt_harmfulness", -1)

        data["prompt_harmfulness"].append(get_label_str(prompt_harmfulness))
        data["cot_harmfulness"].append(get_label_str(cot_pred))
        data["response_harmfulness"].append(get_label_str(response_pred))
        data["cummax_mean_harmful_prob"].append(cummax_stats["mean"][i])
        data["cummax_last_token_harmful_prob"].append(cummax_stats["last"][i])
        data["pertoken_mean_harmful_prob"].append(pertoken_stats["mean"][i])
        data["pertoken_last_token_harmful_prob"].append(pertoken_stats["last"][i])

    df = pd.DataFrame(data)
    plot_df = df[df["response_harmfulness"] != "Unknown"].copy()
    plot_df["response+cot_harmfulness"] = (
        plot_df["response_harmfulness"] + " / " + plot_df["cot_harmfulness"]
    )

    # --- Cummax plots ---
    plt.figure()
    sns.histplot(
        data=plot_df,
        x="cummax_mean_harmful_prob",
        hue="response_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (cumsum): Mean Harmful Prob by Response Harmfulness")
    plt.xlabel("Mean Harmful Token Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "cummax_mean_harmful_prob_by_response.png")
    plt.close()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="cummax_mean_harmful_prob",
        hue="response+cot_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (cumsum): Mean Harmful Prob by CoT Harmfulness")
    plt.xlabel("Mean Harmful Token Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "cummax_mean_harmful_prob_by_cot.png")
    plt.close()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="cummax_last_token_harmful_prob",
        hue="response_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (cumsum): Last Token Harmful Prob by Response Harmfulness")
    plt.xlabel("Last Token Harmful Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "cummax_last_token_harmful_prob_by_response.png")
    plt.close()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="cummax_last_token_harmful_prob",
        hue="response+cot_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (cumsum): Last Token Harmful Prob by CoT Harmfulness")
    plt.xlabel("Last Token Harmful Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "cummax_last_token_harmful_prob_by_cot.png")
    plt.close()

    plt.figure()
    sns.histplot(
        data=plot_df[plot_df["prompt_harmfulness"] == "Harmful"],
        x="cummax_last_token_harmful_prob",
        hue="response+cot_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (cumsum): Last Token Harmful Prob by CoT Harmfulness")
    plt.xlabel("Last Token Harmful Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "cummax_last_token_harmful_prob_by_cot_harmful.png")
    plt.close()

    # --- Per-token plots ---
    plt.figure()
    sns.histplot(
        data=plot_df,
        x="pertoken_mean_harmful_prob",
        hue="response_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (per-token): Mean Harmful Prob by Response Harmfulness")
    plt.xlabel("Mean Harmful Token Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "pertoken_mean_harmful_prob_by_response.png")
    plt.close()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="pertoken_mean_harmful_prob",
        hue="response+cot_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (per-token): Mean Harmful Prob by CoT Harmfulness")
    plt.xlabel("Mean Harmful Token Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "pertoken_mean_harmful_prob_by_cot.png")
    plt.close()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="pertoken_last_token_harmful_prob",
        hue="response_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (per-token): Last Token Harmful Prob by Response Harmfulness")
    plt.xlabel("Last Token Harmful Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "pertoken_last_token_harmful_prob_by_response.png")
    plt.close()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="pertoken_last_token_harmful_prob",
        hue="response+cot_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Multilayer Probe (per-token): Last Token Harmful Prob by CoT Harmfulness")
    plt.xlabel("Last Token Harmful Probability")
    plt.ylabel("Probability")
    plt.savefig(args.save_folder / "pertoken_last_token_harmful_prob_by_cot.png")
    plt.close()

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Prompt harmfulness: {df['prompt_harmfulness'].value_counts().to_dict()}")
    print(
        f"Response harmfulness: {df['response_harmfulness'].value_counts().to_dict()}"
    )
    print(f"CoT harmfulness: {df['cot_harmfulness'].value_counts().to_dict()}")
    print(
        f"\nCumsum - Mean harmful prob:  mean={df['cummax_mean_harmful_prob'].mean():.4f}, std={df['cummax_mean_harmful_prob'].std():.4f}"
    )
    print(
        f"Cumsum - Last token prob:    mean={df['cummax_last_token_harmful_prob'].mean():.4f}, std={df['cummax_last_token_harmful_prob'].std():.4f}"
    )
    print(
        f"Per-token - Mean harmful prob: mean={df['pertoken_mean_harmful_prob'].mean():.4f}, std={df['pertoken_mean_harmful_prob'].std():.4f}"
    )
    print(
        f"Per-token - Last token prob:   mean={df['pertoken_last_token_harmful_prob'].mean():.4f}, std={df['pertoken_last_token_harmful_prob'].std():.4f}"
    )

    # Save DataFrame
    df.to_csv(args.save_folder / "multilayer_probe_summary.csv", index=False)
    print(f"\nSaved summary CSV to: {args.save_folder / 'multilayer_probe_summary.csv'}")
    print(f"Saved plots to: {args.save_folder}")

    # Calculate and save metrics with CIs and std
    y_true = (df["response_harmfulness"] == "Harmful").astype(int)
    y_pred = df["cummax_last_token_harmful_prob"].fillna(0.0)

    metrics_dict = {
        "F1 Score": lambda y_t, y_p: f1_score(y_t, y_p > 0.5),
        "Balanced Accuracy": lambda y_t, y_p: balanced_accuracy_score(y_t, y_p > 0.5),
        "AUROC": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "Average Precision": lambda y_t, y_p: average_precision_score(y_t, y_p),
    }

    print("\n=== Bootstrapped Metrics (CI and STD) ===")
    if y_true.nunique() > 1:
        results = bootstrap_metrics(y_true.values, y_pred.values, metrics_dict)
        print(json.dumps(results, indent=4))
        with open(args.save_folder / "metrics.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved metrics with CIs and std to: {args.save_folder / 'metrics.json'}")
    else:
        print("Warning: Only one class present in y_true. Cannot calculate AUROC/AP.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
