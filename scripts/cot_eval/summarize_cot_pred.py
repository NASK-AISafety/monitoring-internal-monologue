import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_from_disk
from tqdm import tqdm
from src.utils import load_predictions

sns.set_style("whitegrid")

DEFAULT_DATASET_PATH = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/data/processed_benchmarks/harmfulness/prompt/wildguardtest"
DEFAULT_WG_RESULTS_FOLDER = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/wildguard/deepseek_r1_distill_llama_8b/"
DEFAULT_PROBE_RESULTS = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/results/cot_eval/probe_multilayer/deepseek_r1_distill_llama_8b/cot_token_probe_preds.json"
DEFAULT_CUMSUMMED_PROBE_RESULTS = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/results/cot_eval/probe_multilayer/deepseek_r1_distill_llama_8b/cot_token_probe_preds_cumsum.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CoT predictions and probe results."
    )
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
        "--probe-results",
        type=Path,
        default=DEFAULT_PROBE_RESULTS,
        help="Path to the (unsummed) probe predictions JSON.",
    )
    parser.add_argument(
        "--cumsummed-probe-results",
        type=Path,
        default=DEFAULT_CUMSUMMED_PROBE_RESULTS,
        help="Path to the cumsummed probe predictions JSON.",
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


def main(args: argparse.Namespace):
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
            f"Warning: Number of predictions mismatch (CoT: {len(cot_preds)}, Response: {len(response_preds)})"
        )
        min_len = min(len(cot_preds), len(response_preds))
        cot_preds = cot_preds[:min_len]
        response_preds = response_preds[:min_len]

    probe_cumsummed_data = load_predictions(args.cumsummed_probe_results)
    probe_unsummed_data = load_predictions(args.probe_results)

    data = {
        "prompt_harmfulness": [],
        "cot_harmfulness": [],
        "response_harmfulness": [],
        "probe_mean_harmful_token_prob": [],
        "probe_last_token_harmful_prob": [],
        "probe_cumsummed_mean_harmful_token_prob": [],
        "probe_cumsummed_last_token_harmful_prob": [],
    }
    for i in tqdm(range(len(probe_cumsummed_data))):
        cot_pred = parse_harmfulness(cot_preds[i] if i < len(cot_preds) else None)
        response_pred = parse_harmfulness(
            response_preds[i] if i < len(response_preds) else None
        )
        prompt_harmfulness = dataset[i].get("prompt_harmfulness", -1)

        data["prompt_harmfulness"].append(get_label_str(prompt_harmfulness))
        data["cot_harmfulness"].append(get_label_str(cot_pred))
        data["response_harmfulness"].append(get_label_str(response_pred))
        cum_summed_preds = probe_cumsummed_data[i]
        token_predictions = cum_summed_preds.get("token_predictions", [])
        token_probs = [
            float(t.get("harmful_prob", 0.0))
            for t in token_predictions
            if isinstance(t, dict)
        ]
        mean_pred = sum(token_probs) / len(token_probs) if token_probs else None
        last_token_pred = (
            token_predictions[-1].get("harmful_prob", None)
            if len(token_predictions) > 0
            else None
        )

        data["probe_cumsummed_last_token_harmful_prob"].append(last_token_pred)
        data["probe_cumsummed_mean_harmful_token_prob"].append(mean_pred)

        unsummed_preds = probe_unsummed_data[i]
        token_predictions_unsummed = unsummed_preds.get("token_predictions", [])
        token_probs_unsummed = [
            float(t.get("harmful_prob", 0.0))
            for t in token_predictions_unsummed
            if isinstance(t, dict)
        ]
        mean_pred_unsummed = (
            (sum(token_probs_unsummed) / len(token_probs_unsummed))
            if token_probs_unsummed
            else None
        )
        last_token_pred_unsummed = (
            token_predictions_unsummed[-1].get("harmful_prob", None)
            if len(token_predictions_unsummed) > 0
            else None
        )

        data["probe_last_token_harmful_prob"].append(last_token_pred_unsummed)
        data["probe_mean_harmful_token_prob"].append(mean_pred_unsummed)

    # Plot histograms of mean harmful token probabilities for harmful vs safe responses
    df = pd.DataFrame(data)
    plot_df = df[df["response_harmfulness"] != "Unknown"]
    plot_df["response+cot_harmfulness"] = (
        plot_df["response_harmfulness"] + " / " + plot_df["cot_harmfulness"]
    )
    plt.figure()
    sns.histplot(
        data=plot_df,
        x="probe_cumsummed_mean_harmful_token_prob",
        hue="response_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title(
        "Distribution of Mean Harmful Token Probabilities by Response Harmfulness"
    )
    plt.xlabel("Mean Harmful Token Probability")
    plt.ylabel("Count")
    plt.savefig("plotting/outputs/mean_harmful_token_prob_distribution.png")
    plt.show()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="probe_cumsummed_mean_harmful_token_prob",
        hue="response+cot_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title("Distribution of Mean Harmful Token Probabilities by CoT Harmfulness")
    plt.xlabel("Mean Harmful Token Probability")
    plt.ylabel("Count")
    plt.savefig("plotting/outputs/mean_harmful_token_prob_by_cot_harmfulness.png")
    plt.show()

    plt.figure()
    sns.histplot(
        data=plot_df,
        x="probe_cumsummed_last_token_harmful_prob",
        hue="response_harmfulness",
        kde=True,
        stat="probability",
        common_norm=False,
    )
    plt.title(
        "Distribution of Last Token Harmful Probabilities by Response Harmfulness"
    )
    plt.xlabel("Last Token Harmful Probability")
    plt.ylabel("Count")
    plt.savefig("plotting/outputs/last_token_harmful_prob_distribution.png")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
