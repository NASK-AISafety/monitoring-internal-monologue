import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotting.utils import set_plot_style
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from src.utils import load_predictions

# sns.set_style("whitegrid")
# sns.set_context("talk")

# Shared palette — "Response / CoT" format, consistent with plot_harmfulness_over_tokens.py
HUE_PALETTE = {
    "Resp: Harmful / CoT: Faithful": "#c0392b",  # dark red   — response harmful, CoT harmful
    "Resp: Harmful / CoT: Unfaithful": "#e67e22",  # orange     — response harmful, CoT safe
    "Resp: Safe / CoT: Unfaithful": "#8e44ad",  # purple     — response safe,    CoT harmful (self-correction)
    "Resp: Safe / CoT: Faithful": "#95a5a6",  # grey       — response safe,    CoT safe
}
RESPONSE_PALETTE = {"Harmful": "#c0392b", "Safe": "#2980b9"}
HUE_ORDER = [
    "Resp: Harmful / CoT: Faithful",
    "Resp: Harmful / CoT: Unfaithful",
    "Resp: Safe / CoT: Unfaithful",
    "Resp: Safe / CoT: Faithful",
]


def safe_histplot(**kwargs):
    try:
        return sns.histplot(**kwargs)
    except np.linalg.LinAlgError:
        kwargs["kde"] = False
        return sns.histplot(**kwargs)

DEFAULT_DATASET_PATH = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/data/processed_benchmarks/harmfulness/prompt/wildguardtest"
DEFAULT_WG_RESULTS_FOLDER = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/wildguard/deepseek_r1_distill_llama_8b/"
DEFAULT_PERTOKEN_RESULTS = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/mil_probe_eval/deepseek_r1_distill_llama_8b_wg_test/cot_mil_probe_preds_pertoken.json"
DEFAULT_CUMMAX_RESULTS = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/mil_probe_eval/deepseek_r1_distill_llama_8b_wg_test/cot_mil_probe_preds_cummax.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MIL probe CoT predictions.")
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
        help="Path to the per-token MIL probe predictions JSON.",
    )
    parser.add_argument(
        "--cummax-results",
        type=Path,
        default=DEFAULT_CUMMAX_RESULTS,
        help="Path to the cumulative max-pool MIL probe predictions JSON.",
    )
    parser.add_argument(
        "--save-folder",
        type=Path,
        default=Path("plotting/outputs/mil_probe"),
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


def violin_plot(
    df: pd.DataFrame,
    y_col: str,
    hue_col: str,
    title: str,
    ylabel: str,
    save_path: Path,
    figsize: tuple = (12, 6),
):
    order = [g for g in HUE_ORDER if g in df[hue_col].unique()]
    palette = {g: HUE_PALETTE[g] for g in order}
    counts = df[hue_col].value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df, x=hue_col, y=y_col, order=order, palette=palette,
        ax=ax, inner="box", cut=0,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticklabels(
        [f"{g}\n(n={counts.get(g, 0)})" for g in order], rotation=15, ha="right"
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def histogram_plot(
    df: pd.DataFrame,
    x_col: str,
    hue_col: str,
    title: str,
    xlabel: str,
    save_path: Path,
    figsize: tuple = (12, 6),
):
    order = [g for g in HUE_ORDER if g in df[hue_col].unique()]
    palette = {g: HUE_PALETTE[g] for g in order}

    fig, ax = plt.subplots(figsize=figsize)
    safe_histplot(
        data=df,
        x=x_col,
        hue=hue_col,
        hue_order=order,
        palette=palette,
        element="step",
        common_norm=False,
        kde=True,
        ax=ax,
        stat="probability",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_roc_pr(
    df: pd.DataFrame,
    score_cols: list[str],
    save_folder: Path,
):
    """ROC and PR curves treating each score column as a classifier for response harmfulness."""
    eval_df = df[df["response_harmfulness"] != "Unknown"].copy()
    y_true = (eval_df["response_harmfulness"] == "Harmful").astype(int)

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in score_cols:
        valid = ~eval_df[col].isna()
        fpr, tpr, _ = roc_curve(y_true[valid], eval_df.loc[valid, col])
        auc_val = roc_auc_score(y_true[valid], eval_df.loc[valid, col])
        ax.plot(fpr, tpr, label=f"{col} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Probe Score → Response Harmfulness")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_folder / "roc_curves.png", dpi=300)
    fig.savefig(save_folder / "roc_curves.pdf")
    plt.close(fig)

    # --- PR ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in score_cols:
        valid = ~eval_df[col].isna()
        prec, rec, _ = precision_recall_curve(y_true[valid], eval_df.loc[valid, col])
        ap = average_precision_score(y_true[valid], eval_df.loc[valid, col])
        ax.plot(rec, prec, label=f"{col} (AP={ap:.3f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="black", linestyle="--", alpha=0.4,
               linewidth=0.8, label=f"Baseline (prevalence={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves — Probe Score → Response Harmfulness")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_folder / "pr_curves.png", dpi=300)
    fig.savefig(save_folder / "pr_curves.pdf")
    plt.close(fig)


def main(args: argparse.Namespace):
    args.save_folder.mkdir(parents=True, exist_ok=True)
    set_plot_style()

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
    plot_df["response+cot_harmfulness"] = plot_df.apply(
        lambda row: f"Resp: {row['response_harmfulness']} / CoT: {'Faithful' if row['response_harmfulness'] == row['cot_harmfulness'] else 'Unfaithful'}",
        axis=1,
    )

    # --- Cummax plots ---
    violin_plot(
        df=plot_df,
        y_col="cummax_mean_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (cummax): Mean Harmful Prob by Response / CoT Harmfulness",
        ylabel="Mean Harmful Token Probability",
        save_path=args.save_folder / "cummax_mean_harmful_prob_by_cot.png",
    )
    histogram_plot(
        df=plot_df,
        x_col="cummax_mean_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (cummax): Mean Harmful Prob Histogram",
        xlabel="Mean Harmful Token Probability",
        save_path=args.save_folder / "cummax_mean_harmful_prob_by_cot_hist.png",
    )
    violin_plot(
        df=plot_df,
        y_col="cummax_last_token_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (cummax): Last Token Harmful Prob by Response / CoT Harmfulness",
        ylabel="Last Token Harmful Probability",
        save_path=args.save_folder / "cummax_last_token_harmful_prob_by_cot.png",
    )
    histogram_plot(
        df=plot_df,
        x_col="cummax_last_token_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (cummax): Last Token Harmful Prob Histogram",
        xlabel="Last Token Harmful Probability",
        save_path=args.save_folder / "cummax_last_token_harmful_prob_by_cot_hist.png",
    )
    violin_plot(
        df=plot_df[plot_df["prompt_harmfulness"] == "Harmful"],
        y_col="cummax_last_token_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (cummax): Last Token Prob — Harmful Prompts Only",
        ylabel="Last Token Harmful Probability",
        save_path=args.save_folder / "cummax_last_token_harmful_prob_by_cot_harmful.png",
    )
    histogram_plot(
        df=plot_df[plot_df["prompt_harmfulness"] == "Harmful"],
        x_col="cummax_last_token_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (cummax): Last Token Prob Histogram — Harmful Prompts Only",
        xlabel="Last Token Harmful Probability",
        save_path=args.save_folder / "cummax_last_token_harmful_prob_by_cot_harmful_hist.png",
    )

    # --- Per-token plots ---
    violin_plot(
        df=plot_df,
        y_col="pertoken_mean_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (per-token): Mean Harmful Prob by Response / CoT Harmfulness",
        ylabel="Mean Harmful Token Probability",
        save_path=args.save_folder / "pertoken_mean_harmful_prob_by_cot.png",
    )
    histogram_plot(
        df=plot_df,
        x_col="pertoken_mean_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (per-token): Mean Harmful Prob Histogram",
        xlabel="Mean Harmful Token Probability",
        save_path=args.save_folder / "pertoken_mean_harmful_prob_by_cot_hist.png",
    )
    violin_plot(
        df=plot_df,
        y_col="pertoken_last_token_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (per-token): Last Token Harmful Prob by Response / CoT Harmfulness",
        ylabel="Last Token Harmful Probability",
        save_path=args.save_folder / "pertoken_last_token_harmful_prob_by_cot.png",
    )
    histogram_plot(
        df=plot_df,
        x_col="pertoken_last_token_harmful_prob",
        hue_col="response+cot_harmfulness",
        title="MIL Probe (per-token): Last Token Harmful Prob Histogram",
        xlabel="Last Token Harmful Probability",
        save_path=args.save_folder / "pertoken_last_token_harmful_prob_by_cot_hist.png",
    )

    # --- ROC and PR curves ---
    score_cols = [
        "cummax_mean_harmful_prob",
        "cummax_last_token_harmful_prob",
        "pertoken_mean_harmful_prob",
        "pertoken_last_token_harmful_prob",
    ]
    plot_roc_pr(df, score_cols, args.save_folder)

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Prompt harmfulness: {df['prompt_harmfulness'].value_counts().to_dict()}")
    print(
        f"Response harmfulness: {df['response_harmfulness'].value_counts().to_dict()}"
    )
    print(f"CoT harmfulness: {df['cot_harmfulness'].value_counts().to_dict()}")
    print(
        f"\nCummax - Mean harmful prob:  mean={df['cummax_mean_harmful_prob'].mean():.4f}, std={df['cummax_mean_harmful_prob'].std():.4f}"
    )
    print(
        f"Cummax - Last token prob:    mean={df['cummax_last_token_harmful_prob'].mean():.4f}, std={df['cummax_last_token_harmful_prob'].std():.4f}"
    )
    print(
        f"Per-token - Mean harmful prob: mean={df['pertoken_mean_harmful_prob'].mean():.4f}, std={df['pertoken_mean_harmful_prob'].std():.4f}"
    )
    print(
        f"Per-token - Last token prob:   mean={df['pertoken_last_token_harmful_prob'].mean():.4f}, std={df['pertoken_last_token_harmful_prob'].std():.4f}"
    )

    # Save DataFrame
    df.to_csv(args.save_folder / "mil_probe_summary.csv", index=False)
    print(f"\nSaved summary CSV to: {args.save_folder / 'mil_probe_summary.csv'}")
    print(f"Saved plots to: {args.save_folder}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
