import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotting.utils import set_plot_style, get_elongated_size
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from src.utils import load_predictions

set_plot_style()
# sns.set_style("whitegrid")
# sns.set_context("talk")

PALETTE = {
    "Correct": "#2ecc71",   # green
    "Incorrect": "#e74c3c", # red
}
LINESTYLES = {"Correct": "-", "Incorrect": "--"}
N_SPAGHETTI = 25
NUM_BINS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, path: Path):
    """Save figure as png + pdf and close."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _safe_histplot(**kwargs):
    try:
        return sns.histplot(**kwargs)
    except np.linalg.LinAlgError:
        kwargs["kde"] = False
        return sns.histplot(**kwargs)


def extract_probe_stats(probe_data: list[dict]) -> dict:
    """Return mean and last-token probe probs for each sample."""
    means, lasts = [], []
    for sample in probe_data:
        preds = sample.get("token_predictions", [])
        probs = [float(t.get("harmful_prob", 0.0)) for t in preds if isinstance(t, dict)]
        means.append(sum(probs) / len(probs) if probs else None)
        lasts.append(preds[-1].get("harmful_prob") if len(preds) > 0 else None)
    return {"mean": means, "last": lasts}


def _bin_region(preds, pct_lo, pct_hi, num_bins, label, region, sample_id):
    """Bin token predictions into equal-width pct buckets. Returns list of row dicts."""
    n = len(preds)
    if n == 0:
        return []
    probs = np.array([t["harmful_prob"] for t in preds])
    pcts = np.linspace(pct_lo, pct_hi, n, endpoint=False)
    edges = np.linspace(pct_lo, pct_hi, num_bins + 1)
    bins = np.clip(np.digitize(pcts, edges) - 1, 0, num_bins - 1)
    rows = []
    for b in range(num_bins):
        mask = bins == b
        if mask.any():
            rows.append({
                "pct_tokens": (edges[b] + edges[b + 1]) / 2,
                "probe_score": float(probs[mask].mean()),
                "correctness": label,
                "region": region,
                "sample_id": sample_id,
            })
    return rows


def _extract_sample_trajectory(sample, has_prompt):
    """Return (pcts, probs) arrays for one sample's full trajectory."""
    prompt_preds = sample.get("prompt_token_predictions", [])
    cot_preds = sample.get("token_predictions", [])

    if has_prompt:
        n_p, n_c = len(prompt_preds), len(cot_preds)
        probs = np.array(
            [t["harmful_prob"] for t in prompt_preds]
            + [t["harmful_prob"] for t in cot_preds]
        )
        pcts = np.concatenate([
            np.linspace(0, 100, n_p, endpoint=False) if n_p else np.array([]),
            np.linspace(100, 200, n_c, endpoint=False) if n_c else np.array([]),
        ])
    else:
        n_c = len(cot_preds)
        probs = np.array([t["harmful_prob"] for t in cot_preds])
        pcts = np.linspace(0, 100, n_c, endpoint=False)

    return pcts, probs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def violin_plot(df, y_col, hue_col, title, ylabel, save_path):
    order = [g for g in ["Correct", "Incorrect"] if g in df[hue_col].unique()]
    counts = df[hue_col].value_counts()

    fig, ax = plt.subplots()
    sns.violinplot(
        data=df, x=hue_col, y=y_col, order=order,
        palette={g: PALETTE[g] for g in order}, ax=ax, inner="box", cut=0,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticklabels([f"{g}\n(n={counts.get(g, 0)})" for g in order])
    ax.set(xlabel="", ylabel=ylabel, title=title, ylim=(0, 1))
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_roc_pr(df, score_cols, save_folder):
    """ROC and PR curves predicting math incorrectness."""
    valid_df = df.dropna(subset=score_cols, how="all")
    y_true = (valid_df["correctness"] == "Incorrect").astype(int)

    for curve_type in ("roc", "pr"):
        fig, ax = plt.subplots()
        for col in score_cols:
            mask = ~valid_df[col].isna()
            if mask.sum() < 2 or y_true[mask].nunique() < 2:
                continue
            y, scores = y_true[mask], valid_df.loc[mask, col]
            if curve_type == "roc":
                x_vals, y_vals, _ = roc_curve(y, scores)
                metric = roc_auc_score(y, scores)
                ax.plot(x_vals, y_vals, label=f"{col} (AUC={metric:.3f})")
            else:
                prec, rec, _ = precision_recall_curve(y, scores)
                metric = average_precision_score(y, scores)
                ax.plot(rec, prec, label=f"{col} (AP={metric:.3f})")

        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
            ax.set(xlabel="FPR", ylabel="TPR", title="ROC — Probe → Incorrectness")
        else:
            baseline = y_true.mean()
            ax.axhline(baseline, color="black", linestyle="--", alpha=0.4,
                       linewidth=0.8, label=f"Baseline ({baseline:.2f})")
            ax.set(xlabel="Recall", ylabel="Precision", title="PR — Probe → Incorrectness")
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save_fig(fig, save_folder / f"{curve_type}_curves.png")


def plot_score_histogram(df, score_col, save_folder):
    """Overlapping histogram of probe scores split by correctness."""
    fig, ax = plt.subplots()
    for label, color in PALETTE.items():
        sub = df[df["correctness"] == label][score_col].dropna()
        if len(sub):
            _safe_histplot(data=sub, ax=ax, color=color, label=label,
                           stat="density", kde=True, alpha=0.45, binrange=(0, 1), bins=40)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set(xlabel="Probe Score", ylabel="Density", title=f"Distribution — {score_col}")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, save_folder / f"hist_{score_col}.png")


# ---------------------------------------------------------------------------
# Over-tokens trajectory plots
# ---------------------------------------------------------------------------

def build_token_rows(probe_data, correctness_labels, has_prompt, num_bins=NUM_BINS):
    """Build long-form dataframe with binned % tokens vs probe score."""
    rows = []
    for i, sample in enumerate(tqdm(probe_data, desc="Building token rows")):
        lbl = correctness_labels[i]
        prompt_preds = sample.get("prompt_token_predictions", [])
        cot_preds = sample.get("token_predictions", [])

        if has_prompt:
            rows.extend(_bin_region(prompt_preds, 0, 100, num_bins, lbl, "Prompt", i))
            rows.extend(_bin_region(cot_preds, 100, 200, num_bins, lbl, "CoT", i))
        else:
            rows.extend(_bin_region(cot_preds, 0, 100, num_bins, lbl, "CoT", i))
    return rows


def plot_score_over_tokens(df, has_prompt, title, save_path):
    """Spaghetti + mean ±95 CI + derivative panel, split by correctness."""
    fig, (ax, ax_d) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True, figsize=get_elongated_size()
    )
    rng = np.random.default_rng(42)

    # Spaghetti
    for group, color in PALETTE.items():
        gdf = df[df["correctness"] == group]
        ids = gdf["sample_id"].unique()
        for sid in rng.choice(ids, size=min(N_SPAGHETTI, len(ids)), replace=False):
            s = gdf[gdf["sample_id"] == sid].sort_values("pct_tokens")
            ax.plot(s["pct_tokens"], s["probe_score"], color=color, alpha=0.2, lw=0.7, zorder=1)

    # Mean + CI
    n_per = df.groupby("correctness")["sample_id"].nunique()
    df_p = df.copy()
    df_p["correctness"] = df_p["correctness"].map(lambda g: f"{g} (n={n_per.get(g, 0)})")
    pal = {f"{g} (n={n_per.get(g, 0)})": c for g, c in PALETTE.items()}
    dashes = {f"{g} (n={n_per.get(g, 0)})": (LINESTYLES.get(g, "-"),) for g in PALETTE.keys()}

    sns.lineplot(data=df_p, x="pct_tokens", y="probe_score", hue="correctness",
                 ax=ax, errorbar=("ci", 95), palette=pal, linewidth=2.0, zorder=3)

    # Derivative
    for group_n, color in pal.items():
        orig_group = group_n.split(" (n=")[0]
        means = df[df["correctness"] == orig_group].groupby("pct_tokens")["probe_score"].mean()
        ls = LINESTYLES.get(orig_group, "-")
        delta = np.gradient(means.values, means.index.values)
        ax_d.plot(means.index, delta, color=color, linestyle=ls, lw=1.5)

    if has_prompt:
        for a in (ax, ax_d):
            a.axvline(100, color="grey", ls="--", alpha=0.6, lw=1.2)
        ax.set_xlim(0, 200)
        ax_d.set_xlabel("% Tokens (0–100% = Prompt, 100–200% = CoT)")
    else:
        ax.set_xlim(0, 100)
        ax_d.set_xlabel("% of CoT Tokens Processed")

    ax.set(ylabel="Probe Score", ylim=(0, 1), title=title)
    ax.axhline(0.5, color="black", ls=":", lw=0.6, alpha=0.4)
    ax.legend(title="Answer Correctness")
    ax_d.set_ylabel("Δ Probe Score")
    ax_d.axhline(0, color="black", lw=0.5, alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, save_path)
    print(f"Saved: {save_path}")


def plot_sample_trajectories(probe_data, correctness_labels, has_prompt,
                             save_path, n_samples_per_group=10, window_size=15):
    """Grid of raw + smoothed trajectories — one panel per correctness group."""
    unique_groups = sorted(set(correctness_labels), reverse=True)
    n_groups = len(unique_groups)
    if n_groups == 0:
        return

    # 1-column layout scaled by number of groups
    fig, axes = plt.subplots(
        n_groups, 1,
        sharex=True, sharey=True
    )
    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    rng = np.random.default_rng(42)

    for i, group in enumerate(unique_groups):
        ax = axes[i]
        color = PALETTE.get(group, "steelblue")
        idxs = [idx for idx, g in enumerate(correctness_labels) if g == group]
        if not idxs:
            continue
        chosen = rng.choice(idxs, size=min(n_samples_per_group, len(idxs)), replace=False)

        for idx in chosen:
            sample = probe_data[idx]
            pcts, probs = _extract_sample_trajectory(sample, has_prompt)

            if len(probs) == 0:
                continue

            # Plot raw
            ax.plot(pcts, probs, color=color, alpha=0.3, lw=0.5, label="Raw" if idx == chosen[0] else None)
            # Plot smoothed
            smoothed = pd.Series(probs).rolling(window=window_size, min_periods=1, center=True).mean()
            ax.plot(pcts, smoothed, color=color, alpha=0.9, lw=1.5, label="Smoothed" if idx == chosen[0] else None)

        if has_prompt:
            ax.axvline(x=100, color="grey", ls="--", alpha=0.6, lw=1.2)
        ax.axhline(0.5, color="black", ls=":", lw=0.6, alpha=0.4)
        ax.set_title(f"Category: {group}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Probe Score")
        if i == n_groups - 1:
            ax.set_xlabel("% Tokens" if has_prompt else "% of CoT Tokens")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_fig(fig, save_path)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Summarise MIL probe predictions on math benchmarks.")
    parser.add_argument("--cummax-results", type=Path, required=True)
    parser.add_argument("--pertoken-results", type=Path, required=True)
    parser.add_argument("--save-folder", type=Path, default=Path("results/math/mil_probe_eval/summary"))
    return parser.parse_args()


def main(args):
    args.save_folder.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    cummax_data = load_predictions(args.cummax_results)
    pertoken_data = load_predictions(args.pertoken_results)

    cummax_stats = extract_probe_stats(cummax_data)
    pertoken_stats = extract_probe_stats(pertoken_data)

    # Build summary dataframe
    df = pd.DataFrame([
        {
            "sample_id": s.get("sample_id", i),
            "correctness": "Correct" if float(s.get("label", 0.0)) == 1.0 else "Incorrect",
            "num_cot_tokens": s.get("num_cot_tokens", 0),
            "cummax_mean_prob": cummax_stats["mean"][i],
            "cummax_last_token_prob": cummax_stats["last"][i],
            "pertoken_mean_prob": pertoken_stats["mean"][i],
            "pertoken_last_token_prob": pertoken_stats["last"][i],
        }
        for i, s in enumerate(cummax_data)
    ])

    score_cols = [c for c in df.columns if c.endswith("_prob")]

    # Violin + histogram plots
    for col in score_cols:
        mode = "cummax" if "cummax" in col else "per-token"
        stat = "Mean" if "mean" in col else "Last Token"
        violin_plot(df, col, "correctness",
                    f"MIL Probe ({mode}): {stat} — Correct vs Incorrect",
                    "Probe Score", args.save_folder / f"violin_{col}.png")
        plot_score_histogram(df, col, args.save_folder)

    # ROC / PR curves
    plot_roc_pr(df, score_cols, args.save_folder)

    # Trajectory plots (probe score over tokens)
    print("\nBuilding token-level trajectory plots...")
    _prompt_preds = cummax_data[0].get("prompt_token_predictions")
    has_prompt = _prompt_preds is not None and len(_prompt_preds) > 0
    labels = [
        "Correct" if float(s.get("label", 0.0)) == 1.0 else "Incorrect"
        for s in cummax_data
    ]
    token_rows = build_token_rows(cummax_data, labels, has_prompt=has_prompt)
    if token_rows:
        token_df = pd.DataFrame(token_rows)
        plot_score_over_tokens(token_df, has_prompt,
                               "Probe Scores — Math Correctness",
                               args.save_folder / "probe_score_over_tokens.png")

        plot_sample_trajectories(
            probe_data=cummax_data,
            correctness_labels=labels,
            has_prompt=has_prompt,
            save_path=args.save_folder / "sample_trajectories_by_correctness.png",
        )

    # CoT length by correctness
    fig, ax = plt.subplots()
    for label, color in PALETTE.items():
        sub = df[df["correctness"] == label]["num_cot_tokens"].dropna()
        if len(sub):
            _safe_histplot(data=sub, ax=ax, color=color, label=label,
                           stat="density", kde=True, alpha=0.45, bins=50)
    ax.set(xlabel="Number of CoT Tokens", ylabel="Density", title="CoT Length — Correct vs Incorrect")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, args.save_folder / "cot_length_by_correctness.png")

    # Summary stats
    print("\n=== Summary ===")
    print(f"Total: {len(df)}  Correct: {(df['correctness']=='Correct').sum()}  "
          f"Incorrect: {(df['correctness']=='Incorrect').sum()}  "
          f"Accuracy: {(df['correctness']=='Correct').mean():.4f}")
    for col in score_cols:
        v = df[col].dropna()
        print(f"  {col:40s}: mean={v.mean():.4f}  std={v.std():.4f}")

    df.to_csv(args.save_folder / "math_probe_summary.csv", index=False)
    print(f"\nSaved to: {args.save_folder}")


if __name__ == "__main__":
    main(parse_args())
