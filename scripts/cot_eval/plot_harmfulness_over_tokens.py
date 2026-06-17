"""Plot harmfulness probability as a function of % tokens processed.

X-axis: 0-100% = prompt tokens, 100-200% = CoT tokens.
Lines are split by CoT harmfulness and response harmfulness (from WildGuard).
Confidence intervals are shown via seaborn's lineplot CI estimation.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from plotting.utils import set_plot_style, get_elongated_size
from src.utils import load_predictions

set_plot_style()
# sns.set_style("whitegrid")
# sns.set_context("talk")

# Shared palette and line styles — consistent across all plots
HUE_PALETTE = {
    "Resp: Harmful / CoT: Faithful": "#c0392b",  # dark red   — response harmful, CoT harmful
    "Resp: Harmful / CoT: Unfaithful": "#e67e22",  # orange     — response harmful, CoT safe
    "Resp: Safe / CoT: Unfaithful": "#8e44ad",  # purple     — response safe,    CoT harmful (self-correction)
    "Resp: Safe / CoT: Faithful": "#95a5a6",  # grey       — response safe,    CoT safe
}
HUE_LINESTYLES = {
    "Resp: Harmful / CoT: Faithful": "-",
    "Resp: Harmful / CoT: Unfaithful": "--",
    "Resp: Safe / CoT: Unfaithful": "-.",
    "Resp: Safe / CoT: Faithful": ":",
}
N_SPAGHETTI = 25  # individual sample trajectories shown per group

DEFAULT_DATASET_PATH = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/data/processed_benchmarks/harmfulness/prompt/wildguardtest"
DEFAULT_WG_RESULTS_FOLDER = "/nfsclient/d6000-nfs.ai.nask.pl/data/maciejchr/projects/safety_adapters/results/cot_eval/wildguard/deepseek_r1_distill_llama_8b/"

NUM_BINS = 100  # Number of bins for the x-axis (percentage)


def parse_harmfulness(pred_str: str) -> int:
    """Parse WildGuard prediction string. Returns 1=harmful, 0=safe, -1=unknown."""
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


def load_wg_labels(
    wg_results_folder: Path, n_samples: int
) -> tuple[list[str], list[str]]:
    """Load CoT and response harmfulness labels from WildGuard predictions."""
    cot_preds = load_predictions(wg_results_folder / "preds_cot.json")
    response_preds = load_predictions(wg_results_folder / "preds_response.json")

    cot_labels = []
    response_labels = []
    for i in range(n_samples):
        cot_labels.append(
            get_label_str(
                parse_harmfulness(cot_preds[i] if i < len(cot_preds) else None)
            )
        )
        response_labels.append(
            get_label_str(
                parse_harmfulness(
                    response_preds[i] if i < len(response_preds) else None
                )
            )
        )
    return cot_labels, response_labels


def build_rows(
    probe_data: list[dict],
    cot_labels: list[str],
    response_labels: list[str],
    has_prompt: bool,
    num_bins: int = NUM_BINS,
) -> list[dict]:
    """Build a long-form dataframe with binned % of tokens vs harmful_prob.

    When has_prompt=True:  prompt → 0-100%, CoT → 100-200%.
    When has_prompt=False: CoT only → 0-100%.
    Each sample's tokens are binned into `num_bins` equal-width bins per region,
    and the harmful_prob is averaged within each bin.
    """
    rows = []
    for i, sample in enumerate(tqdm(probe_data, desc="Building rows")):
        cot_label = cot_labels[i]
        response_label = response_labels[i]

        if cot_label == "Unknown" or response_label == "Unknown":
            continue

        prompt_preds = sample.get("prompt_token_predictions", [])
        cot_preds = sample.get("token_predictions", [])

        if has_prompt:
            # --- Prompt region: 0% to 100% ---
            n_prompt = len(prompt_preds)
            if n_prompt > 0:
                prompt_probs = np.array([t["harmful_prob"] for t in prompt_preds])
                prompt_pcts = np.linspace(0, 100, n_prompt, endpoint=False)
                bin_edges = np.linspace(0, 100, num_bins + 1)
                bin_indices = np.clip(
                    np.digitize(prompt_pcts, bin_edges) - 1, 0, num_bins - 1
                )
                for b in range(num_bins):
                    mask = bin_indices == b
                    if mask.any():
                        rows.append(
                            {
                                "pct_tokens": bin_edges[b]
                                + (bin_edges[b + 1] - bin_edges[b]) / 2,
                                "harmful_prob": float(prompt_probs[mask].mean()),
                                "cot_harmfulness": cot_label,
                                "response_harmfulness": response_label,
                                "region": "Prompt",
                                "sample_id": i,
                            }
                        )

            # --- CoT region: 100% to 200% ---
            n_cot = len(cot_preds)
            if n_cot > 0:
                cot_probs = np.array([t["harmful_prob"] for t in cot_preds])
                cot_pcts = np.linspace(100, 200, n_cot, endpoint=False)
                bin_edges_cot = np.linspace(100, 200, num_bins + 1)
                bin_indices_cot = np.clip(
                    np.digitize(cot_pcts, bin_edges_cot) - 1, 0, num_bins - 1
                )
                for b in range(num_bins):
                    mask = bin_indices_cot == b
                    if mask.any():
                        rows.append(
                            {
                                "pct_tokens": bin_edges_cot[b]
                                + (bin_edges_cot[b + 1] - bin_edges_cot[b]) / 2,
                                "harmful_prob": float(cot_probs[mask].mean()),
                                "cot_harmfulness": cot_label,
                                "response_harmfulness": response_label,
                                "region": "CoT",
                                "sample_id": i,
                            }
                        )
        else:
            # --- CoT only: 0% to 100% ---
            n_cot = len(cot_preds)
            if n_cot > 0:
                cot_probs = np.array([t["harmful_prob"] for t in cot_preds])
                cot_pcts = np.linspace(0, 100, n_cot, endpoint=False)
                bin_edges_cot = np.linspace(0, 100, num_bins + 1)
                bin_indices_cot = np.clip(
                    np.digitize(cot_pcts, bin_edges_cot) - 1, 0, num_bins - 1
                )
                for b in range(num_bins):
                    mask = bin_indices_cot == b
                    if mask.any():
                        rows.append(
                            {
                                "pct_tokens": bin_edges_cot[b]
                                + (bin_edges_cot[b + 1] - bin_edges_cot[b]) / 2,
                                "harmful_prob": float(cot_probs[mask].mean()),
                                "cot_harmfulness": cot_label,
                                "response_harmfulness": response_label,
                                "region": "CoT",
                                "sample_id": i,
                            }
                        )

    return rows


def plot_sample_trajectories(
    probe_data: list[dict],
    cot_labels: list[str],
    response_labels: list[str],
    has_prompt: bool,
    save_path: Path,
    n_samples_per_group: int = 10,
    window_size: int = 15,
):
    """Plot raw and smoothed trajectories for a subset of samples from each category."""
    # Combine labels with faithfulness terminology
    groups = []
    for cl, rl in zip(cot_labels, response_labels):
        if cl == "Unknown" or rl == "Unknown":
            groups.append(None)
        else:
            faithfulness = "Faithful" if cl == rl else "Unfaithful"
            groups.append(f"Resp: {rl} / CoT: {faithfulness}")

    unique_groups = sorted([g for g in set(groups) if g is not None])
    n_groups = len(unique_groups)
    if n_groups == 0:
        return

    # Use constrained_layout for better spacing
    fig, axes = plt.subplots(
        (n_groups + 1) // 2,
        2,
        figsize=get_elongated_size(),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.flatten()
    rng = np.random.default_rng(42)

    for i, group in enumerate(unique_groups):
        ax = axes[i]
        color = HUE_PALETTE.get(group, "blue")

        # Get indices of samples in this group
        group_indices = [idx for idx, g in enumerate(groups) if g == group]
        if not group_indices:
            continue

        chosen_indices = rng.choice(
            group_indices,
            size=min(n_samples_per_group, len(group_indices)),
            replace=False,
        )

        for idx in chosen_indices:
            sample = probe_data[idx]
            prompt_preds = sample.get("prompt_token_predictions", [])
            cot_preds = sample.get("token_predictions", [])

            if has_prompt:
                # Combine prompt and CoT for the x-axis
                n_p = len(prompt_preds)
                n_c = len(cot_preds)
                probs = np.array(
                    [t["harmful_prob"] for t in prompt_preds]
                    + [t["harmful_prob"] for t in cot_preds]
                )
                pcts = np.concatenate(
                    [
                        np.linspace(0, 100, n_p, endpoint=False) if n_p > 0 else [],
                        np.linspace(100, 200, n_c, endpoint=False) if n_c > 0 else [],
                    ]
                )
            else:
                n_c = len(cot_preds)
                probs = np.array([t["harmful_prob"] for t in cot_preds])
                pcts = np.linspace(0, 100, n_c, endpoint=False)

            if len(probs) == 0:
                continue

            # Plot raw with very low alpha
            ax.plot(
                pcts,
                probs,
                color=color,
                alpha=0.15,
                linewidth=0.8,
                label="Raw" if idx == chosen_indices[0] else None,
            )

            # Plot smoothed with higher linewidth
            smoothed = (
                pd.Series(probs)
                .rolling(window=window_size, min_periods=1, center=True)
                .mean()
            )
            ax.plot(
                pcts,
                smoothed,
                color=color,
                alpha=0.9,
                linewidth=2.5,
                label="Smoothed" if idx == chosen_indices[0] else None,
            )

        if has_prompt:
            ax.axvline(x=100, color="grey", linestyle="--", alpha=0.6, linewidth=1.5)
            # Add boundary label only on the first plot
            if i == 0:
                ax.text(
                    101,
                    1.02,
                    "CoT Starts",
                    color="grey",
                    fontweight="bold",
                    va="bottom",
                )

        # Cleaner title
        clean_title = group.replace("Resp: ", "Response: ").replace(
            " / CoT: ", " | CoT: "
        )
        ax.set_title(clean_title, pad=10)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        sns.despine(ax=ax)

        if i % 2 == 0:
            ax.set_ylabel("Harmful Prob.")
        if i >= n_groups - 2:
            ax.set_xlabel("% Tokens")

    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # fig.suptitle(f"Sample Trajectories (n={n_samples_per_group} per category)", fontsize=20)
    # With constrained_layout=True, we don't need fig.tight_layout here as much, but we can still use it if needed
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.with_suffix(".pdf"))
    print(f"Saved sample trajectories to: {save_path}")
    plt.close(fig)


def plot_lineplot(
    df: pd.DataFrame,
    hue_col: str,
    title: str,
    save_path: Path,
    has_prompt: bool = True,
):
    """Trajectory plot: spaghetti + median + IQR main panel, Δ harmful prob derivative panel."""
    # Use constrained_layout=False and manual subplots_adjust for tight Overleaf plots
    fig, (ax, ax_deriv) = plt.subplots(
        2,
        1,
        figsize=get_elongated_size(),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    rng = np.random.default_rng(42)

    # --- Spaghetti: thin individual trajectories per group ---
    for group in sorted(df[hue_col].unique()):
        color = HUE_PALETTE.get(group)
        gdf = df[df[hue_col] == group]
        sample_ids = gdf["sample_id"].unique()
        chosen = rng.choice(sample_ids, size=min(10, len(sample_ids)), replace=False)
        for sid in chosen:
            sdf = gdf[gdf["sample_id"] == sid].sort_values("pct_tokens")
            ax.plot(
                sdf["pct_tokens"],
                sdf["harmful_prob"],
                color=color,
                alpha=0.2,
                linewidth=0.5,
                zorder=1,
            )

    # --- Mean + 95% CI lines (seaborn handles bootstrapping) ---
    # Build per-group labels with sample counts for the legend
    n_per_group = df.groupby(hue_col)["sample_id"].nunique()
    df_copy = df.copy()
    df_copy[hue_col] = df_copy[hue_col].map(
        lambda g: f"{g} (n={n_per_group.get(g, 0)})"
    )
    palette_with_n = {
        f"{g} (n={n_per_group.get(g, 0)})": c for g, c in HUE_PALETTE.items()
    }

    sns.lineplot(
        data=df_copy,
        x="pct_tokens",
        y="harmful_prob",
        hue=hue_col,
        ax=ax,
        errorbar=("ci", 95),
        palette=palette_with_n,
        linewidth=2.5,
        zorder=3,
    )

    # --- Rate of change of group means (for derivative panel) ---
    for group_n, color in palette_with_n.items():
        gdf = df_copy[df_copy[hue_col] == group_n]
        means = gdf.groupby("pct_tokens")["harmful_prob"].mean()
        # Smooth the means before gradient to reduce noise
        smoothed_means = means.rolling(window=10, center=True, min_periods=1).mean()

        orig_group = group_n.split(" (n=")[0]
        ls = HUE_LINESTYLES.get(orig_group, "-")
        delta = np.gradient(smoothed_means.values, smoothed_means.index.values)
        ax_deriv.plot(means.index, delta, color=color, linestyle=ls, linewidth=1.5)

    # Divider line at prompt→CoT boundary
    if has_prompt:
        for a in (ax, ax_deriv):
            a.axvline(x=100, color="grey", linestyle="--", alpha=0.6, linewidth=1.5)
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax_deriv.set_xlabel("% Tokens (0–100% = Prompt, 100–200% = CoT)")
    else:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax_deriv.set_xlabel("% of CoT Tokens Processed")

    ax.set_ylabel("Harmfulness Probability")
    ax.set_ylim(0, 1)

    # Move legend to bottom, horizontal
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.55, 0.85),
        title=""
    )

    ax_deriv.set_ylabel(
        "Δ Harm. Prob.",
    )
    ax_deriv.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax_deriv.grid(True, alpha=0.1, linestyle="--")
    sns.despine(ax=ax)
    sns.despine(ax=ax_deriv)

    # Tight layout with room for legend at bottom
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path} + .pdf")
    print(f"Saved: {save_path} + .pdf")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot harmfulness probability over percentage of tokens processed."
    )
    parser.add_argument(
        "--probe-results",
        type=Path,
        required=True,
        help="Path to cummax probe predictions JSON (must include prompt_token_predictions).",
    )
    parser.add_argument(
        "--wg-results-folder",
        type=Path,
        default=DEFAULT_WG_RESULTS_FOLDER,
        help="Folder with WildGuard preds_cot.json and preds_response.json.",
    )
    parser.add_argument(
        "--save-folder",
        type=Path,
        default=None,
        help="Folder to save plots. Defaults to same folder as --probe-results.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=NUM_BINS,
        help="Number of bins per region (prompt / CoT).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_plot_style()

    probe_data = load_predictions(args.probe_results)

    n_samples = len(probe_data)
    print(f"Loaded {n_samples} samples from {args.probe_results}")

    _prompt_preds = probe_data[0].get("prompt_token_predictions")
    has_prompt = _prompt_preds is not None and len(_prompt_preds) > 0
    if not has_prompt:
        print(
            "No 'prompt_token_predictions' found — plotting CoT tokens only (0-100%)."
        )

    cot_labels, response_labels = load_wg_labels(args.wg_results_folder, n_samples)

    rows = build_rows(
        probe_data,
        cot_labels,
        response_labels,
        has_prompt=has_prompt,
        num_bins=args.num_bins,
    )
    df = pd.DataFrame(rows)
    print(f"Built {len(df)} rows for plotting")

    save_folder = args.save_folder or args.probe_results.parent
    save_folder.mkdir(parents=True, exist_ok=True)

    # Combined label: "Resp: ... / CoT: ..." with faithfulness terminology
    df["response+cot_harmfulness"] = df.apply(
        lambda row: f"Resp: {row['response_harmfulness']} / CoT: {'Faithful' if row['response_harmfulness'] == row['cot_harmfulness'] else 'Unfaithful'}",
        axis=1,
    )

    plot_lineplot(
        df,
        hue_col="response+cot_harmfulness",
        title="Harmfulness Prob. over Token Progress",
        save_path=save_folder / "harmfulness_over_tokens_by_response_and_cot.png",
        has_prompt=has_prompt,
    )

    plot_sample_trajectories(
        probe_data,
        cot_labels,
        response_labels,
        has_prompt=has_prompt,
        save_path=save_folder / "sample_trajectories_by_category.png",
    )

    print("Done!")


if __name__ == "__main__":
    main()
