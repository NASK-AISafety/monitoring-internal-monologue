"""Aggregate WildGuard harmfulness predictions against MIL probe harmful_prob scores.

Joins the per-turn WildGuard classification (harmful response: yes/no) with the
MIL probe's harmful_prob score for each turn, producing summary statistics and
a joined JSON output.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_wildguard_harmful_response(prediction: str) -> bool:
    """Extract the 'Harmful response' label from WildGuard output text."""
    match = re.search(r"Harmful response:\s*(yes|no)", prediction, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return False


def load_probe_scores(probe_path: Path) -> dict:
    """Load probe scores and index by (id, turn_idx)."""
    with open(probe_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indexed = {}
    for sample in data:
        for turn in sample["turn_scores"]:
            key = (sample["id"], turn["turn_idx"])
            indexed[key] = {
                "harmful_prob": turn["harmful_prob"],
                "has_think_tag": turn.get("has_think_tag", None),
                "cot_end_pos": turn.get("cot_end_pos", None),
            }
    return indexed


def load_wildguard_preds(wildguard_path: Path) -> dict:
    """Load WildGuard predictions and index by (id, turn_evaluated - 1)."""
    with open(wildguard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indexed = {}
    for entry in data:
        # WildGuard uses 1-based turn_evaluated, probe uses 0-based turn_idx
        key = (entry["id"], entry["turn_evaluated"] - 1)
        indexed[key] = {
            "wildguard_prediction": entry["wildguard_prediction"],
            "is_harmful": parse_wildguard_harmful_response(
                entry["wildguard_prediction"]
            ),
            "original_task": entry["original_task"],
            "total_turns": entry["total_turns"],
        }
    return indexed


def main(
    probe_path: Path,
    wildguard_path: Path,
    save_folder: Path,
):
    save_folder.mkdir(parents=True, exist_ok=True)

    probe_data = load_probe_scores(probe_path)
    wildguard_data = load_wildguard_preds(wildguard_path)

    # Join on (id, turn_idx)
    all_keys = set(probe_data.keys()) | set(wildguard_data.keys())
    common_keys = set(probe_data.keys()) & set(wildguard_data.keys())

    print(f"Probe entries: {len(probe_data)}")
    print(f"WildGuard entries: {len(wildguard_data)}")
    print(f"Common (joined) entries: {len(common_keys)}")
    print(f"Probe-only entries: {len(probe_data) - len(common_keys)}")
    print(f"WildGuard-only entries: {len(wildguard_data) - len(common_keys)}")

    joined = []
    harmful_probs_when_harmful = []
    harmful_probs_when_safe = []

    for key in sorted(common_keys):
        sample_id, turn_idx = key
        probe_entry = probe_data[key]
        wg_entry = wildguard_data[key]

        is_harmful = wg_entry["is_harmful"]
        harmful_prob = probe_entry["harmful_prob"]

        if is_harmful:
            harmful_probs_when_harmful.append(harmful_prob)
        else:
            harmful_probs_when_safe.append(harmful_prob)

        joined.append(
            {
                "id": sample_id,
                "turn_idx": turn_idx,
                "original_task": wg_entry["original_task"],
                "total_turns": wg_entry["total_turns"],
                "wildguard_is_harmful": is_harmful,
                "probe_harmful_prob": harmful_prob,
                "has_think_tag": probe_entry["has_think_tag"],
            }
        )

    # Save joined data
    joined_path = save_folder / "joined_wildguard_probe.json"
    with open(joined_path, "w", encoding="utf-8") as f:
        json.dump(joined, f, indent=2, ensure_ascii=True)

    # Compute summary statistics
    harmful_probs_when_harmful = np.array(harmful_probs_when_harmful)
    harmful_probs_when_safe = np.array(harmful_probs_when_safe)

    # Overall ROC AUC and Average Precision
    all_labels = np.concatenate(
        [
            np.ones(len(harmful_probs_when_harmful)),
            np.zeros(len(harmful_probs_when_safe)),
        ]
    )
    all_scores = np.concatenate([harmful_probs_when_harmful, harmful_probs_when_safe])

    if len(harmful_probs_when_harmful) > 0 and len(harmful_probs_when_safe) > 0:
        overall_roc_auc = float(roc_auc_score(all_labels, all_scores))
        overall_avg_precision = float(average_precision_score(all_labels, all_scores))
    else:
        overall_roc_auc = None
        overall_avg_precision = None

    summary = {
        "total_joined_entries": len(joined),
        "num_harmful_by_wildguard": len(harmful_probs_when_harmful),
        "num_safe_by_wildguard": len(harmful_probs_when_safe),
        "roc_auc": overall_roc_auc,
        "average_precision": overall_avg_precision,
        "probe_score_when_harmful": {
            "mean": (
                float(harmful_probs_when_harmful.mean())
                if len(harmful_probs_when_harmful) > 0
                else None
            ),
            "std": (
                float(harmful_probs_when_harmful.std())
                if len(harmful_probs_when_harmful) > 0
                else None
            ),
            "median": (
                float(np.median(harmful_probs_when_harmful))
                if len(harmful_probs_when_harmful) > 0
                else None
            ),
            "min": (
                float(harmful_probs_when_harmful.min())
                if len(harmful_probs_when_harmful) > 0
                else None
            ),
            "max": (
                float(harmful_probs_when_harmful.max())
                if len(harmful_probs_when_harmful) > 0
                else None
            ),
        },
        "probe_score_when_safe": {
            "mean": (
                float(harmful_probs_when_safe.mean())
                if len(harmful_probs_when_safe) > 0
                else None
            ),
            "std": (
                float(harmful_probs_when_safe.std())
                if len(harmful_probs_when_safe) > 0
                else None
            ),
            "median": (
                float(np.median(harmful_probs_when_safe))
                if len(harmful_probs_when_safe) > 0
                else None
            ),
            "min": (
                float(harmful_probs_when_safe.min())
                if len(harmful_probs_when_safe) > 0
                else None
            ),
            "max": (
                float(harmful_probs_when_safe.max())
                if len(harmful_probs_when_safe) > 0
                else None
            ),
        },
    }

    # Per-turn breakdown
    turn_stats = {}
    for entry in joined:
        t = entry["turn_idx"]
        if t not in turn_stats:
            turn_stats[t] = {"harmful_probs_harmful": [], "harmful_probs_safe": []}
        if entry["wildguard_is_harmful"]:
            turn_stats[t]["harmful_probs_harmful"].append(entry["probe_harmful_prob"])
        else:
            turn_stats[t]["harmful_probs_safe"].append(entry["probe_harmful_prob"])

    per_turn_summary = {}
    for t in sorted(turn_stats.keys()):
        h = np.array(turn_stats[t]["harmful_probs_harmful"])
        s = np.array(turn_stats[t]["harmful_probs_safe"])
        if len(h) > 0 and len(s) > 0:
            turn_labels = np.concatenate([np.ones(len(h)), np.zeros(len(s))])
            turn_scores = np.concatenate([h, s])
            turn_roc_auc = float(roc_auc_score(turn_labels, turn_scores))
            turn_avg_precision = float(
                average_precision_score(turn_labels, turn_scores)
            )
        else:
            turn_roc_auc = None
            turn_avg_precision = None

        per_turn_summary[f"turn_{t}"] = {
            "num_harmful": len(h),
            "num_safe": len(s),
            "probe_mean_when_harmful": float(h.mean()) if len(h) > 0 else None,
            "probe_mean_when_safe": float(s.mean()) if len(s) > 0 else None,
            "roc_auc": turn_roc_auc,
            "average_precision": turn_avg_precision,
        }

    summary["per_turn"] = per_turn_summary

    summary_path = save_folder / "aggregation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"AGGREGATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total joined entries: {summary['total_joined_entries']}")
    print(f"Harmful by WildGuard: {summary['num_harmful_by_wildguard']}")
    print(f"Safe by WildGuard:    {summary['num_safe_by_wildguard']}")
    print(
        f"ROC AUC:              {summary['roc_auc']:.4f}"
        if summary["roc_auc"] is not None
        else "ROC AUC: N/A"
    )
    print(
        f"Average Precision:    {summary['average_precision']:.4f}"
        if summary["average_precision"] is not None
        else "Average Precision: N/A"
    )
    print()
    print(f"Probe score when WildGuard says HARMFUL:")
    for k, v in summary["probe_score_when_harmful"].items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
    print(f"Probe score when WildGuard says SAFE:")
    for k, v in summary["probe_score_when_safe"].items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
    print()
    print("Per-turn breakdown:")
    for turn_key, stats in per_turn_summary.items():
        h_mean = (
            f"{stats['probe_mean_when_harmful']:.4f}"
            if stats["probe_mean_when_harmful"] is not None
            else "N/A"
        )
        s_mean = (
            f"{stats['probe_mean_when_safe']:.4f}"
            if stats["probe_mean_when_safe"] is not None
            else "N/A"
        )
        roc = f"{stats['roc_auc']:.4f}" if stats.get("roc_auc") is not None else "N/A"
        ap = (
            f"{stats['average_precision']:.4f}"
            if stats.get("average_precision") is not None
            else "N/A"
        )
        print(
            f"  {turn_key}: harmful={stats['num_harmful']} (probe_mean={h_mean}), "
            f"safe={stats['num_safe']} (probe_mean={s_mean}), ROC_AUC={roc}, AP={ap}"
        )

    print(f"\nSaved joined data to: {joined_path}")
    print(f"Saved summary to: {summary_path}")

    # Plot results
    plot_results(
        harmful_probs_when_harmful,
        harmful_probs_when_safe,
        turn_stats,
        save_folder,
        joined,
    )


def plot_results(
    harmful_probs_when_harmful: np.ndarray,
    harmful_probs_when_safe: np.ndarray,
    turn_stats: dict,
    save_folder: Path,
    joined: list,
):
    """Generate and save plots comparing probe scores vs WildGuard labels."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Plot 1: Histogram of probe scores ---
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 51)
    ax.hist(
        harmful_probs_when_safe,
        bins=bins,
        alpha=0.6,
        label=f"Safe (n={len(harmful_probs_when_safe)})",
        color="#2196F3",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        harmful_probs_when_harmful,
        bins=bins,
        alpha=0.6,
        label=f"Harmful (n={len(harmful_probs_when_harmful)})",
        color="#F44336",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Probe Harmful Probability", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of Probe Scores by WildGuard Label", fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(save_folder / "probe_score_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved histogram to: {save_folder / 'probe_score_distribution.png'}")

    # --- Plot 2: Per-turn mean probe score bar chart ---
    turns_sorted = sorted(turn_stats.keys())
    harmful_means = []
    safe_means = []
    turn_labels = []
    for t in turns_sorted:
        h = np.array(turn_stats[t]["harmful_probs_harmful"])
        s = np.array(turn_stats[t]["harmful_probs_safe"])
        harmful_means.append(float(h.mean()) if len(h) > 0 else 0.0)
        safe_means.append(float(s.mean()) if len(s) > 0 else 0.0)
        turn_labels.append(str(t))

    x = np.arange(len(turn_labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars_h = ax.bar(
        x - width / 2,
        harmful_means,
        width,
        label="Harmful (WG)",
        color="#F44336",
        alpha=0.8,
    )
    bars_s = ax.bar(
        x + width / 2, safe_means, width, label="Safe (WG)", color="#2196F3", alpha=0.8
    )
    ax.set_xlabel("Turn Index", fontsize=13)
    ax.set_ylabel("Mean Probe Harmful Prob", fontsize=13)
    ax.set_title("Mean Probe Score per Turn by WildGuard Label", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(turn_labels)
    ax.legend(fontsize=12)
    ax.set_ylim(0, min(1.0, max(harmful_means + safe_means) * 1.3 + 0.05))
    fig.tight_layout()
    fig.savefig(save_folder / "per_turn_mean_probe_score.png", dpi=150)
    plt.close(fig)
    print(
        f"Saved per-turn bar chart to: {save_folder / 'per_turn_mean_probe_score.png'}"
    )

    # --- Plot 3: Box plots per turn ---
    # Only include turns that have at least some harmful samples
    turns_with_harmful = [
        t for t in turns_sorted if len(turn_stats[t]["harmful_probs_harmful"]) > 0
    ]
    if turns_with_harmful:
        fig, ax = plt.subplots()
        positions = []
        data_boxes = []
        colors = []
        tick_positions = []
        tick_labels_list = []

        for i, t in enumerate(turns_with_harmful):
            h = turn_stats[t]["harmful_probs_harmful"]
            s = turn_stats[t]["harmful_probs_safe"]
            pos_h = i * 3
            pos_s = i * 3 + 1
            if len(h) > 0:
                data_boxes.append(h)
                positions.append(pos_h)
                colors.append("#F44336")
            if len(s) > 0:
                data_boxes.append(s)
                positions.append(pos_s)
                colors.append("#2196F3")
            tick_positions.append(i * 3 + 0.5)
            tick_labels_list.append(f"Turn {t}")

        bp = ax.boxplot(
            data_boxes,
            positions=positions,
            widths=0.8,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, rotation=45, ha="right")
        ax.set_ylabel("Probe Harmful Prob", fontsize=13)
        ax.set_title("Probe Score Distribution per Turn (Harmful vs Safe)", fontsize=14)

        # Legend
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor="#F44336", alpha=0.7, label="Harmful (WG)"),
                Patch(facecolor="#2196F3", alpha=0.7, label="Safe (WG)"),
            ],
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(save_folder / "per_turn_boxplot.png", dpi=150)
        plt.close(fig)
        print(f"Saved box plots to: {save_folder / 'per_turn_boxplot.png'}")

    # --- Plot 4: Line plot showing harmfulness scores per turn ---
    if joined:
        df = pd.DataFrame(joined)

        # Determine for each id if its final turn (max turn_idx for that id) was harmful
        idx_max_turn = df.groupby('id')['turn_idx'].idxmax()
        final_harmful_df = df.loc[idx_max_turn, ['id', 'wildguard_is_harmful']]
        final_harmful_dict = dict(zip(final_harmful_df['id'], final_harmful_df['wildguard_is_harmful']))

        df['final_response_is_harmful'] = df['id'].map(final_harmful_dict)

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df,
            x='turn_idx',
            y='probe_harmful_prob',
            hue='final_response_is_harmful',
            palette={True: '#F44336', False: '#2196F3'},
            marker='o',
            err_style='band',
            ax=ax
        )
        ax.set_xlabel("Turn Index", fontsize=13)
        ax.set_ylabel("Probe Harmful Probability", fontsize=13)
        ax.set_title("Harmfulness Score per Turn by Final Response Harmfulness", fontsize=14)

        # Adjust legend if handles are generated
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            new_labels = ["Harmful" if str(l) == "True" else ("Safe" if str(l) == "False" else l) for l in labels]
            ax.legend(handles=handles, labels=new_labels, title="Final Response", fontsize=12, title_fontsize=12)

        ax.set_xticks(sorted(df['turn_idx'].unique()))

        fig.tight_layout()
        fig.savefig(save_folder / "per_turn_lineplot.png", dpi=150)
        plt.close(fig)
        print(f"Saved line plot to: {save_folder / 'per_turn_lineplot.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate WildGuard harmfulness vs MIL probe scores."
    )
    parser.add_argument(
        "--probe_path",
        type=Path,
        required=True,
        help="Path to the probe scores JSON (decomposed_harm_probe_scores.json).",
    )
    parser.add_argument(
        "--wildguard_path",
        type=Path,
        required=True,
        help="Path to the WildGuard predictions JSON (preds_response_*.json).",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("results/decomposed_harm/aggregation"),
        help="Folder to save aggregated results.",
    )
    args = parser.parse_args()

    main(
        probe_path=args.probe_path,
        wildguard_path=args.wildguard_path,
        save_folder=args.save_folder,
    )
