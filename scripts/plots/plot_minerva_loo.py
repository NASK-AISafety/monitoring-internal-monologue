import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.plots.utils import set_plot_style, get_flattened_figsize


def clean_model_name(name):
    model_mapping = {
        "deepseek_r1_distill_llama_8b": "R1-Llama-8B",
        "qwen3_4b": "Qwen3-4B",
        "qwen3_8b": "Qwen3-8B",
        "qwen3_14b": "Qwen3-14B",
    }
    return model_mapping.get(name, name.replace("_", "-").title())


def plot_minerva_loo():
    file_path = "results/minerva_loo_performance.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    df["Model Name"] = df["model"].apply(clean_model_name)

    # Calculate Best AUROC and its CI
    # We compare LR, RF, XGB, and Ensemble
    clf_cols = ["lr_auc", "rf_auc", "xgb_auc", "ensemble_auc"]
    df["Best AUROC"] = df[clf_cols].max(axis=1)

    def get_best_ci(row):
        vals = [row[c] for c in clf_cols]
        idx = np.argmax(vals)
        prefix = clf_cols[idx].replace("_auc", "")
        return row[f"{prefix}_auc_ci_lower"], row[f"{prefix}_auc_ci_upper"]

    df[["Best_CI_Lower", "Best_CI_Upper"]] = df.apply(
        get_best_ci, axis=1, result_type="expand"
    )

    set_plot_style()
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    models = df["Model Name"].unique()
    categories = df["eval_category"].unique()
    category_order = sorted(categories)

    # 1. Grouped Bar Plot: Baseline vs Max per Category
    # Following the style of plot_mean_improvement_per_model.py
    fig, axes = plt.subplots(
        len(models), 1, figsize=(get_flattened_figsize([2, len(models)])), sharex=False
    )
    if len(models) == 1:
        axes = [axes]

    hatch_pattern = "///"
    palette = sns.color_palette("husl", len(category_order))
    color_map = dict(zip(category_order, palette))

    for i, model in enumerate(models):
        ax = axes[i]
        m_df = (
            df[df["Model Name"] == model]
            .set_index("eval_category")
            .reindex(category_order)
            .reset_index()
        )

        num_cats = len(category_order)
        x = np.arange(num_cats)
        width = 0.35

        for j, cat in enumerate(category_order):
            row = m_df[m_df["eval_category"] == cat]
            if row.empty:
                continue
            row = row.iloc[0]

            x_p = j - width / 2
            x_b = j + width / 2

            y_p = row["baseline_auc"]
            y_p_l, y_p_h = row["baseline_auc_ci_lower"], row["baseline_auc_ci_upper"]

            y_b = row["Best AUROC"]
            y_b_l, y_b_h = row["Best_CI_Lower"], row["Best_CI_Upper"]

            # Plot bars
            ax.bar(
                x_p,
                y_p,
                width,
                color=color_map[cat],
                alpha=0.6,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.bar(
                x_b,
                y_b,
                width,
                color=color_map[cat],
                alpha=0.9,
                hatch=hatch_pattern,
                edgecolor="white",
                linewidth=0.5,
            )

            # Error bars
            ax.errorbar(
                x_p,
                y_p,
                yerr=[[max(0, y_p - y_p_l)], [max(0, y_p_h - y_p)]],
                fmt="none",
                c="black",
                capsize=2,
                elinewidth=0.8,
            )
            ax.errorbar(
                x_b,
                y_b,
                yerr=[[max(0, y_b - y_b_l)], [max(0, y_b_h - y_b)]],
                fmt="none",
                c="black",
                capsize=2,
                elinewidth=0.8,
            )

        ax.set_title(
            f"Minerva LOO Category Generalization: {model}",
            fontweight="bold",
            fontsize=14,
        )
        ax.set_ylabel("AUROC")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace("_", " ").title() for c in category_order], rotation=15
        )
        ax.set_ylim(0.45, 1.0)

    # Custom Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.6, label="Probe Baseline"),
        Patch(
            facecolor="gray",
            alpha=0.9,
            hatch=hatch_pattern,
            label="Trajectory Classifier (Best)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "minerva_loo_baseline_vs_max.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, "minerva_loo_baseline_vs_max.pdf"), bbox_inches="tight"
    )

    # 2. Summary Plot: Mean Baseline vs Mean Max per Model
    plt.figure(figsize=get_flattened_figsize())
    summary_stats = df.groupby("Model Name").agg({
        "baseline_auc": ["mean", "std", "count"],
        "Best AUROC": ["mean", "std", "count"]
    })

    x = np.arange(len(summary_stats))
    model_names = summary_stats.index.tolist()

    y_p = summary_stats[("baseline_auc", "mean")]
    # 95% CI of the mean across categories: 1.96 * std / sqrt(n)
    y_p_err = 1.96 * summary_stats[("baseline_auc", "std")] / np.sqrt(summary_stats[("baseline_auc", "count")])

    y_b = summary_stats[("Best AUROC", "mean")]
    y_b_err = 1.96 * summary_stats[("Best AUROC", "std")] / np.sqrt(summary_stats[("Best AUROC", "count")])

    plt.bar(
        x - width / 2,
        y_p,
        width,
        label="Mean Baseline",
        color="gray",
        alpha=0.6,
    )
    plt.bar(
        x + width / 2,
        y_b,
        width,
        label="Mean Trajectory",
        color="forestgreen",
        alpha=0.8,
        hatch=hatch_pattern,
    )

    # Error bars
    plt.errorbar(
        x - width / 2,
        y_p,
        yerr=y_p_err,
        fmt="none",
        c="black",
        capsize=2,
        elinewidth=0.8,
    )
    plt.errorbar(
        x + width / 2,
        y_b,
        yerr=y_b_err,
        fmt="none",
        c="black",
        capsize=2,
        elinewidth=0.8,
    )

    plt.title("MATH LOO Generalization", fontweight="bold", fontsize=16)
    plt.ylabel("Mean AUROC")
    plt.xticks(x, model_names)
    plt.ylim(0.45, 1.0)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "minerva_loo_summary_baseline_vs_max.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, "minerva_loo_summary_baseline_vs_max.pdf"),
        bbox_inches="tight",
    )

    # 3. Global Summary Plot: Single comparison across all models and categories
    plt.figure(figsize=get_flattened_figsize([1,1]))
    global_baseline = df["baseline_auc"].mean()
    gb_err = 1.96 * df["baseline_auc"].std() / np.sqrt(len(df))

    global_best = df["Best AUROC"].mean()
    gt_err = 1.96 * df["Best AUROC"].std() / np.sqrt(len(df))

    plt.bar(
        "Probe Baseline",
        global_baseline,
        color="gray",
        alpha=0.6,
        label="Probe Baseline",
    )
    plt.bar(
        "Trajectory",
        global_best,
        color="forestgreen",
        alpha=0.8,
        hatch=hatch_pattern,
        label="Trajectory",
    )

    # Error bars
    plt.errorbar(
        0,
        global_baseline,
        yerr=gb_err,
        fmt="none",
        c="black",
        capsize=3,
        elinewidth=1.0,
    )
    plt.errorbar(
        1,
        global_best,
        yerr=gt_err,
        fmt="none",
        c="black",
        capsize=3,
        elinewidth=1.0,
    )

    plt.title("Global Minerva LOO Generalization", fontweight="bold", fontsize=16)
    plt.ylabel("Mean AUROC")
    plt.ylim(0.45, 1.0)

    # Add text labels on top of bars for clarity
    text_offset = 0.04
    plt.text(
        0,
        global_baseline + gb_err + text_offset / 2,
        f"{global_baseline:.3f}",
        ha="center",
        fontweight="bold",
    )
    plt.text(
        1,
        global_best + gt_err + text_offset / 2,
        f"{global_best:.3f}",
        ha="center",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "minerva_loo_global_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, "minerva_loo_global_summary.pdf"), bbox_inches="tight"
    )
    print(f"Minerva LOO plots saved to {output_dir}")


if __name__ == "__main__":
    plot_minerva_loo()
