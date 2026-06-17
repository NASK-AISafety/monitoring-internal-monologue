import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.plots.utils import set_plot_style, get_flattened_figsize


def clean_model_name(name):
    """Simplifies model names for display."""
    if "deepseek" in name.lower():
        return "Llama-8B-R1-Distil"
    if "qwen" in name.lower():
        if "_4b" in name.lower():
            return "Qwen 4B"
        if "8b" in name.lower():
            return "Qwen 8B"
        if "14b" in name.lower():
            return "Qwen 14B"
    return name


def get_dataset_group(exp_id):
    """Categorizes experiment IDs into high-level groups."""
    exp_id = exp_id.lower()
    if "minerva_math" in exp_id:
        return "MATH"
    if "gsm8k" in exp_id:
        return "GSM8K"
    if "aegis" in exp_id:
        return "Aegis"
    if "wildguard" in exp_id:
        return "WGMix"
    return None


def get_task_type(group):
    """Categorizes groups into Harmfulness or Math."""
    if group in ["Aegis", "WGMix"]:
        return "Harmfulness"
    if group in ["MATH", "GSM8K"]:
        return "Math"
    return None


def load_and_prepare_data(pattern, value_col, pool_over_prompt_only=False):
    """Loads CSV files matching a pattern and prepares the dataframe."""
    files = glob.glob(pattern)
    all_dfs = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r"(\d+)(pct|tokens)", basename)
        if not match:
            continue
        val = int(match.group(1))

        df = pd.read_csv(f)
        df[value_col] = val
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Exclude scenario_c and gsm8k (following user request)
    df = df[~df["experiment_id"].str.contains("scenario_c", case=False)]
    # df = df[~df['experiment_id'].str.contains('gsm8k', case=False)]

    if pool_over_prompt_only:
        df = df[
            df["experiment_id"].str.contains(
                "prompt_and_response_pool_over_prompt", case=False
            )
        ]

    df["Dataset Group"] = df["experiment_id"].apply(get_dataset_group)
    df = df.dropna(subset=["Dataset Group"])
    df["Task Type"] = df["Dataset Group"].apply(get_task_type)

    df["Trajectory-Based"] = df[["lr_auc", "rf_auc", "xgb_auc"]].max(axis=1)

    # Flip baseline AUROC if < 0.5
    df.loc[df["baseline_auc"] < 0.5, "baseline_auc"] = (
        1 - df.loc[df["baseline_auc"] < 0.5, "baseline_auc"]
    )

    df = df.rename(columns={"baseline_auc": "Probe"})
    df["Model Name"] = df["model"].apply(clean_model_name)

    return df


def plot_on_ax(ax, df, x_col, task_type, title, model_colors, model_order):
    data_subset = df[df["Task Type"] == task_type].copy()
    if data_subset.empty:
        ax.set_title(f"No {task_type} data for {title}")
        return

    # Melt to long format to use seaborn's hue and style
    melted = data_subset.melt(
        id_vars=["Model Name", x_col],
        value_vars=["Probe", "Trajectory-Based"],
        var_name="Method",
        value_name="AUROC",
    )

    # Define markers and dashes for methods
    method_markers = {"Probe": "X", "Trajectory-Based": "o"}
    method_dashes = {
        "Probe": (2, 2),
        "Trajectory-Based": "",
    }  # (2,2) is dashed, '' is solid

    sns.lineplot(
        data=melted,
        x=x_col,
        y="AUROC",
        hue="Model Name",
        style="Method",
        palette=model_colors,
        hue_order=model_order,
        markers=method_markers,
        dashes=method_dashes,
        ax=ax,
        errorbar=None,
        markersize=8,
        linewidth=2.5,
    )

    ax.set_title(title)
    ax.set_ylabel("Mean AUROC")
    ax.set_xlabel(x_col)
    if x_col == "Percentage of CoT Tokens":
        ax.set_xticks([5, 10, 25, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.3)

    # Remove individual legend
    if ax.get_legend():
        ax.get_legend().remove()


def plot_trajectory_evolution(pool_over_prompt_only=False):
    # Load data
    df_pct = load_and_prepare_data(
        "results/classifier_performance_aggregation*pct.csv",
        "Percentage of CoT Tokens",
        pool_over_prompt_only=pool_over_prompt_only,
    )
    df_tokens = load_and_prepare_data(
        "results/classifier_performance_aggregation*tokens.csv",
        "Number of CoT Tokens",
        pool_over_prompt_only=pool_over_prompt_only,
    )

    if df_pct.empty and df_tokens.empty:
        print("No data found to plot.")
        return

    set_plot_style()
    model_order = ["Llama-8B-R1-Distil", "Qwen 4B", "Qwen 8B", "Qwen 14B"]
    # Use a cleaner palette
    palette = sns.color_palette("muted", len(model_order))
    model_colors = dict(zip(model_order, palette))

    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Helper to clean and format the legend
    def setup_fig_legend(fig, ax):
        handles, labels = ax.get_legend_handles_labels()
        # Filter out the headers "Model Name" and "Method" to save space
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if l not in ["Model Name", "Method"]:
                filtered_handles.append(h)
                filtered_labels.append(l)

        # Top legend with slight offset to avoid title overlap
        fig.legend(
            filtered_handles,
            filtered_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.9),
            ncol=6,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.2,
        )

    # 1. Plot Percentage-based evolution
    if not df_pct.empty:
        fig1, axes1 = plt.subplots(1, 2, figsize=get_flattened_figsize([2,1]))
        plot_on_ax(
            axes1[0],
            df_pct,
            "Percentage of CoT Tokens",
            "Harmfulness",
            "Harmfulness",
            model_colors,
            model_order,
        )
        plot_on_ax(
            axes1[1],
            df_pct,
            "Percentage of CoT Tokens",
            "Math",
            "Math",
            model_colors,
            model_order,
        )

        setup_fig_legend(fig1, axes1[0])

        plt.tight_layout()
        fig1.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
        fig1.savefig(
            os.path.join(output_dir, "trajectory_evolution_pct.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig1.savefig(
            os.path.join(output_dir, "trajectory_evolution_pct.pdf"),
            bbox_inches="tight",
        )
        print(f"Percentage plots saved to {output_dir}")

    # 2. Plot Token-based evolution
    if not df_tokens.empty:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12.8, 8.8))
        plot_on_ax(
            axes2[0],
            df_tokens,
            "Number of CoT Tokens",
            "Harmfulness",
            "Harmfulness",
            model_colors,
            model_order,
        )
        plot_on_ax(
            axes2[1],
            df_tokens,
            "Number of CoT Tokens",
            "Math",
            "Math",
            model_colors,
            model_order,
        )

        setup_fig_legend(fig2, axes2[0])

        plt.tight_layout()
        fig2.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
        fig2.savefig(
            os.path.join(output_dir, "trajectory_evolution_tokens.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(output_dir, "trajectory_evolution_tokens.pdf"),
            bbox_inches="tight",
        )
        print(f"Token plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-over-prompt",
        action="store_true",
        help="Only use prompt_and_response_pool_over_prompt experiments",
    )
    args = parser.parse_args()

    plot_trajectory_evolution(pool_over_prompt_only=args.pool_over_prompt)
