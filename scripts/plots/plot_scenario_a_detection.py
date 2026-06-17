import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.plots.utils import get_flattened_figsize, set_plot_style


def clean_model_name(name):
    """Simplifies model names for display."""
    model_mapping = {
        "deepseek_r1_distill_llama_8b": "R1-Llama-8B",
        "qwen3_4b": "Qwen3-4B",
        "qwen3_8b": "Qwen3-8B",
        "qwen3_14b": "Qwen3-14B",
    }
    return model_mapping.get(name, name.replace("_", "-").title())


def get_dataset_group(exp_id):
    exp_id = exp_id.lower()
    if "aegis" in exp_id:
        return "Aegis"
    if "wildguard" in exp_id:
        return "WildGuard"
    return "Other"


def plot_scenario_a_detection():
    file_path = "results/harmfulness_discrepancy_performance.csv"
    if not os.path.exists(file_path):
        # Try test files
        test_files = [
            "results/test_discrepancy_std.csv",
            "results/test_discrepancy_scenarios.csv",
        ]
        found = False
        for tf in test_files:
            if os.path.exists(tf):
                file_path = tf
                found = True
                break
        if not found:
            print(f"Error: {file_path} not found.")
            return

    df = pd.read_csv(file_path)

    # Process metadata
    df["Dataset"] = df["experiment_id"].apply(get_dataset_group)
    df["Model Name"] = df["model"].apply(clean_model_name)

    # Filter to Aegis/WildGuard and exclude Scenario C
    df = df[df["Dataset"].isin(["Aegis", "WildGuard"])]
    df = df[~df["experiment_id"].str.contains("scenario_c")]

    # Calculate Best Trajectory-Based Detection Rate and its Std for Scenario A
    def get_best_scenario_a(row):
        vals = []
        prefixes = []
        for p in ["lr", "rf", "xgb"]:
            col = f"{p}_scenario_a_detection_rate"
            if col in row and pd.notnull(row[col]):
                vals.append(row[col])
                prefixes.append(p)
        if not vals:
            return None, None
        idx = np.argmax(vals)
        p = prefixes[idx]
        return row[f"{p}_scenario_a_detection_rate"], row.get(
            f"{p}_scenario_a_detection_rate_std", 0
        )

    df[["Best Trajectory Detection", "Best Std"]] = df.apply(
        get_best_scenario_a, axis=1, result_type="expand"
    )

    # Melt for plotting
    melted_rows = []
    for _, row in df.iterrows():
        # Baseline
        melted_rows.append(
            {
                "Model Name": row["Model Name"],
                "Dataset": row["Dataset"],
                "Method": "Probe (Baseline)",
                "Detection Rate": row["baseline_scenario_a_detection_rate"],
                "Std": row.get("baseline_scenario_a_detection_rate_std", 0),
            }
        )
        # Trajectory
        melted_rows.append(
            {
                "Model Name": row["Model Name"],
                "Dataset": row["Dataset"],
                "Method": "Trajectory-Based Classifier",
                "Detection Rate": row["Best Trajectory Detection"],
                "Std": row["Best Std"],
            }
        )

    df_melt = pd.DataFrame(melted_rows)

    # Average duplicate rows (e.g. from different experiment variants)
    df_melt = (
        df_melt.groupby(["Model Name", "Dataset", "Method"])
        .agg({"Detection Rate": "mean", "Std": "mean"})
        .reset_index()
    )

    # Sorting
    model_order = ["R1-Llama-8B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
    available_models = df_melt["Model Name"].unique()
    model_order = [m for m in model_order if m in available_models]

    dataset_groups = ["WildGuard", "Aegis"]
    dataset_groups = [d for d in dataset_groups if d in df_melt["Dataset"].unique()]

    set_plot_style()
    num_datasets = len(dataset_groups)
    if num_datasets == 0:
        print("No datasets found to plot.")
        return

    fig, axes = plt.subplots(
        1, num_datasets,
        # figsize=get_flattened_figsize([num_datasets, 1]),
        figsize=get_flattened_figsize([num_datasets, 1]),
        sharey=True
    )
    if num_datasets == 1:
        axes = [axes]

    # Colors and styles
    palette = sns.color_palette("husl", 4)
    probe_color = palette[2]  # Greenish/Blue
    hatch_pattern = "xxx"

    for i, ds_name in enumerate(dataset_groups):
        ax = axes[i]
        ds_subset = df_melt[df_melt["Dataset"] == ds_name]

        if ds_subset.empty:
            continue

        x = np.arange(len(model_order))
        bar_width = 0.35

        # Extract values
        m_probe = (
            ds_subset[ds_subset["Method"] == "Probe (Baseline)"]
            .set_index("Model Name")
            .reindex(model_order)
        )
        y_p, y_p_std = m_probe["Detection Rate"].values, m_probe["Std"].values

        m_best = (
            ds_subset[ds_subset["Method"] == "Trajectory-Based Classifier"]
            .set_index("Model Name")
            .reindex(model_order)
        )
        y_b, y_b_std = m_best["Detection Rate"].values, m_best["Std"].values

        x_p, x_b = x - bar_width / 2, x + bar_width / 2

        # Plot bars
        ax.bar(
            x_p,
            y_p,
            bar_width,
            color=probe_color,
            alpha=0.8,
            edgecolor="white",
            label="Probe" if i == 0 else "",
        )
        ax.bar(
            x_b,
            y_b,
            bar_width,
            color=probe_color,
            hatch=hatch_pattern,
            alpha=0.9,
            edgecolor="white",
            label="Trajectory" if i == 0 else "",
        )

        # Error bars (95% CI: 1.96 * Std)
        ax.errorbar(
            x_p,
            y_p,
            yerr=1.96 * y_p_std,
            fmt="none",
            c="black",
            capsize=2,
            elinewidth=0.8,
        )
        ax.errorbar(
            x_b,
            y_b,
            yerr=1.96 * y_b_std,
            fmt="none",
            c="black",
            capsize=2,
            elinewidth=0.8,
        )

        ax.set_title(ds_name, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Detection Rate" if i == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontweight="bold", rotation=0)
        ax.set_xlabel("Model")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=probe_color, label="Static Probe", alpha=0.8),
        Patch(
            facecolor=probe_color,
            hatch=hatch_pattern,
            label="Trajectory-Based Classifier",
            alpha=0.9,
            edgecolor="white",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
    )

    # fig.suptitle(
    #     "Unsafe response with unfaithful CoT detection rate", fontweight="bold", y=1.05
    # )
    plt.tight_layout(pad=0.5)

    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "scenario_a_detection_rate.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, "scenario_a_detection_rate.pdf"), bbox_inches="tight"
    )
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    plot_scenario_a_detection()
