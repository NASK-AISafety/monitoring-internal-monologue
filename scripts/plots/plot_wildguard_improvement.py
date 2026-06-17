import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.plots.utils import set_plot_style


def clean_experiment_name(name):
    """Simplifies the experiment_id for better display."""
    name = name.replace("_pool_over_prompt", "")
    if "wildguardtest" in name:
        name = name.replace("wildguardtest_", "WildGuard ")

    if "prompt_and_response" in name:
        name = name.replace("prompt_and_response", "P&R")
    if "response_only" in name:
        name = name.replace("response_only", "Resp")
    if "scenario_b" in name:
        name = name.replace("scenario_b", "Scenario B")
    if "scenario_c" in name:
        name = name.replace("scenario_c", "Scenario C")

    if "+" in name:
        parts = name.split("+")
        base_name = (
            parts[0].split(" P&R")[0].split(" Resp")[0].split(" Scenario")[0].strip()
        )
        if "wildguard" in base_name.lower():
            return "WildGuard (Merged)"
        return f"{base_name} (Merged)"

    return name


def plot_improvement():
    file1 = "results/classifier_performance_aggregation_merged_cv_optimized.csv"
    file2 = "results/classifier_performance_aggregation_cv_optimized.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True)

    # Filter for WildGuard
    df = df[df["experiment_id"].str.contains("wildguard", case=False)]

    # Apply cleaning
    df["Dataset"] = df["experiment_id"].apply(clean_experiment_name)

    # Skip Scenario C
    df = df[~df["Dataset"].str.contains("Scenario C")]

    # Calculate Improvements (Advantage)
    df["LR AUROC Δ"] = df["lr_auc"] - df["baseline_auc"]
    df["RF AUROC Δ"] = df["rf_auc"] - df["baseline_auc"]
    df["XGB AUROC Δ"] = df["xgb_auc"] - df["baseline_auc"]

    df["LR AP Δ"] = df["lr_ap"] - df["baseline_ap"]
    df["RF AP Δ"] = df["rf_ap"] - df["baseline_ap"]
    df["XGB AP Δ"] = df["xgb_ap"] - df["baseline_ap"]

    # Melt for easier plotting
    auc_df = df.melt(
        id_vars=["Dataset", "model"],
        value_vars=["LR AUROC Δ", "RF AUROC Δ", "XGB AUROC Δ"],
        var_name="Model Type",
        value_name="Improvement",
    )
    auc_df["Metric"] = "AUROC"

    ap_df = df.melt(
        id_vars=["Dataset", "model"],
        value_vars=["LR AP Δ", "RF AP Δ", "XGB AP Δ"],
        var_name="Model Type",
        value_name="Improvement",
    )
    ap_df["Metric"] = "AP"

    plot_df = pd.concat([auc_df, ap_df])
    plot_df["Model Type"] = (
        plot_df["Model Type"].str.replace(" AUROC Δ", "").str.replace(" AP Δ", "")
    )

    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), sharex=True)
    # sns.set_style("whitegrid")  # Already handled by set_plot_style()

    sns.barplot(
        data=plot_df[plot_df["Metric"] == "AUROC"],
        x="Dataset",
        y="Improvement",
        hue="Model Type",
        ax=axes[0],
    )
    axes[0].set_title("WildGuard: AUROC Improvement over Baseline", fontsize=14)
    axes[0].set_ylabel("Δ AUROC")
    axes[0].axhline(0, color="black", linewidth=0.8)

    sns.barplot(
        data=plot_df[plot_df["Metric"] == "AP"],
        x="Dataset",
        y="Improvement",
        hue="Model Type",
        ax=axes[1],
    )
    axes[1].set_title(
        "WildGuard: Average Precision Improvement over Baseline", fontsize=14
    )
    axes[1].set_ylabel("Δ AP")
    axes[1].axhline(0, color="black", linewidth=0.8)

    plt.xticks(rotation=15)
    plt.tight_layout()

    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "wildguard_improvement_metrics.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    plot_improvement()
