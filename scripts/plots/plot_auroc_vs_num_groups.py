import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.plots.utils import set_plot_style, get_flattened_figsize


def plot_auroc_vs_num_groups():
    csv_path = "results/qwen3_8b_feature_combinations.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Filter for relevant columns
    # num_groups, auc, mode

    set_plot_style()

    plt.figure(figsize=get_flattened_figsize())

    palette = {"harmfulness": "#8B0000", "math": "#FFB6C1"}  # Dark Red  # Light Pink

    # Use lineplot which automatically aggregates and shows 95% confidence interval
    # markers=True adds points at each num_groups
    ax = sns.lineplot(
        data=df,
        x="num_groups",
        y="auc",
        hue="mode",
        style="mode",
        palette=palette,
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
    )

    # Customize the appearance
    plt.title(
        "Performance Scaling with Feature Groups",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Number of Feature Groups Used", fontsize=14)
    plt.ylabel("Mean AUROC", fontsize=14)

    # Format axes
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(0.75, 1.0)  # Adjust based on the data seen in the CSV

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    # Capitalize labels for better look
    labels = [l.capitalize() for l in labels]
    plt.legend(
        handles,
        labels,
        title="Evaluation Task",
        title_fontsize=12,
        fontsize=11,
        frameon=True,
        loc="lower center",
        ncol=2
    )

    # Tight layout
    plt.tight_layout()

    # Save the plot
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, "auroc_vs_num_groups.pdf")
    png_path = os.path.join(output_dir, "auroc_vs_num_groups.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)

    print("Plot saved successfully:")
    print(f"  - {pdf_path}")
    print(f"  - {png_path}")


if __name__ == "__main__":
    plot_auroc_vs_num_groups()
