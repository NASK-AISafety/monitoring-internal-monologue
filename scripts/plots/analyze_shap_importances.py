import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.plots.utils import set_plot_style, get_flattened_figsize

# Set standard project plot style
set_plot_style()


def clean_feature_name(name):
    """Make feature names more readable for the plot."""
    return name.replace("_", " ").title()


def main():
    csv_path = "results/shap_feature_importances.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Filter for specific experiment type
    target_exp = "prompt_and_response_pool_over_prompt"
    df = df[df["experiment_id"].str.contains(target_exp)]

    if df.empty:
        print(f"Error: No data found for experiment type '{target_exp}'")
        return

    # Separate modes
    modes = ["harmfulness", "math"]
    top_n = 10

    # Aggregated results
    results = {}

    for mode in modes:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            print(f"Warning: No data found for mode '{mode}'")
            continue

        # Aggregate mean_abs_shap across models and experiments
        agg_df = (
            mode_df.groupby("feature")["mean_abs_shap"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        results[mode] = agg_df.head(top_n)

    if not results:
        print("No data to plot.")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=get_flattened_figsize([2,1]), sharey=False)

    # Custom color palettes
    harm_palette = sns.light_palette("#C0392B", n_colors=top_n, reverse=True)  # Crimson
    math_palette = sns.light_palette("#2980B9", n_colors=top_n, reverse=True)  # Blue

    # Harmfulness Plot
    if "harmfulness" in results:
        harm_data = results["harmfulness"].copy()
        harm_data["feature"] = harm_data["feature"].apply(clean_feature_name)
        sns.barplot(
            data=harm_data,
            x="mean_abs_shap",
            y="feature",
            ax=axes[0],
            palette=harm_palette,
            hue="feature",
            legend=False,
        )
        axes[0].set_title("Top 10 Features for Harmfulness")
        axes[0].set_xlabel("|SHAP|")
        axes[0].set_ylabel("")

    # Math Plot
    if "math" in results:
        math_data = results["math"].copy()
        math_data["feature"] = math_data["feature"].apply(clean_feature_name)
        sns.barplot(
            data=math_data,
            x="mean_abs_shap",
            y="feature",
            ax=axes[1],
            palette=math_palette,
            hue="feature",
            legend=False,
        )
        axes[1].set_title("Top 10 Features for Math")
        axes[1].set_xlabel("|SHAP|")
        axes[1].set_ylabel("")

    # Global adjustments
    plt.tight_layout()

    # Save results
    output_plot = "results/shap_top_features_comparison.pdf"
    plt.savefig(output_plot, bbox_inches="tight", dpi=300)
    plt.savefig(output_plot.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

    print(f"Plot saved to {output_plot}")

    # Print summary to console
    print("\n" + "=" * 50)
    print("TOP FEATURES SUMMARY")
    print("=" * 50)
    for mode in modes:
        if mode in results:
            print(f"\n[{mode.upper()}]")
            for i, row in results[mode].iterrows():
                print(
                    f"{i+1:2d}. {row['feature']:30s} (SHAP: {row['mean_abs_shap']:.4f})"
                )
    print("=" * 50)


if __name__ == "__main__":
    main()
