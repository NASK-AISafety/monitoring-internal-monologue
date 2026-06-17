import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.plots.utils import set_plot_style


def clean_model_name(name):
    """Simplifies model names for display."""
    model_mapping = {
        "deepseek_r1_distill_llama_8b": "R1-Llama-8B",
        "qwen3_4b": "Qwen3-4B",
        "qwen3_8b": "Qwen3-8B",
        "qwen3_14b": "Qwen3-14B",
    }
    return model_mapping.get(name, name.replace("_", "-").title())


def get_scenario_label(mode):
    if mode == "harmfulness":
        return "Generalization (Safety)"
    if mode == "math":
        return "Generalization (Math)"
    return "Other"


def get_probe_type(exp_id):
    exp_id = exp_id.lower()
    if "prompt_and_response" in exp_id:
        return "Template-Based"
    if "response_only" in exp_id:
        return "Template-Based Responses"
    if "scenario_b" in exp_id:
        return "Messages-Based"
    if "scenario_c" in exp_id:
        return "Exact Model Messages-Based"
    return "Other"


def plot_cross_dataset_improvement():
    file1 = "results/cross_dataset_classifier_performance.csv"
    if not os.path.exists(file1):
        print(f"Error: {file1} not found.")
        return

    df = pd.read_csv(file1)

    model_order = ["R1-Llama-8B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
    available_models = df["model"].apply(clean_model_name).unique()
    for m in available_models:
        if m not in model_order:
            model_order.append(m)

    # Process metadata
    df["Scenario"] = df["mode"].apply(get_scenario_label)
    df["Probe Type"] = df["experiment_id"].apply(get_probe_type)
    df["Model Name"] = df["model"].apply(clean_model_name)

    # Calculate Best Classifier for each row
    df["Best AUROC"] = df[["lr_auc", "rf_auc"]].max(
        axis=1
    )  # Note: assuming xgb might be skipped
    if "xgb_auc" in df.columns:
        df["Best AUROC"] = df[["lr_auc", "rf_auc", "xgb_auc"]].max(axis=1)

    def get_best_ci(row):
        cols = ["lr_auc", "rf_auc"]
        if "xgb_auc" in row.index and not pd.isna(row["xgb_auc"]):
            cols.append("xgb_auc")

        vals = [row[c] for c in cols]
        idx = np.argmax(vals)
        p = cols[idx].split("_")[0]
        return row[f"{p}_auc_ci_lower"], row[f"{p}_auc_ci_upper"]

    df[["Best_CI_Lower", "Best_CI_Upper"]] = df.apply(
        get_best_ci, axis=1, result_type="expand"
    )

    # Melt into two methods: Probe vs Best
    melted_rows = []
    for _, row in df.iterrows():
        # Baseline (Probe)
        auc = row["baseline_auc"]
        ci_l = row["baseline_auc_ci_lower"]
        ci_u = row["baseline_auc_ci_upper"]
        # Normalization if AUC < 0.5 (alignment with performance signature)
        if auc < 0.5:
            auc = 1 - auc
            ci_l, ci_u = 1 - ci_u, 1 - ci_l

        melted_rows.append(
            {
                "Model Name": row["Model Name"],
                "Scenario": row["Scenario"],
                "Probe Type": row["Probe Type"],
                "Method": "Probe",
                "AUROC": auc,
                "CI_Lower": ci_l,
                "CI_Upper": ci_u,
            }
        )
        # Trajectory-Based Classifier
        melted_rows.append(
            {
                "Model Name": row["Model Name"],
                "Scenario": row["Scenario"],
                "Probe Type": row["Probe Type"],
                "Method": "Trajectory-Based Classifier",
                "AUROC": row["Best AUROC"],
                "CI_Lower": row["Best_CI_Lower"],
                "CI_Upper": row["Best_CI_Upper"],
            }
        )
    df_melt = pd.DataFrame(melted_rows)

    # Custom colors
    set_plot_style()
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    hatch_pattern = "///"

    # Custom colors
    df_melt_raw = pd.DataFrame(melted_rows)

    def save_scenario_plot(scenario_name, df_raw, filename, average_probes=False):
        subset = df_raw[df_raw["Scenario"] == scenario_name]
        if subset.empty:
            print(f"No data for {scenario_name}")
            return

        group_cols = ["Model Name", "Method"]
        if not average_probes:
            group_cols.append("Probe Type")

        stats = subset.groupby(group_cols)["AUROC"].agg(["mean", "std", "count"]).reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))

        models = [m for m in model_order if m in stats["Model Name"].unique()]
        x = np.arange(len(models))
        width = 0.35

        y_p = []
        y_p_err = []
        y_b = []
        y_b_err = []

        for m in models:
            m_data = stats[stats["Model Name"] == m]

            p_row = m_data[m_data["Method"] == "Probe"]
            b_row = m_data[m_data["Method"] == "Trajectory-Based Classifier"]

            def get_val_err(row):
                if row.empty: return 0, 0
                m = row["mean"].iloc[0]
                s = row["std"].iloc[0]
                n = row["count"].iloc[0]
                # Using SEM (1.0 * std / sqrt(n)) instead of 95% CI (1.96 * SEM) to make them smaller as requested
                err = s / np.sqrt(n) if n > 1 and not pd.isna(s) else 0
                return m, err

            val_p, err_p = get_val_err(p_row)
            val_b, err_b = get_val_err(b_row)

            y_p.append(val_p)
            y_p_err.append(err_p)
            y_b.append(val_b)
            y_b_err.append(err_b)

        ax.bar(x - width/2, y_p, width, label="Probe Baseline", color="gray", alpha=0.6, edgecolor="white")
        ax.bar(x + width/2, y_b, width, label="Trajectory Classifier", color="royalblue", alpha=0.8, hatch=hatch_pattern, edgecolor="white")

        ax.errorbar(x - width/2, y_p, yerr=y_p_err, fmt="none", c="black", capsize=2, elinewidth=1)
        ax.errorbar(x + width/2, y_b, yerr=y_b_err, fmt="none", c="black", capsize=2, elinewidth=1)

        # ax.set_title("", fontweight="bold", fontsize=16)
        ax.set_ylabel("AUROC")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight="bold")
        ax.set_ylim(0.45, 1.05)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, filename.replace(".png", ".pdf")), bbox_inches="tight")
        print(f"Saved {scenario_name} plot to {filename} (and .pdf)")

    # Save Safety Plot (Averaged)
    save_scenario_plot("Generalization (Safety)", df_melt_raw, "cross_dataset_safety_averaged.png", average_probes=True)

    # --- 2. MATH PLOT (From minerva_loo_performance.csv) ---
    math_file = "results/minerva_loo_performance.csv"
    if os.path.exists(math_file):
        df_math_raw = pd.read_csv(math_file)
        df_math_raw["Model Name"] = df_math_raw["model"].apply(clean_model_name)

        # Melt math data to match format
        math_melted = []
        for _, row in df_math_raw.iterrows():
            # Find best classifier
            clf_cols = ["lr_auc", "rf_auc", "xgb_auc", "ensemble_auc"]
            best_auc = row[clf_cols].max()

            math_melted.append({
                "Model Name": row["Model Name"],
                "Scenario": "Generalization (Math)",
                "Method": "Probe",
                "AUROC": row["baseline_auc"]
            })
            math_melted.append({
                "Model Name": row["Model Name"],
                "Scenario": "Generalization (Math)",
                "Method": "Trajectory-Based Classifier",
                "AUROC": best_auc
            })

        df_math_melt = pd.DataFrame(math_melted)
        save_scenario_plot("Generalization (Math)", df_math_melt, "cross_dataset_math_averaged.png", average_probes=True)
    else:
        print("Math LOO file not found, skipping math plot.")

    print("Plots updated and saved separately.")


if __name__ == "__main__":
    plot_cross_dataset_improvement()
