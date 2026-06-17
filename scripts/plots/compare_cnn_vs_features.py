import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compare CNN vs Best Feature-based method.")
    parser.add_argument("--classifier_csv", type=str, default="results/classifier_performance_aggregation.csv")
    parser.add_argument("--cnn_csv", type=str, default="results/cnn_performance_aggregation.csv")
    parser.add_argument("--output_dir", type=str, default="results/plots/comparison")
    parser.add_argument("--metric", type=str, default="auc", choices=["auc", "ap", "f1"])
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_feat = pd.read_csv(args.classifier_csv)
    df_cnn = pd.read_csv(args.cnn_csv)

    common_cols = ["model", "experiment_id", "mode"]
    df = pd.merge(df_feat, df_cnn[common_cols + [c for c in df_cnn.columns if c.startswith("cnn_")]], on=common_cols, how="inner")

    if df.empty:
        print("No matching experiments found.")
        return

    metric = args.metric
    feature_methods = ["lr", "rf", "xgb"]
    
    plot_data = []
    for _, row in df.iterrows():
        # Baseline
        plot_data.append({
            "Model": row["model"],
            "Experiment": row["experiment_id"],
            "Mode": row["mode"],
            "Method": "Baseline",
            "Value": row[f"baseline_{metric}"],
            "STD": row[f"baseline_{metric}_std"]
        })
        
        # Best Feature Model
        best_feat_val = -1
        best_feat_name = ""
        best_feat_std = 0
        
        for m in feature_methods:
            val = row[f"{m}_{metric}"]
            if val > best_feat_val:
                best_feat_val = val
                best_feat_name = f"Best Feature ({m.upper()})"
                best_feat_std = row[f"{m}_{metric}_std"]
        
        plot_data.append({
            "Model": row["model"],
            "Experiment": row["experiment_id"],
            "Mode": row["mode"],
            "Method": "Classifier on Trajectory Features",
            "Value": best_feat_val,
            "STD": best_feat_std
        })
        
        # CNN
        plot_data.append({
            "Model": row["model"],
            "Experiment": row["experiment_id"],
            "Mode": row["mode"],
            "Method": "CNN on Trajectory",
            "Value": row[f"cnn_{metric}"],
            "STD": row[f"cnn_{metric}_std"]
        })

    plot_df = pd.DataFrame(plot_data)

    for mode in plot_df["Mode"].unique():
        mode_df = plot_df[plot_df["Mode"] == mode]
        
        for model in mode_df["Model"].unique():
            model_df = mode_df[mode_df["Model"] == model]
            
            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            unique_exps = model_df["Experiment"].unique()
            x = np.arange(len(unique_exps))
            width = 0.25
            
            methods = ["Baseline", "Classifier on Trajectory Features", "CNN on Trajectory"]
            colors = ["#95a5a6", "#3498db", "#e74c3c"]
            
            for i, (method, color) in enumerate(zip(methods, colors)):
                method_data = model_df[model_df["Method"] == method]
                vals = []
                stds = []
                for exp in unique_exps:
                    row_data = method_data[method_data["Experiment"] == exp]
                    vals.append(row_data["Value"].iloc[0] if not row_data.empty else 0)
                    stds.append(row_data["STD"].iloc[0] if not row_data.empty else 0)
                
                plt.bar(x + (i - 1) * width, vals, width, label=method, yerr=stds, capsize=5, color=color, alpha=0.85)

            plt.xlabel("Experiment")
            plt.ylabel(f"Mean {metric.upper()}")
            plt.title(f"Performance Comparison: CNN vs Best Baseline - {model} ({mode})")
            plt.xticks(x, [e.split('_')[0] for e in unique_exps], rotation=0)
            plt.legend(title="Method", loc="lower right")
            plt.ylim(0.45, 1.05)
            plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            save_path = output_dir / f"cnn_vs_best_{model}_{mode}_{metric}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Saved plot to {save_path}")
            plt.close()

if __name__ == "__main__":
    main()
