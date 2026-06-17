import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scripts.plots.utils import set_plot_style

set_plot_style()

def parse_harmfulness(pred_str: str) -> int:
    """
    Parses the prediction string to determine if the response is harmful.
    Returns 1 if harmful, 0 if safe, -1 if unknown.
    """
    if not isinstance(pred_str, str):
        return -1
    lower_pred = pred_str.lower()
    # Logic consistent with scripts/cot_eval/summarize_mil_probe_pred.py
    if "harmful response: yes" in lower_pred:
        return 1
    elif "harmful response: no" in lower_pred:
        return 0
    return -1

def clean_model_name(name):
    """Simplifies model names for display, matching plot_math_baseline_vs_adapter.py."""
    if 'deepseek' in name.lower():
        return 'Llama-8B-R1-Distil'
    if 'qwen' in name.lower():
        if '_4b' in name.lower():
            return 'Qwen 4B'
        if '8b' in name.lower():
            return 'Qwen 8B'
        if '14b' in name.lower():
            return 'Qwen 14B'
    return name

def main():
    parser = argparse.ArgumentParser(description="Plot percentage of Faithful CoT across models and tasks.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["wildguardtest", "aegis_test"], help="Task directories in eval_wildguard.")
    parser.add_argument("--base-dir", type=Path, default=Path("results/harmfulness/eval_wildguard"), help="Base directory for WildGuard results.")
    parser.add_argument("--output-dir", type=Path, default=Path("plotting/outputs"), help="Directory to save the plot.")
    args = parser.parse_args()

    # Original model IDs
    models = [
        "deepseek_r1_distill_llama_8b",
        "qwen3_4b",
        "qwen3_8b",
        "qwen3_14b"
    ]

    # Desired display order
    model_order = ['Llama-8B-R1-Distil', 'Qwen 4B', 'Qwen 8B', 'Qwen 14B']

    results = []

    for task in args.tasks:
        print(f"\nProcessing task: {task}")
        for model in models:
            # Check in task subdir first
            model_path = args.base_dir / task / model
            if not model_path.exists():
                # Fallback to base dir
                model_path = args.base_dir / model

            if not model_path.exists():
                print(f"Warning: Results not found for {model} in task {task} at {model_path}")
                continue

            cot_file = model_path / "preds_cot.json"
            resp_file = model_path / "preds_response.json"

            if not cot_file.exists() or not resp_file.exists():
                print(f"Warning: Missing preds_cot.json or preds_response.json for {model} in {model_path}")
                continue

            with open(cot_file, "r", encoding="utf-8") as f:
                cot_preds = json.load(f)["preds"]
            with open(resp_file, "r", encoding="utf-8") as f:
                resp_preds = json.load(f)["preds"]

            total = 0
            faithful = 0

            # Zip allows iterating over both simultaneously
            for cp, rp in zip(cot_preds, resp_preds):
                cot_label = parse_harmfulness(cp)
                resp_label = parse_harmfulness(rp)

                # Only count if both were successfully parsed
                if cot_label != -1 and resp_label != -1:
                    total += 1
                    if cot_label == resp_label:
                        faithful += 1

            percentage = (faithful / total * 100) if total > 0 else 0
            results.append({
                "Model": model,
                "Model Name": clean_model_name(model),
                "Task": task,
                "Faithful CoT (%)": percentage,
                "Total Valid Samples": total,
                "Faithful Samples": faithful
            })
            print(f"  Model: {model:30} | Faithful CoT: {percentage:6.2f}% ({faithful}/{total})")

    if not results:
        print("No results found to plot.")
        return

    df = pd.DataFrame(results)

    task_name_map = {
        "wildguardtest": "WildGuard",
        "aegis_test": "Aegis"
    }
    df["Task Name"] = df["Task"].map(lambda x: task_name_map.get(x, x))

    # Plotting
    set_plot_style()
    plt.figure()

    # Using a professional color palette
    ax = sns.barplot(data=df, x="Model Name", y="Faithful CoT (%)", hue="Task Name",
                     palette="viridis", order=model_order)

    # Annotate bars with percentage values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                           (p.get_x() + p.get_width() / 2., height),
                           ha = 'center', va = 'center',
                           xytext = (0, 10),
                           textcoords = 'offset points',
                           fontsize=10,
                           fontweight='bold')

    plt.ylabel("Faithful CoT (%)")
    plt.xlabel("Base Model")
    plt.ylim(0, 115) # Leave space for annotations
    plt.legend(title="Task", loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"faithful_cot_combined.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # Also save as PDF for reports
    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    print(f"\nCombined plots saved to:\n - {save_path}\n - {pdf_path}")


if __name__ == "__main__":
    main()
