import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scripts.plots.utils import set_plot_style, get_flattened_figsize

def clean_model_name(name):
    """Simplifies model names for display."""
    model_mapping = {
        'deepseek_r1_distill_llama_8b': 'R1-Llama-8B',
        'qwen3_4b': 'Qwen3-4B',
        'qwen3_8b': 'Qwen3-8B',
        'qwen3_14b': 'Qwen3-14B'
    }
    return model_mapping.get(name, name.replace('_', '-').title())

def get_group(exp_id):
    exp_id = exp_id.lower()
    if 'minerva_math' in exp_id: return 'Math'
    if 'gsm8k' in exp_id: return 'GSM8K'
    return None

def get_probe_type(exp_id):
    exp_id = exp_id.lower()
    if 'prompt_and_response' in exp_id: return 'Template-Based'
    if 'response_only' in exp_id: return 'Template-Based Responses'
    return 'Other'

def plot_math_baseline_vs_adapter():
    file_path = 'results/classifier_performance_aggregation.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    # Process metadata
    df['Dataset'] = df['experiment_id'].apply(get_group)
    df['Probe Type'] = df['experiment_id'].apply(get_probe_type)
    df['Model Name'] = df['model'].apply(clean_model_name)

    # Filter to math datasets
    df = df[df['Dataset'].isin(['Math', 'GSM8K'])]

    # Calculate Trajectory-Based Classifier for each row
    df['Best AUROC'] = df[['lr_auc', 'rf_auc', 'xgb_auc']].max(axis=1)
    def get_best_stats(row):
        vals = [row["lr_auc"], row["rf_auc"], row["xgb_auc"]]
        idx = np.argmax(vals)
        prefixes = ["lr", "rf", "xgb"]
        p = prefixes[idx]
        return row[f"{p}_auc_ci_lower"], row[f"{p}_auc_ci_upper"], row[f"{p}_auc_std"]

    df[["Best_CI_Lower", "Best_CI_Upper", "Best_Std"]] = df.apply(
        get_best_stats, axis=1, result_type="expand"
    )

    # Melt into two methods: Probe vs Trajectory
    melted_rows = []
    for _, row in df.iterrows():
        # Baseline (Probe)
        auc = row['baseline_auc']
        ci_l = row['baseline_auc_ci_lower']
        ci_u = row['baseline_auc_ci_upper']
        if auc < 0.5:
            auc = 1 - auc
            ci_l, ci_u = 1 - ci_u, 1 - ci_l

        melted_rows.append({
            'Model Name': row['Model Name'],
            'Dataset': row['Dataset'],
            'Probe Type': row['Probe Type'],
            'Method': 'Probe',
            'AUROC': auc,
            'Std': row['baseline_auc_std']
        })
        # Trajectory-Based
        melted_rows.append({
            'Model Name': row['Model Name'],
            'Dataset': row['Dataset'],
            'Probe Type': row['Probe Type'],
            'Method': 'Trajectory-Based Classifier',
            'AUROC': row['Best AUROC'],
            'Std': row['Best_Std']
        })
    df_melt = pd.DataFrame(melted_rows)

    # Average duplicate rows if any
    df_melt = df_melt.groupby(['Model Name', 'Dataset', 'Probe Type', 'Method']).agg({
        'AUROC': 'mean',
        'Std': 'mean'
    }).reset_index()

    # Ordering
    model_order = ['R1-Llama-8B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B']
    available_models = df_melt['Model Name'].unique()
    for m in available_models:
        if m not in model_order:
            model_order.append(m)

    probe_order = ['Template-Based']
    dataset_groups = ['Math', 'GSM8K']

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=get_flattened_figsize([2, 1]), sharey=True)

    # Matching safety plot's first color (Pink/Red) and hatch
    probe_color = sns.color_palette("husl", 4)[0]
    hatch_pattern = '///'

    for i, ds_name in enumerate(dataset_groups):
        ax = axes[i]
        ds_subset = df_melt[(df_melt['Dataset'] == ds_name) & (df_melt['Probe Type'] == 'Template-Based')]

        if ds_subset.empty:
            continue

        num_models = len(model_order)
        x = np.arange(num_models)
        bar_width = 0.35

        # Extract values
        m_probe = ds_subset[ds_subset['Method'] == 'Probe'].set_index('Model Name').reindex(model_order)
        y_p, y_p_std = m_probe['AUROC'].values, m_probe['Std'].values

        m_best = ds_subset[ds_subset['Method'] == 'Trajectory-Based Classifier'].set_index('Model Name').reindex(model_order)
        y_b, y_b_std = m_best['AUROC'].values, m_best['Std'].values

        x_p, x_b = x - bar_width/2, x + bar_width/2

        # Plot bars
        ax.bar(x_p, y_p, bar_width, color=probe_color, alpha=0.9, edgecolor='white', linewidth=0.1)
        ax.bar(x_b, y_b, bar_width, color=probe_color, hatch=hatch_pattern, alpha=0.9, edgecolor='white', linewidth=0.1)

        # Error bars (Parametric 95% CI: 1.96 * SE)
        ax.errorbar(x_p, y_p, yerr=1.96 * y_p_std, fmt='none', c='black', capsize=2, elinewidth=0.8)
        ax.errorbar(x_b, y_b, yerr=1.96 * y_b_std, fmt='none', c='black', capsize=2, elinewidth=0.8)

        ax.set_title(f"{ds_name} Dataset", fontweight='bold')
        ax.set_ylim(0.45, 1.)
        ax.set_ylabel('AUROC' if i == 0 else '')
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontweight='bold', rotation=0)
        ax.set_xlabel('Model')

    # Legend with exactly 2 things: Probe and Trajectory
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=probe_color, label='Template-Based Probe', alpha=0.9),
        Patch(facecolor=probe_color, hatch=hatch_pattern, label='Trajectory-Based Classifier', alpha=0.9, edgecolor='white', linewidth=0.2)
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'math_gsm8k_baseline_vs_adapter.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'math_gsm8k_baseline_vs_adapter.pdf'), bbox_inches='tight')
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_math_baseline_vs_adapter()
