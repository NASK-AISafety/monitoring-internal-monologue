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

def get_dataset(exp_id):
    exp_id = exp_id.lower()
    if 'gsm8k' in exp_id: return 'GSM8K'
    if 'minerva' in exp_id: return 'Minerva Math'
    if 'wildguard' in exp_id: return 'Wildguardtest'
    if 'aegis' in exp_id: return 'Aegis'
    return 'Other'

def get_probe_type(exp_id):
    exp_id = exp_id.lower()
    if 'prompt_and_response' in exp_id: return 'Template-Based'
    if 'response_only' in exp_id: return 'Template-Based Responses'
    if 'scenario_b' in exp_id: return 'Messages-Based'
    if 'scenario_c' in exp_id: return 'Exact Model Messages-Based'
    return 'Other'

def plot_mean_improvement_per_model():
    file1 = 'results/classifier_performance_aggregation.csv'
    if not os.path.exists(file1):
        print(f"Error: {file1} not found.")
        return

    df = pd.read_csv(file1)

    # Process metadata
    df['Dataset'] = df['experiment_id'].apply(get_dataset)
    df['Probe Type'] = df['experiment_id'].apply(get_probe_type)
    df['Model Name'] = df['model'].apply(clean_model_name)

    # Calculate Best Classifier for each row
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

    # Melt into two methods: Probe vs Best
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
        # Trajectory-Based Classifier
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
    # Since we want SE-based CIs, we'll keep mean and also calculate SEM if multiple rows exist
    df_melt = df_melt.groupby(['Model Name', 'Dataset', 'Probe Type', 'Method']).agg({
        'AUROC': 'mean',
        'Std': 'mean' # Averaging the SEs is a reasonable proxy if n=1 per group
    }).reset_index()

    # Ordering
    model_order = ['R1-Llama-8B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B']
    # Append any missing models present in data
    available_models = df_melt['Model Name'].unique()
    for m in available_models:
        if m not in model_order:
            model_order.append(m)

    probe_order = ['Template-Based', 'Template-Based Responses', 'Messages-Based', 'Exact Model Messages-Based']
    method_order = ['Probe', 'Trajectory-Based Classifier']
    dataset_configs = [
        ('Wildguardtest', 'wildguardtest_performance_comparison'),
        ('Aegis', 'aegis_performance_comparison')
    ]

    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    # Custom colors for probe types matching the provided images
    probe_colors = sns.color_palette("husl", len(probe_order))
    color_map = dict(zip(probe_order, probe_colors))
    hatch_pattern = '///'

    # Combined figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=get_flattened_figsize([2,1]), sharey=True)

    for i, (ds_name, _) in enumerate(dataset_configs):
        ax = axes[i]
        ds_subset = df_melt[df_melt['Dataset'] == ds_name]

        if ds_subset.empty: continue

        num_models = len(model_order)
        num_probes = len(probe_order)
        x = np.arange(num_models)

        # We have num_probes groups, each group has 2 bars
        total_group_width = 0.8
        probe_type_width = total_group_width / num_probes
        bar_width = probe_type_width * 0.4

        for j, pt in enumerate(probe_order):
            offset = (j - (num_probes - 1) / 2) * probe_type_width
            pt_data = ds_subset[ds_subset['Probe Type'] == pt]

            if pt_data.empty: continue

            # Extract values
            m_probe = pt_data[pt_data['Method'] == 'Probe'].set_index('Model Name').reindex(model_order)
            y_p, y_p_std = m_probe['AUROC'].values, m_probe['Std'].values

            m_best = pt_data[pt_data['Method'] == 'Trajectory-Based Classifier'].set_index('Model Name').reindex(model_order)
            y_b, y_b_std = m_best['AUROC'].values, m_best['Std'].values

            x_p, x_b = x + offset - bar_width/2, x + offset + bar_width/2

            # Plot bars
            r1 = ax.bar(x_p, y_p, bar_width, color=color_map[pt], alpha=0.9, edgecolor='white', linewidth=0.1)
            r2 = ax.bar(x_b, y_b, bar_width, color=color_map[pt], hatch=hatch_pattern, alpha=0.9, edgecolor='white', linewidth=0.1)

            # Error bars (Parametric 95% CI: 1.96 * SE)
            ax.errorbar(x_p, y_p, yerr=1.96 * y_p_std, fmt='none', c='black', capsize=2, elinewidth=0.8)
            ax.errorbar(x_b, y_b, yerr=1.96 * y_b_std, fmt='none', c='black', capsize=2, elinewidth=0.8)

        ax.set_title(f"{ds_name} Dataset", fontweight='bold')
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel('AUROC' if i == 0 else '')
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontweight='bold', rotation=0)
        ax.set_xlabel('Model')

    # Custom Shared Legend with Probe Types and Methods
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[pt], label=pt) for pt in probe_order]
    legend_elements.append(Patch(facecolor='gray', label='Probe baseline', alpha=0.5))
    legend_elements.append(Patch(facecolor='gray', hatch=hatch_pattern, label='Trajectory-Based Classifier', alpha=0.5))

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(os.path.join(output_dir, 'combined_safety_performance_consolidated.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'combined_safety_performance_consolidated.pdf'), bbox_inches='tight')
    print(f"Combined consolidated safety plot saved to {output_dir}")

if __name__ == "__main__":
    plot_mean_improvement_per_model()
