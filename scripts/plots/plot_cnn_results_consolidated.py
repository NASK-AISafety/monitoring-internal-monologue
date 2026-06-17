import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from pathlib import Path

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
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

def get_dataset(exp_id, mode):
    exp_id = exp_id.lower()
    if mode == 'math':
        if 'gsm8k' in exp_id: return 'GSM8K'
        if 'minerva' in exp_id: return 'Minerva Math'
    else:
        if 'wildguard' in exp_id: return 'WildGuard'
        if 'aegis' in exp_id: return 'Aegis'
    return 'Other'

def get_probe_type(exp_id):
    exp_id = exp_id.lower()
    if 'prompt_and_response' in exp_id: return 'Template-Based'
    if 'response_only' in exp_id: return 'Template-Based Responses'
    if 'scenario_b' in exp_id: return 'Messages-Based'
    if 'scenario_c' in exp_id: return 'Exact Model Messages-Based'
    return 'Other'

def prepare_data():
    file_feat = 'results/classifier_performance_aggregation.csv'
    file_cnn = 'results/cnn_performance_aggregation.csv'
    
    if not os.path.exists(file_feat) or not os.path.exists(file_cnn):
        print("Required CSV files not found.")
        return None

    df_feat = pd.read_csv(file_feat)
    df_cnn = pd.read_csv(file_cnn)
    
    common_cols = ["model", "experiment_id", "mode"]
    df = pd.merge(df_feat, df_cnn[common_cols + [c for c in df_cnn.columns if c.startswith("cnn_")]], on=common_cols, how="inner")
    
    df['Dataset'] = df.apply(lambda x: get_dataset(x['experiment_id'], x['mode']), axis=1)
    df['Probe Type'] = df['experiment_id'].apply(get_probe_type)
    df['Model Name'] = df['model'].apply(clean_model_name)
    
    # Calculate Best Feature-based AUROC
    df['Best Feature AUROC'] = df[['lr_auc', 'rf_auc', 'xgb_auc']].max(axis=1)
    def get_best_feat_std(row):
        vals = [row["lr_auc"], row["rf_auc"], row["xgb_auc"]]
        idx = np.argmax(vals)
        prefixes = ["lr", "rf", "xgb"]
        p = prefixes[idx]
        return row[f"{p}_auc_std"]

    df["Best Feature Std"] = df.apply(get_best_feat_std, axis=1)
    
    return df

def plot_consolidated(df, mode_name, datasets, filename, averaged=False):
    set_plot_style()
    model_order = ['R1-Llama-8B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B']
    
    fig, axes = plt.subplots(1, len(datasets), figsize=get_flattened_figsize([len(datasets), 1]), sharey=True)
    if len(datasets) == 1: axes = [axes]
    
    methods = ['Probe Baseline', 'Classifier on Trajectory Features', 'CNN on Trajectory']
    method_cols = ['baseline_auc', 'Best Feature AUROC', 'cnn_auc']
    std_cols = ['baseline_auc_std', 'Best Feature Std', 'cnn_auc_std']
    
    # Use distinct colors for methods
    colors = sns.color_palette("Set2", 3)
    method_colors = dict(zip(methods, colors))
    
    probe_order = ['Template-Based', 'Template-Based Responses', 'Messages-Based', 'Exact Model Messages-Based']
    hatch_styles = ['', '\\\\', '///', 'xxx']
    probe_hatches = dict(zip(probe_order, hatch_styles))
    
    for i, ds_name in enumerate(datasets):
        ax = axes[i]
        ds_subset = df[df['Dataset'] == ds_name]
        
        if ds_subset.empty: continue
        
        if averaged:
            # Aggregate over all probe types for each model
            agg_df = ds_subset.groupby('Model Name').agg({
                'baseline_auc': 'mean', 'baseline_auc_std': 'mean',
                'Best Feature AUROC': 'mean', 'Best Feature Std': 'mean',
                'cnn_auc': 'mean', 'cnn_auc_std': 'mean'
            }).reindex(model_order)
            
            x = np.arange(len(model_order))
            total_width = 0.6
            bar_width = total_width / 3
            
            for m_idx, (method, col, std_col) in enumerate(zip(methods, method_cols, std_cols)):
                offset = (m_idx - 1) * bar_width
                
                y = agg_df[col].values
                y_std = agg_df[std_col].values
                
                ax.bar(x + offset, y, bar_width, color=method_colors[method], alpha=0.9, edgecolor='black', linewidth=0.5)
                ax.errorbar(x + offset, y, yerr=1.96 * np.nan_to_num(y_std), fmt='none', c='black', capsize=2, elinewidth=0.8)
                
        else:
            # Unmerged: distinct bars per probe type
            available_probes = ds_subset['Probe Type'].unique()
            active_probes = [p for p in probe_order if p in available_probes]
            num_probes = len(active_probes)
            
            x = np.arange(len(model_order))
            total_width = 0.8
            group_width = total_width / num_probes
            bar_width = group_width * 0.3
            
            for p_idx, pt in enumerate(active_probes):
                pt_data = ds_subset[ds_subset['Probe Type'] == pt]
                if pt_data.empty: continue
                
                pt_data = pt_data.groupby('Model Name').agg({
                    'baseline_auc': 'mean', 'baseline_auc_std': 'mean',
                    'Best Feature AUROC': 'mean', 'Best Feature Std': 'mean',
                    'cnn_auc': 'mean', 'cnn_auc_std': 'mean'
                }).reindex(model_order)
                
                probe_offset = (p_idx - (num_probes - 1) / 2) * group_width
                hatch = probe_hatches[pt]
                
                for m_idx, (method, col, std_col) in enumerate(zip(methods, method_cols, std_cols)):
                    method_offset = (m_idx - 1) * bar_width
                    final_offset = probe_offset + method_offset
                    
                    y = pt_data[col].values
                    y_std = pt_data[std_col].values
                    
                    ax.bar(x + final_offset, y, bar_width, color=method_colors[method], hatch=hatch, alpha=0.9, edgecolor='black', linewidth=0.5)
                    ax.errorbar(x + final_offset, y, yerr=1.96 * np.nan_to_num(y_std), fmt='none', c='black', capsize=2, elinewidth=0.8)

        ax.set_title(f"{ds_name} Dataset", fontweight='bold')
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel('AUROC' if i == 0 else '')
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontweight='bold', rotation=0)
        ax.set_xlabel('Model')
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.2, linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    
    # Method legend
    for method in methods:
        legend_elements.append(Patch(facecolor=method_colors[method], label=method, alpha=0.9, edgecolor='black', linewidth=0.5))
        
    # Probe legend (if unmerged)
    if not averaged:
        active_pts = [pt for pt in probe_order if pt in df['Probe Type'].unique()]
        for pt in active_pts:
            legend_elements.append(Patch(facecolor='white', hatch=probe_hatches[pt], label=pt, alpha=0.9, edgecolor='black', linewidth=0.5))

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=min(3, len(legend_elements)), frameon=False)

    plt.tight_layout(rect=[0, 0.15, 1, 1] if not averaged and len(legend_elements) > 3 else [0, 0.08, 1, 1])
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, filename + '.pdf'), bbox_inches='tight')
    print(f"Saved {mode_name} plot to {output_dir}/{filename}.png")

def main():
    df = prepare_data()
    if df is None: return
    
    # 1. Math Plots
    math_df = df[df['mode'] == 'math']
    plot_consolidated(math_df, 'Math', ['GSM8K', 'Minerva Math'], 'cnn_comparison_math_unmerged', averaged=False)
    plot_consolidated(math_df, 'Math', ['GSM8K', 'Minerva Math'], 'cnn_comparison_math_averaged', averaged=True)
    
    # 2. Safety Plots
    safety_df = df[df['mode'] == 'harmfulness']
    plot_consolidated(safety_df, 'Safety', ['WildGuard', 'Aegis'], 'cnn_comparison_safety_unmerged', averaged=False)
    plot_consolidated(safety_df, 'Safety', ['WildGuard', 'Aegis'], 'cnn_comparison_safety_averaged', averaged=True)

if __name__ == "__main__":
    main()
