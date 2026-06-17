import json
import matplotlib.pyplot as plt
import argparse
import os
import random
import pandas as pd
import numpy as np
from src.utils import load_predictions
from plotting.utils import set_plot_style

set_plot_style()
def load_data(filepath):
    return load_predictions(filepath)

def plot_single_sample(multi_sample, mil_sample, sample_id, output_dir):
    """Plots a single sample trajectory and saves it to a separate file."""
    multi_preds = multi_sample.get('token_predictions', [])
    mil_preds = mil_sample.get('token_predictions', [])

    if isinstance(multi_preds, str): multi_preds = json.loads(multi_preds)
    if isinstance(mil_preds, str): mil_preds = json.loads(mil_preds)

    multi_x = [p['token_idx'] for p in multi_preds]
    multi_y = [p['harmful_prob'] for p in multi_preds]

    mil_x = [p['token_idx'] for p in mil_preds]
    mil_y = [p['harmful_prob'] for p in mil_preds]

    plt.figure()
    plt.plot(multi_x, multi_y, label='Average Pooling', color='green', alpha=0.7, linewidth=1.5)
    plt.plot(mil_x, mil_y, label='Max Pooling', color='red', alpha=0.7, linewidth=1.5, linestyle='--')

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Token Index", fontsize=12)
    plt.ylabel("Probe Response", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    os.makedirs(os.path.join(output_dir, "individual"), exist_ok=True)
    out_path = os.path.join(output_dir, "individual", f"sample_{sample_id}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace(".png", ".pdf"))
    plt.close()

def plot_trajectories(multi_dict, mil_dict, selected_ids, output_dir):
    """Generates a 4x4 grid of trajectories and individual sample plots."""
    # 1. 4x4 Grid Plot
    fig, axes = plt.subplots(4, 4)
    axes_flat = axes.flatten()

    for i in range(16):
        ax = axes_flat[i]
        if i < len(selected_ids):
            sample_id = selected_ids[i]
            multi_sample = multi_dict[sample_id]
            mil_sample = mil_dict[sample_id]

            multi_preds = multi_sample.get('token_predictions', [])
            mil_preds = mil_sample.get('token_predictions', [])

            if isinstance(multi_preds, str): multi_preds = json.loads(multi_preds)
            if isinstance(mil_preds, str): mil_preds = json.loads(mil_preds)

            multi_x = [p['token_idx'] for p in multi_preds]
            multi_y = [p['harmful_prob'] for p in multi_preds]

            mil_x = [p['token_idx'] for p in mil_preds]
            mil_y = [p['harmful_prob'] for p in mil_preds]

            ax.plot(multi_x, multi_y, label='Average Pooling', color='green', alpha=0.7, linewidth=1.0)
            ax.plot(mil_x, mil_y, label='Max Pooling', color='red', alpha=0.7, linewidth=1.0, linestyle='--')

            # Remove title as requested (do not show sample id etc.)
            ax.set_title("")
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(axis='both', which='major', labelsize=5, pad=1)
            ax.grid(True, alpha=0.2)

            # Labels as requested
            if i >= 12:
                ax.set_xlabel("Token Index", fontsize=6)
            if i % 4 == 0:
                ax.set_ylabel("Probe Response", fontsize=6)
            if i == 0:
                ax.legend(fontsize=5, loc='upper right', frameon=True)

            # 2. Individual Plot for this sample
            plot_single_sample(multi_sample, mil_sample, sample_id, output_dir)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "trajectories_4x4_grid.png")
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace(".png", ".pdf"), dpi=300)
    plt.close()
    print(f"Saved 4x4 grid to {out_path}")
    print(f"Saved individual plots to {os.path.join(output_dir, 'individual')}")

def main(multilayer_file, mil_file, output_dir, num_samples=16, random_seed=42):
    print(f"Loading data from {multilayer_file}...")
    multi_data = load_data(multilayer_file)
    print(f"Loading data from {mil_file}...")
    mil_data = load_data(mil_file)

    multi_dict = {str(item['sample_id']): item for item in multi_data}
    mil_dict = {str(item['sample_id']): item for item in mil_data}

    common_ids = list(set(multi_dict.keys()).intersection(set(mil_dict.keys())))
    print(f"Found {len(common_ids)} common samples.")

    if len(common_ids) == 0:
        print("No common samples found!")
        return

    random.seed(random_seed)
    selected_ids = random.sample(common_ids, min(num_samples, len(common_ids)))

    os.makedirs(output_dir, exist_ok=True)
    plot_trajectories(multi_dict, mil_dict, selected_ids, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sample trajectories for a comparison between Average Pooling and Max Pooling probes.")
    parser.add_argument("--multilayer_file", type=str,
                        default="results/harmfulness/probe_multilayer/deepseek_r1_distill_llama_8b_template_with_prompt_and_response_wildguard/cot_token_probe_preds_cumsum.parquet",
                        help="Path to Multilayer (Average Pooling) probe predictions.")
    parser.add_argument("--mil_file", type=str,
                        default="results/harmfulness/mil_probe_eval/deepseek_r1_distill_llama_8b/wildguardtest_response_only/cot_mil_probe_preds_cummax.parquet",
                        help="Path to MIL (Max Pooling) probe predictions.")
    parser.add_argument("--output_dir", type=str, default="results/harmfulness/trajectory_plots", help="Directory to save the trajectory plots.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of random samples to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")

    args = parser.parse_args()

    main(args.multilayer_file, args.mil_file, args.output_dir, args.num_samples, args.seed)
