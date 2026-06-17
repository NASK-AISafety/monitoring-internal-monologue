import pandas as pd
import os

def clean_experiment_name(name):
    """Simplifies the experiment_id for better display in tables."""
    name = name.replace('_pool_over_prompt', '')
    name = name.replace('wildguardtest_', 'WildGuard ')
    name = name.replace('aegis_test_', 'Aegis ')
    name = name.replace('minerva_math_', 'Math (Minerva) ')
    name = name.replace('gsm8k_', 'Math (GSM8K) ')
    name = name.replace('prompt_and_response', 'P&R')
    name = name.replace('response_only', 'Resp')
    name = name.replace('scenario_b', 'Scenario B')
    name = name.replace('scenario_c', 'Scenario C')
    
    # Handle merged names (joined by +)
    if '+' in name:
        parts = name.split('+')
        # Extract the base name before the specific format (P&R, Resp, etc.)
        base_name = parts[0].split(' P&R')[0].split(' Resp')[0].split(' Scenario')[0].strip()
        return f"{base_name} (Merged)"
    
    return name

def generate_tables():
    file1 = 'results/classifier_performance_aggregation_merged_cv_optimized.csv'
    file2 = 'results/classifier_performance_aggregation_cv_optimized.csv'
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Error: One or both files not found: {file1}, {file2}")
        return

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Combine datasets
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Ensure output directory exists
    output_dir = 'results/latex_tables'
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'auroc': {
            'cols': ['baseline_auc', 'lr_auc', 'rf_auc', 'xgb_auc'],
            'labels': ['Baseline AUC', 'LR AUC', 'RF AUC', 'XGB AUC'],
            'caption': "Baseline vs Trained Model AUROC Comparison"
        },
        'ap': {
            'cols': ['baseline_ap', 'lr_ap', 'rf_ap', 'xgb_ap'],
            'labels': ['Baseline AP', 'LR AP', 'RF AP', 'XGB AP'],
            'caption': "Baseline vs Trained Model Average Precision Comparison"
        }
    }

    for metric_key, config in metrics.items():
        # Select relevant columns
        cols = ['model', 'experiment_id'] + config['cols']
        df_filtered = df[cols].copy()
        
        # Clean up names
        df_filtered['Dataset'] = df_filtered['experiment_id'].apply(clean_experiment_name)
        
        rename_dict = {'model': 'Model'}
        for i, col in enumerate(config['cols']):
            rename_dict[col] = config['labels'][i]
        
        df_filtered = df_filtered.rename(columns=rename_dict)
        
        # Format values to 3 decimal places
        for label in config['labels']:
            df_filtered[label] = df_filtered[label].map(lambda x: f"{x:.3f}")
            
        # Reorder columns
        df_final = df_filtered[['Model', 'Dataset'] + config['labels']]
        
        # Sort for better presentation
        df_final = df_final.sort_values(['Model', 'Dataset'])
        
        # Save CSV
        csv_path = os.path.join(output_dir, f'{metric_key}_table.csv')
        df_final.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
        
        # Save LaTeX
        latex_path = os.path.join(output_dir, f'{metric_key}_table.tex')
        df_final.to_latex(latex_path, index=False, caption=config['caption'], label=f"tab:{metric_key}_comparison")
        print(f"Saved LaTeX to {latex_path}")

if __name__ == "__main__":
    generate_tables()
