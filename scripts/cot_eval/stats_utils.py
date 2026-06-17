import numpy as np
from sklearn.utils import resample
import logging

logger = logging.getLogger(__name__)

def bootstrap_metrics(y_true, y_pred, metrics_dict, n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Calculate bootstrapping confidence intervals for multiple metrics.
    metrics_dict: { 'metric_name': metric_fn }
    """
    bootstrapped_results = {name: [] for name in metrics_dict}
    
    rng = np.random.RandomState(random_state)
    
    for _ in range(n_bootstrap):
        # resample indices with replacement
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]
        
        if len(np.unique(y_true_resampled)) < 2:
            continue
            
        for name, metric_fn in metrics_dict.items():
            try:
                score = metric_fn(y_true_resampled, y_pred_resampled)
                bootstrapped_results[name].append(score)
            except Exception as e:
                logger.warning(f"Error calculating metric {name} during bootstrap: {e}")
                continue
    
    final_results = {}
    for name in metrics_dict:
        scores = bootstrapped_results[name]
        if not scores:
            final_results[name] = None
            final_results[f"{name}_ci_lower"] = None
            final_results[f"{name}_ci_upper"] = None
        else:
            lower_bound = np.percentile(scores, (1 - ci) / 2 * 100)
            upper_bound = np.percentile(scores, (1 + ci) / 2 * 100)
            final_results[name] = np.mean(scores)
            final_results[f"{name}_ci_lower"] = lower_bound
            final_results[f"{name}_ci_upper"] = upper_bound
            final_results[f"{name}_std"] = np.std(scores)
            
    return final_results
