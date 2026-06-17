import itertools
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks

FEATURE_GROUPS = {
    "global_stats": [
        "prompt_mean", "cot_mean", "prompt_max", "cot_max", "prompt_last", "cot_last",
        "cot_var", "cot_median", "cot_iqr", "cot_rms", "cot_last_to_max_ratio",
        "prompt_slope", "cot_slope", "cot_running_mean_slope", "prompt_late_slope",
        "cot_prop_high", "cot_prop_low", "cot_prop_mid"
    ],
    "shape_dynamics": [
        "cot_concavity", "cot_smoothed_slope", "cot_max_drawdown", "cot_recovery_ratio",
        "cot_delta_var", "cot_accel_mean", "cot_accel_var", "surge_speed_first_5_percent",
        "peak_to_end_drop", "terminal_derivative_max", "terminal_derivative_min",
        "terminal_derivative_smoothed"
    ],
    "tertiles": [
        "prompt_tertile_1_mean", "prompt_tertile_2_mean", "prompt_tertile_3_mean",
        "cot_tertile_1_mean", "cot_tertile_2_mean", "cot_tertile_3_mean",
        "cot_tertile_delta_1_to_2", "cot_tertile_delta_2_to_3", "cot_tertile_3_slope"
    ],
    "boundary": [
        "prompt_to_cot_jump", "boundary_max_spike", "boundary_max_dip",
        "boundary_volatility", "prompt_to_cot_trend_delta"
    ],
    "signal_processing": [
        "cot_peak_count", "cot_peaks_per_token", "cot_max_consecutive_70",
        "cot_max_consecutive_90", "cot_first_crossing_idx_80", "cot_dwell_time_70",
        "cot_autocorr_lag1", "cot_mean_crossing_rate"
    ],
    "landmarks": [
        "cot_argmax_percent", "cot_to_prompt_length_ratio",
        "cot_to_prompt_mean_ratio", "cot_to_prompt_max_ratio"
    ]
}


def _get_linear_slope(probs: np.ndarray) -> float:
    """Helper to calculate linear regression slope."""
    if len(probs) > 1:
        slope, _, _, _, _ = linregress(np.arange(len(probs)), probs)
        return float(slope)
    return 0.0


def process_sample(sample: dict) -> dict:
    """
    Extract features categorized into 6 groups:
    1. Global Statistical State
    2. Shape and Trend Dynamics
    3. Temporal Segmentation (Tertiles)
    4. Boundary Transients and Volatility
    5. Signal Processing and Sustained Intents
    6. Temporal and Relational Landmarks
    """
    num_cot_tokens = sample.get("num_cot_tokens", 0)

    # Identifiers and Metadata
    features = {
        "sample_id": sample.get("sample_id"),
        "label": sample.get("label"),
        "task": sample.get("task") or "unknown",
        "num_cot_tokens": num_cot_tokens,
        "response_label": sample.get("response_label"),
        "experiment_id": sample.get("experiment_id"),
        "model": sample.get("model"),
    }

    prompt_preds = sample.get("prompt_token_predictions", [])
    cot_preds = sample.get("token_predictions", [])

    prompt_probs = np.array([t.get("harmful_prob", 0.0) for t in prompt_preds])
    cot_probs = np.array([t.get("harmful_prob", 0.0) for t in cot_preds])

    # --- Group 1: Global Statistical State ---
    if len(prompt_probs) > 0:
        features["prompt_mean"] = float(np.mean(prompt_probs))
        features["prompt_max"] = float(np.max(prompt_probs))
        features["prompt_last"] = float(prompt_probs[-1])
        features["prompt_slope"] = _get_linear_slope(prompt_probs)
        if len(prompt_probs) >= 5:
            late_window = max(5, int(len(prompt_probs) * 0.2))
            features["prompt_late_slope"] = _get_linear_slope(prompt_probs[-late_window:])
        else:
            features["prompt_late_slope"] = features["prompt_slope"]
    else:
        features["prompt_mean"] = 0.0
        features["prompt_max"] = 0.0
        features["prompt_last"] = 0.0
        features["prompt_slope"] = 0.0
        features["prompt_late_slope"] = 0.0

    if len(cot_probs) > 0:
        features["cot_mean"] = float(np.mean(cot_probs))
        features["cot_max"] = float(np.max(cot_probs))
        features["cot_var"] = float(np.var(cot_probs))
        features["cot_last"] = float(cot_probs[-1])
        features["cot_last_to_max_ratio"] = float(cot_probs[-1] / (features["cot_max"] + 1e-9))
        features["cot_slope"] = _get_linear_slope(cot_probs)
        running_mean = np.cumsum(cot_probs) / (np.arange(len(cot_probs)) + 1)
        features["cot_running_mean_slope"] = _get_linear_slope(running_mean)
        
        features["cot_prop_high"] = float(np.mean(cot_probs > 0.8))
        features["cot_prop_low"] = float(np.mean(cot_probs < 0.2))
        features["cot_prop_mid"] = float(np.mean((cot_probs >= 0.2) & (cot_probs <= 0.8)))
        
        if len(cot_probs) > 2:
            q75, q25 = np.percentile(cot_probs, [75, 25])
            features["cot_iqr"] = float(q75 - q25)
            features["cot_median"] = float(np.median(cot_probs))
        else:
            features["cot_iqr"] = 0.0
            features["cot_median"] = features["cot_mean"]
            
        features["cot_rms"] = float(np.sqrt(np.mean(cot_probs**2)))
    else:
        features["cot_mean"] = 0.0
        features["cot_max"] = 0.0
        features["cot_var"] = 0.0
        features["cot_last"] = 0.0
        features["cot_last_to_max_ratio"] = 0.0
        features["cot_slope"] = 0.0
        features["cot_running_mean_slope"] = 0.0
        features["cot_prop_high"] = 0.0
        features["cot_prop_low"] = 0.0
        features["cot_prop_mid"] = 0.0
        features["cot_iqr"] = 0.0
        features["cot_median"] = 0.0
        features["cot_rms"] = 0.0

    # --- Group 2: Shape and Trend Dynamics ---
    if len(cot_probs) > 1:
        # Trajectory Concavity
        if len(cot_probs) > 2:
            coeffs = np.polyfit(np.arange(len(cot_probs)), cot_probs, 2)
            features["cot_concavity"] = float(coeffs[0])
        else:
            features["cot_concavity"] = 0.0
            
        # Smoothed Slope
        if len(cot_probs) >= 5:
            window_size = 3
            weights = np.ones(window_size) / window_size
            cot_smoothed = np.convolve(cot_probs, weights, mode='valid')
            features["cot_smoothed_slope"] = _get_linear_slope(cot_smoothed)
        else:
            features["cot_smoothed_slope"] = features["cot_slope"]

        # Max Drawdown & Recovery
        rolling_max = np.maximum.accumulate(cot_probs)
        drawdowns = rolling_max - cot_probs
        max_drawdown_idx = int(np.argmax(drawdowns))
        features["cot_max_drawdown"] = float(drawdowns[max_drawdown_idx])
        if features["cot_max_drawdown"] > 0:
            trough_val = cot_probs[max_drawdown_idx]
            max_after_trough = np.max(cot_probs[max_drawdown_idx:])
            features["cot_recovery_ratio"] = float((max_after_trough - trough_val) / features["cot_max_drawdown"])
        else:
            features["cot_recovery_ratio"] = 1.0

        cot_diffs = np.diff(cot_probs)
        features["cot_delta_var"] = float(np.var(cot_diffs))
        
        if len(cot_diffs) > 1:
            cot_accel = np.diff(cot_diffs)
            features["cot_accel_mean"] = float(np.mean(cot_accel))
            features["cot_accel_var"] = float(np.var(cot_accel))
        else:
            features["cot_accel_mean"] = 0.0
            features["cot_accel_var"] = 0.0
            
        # Surge speed
        first_5_percent_len = max(1, int(len(cot_probs) * 0.05))
        first_5_percent_probs = cot_probs[:first_5_percent_len]
        features["surge_speed_first_5_percent"] = float(np.max(np.diff(first_5_percent_probs))) if len(first_5_percent_probs) > 1 else 0.0
        
        features["peak_to_end_drop"] = float(np.max(cot_probs) - cot_probs[-1])
        
        # Terminal Derivatives
        terminal_len = min(11, len(cot_probs))
        terminal_probs = cot_probs[-terminal_len:]
        if len(terminal_probs) > 1:
            diffs = np.diff(terminal_probs)
            features["terminal_derivative_max"] = float(np.max(diffs))
            features["terminal_derivative_min"] = float(np.min(diffs))
            
            # Smoothed Terminal
            window_size = 3
            if len(terminal_probs) >= window_size:
                weights = np.ones(window_size) / window_size
                smoothed_tail = np.convolve(terminal_probs, weights, mode='valid')
                features["terminal_derivative_smoothed"] = float(np.mean(np.diff(smoothed_tail))) if len(smoothed_tail) > 1 else 0.0
            else:
                features["terminal_derivative_smoothed"] = float(np.mean(np.diff(terminal_probs)))
        else:
            features["terminal_derivative_max"] = 0.0
            features["terminal_derivative_min"] = 0.0
            features["terminal_derivative_smoothed"] = 0.0
    else:
        features["cot_concavity"] = 0.0
        features["cot_smoothed_slope"] = 0.0
        features["cot_max_drawdown"] = 0.0
        features["cot_recovery_ratio"] = 0.0
        features["cot_delta_var"] = 0.0
        features["cot_accel_mean"] = 0.0
        features["cot_accel_var"] = 0.0
        features["surge_speed_first_5_percent"] = 0.0
        features["peak_to_end_drop"] = 0.0
        features["terminal_derivative_max"] = 0.0
        features["terminal_derivative_min"] = 0.0
        features["terminal_derivative_smoothed"] = 0.0

    # --- Group 3: Temporal Segmentation (Tertiles) ---
    if len(prompt_probs) > 0:
        prompt_tertiles = np.array_split(prompt_probs, 3)
        for i, slice_data in enumerate(prompt_tertiles):
            features[f"prompt_tertile_{i+1}_mean"] = float(np.mean(slice_data)) if len(slice_data) > 0 else 0.0
    else:
        for i in range(1, 4):
            features[f"prompt_tertile_{i}_mean"] = 0.0

    if len(cot_probs) > 0:
        cot_tertiles = np.array_split(cot_probs, 3)
        for i, slice_data in enumerate(cot_tertiles):
            features[f"cot_tertile_{i+1}_mean"] = float(np.mean(slice_data)) if len(slice_data) > 0 else 0.0
        
        features["cot_tertile_delta_1_to_2"] = features["cot_tertile_2_mean"] - features["cot_tertile_1_mean"]
        features["cot_tertile_delta_2_to_3"] = features["cot_tertile_3_mean"] - features["cot_tertile_2_mean"]
        features["cot_tertile_3_slope"] = _get_linear_slope(cot_tertiles[2])
    else:
        for i in range(1, 4):
            features[f"cot_tertile_{i}_mean"] = 0.0
        features["cot_tertile_delta_1_to_2"] = 0.0
        features["cot_tertile_delta_2_to_3"] = 0.0
        features["cot_tertile_3_slope"] = 0.0

    # --- Group 4: Boundary Transients and Volatility ---
    if len(prompt_probs) > 0 and len(cot_probs) > 0:
        features["prompt_to_cot_jump"] = float(cot_probs[0] - prompt_probs[-1])
        
        p_window_size = max(1, int(len(prompt_probs) * 0.01))
        c_window_size = max(1, int(len(cot_probs) * 0.01))
        boundary_window = np.concatenate((prompt_probs[-p_window_size:], cot_probs[:c_window_size]))
        if len(boundary_window) > 1:
            boundary_diffs = np.diff(boundary_window)
            features["boundary_max_spike"] = float(np.max(boundary_diffs))
            features["boundary_max_dip"] = float(np.min(boundary_diffs))
            features["boundary_volatility"] = float(np.max(np.abs(boundary_diffs)))
        else:
            features["boundary_max_spike"] = 0.0
            features["boundary_max_dip"] = 0.0
            features["boundary_volatility"] = 0.0
    else:
        features["prompt_to_cot_jump"] = 0.0
        features["boundary_max_spike"] = 0.0
        features["boundary_max_dip"] = 0.0
        features["boundary_volatility"] = 0.0
    
    features["prompt_to_cot_trend_delta"] = features.get("cot_slope", 0.0) - features.get("prompt_slope", 0.0)

    # --- Group 5: Signal Processing and Sustained Intents ---
    if len(cot_probs) > 0:
        peaks, _ = find_peaks(cot_probs, prominence=0.05)
        features["cot_peak_count"] = float(len(peaks))
        features["cot_peaks_per_token"] = float(len(peaks) / len(cot_probs))
        
        for thresh in [0.7, 0.9]:
            is_above = cot_probs > thresh
            key = f"cot_max_consecutive_{int(thresh*100)}"
            if np.any(is_above):
                features[key] = float(max(sum(1 for _ in g) for k, g in itertools.groupby(is_above) if k))
            else:
                features[key] = 0.0
                
        above_80 = np.where(cot_probs > 0.8)[0]
        features["cot_first_crossing_idx_80"] = float(above_80[0] / len(cot_probs)) if len(above_80) > 0 else -1.0
        features["cot_dwell_time_70"] = float(np.mean(cot_probs > 0.7))
        
        if len(cot_probs) > 1:
            if np.var(cot_probs) > 1e-9:
                corr = np.corrcoef(cot_probs[:-1], cot_probs[1:])[0, 1]
                features["cot_autocorr_lag1"] = float(corr) if not np.isnan(corr) else 0.0
            else:
                features["cot_autocorr_lag1"] = 0.0
            
            crossings = np.sum(np.diff(np.sign(cot_probs - features["cot_mean"])) != 0)
            features["cot_mean_crossing_rate"] = float(crossings / len(cot_probs))
        else:
            features["cot_autocorr_lag1"] = 0.0
            features["cot_mean_crossing_rate"] = 0.0
    else:
        features["cot_peak_count"] = 0.0
        features["cot_peaks_per_token"] = 0.0
        features["cot_max_consecutive_70"] = 0.0
        features["cot_max_consecutive_90"] = 0.0
        features["cot_first_crossing_idx_80"] = -1.0
        features["cot_dwell_time_70"] = 0.0
        features["cot_autocorr_lag1"] = 0.0
        features["cot_mean_crossing_rate"] = 0.0

    # --- Group 6: Temporal and Relational Landmarks ---
    if len(cot_probs) > 0:
        features["cot_argmax_percent"] = float(np.argmax(cot_probs) / len(cot_probs))
        features["cot_to_prompt_length_ratio"] = float(len(cot_probs) / max(1, len(prompt_probs)))
        
        if features.get("prompt_mean", 0.0) > 1e-9:
            features["cot_to_prompt_mean_ratio"] = float(features["cot_mean"] / features["prompt_mean"])
        else:
            features["cot_to_prompt_mean_ratio"] = 0.0
            
        if features.get("prompt_max", 0.0) > 1e-9:
            features["cot_to_prompt_max_ratio"] = float(features["cot_max"] / features["prompt_max"])
        else:
            features["cot_to_prompt_max_ratio"] = 0.0
    else:
        features["cot_argmax_percent"] = 0.0
        features["cot_to_prompt_length_ratio"] = 0.0
        features["cot_to_prompt_mean_ratio"] = 0.0
        features["cot_to_prompt_max_ratio"] = 0.0

    return features
