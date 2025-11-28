import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

def compute_cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def compute_mmd(x, y, gamma=1.0):
    xx = rbf_kernel(x.reshape(-1, 1), x.reshape(-1, 1), gamma=gamma)
    yy = rbf_kernel(y.reshape(-1, 1), y.reshape(-1, 1), gamma=gamma)
    xy = rbf_kernel(x.reshape(-1, 1), y.reshape(-1, 1), gamma=gamma)
    return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)

def run_statistical_tests(real_windows, synth_windows, sensor_type: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if synthetic windows are empty
    if len(synth_windows) == 0:
        print(f"Warning: No synthetic data available for {sensor_type}. Skipping statistical tests.")
        return pd.DataFrame()
    
    from ..eda.gait_signal_analysis import compute_features
    
    real_features = compute_features(real_windows)
    synth_features = compute_features(synth_windows)
    
    feature_cols = ["mean_mag", "std_mag", "rms_mag", "jerk_std", "dominant_freq", "corr_xy", "corr_xz", "corr_yz"]
    
    # Check if synth_features has the required columns
    if len(synth_features) == 0 or not all(col in synth_features.columns for col in feature_cols):
        print(f"Warning: Synthetic features are empty or incomplete for {sensor_type}. Skipping statistical tests.")
        return pd.DataFrame()
    
    results = []
    
    for feat_name in feature_cols:
        real_vals = real_features[feat_name].values
        synth_vals = synth_features[feat_name].values
        
        ks_stat, ks_pval = ks_2samp(real_vals, synth_vals)
        cohens_d = compute_cohens_d(real_vals, synth_vals)
        wass_dist = wasserstein_distance(real_vals, synth_vals)
        
        results.append({
            "feature": feat_name,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "cohens_d": cohens_d,
            "wasserstein": wass_dist
        })
    
    mmd_values = []
    for feat_name in feature_cols:
        real_vals = real_features[feat_name].values
        synth_vals = synth_features[feat_name].values
        mmd = compute_mmd(real_vals, synth_vals)
        mmd_values.append(mmd)
    
    results_df = pd.DataFrame(results)
    results_df["mmd"] = mmd_values
    
    output_path = os.path.join(save_dir, f"statistical_tests_{sensor_type.lower()}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Statistical tests complete. Results saved to {output_path}")
    print(results_df.to_string())
    
    return results_df

