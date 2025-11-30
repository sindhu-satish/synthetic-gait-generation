import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import umap
from typing import Tuple, List
from ..config import IMU_FS_HZ

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

def compute_magnitude(windows):
    if len(windows) == 0:
        return np.array([])
    windows = np.asarray(windows)
    if windows.ndim == 0:
        return np.array([])
    return np.sqrt((windows ** 2).sum(axis=-1) + 1e-8)

def compute_jerk(windows):
    mag = compute_magnitude(windows)
    if mag.size == 0:
        return np.array([])
    if mag.ndim < 2:
        windows = np.asarray(windows)
        if windows.ndim >= 2 and len(windows) > 0:
            n_windows = windows.shape[0]
            n_timesteps = windows.shape[1] if windows.shape[1] > 1 else 0
            return np.zeros((n_windows, max(0, n_timesteps - 1)))
        return np.array([])
    if mag.shape[1] < 2:
        return np.zeros((mag.shape[0], 0))
    jerk = np.diff(mag, axis=1)
    return jerk

def compute_features(windows, fs_hz: float = IMU_FS_HZ):
    features = []
    mag = compute_magnitude(windows)
    jerk = compute_jerk(windows)
    
    for i in range(len(windows)):
        win = windows[i]
        m = mag[i]
        j = jerk[i] if len(jerk) > i else np.array([])
        
        feat = {
            "mean_mag": np.mean(m),
            "std_mag": np.std(m),
            "rms_mag": np.sqrt(np.mean(m**2)),
            "energy_mag": float(np.sum(m ** 2)),
            "jerk_std": np.std(j) if len(j) > 0 else 0.0,
        }
        
        if len(win) >= 2:
            feat["corr_xy"] = np.corrcoef(win[:, 0], win[:, 1])[0, 1] if not np.isnan(np.corrcoef(win[:, 0], win[:, 1])[0, 1]) else 0.0
            feat["corr_xz"] = np.corrcoef(win[:, 0], win[:, 2])[0, 1] if not np.isnan(np.corrcoef(win[:, 0], win[:, 2])[0, 1]) else 0.0
            feat["corr_yz"] = np.corrcoef(win[:, 1], win[:, 2])[0, 1] if not np.isnan(np.corrcoef(win[:, 1], win[:, 2])[0, 1]) else 0.0
        else:
            feat["corr_xy"] = 0.0
            feat["corr_xz"] = 0.0
            feat["corr_yz"] = 0.0
        
        hann_window = np.hanning(len(m))
        m_detrended = m - np.mean(m)
        fft_vals = np.fft.rfft(m_detrended * hann_window)
        freqs = np.fft.rfftfreq(len(m), d=1/float(fs_hz))
        power = np.abs(fft_vals)**2
        if len(power) > 0:
            dominant_freq_idx = np.argmax(power[1:]) + 1
            feat["dominant_freq"] = freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0.0
        else:
            feat["dominant_freq"] = 0.0
        
        features.append(feat)
    
    return pd.DataFrame(features)

def load_windows_from_data(data_module, sensor_type: str):
    windows_list = []
    user_ids_list = []
    
    for split_name, dataset in [("train", data_module.train_ds), ("val", data_module.val_ds), ("test", data_module.test_ds)]:
        if dataset is not None:
            for i in range(len(dataset)):
                item = dataset[i]
                windows_list.append(item["window"].numpy())
                cond_idx = item["cond"]
                if isinstance(cond_idx, torch.Tensor):
                    cond_idx = cond_idx.item()
                user_id = data_module.rev_cond_map.get(cond_idx, cond_idx)
                user_ids_list.append(user_id)
    
    return np.array(windows_list), np.array(user_ids_list)

def run_eda(
    real_windows,
    real_user_ids,
    synth_windows,
    synth_user_ids,
    sensor_type: str,
    save_dir: str,
    fs_hz: float = IMU_FS_HZ,
    synth_windows_post=None,
):
    os.makedirs(save_dir, exist_ok=True)
    
    real_features = compute_features(real_windows, fs_hz=fs_hz)
    synth_features = compute_features(synth_windows, fs_hz=fs_hz)
    post_features = compute_features(synth_windows_post, fs_hz=fs_hz) if synth_windows_post is not None else None
    
    real_mag = compute_magnitude(real_windows)
    synth_mag = compute_magnitude(synth_windows)
    post_mag = compute_magnitude(synth_windows_post) if synth_windows_post is not None else np.array([])
    real_jerk = compute_jerk(real_windows)
    synth_jerk = compute_jerk(synth_windows)
    post_jerk = compute_jerk(synth_windows_post) if synth_windows_post is not None else np.array([])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Time Series Overlays - {sensor_type}", fontsize=16, fontweight='bold', y=0.995)
    
    for idx in range(min(4, len(real_windows))):
        ax = axes[idx // 2, idx % 2]
        time_real = np.arange(len(real_mag[idx])) / float(fs_hz)
        ax.plot(time_real, real_mag[idx], label="Real", alpha=0.7, linewidth=1.5)
        if len(synth_mag) > 0 and idx < len(synth_mag):
            time_synth = np.arange(len(synth_mag[idx])) / float(fs_hz)
            ax.plot(time_synth, synth_mag[idx], label="Synthetic (raw)", alpha=0.7, linewidth=1.5, linestyle="--")
        if len(post_mag) > 0 and idx < len(post_mag):
            time_post = np.arange(len(post_mag[idx])) / float(fs_hz)
            ax.plot(time_post, post_mag[idx], label="Synthetic (post)", alpha=0.7, linewidth=1.5, linestyle=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Window {idx+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "time_series_overlays_real_raw_post.png" if synth_windows_post is not None else "time_series_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Jerk Distribution - {sensor_type}", fontsize=16, fontweight='bold', y=0.995)
    
    real_jerk_flat = real_jerk.flatten()
    synth_jerk_flat = synth_jerk.flatten() if len(synth_jerk) > 0 else np.array([])
    post_jerk_flat = post_jerk.flatten() if len(post_jerk) > 0 else np.array([])
    
    axes[0].hist(real_jerk_flat, bins=50, alpha=0.6, label="Real", density=True, color="blue")
    if len(synth_jerk_flat) > 0:
        axes[0].hist(synth_jerk_flat, bins=50, alpha=0.45, label="Synthetic (raw)", density=True, color="red")
    if len(post_jerk_flat) > 0:
        axes[0].hist(post_jerk_flat, bins=50, alpha=0.45, label="Synthetic (post)", density=True, color="green")
    axes[0].set_xlabel("Jerk")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Jerk Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    try:
        if len(real_jerk_flat) > 0:
            kde_real = gaussian_kde(real_jerk_flat)
            kde_synth = gaussian_kde(synth_jerk_flat) if len(synth_jerk_flat) > 0 else None
            kde_post = gaussian_kde(post_jerk_flat) if len(post_jerk_flat) > 0 else None
            x_range = np.linspace(real_jerk_flat.min(), real_jerk_flat.max(), 200)
            axes[1].plot(x_range, kde_real(x_range), label="Real", linewidth=2, color="blue")
            if kde_synth is not None:
                axes[1].plot(x_range, kde_synth(x_range), label="Synthetic (raw)", linewidth=2, color="red", linestyle="--")
            if kde_post is not None:
                axes[1].plot(x_range, kde_post(x_range), label="Synthetic (post)", linewidth=2, color="green", linestyle=":")
            axes[1].set_xlabel("Jerk")
            axes[1].set_ylabel("Density")
            axes[1].set_title("Jerk KDE")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    except:
        pass
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "jerk_distribution_real_raw_post.png" if synth_windows_post is not None else "jerk_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Feature Boxplots - {sensor_type}", fontsize=16, fontweight='bold', y=0.995)
    feature_names = ["mean_mag", "std_mag", "rms_mag", "energy_mag"]
    
    for idx, feat_name in enumerate(feature_names):
        data_to_plot = [real_features[feat_name].values]
        labels = ["Real"]
        if len(synth_features) > 0 and feat_name in synth_features.columns:
            data_to_plot.append(synth_features[feat_name].values)
            labels.append("Synthetic (raw)")
        if post_features is not None and len(post_features) > 0 and feat_name in post_features.columns:
            data_to_plot.append(post_features[feat_name].values)
            labels.append("Synthetic (post)")
        axes[idx].boxplot(data_to_plot, labels=labels)
        axes[idx].set_ylabel(feat_name.replace("_", " ").title())
        axes[idx].set_title(f"{feat_name.replace('_', ' ').title()} Comparison")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_boxplots_real_raw_post.png" if synth_windows_post is not None else "feature_boxplots.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"Dominant Frequency Distribution - {sensor_type}", fontsize=16, fontweight='bold', y=0.995)
    ax.hist(real_features["dominant_freq"], bins=30, alpha=0.6, label="Real", density=True, color="blue")
    if len(synth_features) > 0 and "dominant_freq" in synth_features.columns:
        ax.hist(synth_features["dominant_freq"], bins=30, alpha=0.45, label="Synthetic (raw)", density=True, color="red")
    if post_features is not None and len(post_features) > 0 and "dominant_freq" in post_features.columns:
        ax.hist(post_features["dominant_freq"], bins=30, alpha=0.45, label="Synthetic (post)", density=True, color="green")
    ax.set_xlabel("Dominant Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dominant_frequency_real_raw_post.png" if synth_windows_post is not None else "dominant_frequency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    
    user_ids_sorted = sorted(np.unique(real_user_ids))[:20]
    fig, axes = plt.subplots(1, 3 if synth_windows_post is not None else 2, figsize=(18, 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    fig.suptitle(f"Per-User Mean Magnitude Distribution - {sensor_type}", fontsize=16, fontweight='bold', y=0.995)

    def _per_user_values(features_df, user_ids):
        return {uid: features_df.iloc[np.where(user_ids == uid)[0]]["mean_mag"].values for uid in np.unique(user_ids)}

    real_user_map = _per_user_values(real_features, real_user_ids)
    synth_user_map = _per_user_values(synth_features, synth_user_ids) if len(synth_features) > 0 else {}
    post_user_map = _per_user_values(post_features, synth_user_ids) if post_features is not None and len(post_features) > 0 else {}

    panels = [("Real", real_user_map), ("Synthetic (raw)", synth_user_map)]
    if synth_windows_post is not None:
        panels.append(("Synthetic (post)", post_user_map))

    for ax, (title, mp) in zip(axes, panels):
        data_to_plot = [mp.get(uid, np.array([])) for uid in user_ids_sorted]
        ax.boxplot(data_to_plot, labels=[str(uid) for uid in user_ids_sorted], showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("User ID")
        ax.set_ylabel("Mean Magnitude")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_user_magnitude_real_raw_post.png" if synth_windows_post is not None else "per_user_magnitude.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    feature_cols = ["mean_mag", "std_mag", "rms_mag", "jerk_std", "dominant_freq", "corr_xy", "corr_xz", "corr_yz"]
    real_feat_matrix = real_features[feature_cols].values
    
    if len(synth_features) > 0 and all(col in synth_features.columns for col in feature_cols):
        synth_feat_matrix = synth_features[feature_cols].values
        combined_features = np.vstack([real_feat_matrix, synth_feat_matrix])
        labels = np.array(["Real"] * len(real_feat_matrix) + ["Synthetic (raw)"] * len(synth_feat_matrix))
        user_labels = np.concatenate([real_user_ids, synth_user_ids])

        if post_features is not None and all(col in post_features.columns for col in feature_cols):
            post_feat_matrix = post_features[feature_cols].values
            combined_features = np.vstack([combined_features, post_feat_matrix])
            labels = np.array(list(labels) + ["Synthetic (post)"] * len(post_feat_matrix))
            user_labels = np.concatenate([user_labels, synth_user_ids])
    else:
        combined_features = real_feat_matrix
        labels = np.array(["Real"] * len(real_feat_matrix))
        user_labels = real_user_ids
    
    try:
        if len(combined_features) > 0:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(combined_features)//4))
            embedding = reducer.fit_transform(combined_features)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            fig.suptitle(f"UMAP Embedding of Physics Features - {sensor_type}", fontsize=16, fontweight='bold', y=0.995)
            real_mask = labels == "Real"
            ax.scatter(embedding[real_mask, 0], embedding[real_mask, 1], c="blue", alpha=0.5, label="Real", s=20)
            if "Synthetic (raw)" in labels:
                synth_mask = labels == "Synthetic (raw)"
                ax.scatter(embedding[synth_mask, 0], embedding[synth_mask, 1], c="red", alpha=0.5, label="Synthetic (raw)", s=20, marker="x")
            if "Synthetic (post)" in labels:
                post_mask = labels == "Synthetic (post)"
                ax.scatter(embedding[post_mask, 0], embedding[post_mask, 1], c="green", alpha=0.5, label="Synthetic (post)", s=20, marker=".")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "umap_embedding_real_raw_post.png" if synth_windows_post is not None else "umap_embedding.png"), dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"UMAP failed: {e}")
    
    print(f"EDA complete. Plots saved to {save_dir}")

