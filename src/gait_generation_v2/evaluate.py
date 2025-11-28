"""
Evaluation utilities for Gait Generation V2.

Includes:
- Reconstruction quality metrics
- Statistical similarity metrics
- User classification evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .config import ConfigV2
from .conv_vae import ConvVAE
from .diffusion import LatentDDPM, EMA
from .preprocessing import GaitPreprocessor


def evaluate_reconstruction(
    vae: ConvVAE,
    windows: np.ndarray,
    user_indices: np.ndarray,
    cfg: ConfigV2,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Evaluate VAE reconstruction quality.
    
    Args:
        vae: Trained ConvVAE
        windows: (N, window_size, 4) test windows
        user_indices: (N,) user indices
        cfg: Configuration
        batch_size: Batch size for evaluation
        
    Returns:
        metrics: Dict with MSE, MAE, per-channel errors
    """
    vae.eval()
    device = cfg.device
    
    all_mse = []
    all_mae = []
    channel_mse = {0: [], 1: [], 2: [], 3: []}  # X, Y, Z, mag
    
    n_samples = len(windows)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_windows = torch.tensor(
                windows[i:i+batch_size], 
                dtype=torch.float32, 
                device=device
            )
            batch_users = torch.tensor(
                user_indices[i:i+batch_size], 
                dtype=torch.long, 
                device=device
            )
            
            # Encode and decode
            mu, logvar = vae.encode(batch_windows, batch_users)
            z = vae.reparameterize(mu, logvar)
            recon = vae.decode(z, batch_users)
            
            # Transpose recon to (B, T, C)
            recon = recon.transpose(1, 2)
            
            # Compute errors
            diff = (recon - batch_windows).cpu().numpy()
            
            all_mse.append(np.mean(diff ** 2))
            all_mae.append(np.mean(np.abs(diff)))
            
            for c in range(4):
                channel_mse[c].append(np.mean(diff[:, :, c] ** 2))
    
    metrics = {
        "mse": float(np.mean(all_mse)),
        "mae": float(np.mean(all_mae)),
        "mse_x": float(np.mean(channel_mse[0])),
        "mse_y": float(np.mean(channel_mse[1])),
        "mse_z": float(np.mean(channel_mse[2])),
        "mse_mag": float(np.mean(channel_mse[3])),
    }
    
    return metrics


def evaluate_statistical_similarity(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray
) -> Dict[str, float]:
    """
    Compare statistical properties of real vs synthetic data.
    
    Args:
        real_windows: (N, window_size, 4) real windows
        synthetic_windows: (M, window_size, 4) synthetic windows
        
    Returns:
        metrics: Dict with mean/std differences per channel
    """
    channel_names = ["X", "Y", "Z", "mag"]
    metrics = {}
    
    for c, name in enumerate(channel_names):
        real_flat = real_windows[:, :, c].flatten()
        synth_flat = synthetic_windows[:, :, c].flatten()
        
        # Mean difference
        mean_diff = abs(real_flat.mean() - synth_flat.mean())
        metrics[f"mean_diff_{name}"] = float(mean_diff)
        
        # Std difference
        std_diff = abs(real_flat.std() - synth_flat.std())
        metrics[f"std_diff_{name}"] = float(std_diff)
        
        # Percentile differences
        for p in [25, 50, 75]:
            real_p = np.percentile(real_flat, p)
            synth_p = np.percentile(synth_flat, p)
            metrics[f"p{p}_diff_{name}"] = float(abs(real_p - synth_p))
    
    return metrics


def evaluate_user_classification(
    real_windows: np.ndarray,
    real_user_ids: np.ndarray,
    synthetic_windows: np.ndarray,
    synthetic_user_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate user classification with and without synthetic data.
    
    Args:
        real_windows: (N, window_size, 4) real windows
        real_user_ids: (N,) real user IDs
        synthetic_windows: (M, window_size, 4) synthetic windows
        synthetic_user_ids: (M,) synthetic user IDs
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        metrics: Classification metrics
    """
    # Flatten windows to feature vectors
    real_features = real_windows.reshape(len(real_windows), -1)
    synth_features = synthetic_windows.reshape(len(synthetic_windows), -1)
    
    # Split real data
    X_train_real, X_test, y_train_real, y_test = train_test_split(
        real_features, real_user_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=real_user_ids
    )
    
    # Model 1: Train on real only
    clf_real = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf_real.fit(X_train_real, y_train_real)
    y_pred_real = clf_real.predict(X_test)
    acc_real = accuracy_score(y_test, y_pred_real)
    
    # Model 2: Train on real + synthetic
    X_train_combined = np.vstack([X_train_real, synth_features])
    y_train_combined = np.concatenate([y_train_real, synthetic_user_ids])
    
    clf_combined = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf_combined.fit(X_train_combined, y_train_combined)
    y_pred_combined = clf_combined.predict(X_test)
    acc_combined = accuracy_score(y_test, y_pred_combined)
    
    metrics = {
        "accuracy_real_only": float(acc_real),
        "accuracy_real_plus_synthetic": float(acc_combined),
        "improvement": float(acc_combined - acc_real),
        "n_train_real": len(X_train_real),
        "n_train_synthetic": len(synth_features),
        "n_test": len(X_test)
    }
    
    return metrics


def plot_training_history(
    history: Dict[str, List[float]],
    title: str,
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        history: Dict with "train" and "val" loss lists
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history.get("train", history.get("train_loss", []))) + 1)
    
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="orange")
    elif "train" in history:
        ax.plot(epochs, history["train"], label="Train Loss", color="blue")
        ax.plot(epochs, history["val"], label="Val Loss", color="orange")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_sample_windows(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    n_samples: int = 3,
    save_path: Optional[str] = None
):
    """
    Plot sample real vs synthetic windows side by side.
    
    Args:
        real_windows: (N, window_size, 4) real windows
        synthetic_windows: (M, window_size, 4) synthetic windows
        n_samples: Number of samples to plot
        save_path: Optional path to save figure
    """
    channel_names = ["X", "Y", "Z", "Magnitude"]
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    for i in range(n_samples):
        real_idx = np.random.randint(len(real_windows))
        synth_idx = np.random.randint(len(synthetic_windows))
        
        for c in range(4):
            ax = axes[i, c] if n_samples > 1 else axes[c]
            ax.plot(real_windows[real_idx, :, c], label="Real", alpha=0.7)
            ax.plot(synthetic_windows[synth_idx, :, c], label="Synthetic", alpha=0.7)
            ax.set_title(f"{channel_names[c]}")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.close()


def comprehensive_evaluation(
    vae: ConvVAE,
    ddpm: LatentDDPM,
    ema: EMA,
    preprocessor: GaitPreprocessor,
    cfg: ConfigV2,
    sensor_type: str,
    n_synthetic_per_user: int = 50
) -> Dict:
    """
    Run comprehensive evaluation.
    
    Args:
        vae, ddpm, ema: Trained models
        preprocessor: Fitted preprocessor
        cfg: Configuration
        sensor_type: Sensor type
        n_synthetic_per_user: Synthetic windows per user
        
    Returns:
        results: Dict with all evaluation metrics
    """
    from .generate import generate_synthetic_windows
    import os
    
    print(f"\n{'='*50}")
    print(f"Comprehensive Evaluation for {sensor_type}")
    print(f"{'='*50}")
    
    results = {}
    
    # Get test data
    test_windows = preprocessor.test_windows
    test_user_ids = np.array([preprocessor.user_to_idx[u] for u in preprocessor.test_user_ids])
    
    print(f"\nTest data: {test_windows.shape}")
    
    # 1. Reconstruction quality
    print("\n1. Evaluating reconstruction quality...")
    recon_metrics = evaluate_reconstruction(vae, test_windows, test_user_ids, cfg)
    results["reconstruction"] = recon_metrics
    print(f"   MSE: {recon_metrics['mse']:.6f}")
    print(f"   MAE: {recon_metrics['mae']:.6f}")
    
    # 2. Generate synthetic data
    print("\n2. Generating synthetic data...")
    all_synthetic = []
    all_synthetic_users = []
    
    for user_id in preprocessor.test_users[:10]:  # First 10 test users
        try:
            windows = generate_synthetic_windows(
                n_windows=n_synthetic_per_user,
                user_id=user_id,
                vae=vae,
                ddpm=ddpm,
                preprocessor=preprocessor,
                cfg=cfg,
                use_ema=True,
                ema=ema,
                return_normalized=True
            )
            all_synthetic.append(windows)
            all_synthetic_users.extend([preprocessor.user_to_idx[user_id]] * n_synthetic_per_user)
        except Exception as e:
            print(f"   Skipping user {user_id}: {e}")
    
    if all_synthetic:
        synthetic_windows = np.concatenate(all_synthetic, axis=0)
        synthetic_user_ids = np.array(all_synthetic_users)
        
        print(f"   Generated: {synthetic_windows.shape}")
        
        # 3. Statistical similarity
        print("\n3. Evaluating statistical similarity...")
        stat_metrics = evaluate_statistical_similarity(test_windows, synthetic_windows)
        results["statistical"] = stat_metrics
        for k, v in list(stat_metrics.items())[:6]:
            print(f"   {k}: {v:.4f}")
        
        # 4. User classification (if enough data)
        if len(np.unique(test_user_ids)) >= 3 and len(np.unique(synthetic_user_ids)) >= 3:
            print("\n4. Evaluating user classification...")
            try:
                clf_metrics = evaluate_user_classification(
                    test_windows, test_user_ids,
                    synthetic_windows, synthetic_user_ids
                )
                results["classification"] = clf_metrics
                print(f"   Real-only accuracy: {clf_metrics['accuracy_real_only']:.4f}")
                print(f"   Real+Synth accuracy: {clf_metrics['accuracy_real_plus_synthetic']:.4f}")
                print(f"   Improvement: {clf_metrics['improvement']:.4f}")
            except Exception as e:
                print(f"   Classification evaluation failed: {e}")
        
        # 5. Save plots
        print("\n5. Saving evaluation plots...")
        save_dir = os.path.join(cfg.save_dir, sensor_type.lower(), "evaluation")
        os.makedirs(save_dir, exist_ok=True)
        
        plot_sample_windows(
            test_windows, synthetic_windows,
            n_samples=3,
            save_path=os.path.join(save_dir, "sample_comparison.png")
        )
    
    return results

