"""
Configuration for Gait Generation V2 pipeline.

Contains all hyperparameters for windowing, preprocessing, VAE, DDPM training.
Sensor-specific clipping values are derived from EDA analysis.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ConfigV2:
    """Configuration for Temporal VAE + DDPM Gait Generation."""
    
    # =========================
    # Data paths
    # =========================
    gait_base_dir: str = "BB-MAS_Dataset/BB-MAS_Dataset"
    sensor_types: Tuple[str, ...] = ("Accelerometer", "Gyroscope")
    save_dir: str = "src/gait_generation_v2/checkpoints/gait_v2"
    
    # =========================
    # Windowing parameters
    # =========================
    window_size: int = 256          # 2.56s at 100Hz
    stride: int = 192               # 0.64s overlap (25% overlap) - reduced from 75% to prevent overfitting
    target_hz: int = 100            # Resample to 100Hz
    gap_threshold_sec: float = 0.2  # Segment on gaps > 0.2s
    min_segment_length: int = 256   # Minimum segment length to keep (samples at target_hz)
    
    # =========================
    # User balancing
    # =========================
    max_datapoints_per_user: int = 30_000  # ~465 windows per user max
    
    # =========================
    # Train/val/test split (by user)
    # =========================
    n_train_users: int = 80
    n_val_users: int = 17
    n_test_users: int = 20  # Remaining users (117 total, some may have no data)
    seed: int = 42
    
    # =========================
    # VAE architecture
    # =========================
    latent_dim_accel: int = 64      # Larger for accelerometer (more complex dynamics)
    latent_dim_gyro: int = 32       # Smaller for gyroscope (simpler dynamics)
    hidden_channels: Tuple[int, ...] = (64, 128, 256, 256)  # Conv channel progression
    user_embed_dim: int = 32        # User embedding dimension
    in_channels: int = 4            # X, Y, Z, magnitude
    
    # =========================
    # VAE training
    # =========================
    vae_epochs: int = 30
    vae_batch_size: int = 64
    vae_lr: float = 1e-4
    vae_weight_decay: float = 1e-5  # Weight decay for regularization
    vae_dropout: float = 0.2        # Dropout rate in encoder/decoder
    kl_beta: float = 1.0            # KL loss weight
    kl_warmup_epochs: int = 10      # Warmup epochs for KL beta
    vae_patience: int = 15          # Early stopping patience
    
    # =========================
    # DDPM architecture
    # =========================
    ddpm_hidden_dim: int = 256      # Hidden dimension for UNet MLP
    ddpm_n_blocks: int = 6          # Number of residual blocks
    
    # =========================
    # DDPM training
    # =========================
    T: int = 1000                   # Diffusion timesteps
    beta_schedule: str = "cosine"   # Beta schedule: "linear" or "cosine"
    ddpm_epochs: int = 30
    ddpm_batch_size: int = 128
    ddpm_lr: float = 1e-4
    ddpm_patience: int = 20         # Early stopping patience
    ema_decay: float = 0.9999       # EMA decay for sampling
    
    # =========================
    # Sensor-specific clipping (from EDA)
    # =========================
    # Accelerometer: Based on EDA analysis
    # X: μ≈0.33, σ≈4.45 → clip to [-18, 18]
    # Y: μ≈-1.36, σ≈9.59 (bimodal) → clip to [-35, 35]
    # Z: μ≈-0.79, σ≈4.39 → clip to [-18, 18]
    # Magnitude: μ≈10.76, right tail to ~68 → clip to [0, 35] (99.5th percentile)
    accel_clip: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Xvalue": (-18.0, 18.0),
        "Yvalue": (-35.0, 35.0),
        "Zvalue": (-18.0, 18.0),
        "magnitude": (0.0, 35.0)
    })
    
    # Gyroscope: Based on EDA analysis
    # X: μ≈0, σ≈1.36 → clip to [-6, 6]
    # Y: μ≈0, σ≈1.28, heavy tails → clip to [-15, 15]
    # Z: μ≈0.025, σ≈0.70 → clip to [-4, 4]
    # Magnitude: μ≈1.61, σ≈1.17, heavy right tail → clip to [0, 10] (99.5th percentile)
    gyro_clip: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Xvalue": (-6.0, 6.0),
        "Yvalue": (-15.0, 15.0),
        "Zvalue": (-4.0, 4.0),
        "magnitude": (0.0, 10.0)
    })
    
    # =========================
    # Generation
    # =========================
    sample_steps: int = 1000        # DDPM sampling steps (can be less for faster sampling)
    n_samples_demo: int = 100       # Number of windows to generate for demo
    
    # =========================
    # Misc
    # =========================
    num_workers: int = 4            # DataLoader workers
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def get_latent_dim(self, sensor_type: str) -> int:
        """Get latent dimension for a specific sensor type."""
        if sensor_type.lower() == "accelerometer":
            return self.latent_dim_accel
        elif sensor_type.lower() == "gyroscope":
            return self.latent_dim_gyro
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    def get_clip_values(self, sensor_type: str) -> Dict[str, Tuple[float, float]]:
        """Get clipping values for a specific sensor type."""
        if sensor_type.lower() == "accelerometer":
            return self.accel_clip
        elif sensor_type.lower() == "gyroscope":
            return self.gyro_clip
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

