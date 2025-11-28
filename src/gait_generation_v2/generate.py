"""
Generation utilities for Gait Generation V2.

Functions for sampling synthetic gait windows.
"""

import torch
import numpy as np
from typing import Optional, Union
from .config import ConfigV2
from .conv_vae import ConvVAE
from .diffusion import LatentDDPM, EMA
from .preprocessing import GaitPreprocessor


def generate_synthetic_windows(
    n_windows: int,
    user_id: str,
    vae: ConvVAE,
    ddpm: LatentDDPM,
    preprocessor: GaitPreprocessor,
    cfg: ConfigV2,
    use_ema: bool = True,
    ema: Optional[EMA] = None,
    return_normalized: bool = False
) -> np.ndarray:
    """
    Generate synthetic gait windows for a specific user.
    
    Args:
        n_windows: Number of windows to generate
        user_id: User ID to condition on
        vae: Trained ConvVAE model
        ddpm: Trained LatentDDPM model
        preprocessor: Fitted GaitPreprocessor
        cfg: Configuration
        use_ema: Whether to use EMA weights for DDPM sampling
        ema: EMA object (required if use_ema=True)
        return_normalized: If True, return normalized values; else physical units
        
    Returns:
        windows: (n_windows, window_size, 4) synthetic windows
    """
    # Get user index
    if user_id not in preprocessor.user_to_idx:
        raise ValueError(f"Unknown user_id: {user_id}. Available: {list(preprocessor.user_to_idx.keys())[:10]}...")
    
    user_idx = preprocessor.user_to_idx[user_id]
    user_idx_tensor = torch.full((n_windows,), user_idx, dtype=torch.long, device=cfg.device)
    
    # Apply EMA weights if requested
    original_state = None
    if use_ema and ema is not None:
        original_state = {k: v.clone() for k, v in ddpm.model.state_dict().items()}
        ema.apply_shadow()
    
    try:
        # Sample latents from DDPM
        with torch.no_grad():
            z = ddpm.sample(n=n_windows, user_idx=user_idx_tensor, steps=cfg.sample_steps)
        
        # Decode with VAE
        vae.eval()
        with torch.no_grad():
            # z: (n_windows, latent_dim)
            windows_normalized = vae.decode(z, user_idx_tensor)  # (n_windows, 4, window_size)
            windows_normalized = windows_normalized.transpose(1, 2)  # (n_windows, window_size, 4)
            windows_normalized = windows_normalized.cpu().numpy()
    
    finally:
        # Restore original weights
        if original_state is not None:
            ddpm.model.load_state_dict(original_state)
    
    if return_normalized:
        return windows_normalized
    
    # Inverse standardization to physical units
    windows_physical = preprocessor.inverse_standardize(windows_normalized)
    
    return windows_physical


def generate_for_multiple_users(
    n_windows_per_user: int,
    user_ids: list,
    vae: ConvVAE,
    ddpm: LatentDDPM,
    preprocessor: GaitPreprocessor,
    cfg: ConfigV2,
    use_ema: bool = True,
    ema: Optional[EMA] = None,
    return_normalized: bool = False
) -> dict:
    """
    Generate synthetic windows for multiple users.
    
    Args:
        n_windows_per_user: Number of windows per user
        user_ids: List of user IDs
        vae: Trained ConvVAE model
        ddpm: Trained LatentDDPM model
        preprocessor: Fitted GaitPreprocessor
        cfg: Configuration
        use_ema: Whether to use EMA weights
        ema: EMA object
        return_normalized: If True, return normalized values
        
    Returns:
        results: Dict mapping user_id -> (n_windows, window_size, 4) windows
    """
    results = {}
    
    for user_id in user_ids:
        try:
            windows = generate_synthetic_windows(
                n_windows=n_windows_per_user,
                user_id=user_id,
                vae=vae,
                ddpm=ddpm,
                preprocessor=preprocessor,
                cfg=cfg,
                use_ema=use_ema,
                ema=ema,
                return_normalized=return_normalized
            )
            results[user_id] = windows
            print(f"  Generated {n_windows_per_user} windows for user {user_id}")
        except ValueError as e:
            print(f"  Skipping user {user_id}: {e}")
    
    return results


def windows_to_dataframe(
    windows: np.ndarray,
    user_id: str,
    sensor_type: str,
    start_timestamp: float = 0.0,
    sample_rate: int = 100
) -> "pd.DataFrame":
    """
    Convert windows back to DataFrame format.
    
    Args:
        windows: (n_windows, window_size, 4) array [X, Y, Z, magnitude]
        user_id: User ID
        sensor_type: Sensor type
        start_timestamp: Starting timestamp
        sample_rate: Sample rate in Hz
        
    Returns:
        df: DataFrame with Timestamp, Xvalue, Yvalue, Zvalue, magnitude, __user_id__, __sensor_type__
    """
    import pandas as pd
    
    n_windows, window_size, n_channels = windows.shape
    
    # Flatten windows
    n_total = n_windows * window_size
    flat_data = windows.reshape(n_total, n_channels)
    
    # Create timestamps
    timestamps = start_timestamp + np.arange(n_total) / sample_rate
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Xvalue": flat_data[:, 0],
        "Yvalue": flat_data[:, 1],
        "Zvalue": flat_data[:, 2],
        "magnitude": flat_data[:, 3],
        "__user_id__": user_id,
        "__sensor_type__": sensor_type
    })
    
    return df

