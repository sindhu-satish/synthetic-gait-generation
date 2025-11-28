"""
Main orchestration script for Gait Generation V2.

Handles the full pipeline:
1. Load/preprocess data (or load from checkpoint)
2. Train VAE (or load from checkpoint)
3. Encode to latents (or load from checkpoint)
4. Train DDPM (or load from checkpoint)
5. Generate synthetic samples
"""

import os
import json
import warnings
import numpy as np
import torch
from typing import Dict, Optional, Tuple

# Suppress RuntimeWarning about module already in sys.modules
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

from .config import ConfigV2, set_seed
from .preprocessing import GaitPreprocessor
from .data import GaitWindowDataModule
from .conv_vae import ConvVAE
from .diffusion import LatentUNet, LatentDDPM, EMA
from .train import train_vae, train_ddpm, encode_to_latents
from .generate import generate_synthetic_windows, windows_to_dataframe


def train_sensor_model(
    sensor_type: str,
    cfg: ConfigV2
) -> Tuple[ConvVAE, LatentDDPM, EMA, GaitPreprocessor, GaitWindowDataModule]:
    """
    Train models for a single sensor type with checkpoint/resume support.
    
    Args:
        sensor_type: "Accelerometer" or "Gyroscope"
        cfg: Configuration
        
    Returns:
        vae, ddpm, ema, preprocessor, dm
    """
    print(f"\n{'='*70}")
    print(f"Training pipeline for {sensor_type}")
    print(f"{'='*70}")
    
    save_dir = os.path.join(cfg.save_dir, sensor_type.lower())
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint paths
    pre_path = os.path.join(save_dir, "preprocessor.pkl")
    vae_path = os.path.join(save_dir, "vae_best.pt")
    latents_train_path = os.path.join(save_dir, "latents_train.npz")
    latents_val_path = os.path.join(save_dir, "latents_val.npz")
    ddpm_path = os.path.join(save_dir, "ddpm_best.pt")
    ema_path = os.path.join(save_dir, "ddpm_ema.pt")
    
    # ============================================================
    # STEP 1: Preprocessor
    # ============================================================
    print("\n" + "="*50)
    print("STEP 1: Preprocessor")
    print("="*50)
    
    if os.path.exists(pre_path):
        print(f"Loading existing preprocessor from {pre_path}")
        preprocessor = GaitPreprocessor.load(pre_path)
        print(f"  Loaded preprocessor with {len(preprocessor.train_users)} train users")
    else:
        print("Creating new preprocessor...")
        preprocessor = GaitPreprocessor(cfg, sensor_type)
        preprocessor.load_and_process()
        preprocessor.save(pre_path)
    
    # ============================================================
    # STEP 2: Data Module
    # ============================================================
    print("\n" + "="*50)
    print("STEP 2: Data Module")
    print("="*50)
    
    dm = GaitWindowDataModule(preprocessor, cfg)
    dm.setup()
    
    # ============================================================
    # STEP 3: VAE
    # ============================================================
    print("\n" + "="*50)
    print("STEP 3: VAE")
    print("="*50)
    
    latent_dim = cfg.get_latent_dim(sensor_type)
    n_users = dm.n_users
    
    print(f"  Latent dim: {latent_dim}")
    print(f"  N users: {n_users}")
    
    vae = ConvVAE(cfg=cfg, latent_dim=latent_dim, n_users=n_users)
    vae = vae.to(cfg.device)
    
    if os.path.exists(vae_path):
        print(f"Loading existing VAE from {vae_path}, skipping VAE training")
        vae.load_state_dict(torch.load(vae_path, map_location=cfg.device))
    else:
        print("Training VAE...")
        train_vae(vae, dm, cfg, sensor_type)
        # Load best weights
        vae.load_state_dict(torch.load(vae_path, map_location=cfg.device))
    
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    # ============================================================
    # STEP 4: Encode to Latents
    # ============================================================
    print("\n" + "="*50)
    print("STEP 4: Encode to Latents")
    print("="*50)
    
    if os.path.exists(latents_train_path) and os.path.exists(latents_val_path):
        print(f"Loading existing latents from {latents_train_path}")
        data = np.load(latents_train_path)
        Z_train, user_idx_train = data['Z'], data['user_idx']
        data = np.load(latents_val_path)
        Z_val, user_idx_val = data['Z'], data['user_idx']
        print(f"  Train latents: {Z_train.shape}")
        print(f"  Val latents: {Z_val.shape}")
    else:
        print("Encoding datasets to latents...")
        Z_train, user_idx_train = encode_to_latents(dm.train_ds, vae, cfg)
        Z_val, user_idx_val = encode_to_latents(dm.val_ds, vae, cfg)
        np.savez(latents_train_path, Z=Z_train, user_idx=user_idx_train)
        np.savez(latents_val_path, Z=Z_val, user_idx=user_idx_val)
        print(f"  Saved latents to {latents_train_path}")
        print(f"  Train latents: {Z_train.shape}")
        print(f"  Val latents: {Z_val.shape}")
    
    # ============================================================
    # STEP 5: DDPM
    # ============================================================
    print("\n" + "="*50)
    print("STEP 5: DDPM")
    print("="*50)
    
    unet = LatentUNet(
        latent_dim=latent_dim,
        hidden_dim=cfg.ddpm_hidden_dim,
        n_blocks=cfg.ddpm_n_blocks,
        n_users=n_users,
        user_embed_dim=cfg.user_embed_dim
    )
    unet = unet.to(cfg.device)
    
    ddpm = LatentDDPM(
        unet,
        T=cfg.T,
        beta_schedule=cfg.beta_schedule,
        device=cfg.device
    )
    ema = EMA(unet, decay=cfg.ema_decay)
    
    if os.path.exists(ddpm_path):
        print(f"Loading existing DDPM from {ddpm_path}, skipping DDPM training")
        unet.load_state_dict(torch.load(ddpm_path, map_location=cfg.device))
        if os.path.exists(ema_path):
            ema.load_state_dict(torch.load(ema_path, map_location=cfg.device))
            print(f"  Loaded EMA weights from {ema_path}")
    else:
        print("Training DDPM...")
        train_ddpm(ddpm, ema, Z_train, user_idx_train, Z_val, user_idx_val, cfg, sensor_type)
        # Load best weights
        unet.load_state_dict(torch.load(ddpm_path, map_location=cfg.device))
        if os.path.exists(ema_path):
            ema.load_state_dict(torch.load(ema_path, map_location=cfg.device))
    
    unet.eval()
    
    print(f"\n{'='*50}")
    print(f"Training complete for {sensor_type}!")
    print(f"{'='*50}")
    print(f"  Preprocessor: {pre_path}")
    print(f"  VAE: {vae_path}")
    print(f"  DDPM: {ddpm_path}")
    print(f"  EMA: {ema_path}")
    
    return vae, ddpm, ema, preprocessor, dm


def demo_generation(
    vae: ConvVAE,
    ddpm: LatentDDPM,
    ema: EMA,
    preprocessor: GaitPreprocessor,
    cfg: ConfigV2,
    sensor_type: str
):
    """
    Demo: generate synthetic samples for a few users.
    """
    print(f"\n{'='*50}")
    print(f"Demo: Generating synthetic {sensor_type} data")
    print(f"{'='*50}")
    
    # Pick first 3 training users
    demo_users = preprocessor.train_users[:3]
    n_windows = cfg.n_samples_demo
    
    for user_id in demo_users:
        print(f"\nGenerating {n_windows} windows for user {user_id}...")
        
        try:
            windows = generate_synthetic_windows(
                n_windows=n_windows,
                user_id=user_id,
                vae=vae,
                ddpm=ddpm,
                preprocessor=preprocessor,
                cfg=cfg,
                use_ema=True,
                ema=ema,
                return_normalized=False
            )
            
            print(f"  Shape: {windows.shape}")
            print(f"  X range: [{windows[:,:,0].min():.2f}, {windows[:,:,0].max():.2f}]")
            print(f"  Y range: [{windows[:,:,1].min():.2f}, {windows[:,:,1].max():.2f}]")
            print(f"  Z range: [{windows[:,:,2].min():.2f}, {windows[:,:,2].max():.2f}]")
            print(f"  Mag range: [{windows[:,:,3].min():.2f}, {windows[:,:,3].max():.2f}]")
            
            # Save sample
            save_dir = os.path.join(cfg.save_dir, sensor_type.lower(), "samples")
            os.makedirs(save_dir, exist_ok=True)
            
            df = windows_to_dataframe(windows, user_id, sensor_type)
            sample_path = os.path.join(save_dir, f"synthetic_user_{user_id}.csv")
            df.to_csv(sample_path, index=False)
            print(f"  Saved to: {sample_path}")
            
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main entry point."""
    cfg = ConfigV2()
    set_seed(cfg.seed)
    
    print("="*70)
    print("Gait Generation V2: Temporal VAE + Latent DDPM")
    print("="*70)
    print(f"Device: {cfg.device}")
    print(f"Seed: {cfg.seed}")
    print(f"Save dir: {cfg.save_dir}")
    print(f"Sensor types: {cfg.sensor_types}")
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(cfg.save_dir, "config.json")
    config_dict = {k: str(v) if not isinstance(v, (int, float, bool, list, tuple, dict, type(None))) else v 
                   for k, v in cfg.__dict__.items()}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Saved config to: {config_path}")
    
    results = {}
    
    for sensor_type in cfg.sensor_types:
        try:
            vae, ddpm, ema, preprocessor, dm = train_sensor_model(sensor_type, cfg)
            results[sensor_type] = {
                "vae": vae,
                "ddpm": ddpm,
                "ema": ema,
                "preprocessor": preprocessor,
                "dm": dm
            }
            
            # Demo generation
            demo_generation(vae, ddpm, ema, preprocessor, cfg, sensor_type)
            
        except Exception as e:
            print(f"\nError training {sensor_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("All training complete!")
    print(f"{'='*70}")
    
    for sensor_type in results:
        print(f"  {sensor_type}: OK")
    
    return results


if __name__ == "__main__":
    main()

