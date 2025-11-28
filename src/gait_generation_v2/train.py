"""
Training loops for VAE and DDPM in Gait Generation V2.

Includes:
- train_vae: Train the convolutional VAE
- train_ddpm: Train the latent DDPM
- encode_to_latents: Encode dataset to latent codes
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, Tuple, Optional

from .config import ConfigV2
from .data import GaitWindowDataModule, LatentDataset
from .conv_vae import ConvVAE
from .diffusion import LatentUNet, LatentDDPM, EMA


def kl_warmup_beta(epoch: int, warmup_epochs: int, max_beta: float = 1.0) -> float:
    """
    Cosine warmup schedule for KL beta.
    
    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of warmup epochs
        max_beta: Maximum beta value
        
    Returns:
        beta: KL weight for this epoch
    """
    if warmup_epochs <= 0:
        return max_beta
    if epoch >= warmup_epochs:
        return max_beta
    x = (epoch + 1) / warmup_epochs
    return float(0.5 * (1 - math.cos(math.pi * x))) * max_beta


def train_vae(
    vae: ConvVAE,
    dm: GaitWindowDataModule,
    cfg: ConfigV2,
    sensor_type: str
) -> Dict:
    """
    Train the VAE.
    
    Args:
        vae: ConvVAE model
        dm: Data module with dataloaders
        cfg: Configuration
        sensor_type: "Accelerometer" or "Gyroscope"
        
    Returns:
        history: Training history dict
    """
    save_dir = os.path.join(cfg.save_dir, sensor_type.lower())
    os.makedirs(save_dir, exist_ok=True)
    
    vae_path = os.path.join(save_dir, "vae_best.pt")
    history_path = os.path.join(save_dir, "vae_history.json")
    
    # Move to device
    vae = vae.to(cfg.device)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_weight_decay)
    
    # Training state
    best_val = float("inf")
    patience_counter = 0
    history = defaultdict(list)
    
    print(f"\n{'='*60}")
    print(f"Training VAE for {sensor_type}")
    print(f"{'='*60}")
    print(f"  Epochs: {cfg.vae_epochs}")
    print(f"  Batch size: {cfg.vae_batch_size}")
    print(f"  Learning rate: {cfg.vae_lr}")
    print(f"  KL beta: {cfg.kl_beta}")
    print(f"  KL warmup: {cfg.kl_warmup_epochs} epochs")
    print(f"  Device: {cfg.device}")
    print()
    
    for epoch in range(cfg.vae_epochs):
        # KL beta warmup
        beta = kl_warmup_beta(epoch, cfg.kl_warmup_epochs, cfg.kl_beta)
        
        # Training
        vae.train()
        train_losses = []
        train_recon = []
        train_kl = []
        
        for batch in dm.train_dataloader():
            window = batch["window"].to(cfg.device)  # (B, window_size, 4)
            user_idx = batch["user_idx"].to(cfg.device)
            
            # Check for NaN/Inf in input
            if torch.isnan(window).any() or torch.isinf(window).any():
                print(f"  WARNING: NaN/Inf detected in input window!")
                print(f"    NaN count: {torch.isnan(window).sum().item()}")
                print(f"    Inf count: {torch.isinf(window).sum().item()}")
                print(f"    Window stats: min={window.min().item():.4f}, max={window.max().item():.4f}, mean={window.mean().item():.4f}")
                continue
            
            optimizer.zero_grad()
            loss, logs = vae(window, user_idx, beta=beta)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss detected!")
                print(f"    Recon: {logs['recon']}, KL: {logs['kl']}")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            train_recon.append(logs["recon"])
            train_kl.append(logs["kl"])
        
        train_loss = float(np.mean(train_losses))
        train_recon_mean = float(np.mean(train_recon))
        train_kl_mean = float(np.mean(train_kl))
        
        # Validation
        vae.eval()
        val_losses = []
        val_recon = []
        val_kl = []
        
        with torch.no_grad():
            for batch in dm.val_dataloader():
                window = batch["window"].to(cfg.device)
                user_idx = batch["user_idx"].to(cfg.device)
                
                # Use same warmup beta as training for fair comparison
                loss, logs = vae(window, user_idx, beta=beta)
                
                val_losses.append(loss.item())
                val_recon.append(logs["recon"])
                val_kl.append(logs["kl"])
        
        val_loss = float(np.mean(val_losses))
        val_recon_mean = float(np.mean(val_recon))
        val_kl_mean = float(np.mean(val_kl))
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon_mean)
        history["train_kl"].append(train_kl_mean)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon_mean)
        history["val_kl"].append(val_kl_mean)
        history["beta"].append(beta)
        
        # Print progress
        print(f"[VAE] Epoch {epoch+1:03d}/{cfg.vae_epochs} | "
              f"train: {train_loss:.4f} (recon: {train_recon_mean:.4f}, kl: {train_kl_mean:.4f}) | "
              f"val: {val_loss:.4f} | beta: {beta:.3f}")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(vae.state_dict(), vae_path)
            print(f"  → Saved best VAE: {vae_path} (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.vae_patience:
                print(f"  → Early stopping at epoch {epoch+1}")
                break
    
    # Save history
    with open(history_path, 'w') as f:
        json.dump(dict(history), f, indent=2)
    print(f"Saved VAE history to: {history_path}")
    
    return dict(history)


@torch.no_grad()
def encode_to_latents(
    dataset,
    vae: ConvVAE,
    cfg: ConfigV2,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a dataset to latent codes using the VAE encoder.
    
    Args:
        dataset: GaitWindowDataset
        vae: Trained ConvVAE
        cfg: Configuration
        batch_size: Batch size for encoding
        
    Returns:
        Z: (N, latent_dim) latent codes
        user_idx: (N,) user indices
    """
    vae.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    latents = []
    user_indices = []
    
    for batch in loader:
        window = batch["window"].to(cfg.device)
        user_idx = batch["user_idx"].to(cfg.device)
        
        # Encode to get mean (not sampled)
        mu, logvar = vae.encode(window, user_idx)
        
        latents.append(mu.cpu())
        user_indices.append(user_idx.cpu())
    
    Z = torch.cat(latents, dim=0).numpy()
    user_idx = torch.cat(user_indices, dim=0).numpy()
    
    return Z, user_idx


def train_ddpm(
    ddpm: LatentDDPM,
    ema: EMA,
    Z_train: np.ndarray,
    user_idx_train: np.ndarray,
    Z_val: np.ndarray,
    user_idx_val: np.ndarray,
    cfg: ConfigV2,
    sensor_type: str
) -> Dict:
    """
    Train the DDPM.
    
    Args:
        ddpm: LatentDDPM model
        ema: EMA for model weights
        Z_train: (N_train, latent_dim) training latents
        user_idx_train: (N_train,) training user indices
        Z_val: (N_val, latent_dim) validation latents
        user_idx_val: (N_val,) validation user indices
        cfg: Configuration
        sensor_type: "Accelerometer" or "Gyroscope"
        
    Returns:
        history: Training history dict
    """
    save_dir = os.path.join(cfg.save_dir, sensor_type.lower())
    os.makedirs(save_dir, exist_ok=True)
    
    ddpm_path = os.path.join(save_dir, "ddpm_best.pt")
    ema_path = os.path.join(save_dir, "ddpm_ema.pt")
    history_path = os.path.join(save_dir, "ddpm_history.json")
    
    # Create datasets
    train_ds = LatentDataset(Z_train, user_idx_train)
    val_ds = LatentDataset(Z_val, user_idx_val)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.ddpm_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.ddpm_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=cfg.ddpm_lr)
    
    # Training state
    best_val = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}
    
    print(f"\n{'='*60}")
    print(f"Training DDPM for {sensor_type}")
    print(f"{'='*60}")
    print(f"  Epochs: {cfg.ddpm_epochs}")
    print(f"  Batch size: {cfg.ddpm_batch_size}")
    print(f"  Learning rate: {cfg.ddpm_lr}")
    print(f"  Diffusion steps: {cfg.T}")
    print(f"  EMA decay: {cfg.ema_decay}")
    print(f"  Device: {cfg.device}")
    print(f"  Train latents: {Z_train.shape}")
    print(f"  Val latents: {Z_val.shape}")
    print()
    
    for epoch in range(cfg.ddpm_epochs):
        # Training
        ddpm.model.train()
        train_losses = []
        
        for z0, user_idx in train_loader:
            z0 = z0.to(cfg.device)
            user_idx = user_idx.to(cfg.device)
            
            # Sample random timesteps
            t = torch.randint(0, ddpm.T, (z0.shape[0],), device=cfg.device, dtype=torch.long)
            
            # Sample noise
            noise = torch.randn_like(z0)
            
            # Forward diffusion
            zt = ddpm.q_sample(z0, t, noise)
            
            # Predict noise
            noise_pred = ddpm.model(zt, t, user_idx)
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update()
            
            train_losses.append(loss.item())
        
        train_loss = float(np.mean(train_losses))
        history["train"].append(train_loss)
        
        # Validation
        ddpm.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for z0, user_idx in val_loader:
                z0 = z0.to(cfg.device)
                user_idx = user_idx.to(cfg.device)
                
                t = torch.randint(0, ddpm.T, (z0.shape[0],), device=cfg.device, dtype=torch.long)
                noise = torch.randn_like(z0)
                zt = ddpm.q_sample(z0, t, noise)
                noise_pred = ddpm.model(zt, t, user_idx)
                
                val_losses.append(F.mse_loss(noise_pred, noise).item())
        
        val_loss = float(np.mean(val_losses))
        history["val"].append(val_loss)
        
        # Print progress
        print(f"[DDPM] Epoch {epoch+1:03d}/{cfg.ddpm_epochs} | "
              f"train: {train_loss:.4f} | val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(ddpm.model.state_dict(), ddpm_path)
            torch.save(ema.state_dict(), ema_path)
            print(f"  → Saved best DDPM: {ddpm_path} (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.ddpm_patience:
                print(f"  → Early stopping at epoch {epoch+1}")
                break
    
    # Save history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved DDPM history to: {history_path}")
    
    return history

