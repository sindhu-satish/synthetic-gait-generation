import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from .vae import WindowVAE
from .diffusion import LatentDDPM, LatentDataset
from .data import GaitDataModule
from .config import (
    VAE_LR, VAE_EPOCHS, VAE_PATIENCE, KL_MAX_BETA, KL_WARMUP_EPOCHS, KL_BETA,
    USE_SMOOTHNESS_LOSS, USE_DISTRIBUTION_LOSS, LAMBDA_SMOOTH, LAMBDA_PHYS, LAMBDA_SPECTRAL,
    DEVICE, DDPM_LR, DDPM_EPOCHS, DDPM_PATIENCE, DDPM_BATCH_SIZE, T, BETA_SCHEDULE,
    NUM_WORKERS, DDPM_USE_MU_ONLY, DDPM_USE_EMA, DDPM_EMA_DECAY, PIN_MEMORY, LAMBDA_VAR,
    DDPM_EVAL_INTERVAL, DDPM_EVAL_BATCH_SIZE, DDPM_EVAL_SEED, DDPM_REALISM_HF_CUTOFF_HZ,
    DDPM_REALISM_JERK_PERCENTILE, DDPM_REALISM_SCORE_WEIGHTS, DDPM_REALISM_PATIENCE,
    DDPM_REALISM_MIN_DELTA, IMU_FS_HZ, WINDOW_SIZE,
    DDPM_W_SMOOTH, DDPM_W_JERK, DDPM_W_SPEC, DDPM_SPEC_CUTOFF_HZ, DDPM_SPEC_FMIN_HZ,
    DDPM_SAMPLING_RATE_HZ, DDPM_SPEC_EPS,
    DDPM_W_SPEC_MAG, DDPM_W_RMS, DDPM_W_CADENCE, DDPM_W_USER,
    DDPM_ENABLE_SPEC_MAG, DDPM_ENABLE_RMS, DDPM_ENABLE_CADENCE, DDPM_ENABLE_USER,
    DDPM_CADENCE_FMIN_HZ, DDPM_CADENCE_FMAX_HZ, SAVE_DIR
)
from .physics_losses import smoothness_loss, distribution_loss, spectral_loss
from .metrics import compute_realism_metrics

_spectral_loss_logged = False

def compute_smoothness_loss_ddpm(x0_pred_windows):
    """
    Smoothness loss on first differences (targets jerk spikes).
    x0_pred_windows: (B, T, 3) tensor of predicted windows
    Returns: L1 loss on first differences
    """
    dx = x0_pred_windows[:, 1:, :] - x0_pred_windows[:, :-1, :]
    return torch.abs(dx).mean()

def compute_jerk_loss_ddpm(x0_pred_windows):
    """
    Jerk suppression loss on second differences (targets impulses).
    x0_pred_windows: (B, T, 3) tensor of predicted windows
    Returns: L1 loss on second differences
    """
    dx = x0_pred_windows[:, 1:, :] - x0_pred_windows[:, :-1, :]
    d2x = dx[:, 1:, :] - dx[:, :-1, :]
    return torch.abs(d2x).mean()

def compute_spectral_loss_ddpm(x0_pred_windows, fs_hz, cutoff_hz, fmin_hz, eps):
    """
    Spectral high-frequency penalty (targets hf_ratio and tail_mass).
    x0_pred_windows: (B, T, 3) tensor of predicted windows
    fs_hz: Sampling rate in Hz
    cutoff_hz: High frequency cutoff (default 10 Hz)
    fmin_hz: Minimum frequency to consider (default 0.5 Hz to ignore DC)
    eps: Small epsilon for numerical stability
    Returns: Mean hf_ratio across windows
    """
    global _spectral_loss_logged
    
    B, T, C = x0_pred_windows.shape
    device = x0_pred_windows.device
    
    if not _spectral_loss_logged:
        n_fft = T
        nyquist_hz = fs_hz / 2.0
        freq_resolution_hz = fs_hz / float(n_fft)
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(fs_hz), device=device)
        cutoff_bin_idx = torch.sum(freqs <= cutoff_hz).item()
        if cutoff_bin_idx >= len(freqs):
            cutoff_bin_idx = len(freqs) - 1
        
        print(f"\n{'='*70}")
        print(f"[DDPM Training - Spectral Loss] Spectral Parameters:")
        print(f"{'='*70}")
        print(f"  Sampling frequency (fs):           {fs_hz:.2f} Hz")
        print(f"  FFT length (n_fft):                {n_fft}")
        print(f"  Nyquist frequency (fs/2):          {nyquist_hz:.2f} Hz")
        print(f"  Frequency resolution (fs/n_fft):   {freq_resolution_hz:.4f} Hz")
        print(f"  Cutoff frequency (HF > {cutoff_hz} Hz): {cutoff_hz:.2f} Hz")
        print(f"  Computed cutoff bin index:         {cutoff_bin_idx}")
        print(f"{'='*70}\n")
        _spectral_loss_logged = True
    
    # Vectorized FFT computation: process all (B, C) axes at once
    x_flat = x0_pred_windows.float()  # (B, T, C)
    x_flat = x_flat.transpose(1, 2).contiguous()  # (B, C, T) for FFT along time dim
    
    # Compute FFT for all windows and channels at once
    X = torch.fft.rfft(x_flat, dim=2)  # (B, C, n_freq)
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fs_hz), device=device)  # (n_freq,)
    power = torch.abs(X) ** 2  # (B, C, n_freq)
    
    # Create masks (same for all windows/channels)
    mask_above_fmin = freqs > float(fmin_hz)  # (n_freq,)
    mask_above_cutoff = freqs > float(cutoff_hz)  # (n_freq,)
    
    # Compute total and HF power for all (B, C) pairs
    # Expand masks to match power shape: (n_freq,) -> (1, 1, n_freq) for broadcasting
    mask_fmin_expanded = mask_above_fmin[None, None, :]  # (1, 1, n_freq)
    mask_cutoff_expanded = mask_above_cutoff[None, None, :]  # (1, 1, n_freq)
    
    total_power = (power * mask_fmin_expanded).sum(dim=2) + eps  # (B, C)
    hf_power = (power * mask_cutoff_expanded).sum(dim=2)  # (B, C)
    
    # Compute hf_ratio for each (window, channel) pair
    hf_ratios_per_axis = hf_power / total_power  # (B, C)
    
    # Average across channels for each window, then across windows
    hf_ratios_per_window = hf_ratios_per_axis.mean(dim=1)  # (B,)
    return hf_ratios_per_window.mean()  # scalar

def compute_spectral_loss_mag(x0_pred_windows, fs_hz, cutoff_hz, fmin_hz, eps):
    """
    Spectral high-frequency penalty computed on magnitude signal (aligned with tail_mass metric).
    x0_pred_windows: (B, T, 3) tensor of predicted windows
    fs_hz: Sampling rate in Hz
    cutoff_hz: High frequency cutoff (default 10 Hz)
    fmin_hz: Minimum frequency to consider (default 0.5 Hz to ignore DC)
    eps: Small epsilon for numerical stability
    Returns: Mean hf_ratio_mag across windows
    """
    B, T, C = x0_pred_windows.shape
    device = x0_pred_windows.device
    
    # Compute magnitude per time-step: m[t] = sqrt(sum_c x[t,c]^2 + eps)
    m_pred = torch.sqrt((x0_pred_windows ** 2).sum(dim=2) + eps)  # (B, T)
    
    # Compute FFT on magnitude signal
    X_mag = torch.fft.rfft(m_pred, dim=1)  # (B, n_freq)
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fs_hz), device=device)  # (n_freq,)
    power_mag = torch.abs(X_mag) ** 2  # (B, n_freq)
    
    # Create masks
    mask_above_fmin = freqs > float(fmin_hz)  # (n_freq,)
    mask_above_cutoff = freqs > float(cutoff_hz)  # (n_freq,)
    
    # Expand masks for broadcasting
    mask_fmin_expanded = mask_above_fmin[None, :]  # (1, n_freq)
    mask_cutoff_expanded = mask_above_cutoff[None, :]  # (1, n_freq)
    
    # Compute total and HF power
    total_power = (power_mag * mask_fmin_expanded).sum(dim=1) + eps  # (B,)
    hf_power = (power_mag * mask_cutoff_expanded).sum(dim=1)  # (B,)
    
    # Compute hf_ratio_mag per window
    hf_ratios_mag = hf_power / total_power  # (B,)
    
    return hf_ratios_mag.mean()  # scalar

def compute_rms_loss(x0_pred_windows, x0_real_windows, eps=1e-8):
    """
    RMS preservation loss to prevent damped dynamics.
    x0_pred_windows: (B, T, 3) predicted windows
    x0_real_windows: (B, T, 3) real windows (clean target)
    eps: Small epsilon for numerical stability
    Returns: Mean squared difference of RMS values
    """
    # Compute magnitude for both
    m_pred = torch.sqrt((x0_pred_windows ** 2).sum(dim=2) + eps)  # (B, T)
    m_real = torch.sqrt((x0_real_windows ** 2).sum(dim=2) + eps)  # (B, T)
    
    # Compute RMS per window: sqrt(mean_t m^2)
    rms_pred = torch.sqrt((m_pred ** 2).mean(dim=1) + eps)  # (B,)
    rms_real = torch.sqrt((m_real ** 2).mean(dim=1) + eps)  # (B,)
    
    # Mean squared difference
    return ((rms_pred - rms_real) ** 2).mean()

def compute_cadence_loss(x0_pred_windows, x0_real_windows, fs_hz, fmin_hz, fmax_hz):
    """
    Cadence preservation loss (dominant frequency matching).
    x0_pred_windows: (B, T, 3) predicted windows
    x0_real_windows: (B, T, 3) real windows
    fs_hz: Sampling rate in Hz
    fmin_hz: Minimum frequency for cadence search
    fmax_hz: Maximum frequency for cadence search
    Returns: Mean squared difference of dominant frequencies
    """
    B, T, C = x0_pred_windows.shape
    device = x0_pred_windows.device
    eps = 1e-8
    
    # Compute magnitude for both
    m_pred = torch.sqrt((x0_pred_windows ** 2).sum(dim=2) + eps)  # (B, T)
    m_real = torch.sqrt((x0_real_windows ** 2).sum(dim=2) + eps)  # (B, T)
    
    # Compute FFT on magnitude
    X_pred = torch.fft.rfft(m_pred, dim=1)  # (B, n_freq)
    X_real = torch.fft.rfft(m_real, dim=1)  # (B, n_freq)
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fs_hz), device=device)  # (n_freq,)
    power_pred = torch.abs(X_pred) ** 2  # (B, n_freq)
    power_real = torch.abs(X_real) ** 2  # (B, n_freq)
    
    # Find dominant frequency in plausible gait band (ignore DC)
    mask_band = (freqs >= fmin_hz) & (freqs <= fmax_hz)  # (n_freq,)
    mask_band_expanded = mask_band[None, :]  # (1, n_freq)
    
    # Mask power to band and find argmax (excluding DC)
    power_pred_band = power_pred * mask_band_expanded  # (B, n_freq)
    power_real_band = power_real * mask_band_expanded  # (B, n_freq)
    
    # Find dominant frequency index per sample
    f_dom_pred_idx = power_pred_band.argmax(dim=1)  # (B,)
    f_dom_real_idx = power_real_band.argmax(dim=1)  # (B,)
    
    # Convert to Hz
    f_dom_pred = freqs[f_dom_pred_idx]  # (B,)
    f_dom_real = freqs[f_dom_real_idx]  # (B,)
    
    # Mean squared difference
    return ((f_dom_pred - f_dom_real) ** 2).mean(), f_dom_pred.mean(), f_dom_real.mean()

def compute_user_magnitude_loss(x0_pred_windows, cond, user_target_mean_mag, eps=1e-8):
    """
    Per-user magnitude supervision loss.
    x0_pred_windows: (B, T, 3) predicted windows
    cond: (B,) conditioning indices (user IDs)
    user_target_mean_mag: dict mapping user_id -> target mean magnitude
    eps: Small epsilon for numerical stability
    Returns: Mean squared difference of mean magnitudes per user
    """
    # Compute magnitude per time-step
    m_pred = torch.sqrt((x0_pred_windows ** 2).sum(dim=2) + eps)  # (B, T)
    
    # Compute mean magnitude per sample
    sample_mean_mag_pred = m_pred.mean(dim=1)  # (B,)
    
    # Lookup target mean magnitude for each sample
    cond_np = cond.cpu().numpy() if isinstance(cond, torch.Tensor) else cond
    target_mags = torch.tensor(
        [user_target_mean_mag.get(int(uid), 10.0) for uid in cond_np],
        device=x0_pred_windows.device,
        dtype=x0_pred_windows.dtype
    )  # (B,)
    
    # Mean squared difference
    return ((sample_mean_mag_pred - target_mags) ** 2).mean()

def compute_user_magnitude_stats(dm: GaitDataModule, sensor_type: str, cache_dir: str = None):
    """
    Precompute per-user target mean magnitude from training set.
    Returns: dict mapping user_id -> mean magnitude
    """
    if cache_dir is None:
        cache_dir = SAVE_DIR
    cache_path = os.path.join(cache_dir, f"{sensor_type.lower() if sensor_type else 'gait'}", "user_mag_stats.json")
    
    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading cached user magnitude stats from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    # Compute stats from training set
    print(f"Computing user magnitude stats from training set...")
    user_mags = defaultdict(list)
    
    for batch in dm.train_dataloader():
        windows = batch["window"]  # (B, T, 3)
        cond = batch["cond"]  # (B,)
        
        # Compute magnitude per window
        m = torch.sqrt((windows ** 2).sum(dim=2) + 1e-8)  # (B, T)
        mean_mag = m.mean(dim=1).numpy()  # (B,)
        
        # Group by user
        cond_np = cond.numpy() if isinstance(cond, torch.Tensor) else cond
        for uid, mag in zip(cond_np, mean_mag):
            user_mags[int(uid)].append(float(mag))
    
    # Average per user
    user_target_mean_mag = {
        int(uid): float(np.mean(mags))
        for uid, mags in user_mags.items()
    }
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(user_target_mean_mag, f, indent=2)
    print(f"Saved user magnitude stats to {cache_path}")
    
    return user_target_mean_mag

def kl_cosine_beta(epoch, warmup_epochs, max_beta=1.0):
    if warmup_epochs <= 0:
        return max_beta
    if epoch >= warmup_epochs:
        return max_beta
    x = (epoch+1)/warmup_epochs
    return float(0.5*(1 - math.cos(math.pi*x)))*max_beta

def train_vae(vae, dm: GaitDataModule, save_dir: str, sensor_type: str = None):
    opt = torch.optim.AdamW(vae.parameters(), lr=VAE_LR)
    best_val = float("inf")
    patience = VAE_PATIENCE
    bad_steps = 0
    history = defaultdict(list)
    
    os.makedirs(save_dir, exist_ok=True)
    
    vae_filename = f"vae_best_{sensor_type.lower()}.pt" if sensor_type else "vae_best.pt"

    for epoch in range(VAE_EPOCHS):
        vae.train()
        beta = kl_cosine_beta(epoch, KL_WARMUP_EPOCHS, KL_MAX_BETA)
        tr_losses = []
        tr_recon = []
        tr_kl = []
        tr_smooth = []
        tr_phys = []
        tr_spectral = []
        
        for batch in dm.train_dataloader():
            windows = batch["window"].to(DEVICE)
            B, T, C = windows.shape
            x_flat = windows.view(B, -1)
            
            opt.zero_grad()
            recon_flat, mu, logvar = vae(x_flat)
            recon_windows = recon_flat.view(B, T, C)
            
            recon_loss = F.mse_loss(recon_windows, windows)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            smooth_loss_val = smoothness_loss(recon_windows) if USE_SMOOTHNESS_LOSS else 0.0
            phys_loss_val = distribution_loss(windows, recon_windows) if USE_DISTRIBUTION_LOSS else 0.0
            spec_loss_val = spectral_loss(windows, recon_windows)
            
            total_loss = (
                recon_loss
                + KL_BETA * beta * kl_loss
                + LAMBDA_SMOOTH * smooth_loss_val
                + LAMBDA_PHYS * phys_loss_val
                + LAMBDA_SPECTRAL * spec_loss_val
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt.step()
            
            tr_losses.append(total_loss.item())
            tr_recon.append(recon_loss.item())
            tr_kl.append(kl_loss.item())
            tr_smooth.append(smooth_loss_val.item() if isinstance(smooth_loss_val, torch.Tensor) else smooth_loss_val)
            tr_phys.append(phys_loss_val.item() if isinstance(phys_loss_val, torch.Tensor) else phys_loss_val)
            tr_spectral.append(spec_loss_val.item())
        
        train_loss = float(np.mean(tr_losses))
        train_recon = float(np.mean(tr_recon))
        train_kl = float(np.mean(tr_kl))
        train_smooth = float(np.mean(tr_smooth))
        train_phys = float(np.mean(tr_phys))
        train_spectral = float(np.mean(tr_spectral))

        vae.eval()
        with torch.no_grad():
            val_losses = []
            for batch in dm.val_dataloader():
                windows = batch["window"].to(DEVICE)
                B, T, C = windows.shape
                x_flat = windows.view(B, -1)
                
                recon_flat, mu, logvar = vae(x_flat)
                recon_windows = recon_flat.view(B, T, C)
                
                recon_loss = F.mse_loss(recon_windows, windows)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                smooth_loss_val = smoothness_loss(recon_windows) if USE_SMOOTHNESS_LOSS else 0.0
                phys_loss_val = distribution_loss(windows, recon_windows) if USE_DISTRIBUTION_LOSS else 0.0
                spec_loss_val = spectral_loss(windows, recon_windows)
                
                total_loss = (
                    recon_loss
                    + KL_BETA * KL_MAX_BETA * kl_loss
                    + LAMBDA_SMOOTH * smooth_loss_val
                    + LAMBDA_PHYS * phys_loss_val
                    + LAMBDA_SPECTRAL * spec_loss_val
                )
                val_losses.append(total_loss.item())
            val_loss = float(np.mean(val_losses))
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["beta"].append(beta)
        history["recon"].append(train_recon)
        history["kl"].append(train_kl)
        history["smooth"].append(train_smooth)
        history["phys"].append(train_phys)
        history["spectral"].append(train_spectral)
        
        print(f"[VAE] epoch {epoch+1:03d} | train {train_loss:.4f} | val {val_loss:.4f} | beta {beta:.3f} | recon {train_recon:.4f} | kl {train_kl:.4f} | smooth {train_smooth:.4f} | phys {train_phys:.4f} | spectral {train_spectral:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_steps = 0
            vae_path = os.path.join(save_dir, vae_filename)
            torch.save(vae.state_dict(), vae_path)
            print(f"  → Saved best VAE model: {vae_path} (val_loss: {val_loss:.4f})")
        else:
            bad_steps += 1
            if bad_steps >= patience:
                print("Early stopping VAE.")
                break

    return history

@torch.no_grad()
def generate_fixed_eval_samples(
    ddpm: LatentDDPM,
    vae: WindowVAE,
    n_samples: int,
    fixed_cond_idx: np.ndarray | None,
    eval_seed: int,
    steps: int = None
) -> np.ndarray:
    """
    Generate fixed evaluation samples with deterministic seed.
    
    Args:
        ddpm: Trained DDPM model
        vae: Trained VAE model
        n_samples: Number of samples to generate
        fixed_cond_idx: Fixed conditioning indices (shape: (n_samples,))
        eval_seed: Fixed random seed for reproducibility
        steps: Number of sampling steps (default: SAMPLE_STEPS)
        
    Returns:
        Generated windows as numpy array (n_samples, T, 3)
    """
    from .config import SAMPLE_STEPS, set_seed
    
    if steps is None:
        steps = SAMPLE_STEPS
    
    
    set_seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)
    
    
    if fixed_cond_idx is not None:
        cond_tensor = torch.tensor(fixed_cond_idx, device=DEVICE, dtype=torch.long)
    else:
        cond_tensor = None
    Z = ddpm.sample(n=n_samples, cond_idx=cond_tensor, steps=steps).to(DEVICE)
    
    
    if hasattr(ddpm, "z_mean") and hasattr(ddpm, "z_std") and ddpm.z_mean is not None and ddpm.z_std is not None:
        Z = Z * ddpm.z_std + ddpm.z_mean
    
    
    recon_flat = vae.decode(Z)
    recon_windows = recon_flat.view(n_samples, WINDOW_SIZE, 3)
    
    
    windows_np = recon_windows.detach().cpu().numpy()
    return windows_np


@torch.no_grad()
def encode_dataset_to_latents(ds, vae, batch_size=1024):
    from torch.utils.data import DataLoader
    latents, conds = [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    vae.eval()
    for batch in loader:
        windows = batch["window"].to(DEVICE)
        B, T, C = windows.shape
        x_flat = windows.view(B, -1)
        mu, logvar = vae.encode(x_flat)
        if DDPM_USE_MU_ONLY:
            latents.append(mu.detach().cpu())
        else:
            z = vae.reparameterize(mu, logvar)
            latents.append(z.detach().cpu())
        if "cond" in batch:
            conds.append(batch["cond"])
    Z = torch.cat(latents, dim=0).numpy()
    C = torch.cat(conds, dim=0).numpy() if conds else None
    return Z, C

def train_ddpm(
    model,
    Z_train,
    C_train,
    Z_val,
    C_val,
    save_dir: str,
    sensor_type: str = None,
    z_mean: np.ndarray | None = None,
    z_std: np.ndarray | None = None,
    vae: WindowVAE = None,
    dm: GaitDataModule = None,
):
    ddpm = LatentDDPM(model, T=T, beta_schedule=BETA_SCHEDULE, device=DEVICE)
    if z_mean is not None and z_std is not None:
        ddpm.z_mean = torch.tensor(z_mean, dtype=torch.float32, device=DEVICE)
        ddpm.z_std = torch.tensor(z_std, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=DDPM_LR)
    
    import copy
    ema_model = copy.deepcopy(model) if DDPM_USE_EMA else None

    def ema_update(ema_model, model, decay):
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    
    train_ds = LatentDataset(Z_train, C_train)
    val_ds = LatentDataset(Z_val, C_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=DDPM_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=DDPM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    best_val = float("inf")
    patience = DDPM_PATIENCE
    bad = 0
    history = {"train": [], "val": []}
    
    
    use_realism_metrics = (vae is not None and dm is not None)
    best_realism_score = float("inf")
    realism_bad = 0
    realism_patience = DDPM_REALISM_PATIENCE
    
    # Precompute user magnitude stats if user supervision is enabled
    user_target_mean_mag = None
    if DDPM_ENABLE_USER and dm is not None:
        user_target_mean_mag = compute_user_magnitude_stats(dm, sensor_type, save_dir)
    
    
    fixed_cond_idx = None
    if use_realism_metrics:
        if C_val is not None and len(C_val) > 0:
            n_eval = min(DDPM_EVAL_BATCH_SIZE, len(C_val))
            unique_conds = np.unique(C_val)
            if len(unique_conds) >= n_eval:
                fixed_cond_idx = np.tile(unique_conds[:n_eval], (DDPM_EVAL_BATCH_SIZE // n_eval + 1))[:DDPM_EVAL_BATCH_SIZE]
            else:
                fixed_cond_idx = np.tile(unique_conds, (DDPM_EVAL_BATCH_SIZE // len(unique_conds) + 1))[:DDPM_EVAL_BATCH_SIZE]
            fixed_cond_idx = fixed_cond_idx.astype(np.int64)
        elif C_train is not None and len(C_train) > 0:
            unique_conds = np.unique(C_train)
            n_eval = min(DDPM_EVAL_BATCH_SIZE, len(unique_conds))
            if len(unique_conds) >= n_eval:
                fixed_cond_idx = np.tile(unique_conds[:n_eval], (DDPM_EVAL_BATCH_SIZE // n_eval + 1))[:DDPM_EVAL_BATCH_SIZE]
            else:
                fixed_cond_idx = np.tile(unique_conds, (DDPM_EVAL_BATCH_SIZE // len(unique_conds) + 1))[:DDPM_EVAL_BATCH_SIZE]
            fixed_cond_idx = fixed_cond_idx.astype(np.int64)
    
    
    real_val_windows = None
    if use_realism_metrics and dm.val_ds is not None:
        real_windows_list = []
        n_real = min(DDPM_EVAL_BATCH_SIZE, len(dm.val_ds))
        for i in range(n_real):
            real_windows_list.append(dm.val_ds[i]["window"].numpy())
        if real_windows_list:
            real_val_windows = np.array(real_windows_list)
    
    ddpm_filename = f"ddpm_best_{sensor_type.lower()}.pt" if sensor_type else "ddpm_best.pt"
    ddpm_realism_filename = f"ddpm_best_by_realism_{sensor_type.lower()}.pt" if sensor_type else "ddpm_best_by_realism.pt"
    
    if vae is not None:
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
    
    for epoch in range(DDPM_EPOCHS):
        model.train()
        epoch_total = []
        epoch_mse = []
        epoch_var_loss = []
        epoch_smooth = []
        epoch_jerk = []
        epoch_spec = []
        epoch_rms = []
        epoch_rms_pred_mean = []
        epoch_rms_real_mean = []
        for z0, cond in train_loader:
            z0 = z0.to(DEVICE)
            cond = cond.to(DEVICE)
            t = torch.randint(0, ddpm.num_timesteps, (z0.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(z0)
            zt = ddpm.q_sample(z0, t, noise)
            noise_pred = model(zt, t, cond)
            
            mse_loss = F.mse_loss(noise_pred, noise)
            
            sqrt_ac = ddpm.sqrt_alphas_cumprod[t][:, None]
            sqrt_om = ddpm.sqrt_one_minus_alphas_cumprod[t][:, None]
            x0_pred_latent = (zt - sqrt_om * noise_pred) / (sqrt_ac + 1e-8)
            
            smooth_loss_val = torch.tensor(0.0, device=DEVICE)
            jerk_loss_val = torch.tensor(0.0, device=DEVICE)
            spec_loss_val = torch.tensor(0.0, device=DEVICE)
            rms_loss_val = torch.tensor(0.0, device=DEVICE)
            rms_pred_mean_val = torch.tensor(0.0, device=DEVICE)
            rms_real_mean_val = torch.tensor(0.0, device=DEVICE)
            
            if vae is not None and (DDPM_W_SMOOTH > 0 or DDPM_W_JERK > 0 or DDPM_W_SPEC > 0 or (DDPM_ENABLE_RMS and DDPM_W_RMS > 0)):
                x0_pred_latent.requires_grad_(True)
                x0_pred_windows_flat = vae.decode(x0_pred_latent)
                x0_pred_windows = x0_pred_windows_flat.view(z0.shape[0], WINDOW_SIZE, 3)
                
                if DDPM_W_SMOOTH > 0:
                    smooth_loss_val = compute_smoothness_loss_ddpm(x0_pred_windows)
                if DDPM_W_JERK > 0:
                    jerk_loss_val = compute_jerk_loss_ddpm(x0_pred_windows)
                if DDPM_W_SPEC > 0:
                    spec_loss_val = compute_spectral_loss_ddpm(
                        x0_pred_windows,
                        fs_hz=DDPM_SAMPLING_RATE_HZ,
                        cutoff_hz=DDPM_SPEC_CUTOFF_HZ,
                        fmin_hz=DDPM_SPEC_FMIN_HZ,
                        eps=DDPM_SPEC_EPS
                    )
                
                # RMS loss: decode z0 to get real windows for comparison
                if DDPM_ENABLE_RMS and DDPM_W_RMS > 0:
                    with torch.no_grad():
                        x0_real_windows_flat = vae.decode(z0)
                        x0_real_windows = x0_real_windows_flat.view(z0.shape[0], WINDOW_SIZE, 3)
                    rms_loss_val = compute_rms_loss(x0_pred_windows, x0_real_windows, eps=1e-8)
                    
                    # Compute mean RMS values for logging
                    eps_log = 1e-8
                    m_pred = torch.sqrt((x0_pred_windows ** 2).sum(dim=2) + eps_log)
                    m_real = torch.sqrt((x0_real_windows ** 2).sum(dim=2) + eps_log)
                    rms_pred = torch.sqrt((m_pred ** 2).mean(dim=1) + eps_log)
                    rms_real = torch.sqrt((m_real ** 2).mean(dim=1) + eps_log)
                    rms_pred_mean_val = rms_pred.mean()
                    rms_real_mean_val = rms_real.mean()
            
            t0 = torch.zeros_like(t)
            noise0 = torch.randn_like(z0)
            zt0 = ddpm.q_sample(z0, t0, noise0)
            noise_pred0 = model(zt0, t0, cond)
            sqrt_ac0 = ddpm.sqrt_alphas_cumprod[t0][:, None]
            sqrt_om0 = ddpm.sqrt_one_minus_alphas_cumprod[t0][:, None]
            recon_x0_0 = (zt0 - sqrt_om0 * noise_pred0) / (sqrt_ac0 + 1e-8)

            real_var = z0.var(dim=0, unbiased=False)
            synthetic_var = recon_x0_0.var(dim=0, unbiased=False)
            eps = 1e-8
            var_loss = F.mse_loss(torch.log(synthetic_var + eps), torch.log(real_var + eps))
            
            loss = (
                mse_loss
                + LAMBDA_VAR * var_loss
                + DDPM_W_SMOOTH * smooth_loss_val
                + DDPM_W_JERK * jerk_loss_val
                + DDPM_W_SPEC * spec_loss_val
                + (DDPM_W_RMS * rms_loss_val if DDPM_ENABLE_RMS else 0.0)
            )
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if ema_model is not None:
                ema_update(ema_model, model, DDPM_EMA_DECAY)
            epoch_total.append(loss.item())
            epoch_mse.append(mse_loss.item())
            epoch_var_loss.append(var_loss.item())
            epoch_smooth.append(smooth_loss_val.item() if isinstance(smooth_loss_val, torch.Tensor) else smooth_loss_val)
            epoch_jerk.append(jerk_loss_val.item() if isinstance(jerk_loss_val, torch.Tensor) else jerk_loss_val)
            epoch_spec.append(spec_loss_val.item() if isinstance(spec_loss_val, torch.Tensor) else spec_loss_val)
            epoch_rms.append(rms_loss_val.item() if isinstance(rms_loss_val, torch.Tensor) else rms_loss_val)
            epoch_rms_pred_mean.append(rms_pred_mean_val.item() if isinstance(rms_pred_mean_val, torch.Tensor) else rms_pred_mean_val)
            epoch_rms_real_mean.append(rms_real_mean_val.item() if isinstance(rms_real_mean_val, torch.Tensor) else rms_real_mean_val)
        tr_total = float(np.mean(epoch_total))
        tr_mse = float(np.mean(epoch_mse))
        tr_var = float(np.mean(epoch_var_loss))
        tr_smooth = float(np.mean(epoch_smooth))
        tr_jerk = float(np.mean(epoch_jerk))
        tr_spec = float(np.mean(epoch_spec))
        tr_rms = float(np.mean(epoch_rms))
        tr_rms_pred_mean = float(np.mean(epoch_rms_pred_mean))
        tr_rms_real_mean = float(np.mean(epoch_rms_real_mean))
        history["train_total"] = history.get("train_total", [])
        history["train_total"].append(tr_total)
        history["train_mse"] = history.get("train_mse", [])
        history["train_mse"].append(tr_mse)
        history["train_var"] = history.get("train_var", [])
        history["train_var"].append(tr_var)
        history["train_smooth"] = history.get("train_smooth", [])
        history["train_smooth"].append(tr_smooth)
        history["train_jerk"] = history.get("train_jerk", [])
        history["train_jerk"].append(tr_jerk)
        history["train_spec"] = history.get("train_spec", [])
        history["train_spec"].append(tr_spec)
        history["train_rms"] = history.get("train_rms", [])
        history["train_rms"].append(tr_rms)
        history["train_rms_pred_mean"] = history.get("train_rms_pred_mean", [])
        history["train_rms_pred_mean"].append(tr_rms_pred_mean)
        history["train_rms_real_mean"] = history.get("train_rms_real_mean", [])
        history["train_rms_real_mean"].append(tr_rms_real_mean)

        model.eval()
        v_total = []
        v_mse = []
        v_var = []
        with torch.no_grad():
            for z0, cond in val_loader:
                z0 = z0.to(DEVICE)
                cond = cond.to(DEVICE)
                t = torch.randint(0, ddpm.num_timesteps, (z0.shape[0],), device=DEVICE).long()
                noise = torch.randn_like(z0)
                zt = ddpm.q_sample(z0, t, noise)
                noise_pred = model(zt, t, cond)
                mse = F.mse_loss(noise_pred, noise)


                t0 = torch.zeros_like(t)
                noise0 = torch.randn_like(z0)
                zt0 = ddpm.q_sample(z0, t0, noise0)
                noise_pred0 = model(zt0, t0, cond)
                sqrt_ac0 = ddpm.sqrt_alphas_cumprod[t0][:, None]
                sqrt_om0 = ddpm.sqrt_one_minus_alphas_cumprod[t0][:, None]
                recon_x0_0 = (zt0 - sqrt_om0 * noise_pred0) / (sqrt_ac0 + 1e-8)

                real_var = z0.var(dim=0, unbiased=False)
                synthetic_var = recon_x0_0.var(dim=0, unbiased=False)
                eps = 1e-8
                var = F.mse_loss(torch.log(synthetic_var + eps), torch.log(real_var + eps))

                total = mse + LAMBDA_VAR * var
                v_total.append(total.item())
                v_mse.append(mse.item())
                v_var.append(var.item())

        va_total = float(np.mean(v_total)) if v_total else float("inf")
        va_mse = float(np.mean(v_mse)) if v_mse else float("inf")
        va_var = float(np.mean(v_var)) if v_var else float("inf")
        history["val_total"] = history.get("val_total", [])
        history["val_total"].append(va_total)
        history["val_mse"] = history.get("val_mse", [])
        history["val_mse"].append(va_mse)
        history["val_var"] = history.get("val_var", [])
        history["val_var"].append(va_var)

        print(
            f"[DDPM] epoch {epoch+1:03d} | "
            f"train_total {tr_total:.4f} | val_total {va_total:.4f} | "
            f"train_mse {tr_mse:.4f} | val_mse {va_mse:.4f} | "
            f"train_var {tr_var:.6f} | val_var {va_var:.6f} | "
            f"train_smooth {tr_smooth:.6f} | train_jerk {tr_jerk:.6f} | train_spec {tr_spec:.6f} | "
            f"train_rms {tr_rms:.6f} | rms_pred_mean {tr_rms_pred_mean:.4f} | rms_real_mean {tr_rms_real_mean:.4f}"
        )
        
        realism_metrics = None
        composite_score = None
        should_eval_realism = use_realism_metrics and ((epoch + 1) % DDPM_EVAL_INTERVAL == 0 or epoch == 0)
        
        if should_eval_realism:
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()
            eval_ddpm = LatentDDPM(eval_model, T=T, beta_schedule=BETA_SCHEDULE, device=DEVICE)
            if z_mean is not None and z_std is not None:
                eval_ddpm.z_mean = torch.tensor(z_mean, dtype=torch.float32, device=DEVICE)
                eval_ddpm.z_std = torch.tensor(z_std, dtype=torch.float32, device=DEVICE)
            
            vae.eval()
            synth_windows = generate_fixed_eval_samples(
                eval_ddpm, vae, DDPM_EVAL_BATCH_SIZE, fixed_cond_idx, DDPM_EVAL_SEED
            )
            
            realism_metrics = compute_realism_metrics(
                synth_windows,
                fs_hz=IMU_FS_HZ,
                hf_cutoff_hz=DDPM_REALISM_HF_CUTOFF_HZ,
                jerk_percentile=DDPM_REALISM_JERK_PERCENTILE
            )
            
            w1, w2, w3 = DDPM_REALISM_SCORE_WEIGHTS
            composite_score = (
                w1 * realism_metrics["hf_power_ratio_mean"] +
                w2 * realism_metrics["jerk_tail_mean"] +
                w3 * realism_metrics["dom_tail_mass_mean"]
            )
            
            print(
                f"  [Realism] hf_ratio={realism_metrics['hf_power_ratio_mean']:.4f} "
                f"jerk_tail={realism_metrics['jerk_tail_mean']:.4f} "
                f"tail_mass={realism_metrics['dom_tail_mass_mean']:.4f} "
                f"composite={composite_score:.4f}"
            )
            
            for key, val in realism_metrics.items():
                history_key = f"realism_{key}"
                if history_key not in history:
                    history[history_key] = []
                history[history_key].append(val)
            
            if "realism_composite_score" not in history:
                history["realism_composite_score"] = []
            history["realism_composite_score"].append(composite_score)
            
            eval_log = {
                "epoch": epoch + 1,
                "val_total": va_total,
                **realism_metrics,
                "composite_score": composite_score,
            }
            realism_log_path = os.path.join(save_dir, f"realism_metrics_{sensor_type.lower() if sensor_type else 'ddpm'}.jsonl")
            with open(realism_log_path, "a") as f:
                f.write(json.dumps(eval_log) + "\n")
            
            if composite_score < (best_realism_score - DDPM_REALISM_MIN_DELTA):
                best_realism_score = composite_score
                realism_bad = 0
                ddpm_realism_path = os.path.join(save_dir, ddpm_realism_filename)
                save_model = ema_model if ema_model is not None else model
                torch.save({
                    "model_state_dict": save_model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "ema_decay": DDPM_EMA_DECAY if ema_model is not None else None,
                    "num_timesteps": ddpm.num_timesteps,
                    "z_mean": z_mean,
                    "z_std": z_std,
                    "realism_metrics": realism_metrics,
                    "composite_score": composite_score,
                    "epoch": epoch + 1,
                }, ddpm_realism_path)
                print(f"  → Saved best realism model: {ddpm_realism_path} (composite_score: {composite_score:.4f})")
            else:
                realism_bad += 1
                if realism_bad >= realism_patience:
                    best_epoch = epoch + 1 - realism_patience * DDPM_EVAL_INTERVAL
                    print(
                        f"Early stopping DDPM (realism metrics). "
                        f"Best epoch: {best_epoch}, "
                        f"best score: {best_realism_score:.4f}, "
                        f"current score: {composite_score:.4f}"
                    )
                    break
        else:
            if "realism_composite_score" in history:
                history["realism_composite_score"].append(None)

        if va_total < best_val:
            best_val = va_total
            bad = 0
            ddpm_path = os.path.join(save_dir, ddpm_filename)
            save_model = ema_model if ema_model is not None else model
            torch.save({
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "ema_decay": DDPM_EMA_DECAY if ema_model is not None else None,
                "num_timesteps": ddpm.num_timesteps,
                "z_mean": z_mean,
                "z_std": z_std,
            }, ddpm_path)
            print(f"  → Saved best val_total model: {ddpm_path} (val_total: {va_total:.4f})")
        else:
            bad += 1
            if not use_realism_metrics and bad >= patience:
                print("Early stopping DDPM (val_total).")
                break

    final_model = ema_model if ema_model is not None else model
    ddpm = LatentDDPM(final_model, T=T, beta_schedule=BETA_SCHEDULE, device=DEVICE)
    if z_mean is not None and z_std is not None:
        ddpm.z_mean = torch.tensor(z_mean, dtype=torch.float32, device=DEVICE)
        ddpm.z_std = torch.tensor(z_std, dtype=torch.float32, device=DEVICE)
    return ddpm, history

