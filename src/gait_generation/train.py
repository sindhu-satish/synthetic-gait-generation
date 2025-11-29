import os
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
    NUM_WORKERS, DDPM_USE_MU_ONLY, DDPM_USE_EMA, DDPM_EMA_DECAY, PIN_MEMORY, LAMBDA_VAR
)
from .physics_losses import smoothness_loss, distribution_loss, spectral_loss

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
    
    ddpm_filename = f"ddpm_best_{sensor_type.lower()}.pt" if sensor_type else "ddpm_best.pt"
    
    for epoch in range(DDPM_EPOCHS):
        model.train()
        epoch_total = []
        epoch_mse = []
        epoch_var_loss = []
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
            recon_x0 = (zt - sqrt_om * noise_pred) / (sqrt_ac + 1e-8)
            
            
            real_var = z0.var(dim=0, unbiased=False)
            synthetic_var = recon_x0.var(dim=0, unbiased=False)
            denom = real_var.detach().mean().clamp_min(1e-8)
            var_loss = F.mse_loss(synthetic_var, real_var) / denom
            
            loss = mse_loss + LAMBDA_VAR * var_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if ema_model is not None:
                ema_update(ema_model, model, DDPM_EMA_DECAY)
            epoch_total.append(loss.item())
            epoch_mse.append(mse_loss.item())
            epoch_var_loss.append(var_loss.item())
        tr_total = float(np.mean(epoch_total))
        tr_mse = float(np.mean(epoch_mse))
        tr_var = float(np.mean(epoch_var_loss))
        history["train_total"] = history.get("train_total", [])
        history["train_total"].append(tr_total)
        history["train_mse"] = history.get("train_mse", [])
        history["train_mse"].append(tr_mse)
        history["train_var"] = history.get("train_var", [])
        history["train_var"].append(tr_var)

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

                sqrt_ac = ddpm.sqrt_alphas_cumprod[t][:, None]
                sqrt_om = ddpm.sqrt_one_minus_alphas_cumprod[t][:, None]
                recon_x0 = (zt - sqrt_om * noise_pred) / (sqrt_ac + 1e-8)

                real_var = z0.var(dim=0, unbiased=False)
                synthetic_var = recon_x0.var(dim=0, unbiased=False)
                denom = real_var.detach().mean().clamp_min(1e-8)
                var = F.mse_loss(synthetic_var, real_var) / denom

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
            f"train_var {tr_var:.6f} | val_var {va_var:.6f}"
        )

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
            print(f"  → Saved best DDPM model: {ddpm_path} (val_total: {va_total:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping DDPM.")
                break

    final_model = ema_model if ema_model is not None else model
    ddpm = LatentDDPM(final_model, T=T, beta_schedule=BETA_SCHEDULE, device=DEVICE)
    if z_mean is not None and z_std is not None:
        ddpm.z_mean = torch.tensor(z_mean, dtype=torch.float32, device=DEVICE)
        ddpm.z_std = torch.tensor(z_std, dtype=torch.float32, device=DEVICE)
    return ddpm, history

