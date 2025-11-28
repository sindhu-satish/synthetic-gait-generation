import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from .vae import VAE
from .diffusion import LatentDDPM, LatentDataset
from .data import KeystrokeDataModule
from .config import Config

def kl_cosine_beta(epoch, warmup_epochs, max_beta=1.0):
    if warmup_epochs <= 0:
        return max_beta
    if epoch >= warmup_epochs:
        return max_beta
    x = (epoch+1)/warmup_epochs
    return float(0.5*(1 - math.cos(math.pi*x)))*max_beta

def train_vae(vae, dm: KeystrokeDataModule, cfg: Config):
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.vae_lr)
    best_val = float("inf")
    patience = cfg.vae_patience
    bad_steps = 0
    history = defaultdict(list)
    
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(cfg.vae_epochs):
        vae.train()
        beta = kl_cosine_beta(epoch, cfg.kl_warmup_epochs, cfg.kl_max_beta)
        tr_losses = []
        for batch in dm.train_dataloader():
            num_x = batch["num"].to(cfg.device)
            cat_idx = batch["cat"].to(cfg.device) if batch["cat"].ndim == 2 else torch.zeros((len(num_x), 0), dtype=torch.long, device=cfg.device)
            opt.zero_grad()
            loss, logs = vae(num_x, cat_idx, beta=beta)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        train_loss = float(np.mean(tr_losses))

        vae.eval()
        with torch.no_grad():
            val_losses = []
            for batch in dm.val_dataloader():
                num_x = batch["num"].to(cfg.device)
                cat_idx = batch["cat"].to(cfg.device) if batch["cat"].ndim == 2 else torch.zeros((len(num_x), 0), dtype=torch.long, device=cfg.device)
                loss, logs = vae(num_x, cat_idx, beta=cfg.kl_max_beta)
                val_losses.append(loss.item())
            val_loss = float(np.mean(val_losses))
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["beta"].append(beta)
        print(f"[VAE] epoch {epoch+1:03d} | train {train_loss:.4f} | val {val_loss:.4f} | beta {beta:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_steps = 0
            torch.save(vae.state_dict(), os.path.join(cfg.save_dir, "vae_best.pt"))
        else:
            bad_steps += 1
            if bad_steps >= patience:
                print("Early stopping VAE.")
                break

    return history

@torch.no_grad()
def encode_dataset_to_latents(ds, vae, cfg: Config, batch_size=1024):
    from torch.utils.data import DataLoader
    latents, conds = [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        num_x = batch["num"].to(cfg.device)
        if batch["cat"].ndim == 1:
            cat_idx = torch.zeros((len(num_x), 0), dtype=torch.long, device=cfg.device)
        else:
            cat_idx = batch["cat"].to(cfg.device)
        mu, logvar = vae.encode(num_x, cat_idx)
        latents.append(mu.detach().cpu())
        if "cond" in batch:
            conds.append(batch["cond"])
    Z = torch.cat(latents, dim=0).numpy()
    C = torch.cat(conds, dim=0).numpy() if conds else None
    return Z, C

def train_ddpm(model, Z_train, C_train, Z_val, C_val, cfg: Config):
    ddpm = LatentDDPM(model, T=cfg.T, beta_schedule=cfg.beta_schedule, device=cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ddpm_lr)
    train_ds = LatentDataset(Z_train, C_train)
    val_ds = LatentDataset(Z_val, C_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.ddpm_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.ddpm_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    best_val = float("inf")
    patience = cfg.ddpm_patience
    bad = 0
    tr_hist, va_hist = [], []
    
    for epoch in range(cfg.ddpm_epochs):
        model.train()
        epoch_loss = []
        for z0, cond in train_loader:
            z0 = z0.to(cfg.device)
            cond = cond.to(cfg.device)
            t = torch.randint(0, ddpm.T, (z0.shape[0],), device=cfg.device).long()
            noise = torch.randn_like(z0)
            zt = ddpm.q_sample(z0, t, noise)
            noise_pred = model(zt, t, cond)
            loss = F.mse_loss(noise_pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
        tr = float(np.mean(epoch_loss))
        tr_hist.append(tr)

        model.eval()
        vlosses = []
        with torch.no_grad():
            for z0, cond in val_loader:
                z0 = z0.to(cfg.device)
                cond = cond.to(cfg.device)
                t = torch.randint(0, ddpm.T, (z0.shape[0],), device=cfg.device).long()
                noise = torch.randn_like(z0)
                zt = ddpm.q_sample(z0, t, noise)
                noise_pred = model(zt, t, cond)
                vlosses.append(F.mse_loss(noise_pred, noise).item())
        va = float(np.mean(vlosses))
        va_hist.append(va)
        print(f"[DDPM] epoch {epoch+1:03d} | train {tr:.4f} | val {va:.4f}")

        if va < best_val:
            best_val = va
            bad = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "ddpm_best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping DDPM.")
                break

    return ddpm

