import os
import json
import numpy as np
import pandas as pd
import torch
from ..config import (
    GAIT_BASE_DIR, WINDOW_SIZE, HIDDEN_DIM, N_LAYERS, DEVICE,
    T, BETA_SCHEDULE, USE_SMOOTHNESS_LOSS, USE_DISTRIBUTION_LOSS
)
from ..vae import WindowVAE
from ..data import GaitDataModule, load_sensor_csvs
from ..train import train_vae, encode_dataset_to_latents, train_ddpm
from ..diffusion import UNetMLP, LatentDDPM
from .statistical_tests import run_statistical_tests
from .real_vs_synthetic_classifier import evaluate_real_vs_synthetic
from .augmentation_effectiveness import run_augmentation_experiment
from ..eda.gait_signal_analysis import load_windows_from_data
from ..generate import sample_synthetic

def train_vae_with_overrides(vae, dm, save_dir, sensor_type, use_physics):
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    from ..config import (
        VAE_LR, VAE_EPOCHS, VAE_PATIENCE, KL_MAX_BETA, KL_WARMUP_EPOCHS, KL_BETA,
        LAMBDA_SMOOTH, LAMBDA_PHYS, DEVICE
    )
    from ..physics_losses import smoothness_loss, distribution_loss
    
    opt = torch.optim.AdamW(vae.parameters(), lr=VAE_LR)
    best_val = float("inf")
    patience = VAE_PATIENCE
    bad_steps = 0
    history = defaultdict(list)
    
    os.makedirs(save_dir, exist_ok=True)
    vae_filename = f"vae_best_{sensor_type.lower()}.pt"
    
    def kl_cosine_beta(epoch, warmup_epochs, max_beta=1.0):
        import math
        if warmup_epochs <= 0:
            return max_beta
        if epoch >= warmup_epochs:
            return max_beta
        x = (epoch+1)/warmup_epochs
        return float(0.5*(1 - math.cos(math.pi*x)))*max_beta

    for epoch in range(VAE_EPOCHS):
        vae.train()
        beta = kl_cosine_beta(epoch, KL_WARMUP_EPOCHS, KL_MAX_BETA)
        tr_losses = []
        
        for batch in dm.train_dataloader():
            windows = batch["window"].to(DEVICE)
            B, T, C = windows.shape
            x_flat = windows.view(B, -1)
            
            opt.zero_grad()
            recon_flat, mu, logvar = vae(x_flat)
            recon_windows = recon_flat.view(B, T, C)
            
            recon_loss = F.mse_loss(recon_windows, windows)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            smooth_loss_val = smoothness_loss(recon_windows) if use_physics else 0.0
            phys_loss_val = distribution_loss(windows, recon_windows) if use_physics else 0.0
            
            total_loss = (
                recon_loss
                + KL_BETA * beta * kl_loss
                + LAMBDA_SMOOTH * smooth_loss_val
                + LAMBDA_PHYS * phys_loss_val
            )
            
            total_loss.backward()
            opt.step()
            tr_losses.append(total_loss.item())
        
        train_loss = float(np.mean(tr_losses))

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
                
                smooth_loss_val = smoothness_loss(recon_windows) if use_physics else 0.0
                phys_loss_val = distribution_loss(windows, recon_windows) if use_physics else 0.0
                
                total_loss = (
                    recon_loss
                    + KL_BETA * KL_MAX_BETA * kl_loss
                    + LAMBDA_SMOOTH * smooth_loss_val
                    + LAMBDA_PHYS * phys_loss_val
                )
                val_losses.append(total_loss.item())
            val_loss = float(np.mean(val_losses))
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["recon"].append(recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad_steps = 0
            vae_path = os.path.join(save_dir, vae_filename)
            torch.save(vae.state_dict(), vae_path)
        else:
            bad_steps += 1
            if bad_steps >= patience:
                break

    return history

def run_vae_ablation(sensor_type: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    latent_dims = [32, 64, 128]
    physics_configs = [True, False]
    
    for latent_dim in latent_dims:
        for use_physics in physics_configs:
            if latent_dim == 64 and use_physics == False:
                continue
            
            print(f"\n{'='*60}")
            print(f"VAE Ablation: latent_dim={latent_dim}, physics={use_physics}")
            print(f"{'='*60}\n")
            
            exp_save_dir = os.path.join(save_dir, f"latent_{latent_dim}_physics_{use_physics}")
            os.makedirs(exp_save_dir, exist_ok=True)
            
            raw_df, _ = load_sensor_csvs(GAIT_BASE_DIR, sensor_type)
            dm = GaitDataModule(raw_df, save_dir=exp_save_dir)
            dm.setup()
            
            input_dim = WINDOW_SIZE * 3
            vae = WindowVAE(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)
            
            vae_hist = train_vae_with_overrides(vae, dm, exp_save_dir, sensor_type, use_physics)
            
            vae_path = os.path.join(exp_save_dir, f"vae_best_{sensor_type.lower()}.pt")
            vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
            vae.eval()
            
            Z_train, C_train = encode_dataset_to_latents(dm.train_ds, vae)
            Z_val, C_val = encode_dataset_to_latents(dm.val_ds, vae)
            
            unet = UNetMLP(
                latent_dim=latent_dim,
                hidden_dim=HIDDEN_DIM,
                n_blocks=8,
                cond_dim=1,
                dropout=0.1,
            ).to(DEVICE)
            
            ddpm, ddpm_hist = train_ddpm(unet, Z_train, C_train, Z_val, C_val, exp_save_dir, sensor_type=sensor_type)
            
            ddpm_path = os.path.join(exp_save_dir, f"ddpm_best_{sensor_type.lower()}.pt")
            unet.load_state_dict(torch.load(ddpm_path, map_location=DEVICE))
            unet.eval()
            
            synth_df = sample_synthetic(1000, ddpm, vae, dm)
            
            real_windows, real_user_ids = load_windows_from_data(dm, sensor_type)
            synth_windows = []
            synth_user_ids = []
            
            for user_id in np.unique(real_user_ids)[:10]:
                user_synth = synth_df[synth_df["__user_id__"] == dm.rev_cond_map.get(user_id, "synthetic")]
                if len(user_synth) >= WINDOW_SIZE:
                    user_data = user_synth[["Xvalue", "Yvalue", "Zvalue"]].values[:WINDOW_SIZE*10]
                    for i in range(0, len(user_data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                        synth_windows.append(user_data[i:i+WINDOW_SIZE])
                        synth_user_ids.append(user_id)
            
            synth_windows = np.array(synth_windows[:len(real_windows)])
            synth_user_ids = np.array(synth_user_ids[:len(real_windows)])
            
            recon_loss = vae_hist["recon"][-1] if "recon" in vae_hist else 0.0
            
            from sklearn.metrics.pairwise import pairwise_distances
            mmd = np.mean(pairwise_distances(Z_train[:1000], Z_val[:1000]))
            
            classifier_results = evaluate_real_vs_synthetic(
                real_windows[:1000], synth_windows[:1000], sensor_type, exp_save_dir
            )
            
            augmentation_results = run_augmentation_experiment(
                real_windows[:1000], real_user_ids[:1000],
                synth_windows[:1000], synth_user_ids[:1000],
                sensor_type, exp_save_dir, n_seeds=3
            )
            
            f1_improvement = np.mean(augmentation_results["full_data"]["real_synth"]) - np.mean(augmentation_results["full_data"]["real_only"])
            
            results.append({
                "latent_dim": latent_dim,
                "physics_loss": use_physics,
                "recon_loss": recon_loss,
                "mmd": mmd,
                "classifier_auc": classifier_results["auc"],
                "f1_improvement": f1_improvement
            })
    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(save_dir, f"ablation_vae_{sensor_type.lower()}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nVAE Ablation Results:")
    print(results_df.to_string())
    print(f"\nResults saved to {output_path}")
    
    return results_df

