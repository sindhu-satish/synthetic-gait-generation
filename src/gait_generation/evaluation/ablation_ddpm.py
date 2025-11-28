import os
import numpy as np
import pandas as pd
import torch
from ..config import (
    GAIT_BASE_DIR, WINDOW_SIZE, LATENT_DIM, HIDDEN_DIM, N_LAYERS, DEVICE,
    T, BETA_SCHEDULE
)
from ..vae import WindowVAE
from ..data import GaitDataModule, load_sensor_csvs
from ..train import train_vae, encode_dataset_to_latents, train_ddpm
from ..diffusion import UNetMLP, LatentDDPM
from .real_vs_synthetic_classifier import evaluate_real_vs_synthetic
from .augmentation_effectiveness import run_augmentation_experiment
from ..eda.gait_signal_analysis import load_windows_from_data
from ..generate import sample_synthetic
from sklearn.metrics.pairwise import pairwise_distances

def run_ddpm_ablation(sensor_type: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    sampling_steps = [50, 100, 300]
    use_ema = [True, False]
    
    print(f"\n{'='*60}")
    print(f"Loading base VAE for DDPM ablation")
    print(f"{'='*60}\n")
    
    base_save_dir = os.path.join(save_dir, "base_vae")
    os.makedirs(base_save_dir, exist_ok=True)
    
    raw_df, _ = load_sensor_csvs(GAIT_BASE_DIR, sensor_type)
    dm = GaitDataModule(raw_df, save_dir=base_save_dir)
    dm.setup()
    
    input_dim = WINDOW_SIZE * 3
    vae = WindowVAE(input_dim=input_dim).to(DEVICE)
    
    vae_path = os.path.join(base_save_dir, f"vae_best_{sensor_type.lower()}.pt")
    if not os.path.exists(vae_path):
        train_vae(vae, dm, base_save_dir, sensor_type=sensor_type)
    
    vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
    vae.eval()
    
    Z_train, C_train = encode_dataset_to_latents(dm.train_ds, vae)
    Z_val, C_val = encode_dataset_to_latents(dm.val_ds, vae)
    
    for steps in sampling_steps:
        for ema in use_ema:
            print(f"\n{'='*60}")
            print(f"DDPM Ablation: steps={steps}, EMA={ema}")
            print(f"{'='*60}\n")
            
            exp_save_dir = os.path.join(save_dir, f"steps_{steps}_ema_{ema}")
            os.makedirs(exp_save_dir, exist_ok=True)
            
            unet = UNetMLP(
                latent_dim=LATENT_DIM,
                hidden_dim=HIDDEN_DIM,
                n_blocks=8,
                cond_dim=1,
                dropout=0.1,
            ).to(DEVICE)
            
            ddpm_path = os.path.join(exp_save_dir, f"ddpm_best_{sensor_type.lower()}.pt")
            if not os.path.exists(ddpm_path):
                ddpm, ddpm_hist = train_ddpm(unet, Z_train, C_train, Z_val, C_val, exp_save_dir, sensor_type=sensor_type)
            else:
                unet.load_state_dict(torch.load(ddpm_path, map_location=DEVICE))
                ddpm = LatentDDPM(unet, T=T, beta_schedule=BETA_SCHEDULE, device=DEVICE)
            
            unet.eval()
            
            synth_df = sample_synthetic(1000, ddpm, vae, dm, steps=steps)
            
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
            
            sample_diversity = np.mean(pairwise_distances(Z_train[:500], Z_val[:500]))
            
            classifier_results = evaluate_real_vs_synthetic(
                real_windows[:1000], synth_windows[:1000], sensor_type, exp_save_dir
            )
            
            augmentation_results = run_augmentation_experiment(
                real_windows[:1000], real_user_ids[:1000],
                synth_windows[:1000], synth_user_ids[:1000],
                sensor_type, exp_save_dir, n_seeds=3
            )
            
            f1_score = np.mean(augmentation_results["full_data"]["real_synth"])
            
            results.append({
                "sampling_steps": steps,
                "use_ema": ema,
                "sample_diversity": sample_diversity,
                "classifier_auc": classifier_results["auc"],
                "downstream_f1": f1_score
            })
    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(save_dir, f"ablation_ddpm_{sensor_type.lower()}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nDDPM Ablation Results:")
    print(results_df.to_string())
    print(f"\nResults saved to {output_path}")
    
    return results_df

