import os
import json
import torch
import numpy as np
from .config import (
    set_seed, GAIT_BASE_DIR, SENSOR_TYPES, SEED, SAVE_DIR, DEVICE,
    WINDOW_SIZE, LATENT_DIM, HIDDEN_DIM, N_LAYERS, T, BETA_SCHEDULE
)
from .data import load_sensor_csvs, GaitDataModule
from .vae import WindowVAE
from .diffusion import UNetMLP, LatentDDPM
from .train import train_vae, train_ddpm, encode_dataset_to_latents
from .generate import sample_synthetic
from .preprocessor import Preprocessor
from .eda.gait_signal_analysis import load_windows_from_data, run_eda
from .evaluation.real_vs_synthetic_classifier import evaluate_real_vs_synthetic
from .evaluation.statistical_tests import run_statistical_tests
from .evaluation.augmentation_effectiveness import run_augmentation_experiment
from .postprocess import PostprocessConfig, postprocess_windows

def train_windowed_vae(sensor_type: str, save_dir: str = None):
    print(f"\n{'='*60}")
    print(f"Training VAE for {sensor_type}")
    print(f"{'='*60}\n")
    
    sensor_save_dir = os.path.join(save_dir or SAVE_DIR, sensor_type.lower())
    os.makedirs(sensor_save_dir, exist_ok=True)
    
    set_seed(SEED)
    
    raw_df, _ = load_sensor_csvs(GAIT_BASE_DIR, sensor_type)
    dm = GaitDataModule(raw_df, save_dir=sensor_save_dir)
    dm.setup()
    
    input_dim = WINDOW_SIZE * 3
    vae = WindowVAE(input_dim=input_dim).to(DEVICE)
    
    vae_path = os.path.join(sensor_save_dir, f"vae_best_{sensor_type.lower()}.pt")
    if not os.path.exists(vae_path):
        print(f"VAE model not found at {vae_path}, starting training...")
        vae_hist = train_vae(vae, dm, sensor_save_dir, sensor_type=sensor_type)
        vae_hist_path = os.path.join(sensor_save_dir, f"vae_history_{sensor_type.lower()}.json")
        with open(vae_hist_path, 'w') as f:
            json.dump({k: [float(v) if isinstance(v, (int, float)) else v for v in vals] for k, vals in vae_hist.items()}, f, indent=2)
    else:
        print(f"Found existing VAE model at {vae_path}, skipping training...")
        vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
    
    vae.eval()
    return vae, dm, sensor_save_dir

def train_ddpm_model(sensor_type: str, vae, dm, save_dir: str):
    print(f"\n{'='*60}")
    print(f"Training DDPM for {sensor_type}")
    print(f"{'='*60}\n")
    
    Z_train, C_train = encode_dataset_to_latents(dm.train_ds, vae)
    Z_val, C_val = encode_dataset_to_latents(dm.val_ds, vae)


    z_mean = Z_train.mean(axis=0)
    z_std = Z_train.std(axis=0)
    z_std = np.maximum(z_std, 1e-6)
    Z_train = (Z_train - z_mean) / z_std
    Z_val = (Z_val - z_mean) / z_std
    
    unet = UNetMLP(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_blocks=8,
        cond_dim=1,
        dropout=0.1,
    ).to(DEVICE)
    
    ddpm_path = os.path.join(save_dir, f"ddpm_best_{sensor_type.lower()}.pt")
    if not os.path.exists(ddpm_path):
        print(f"DDPM model not found at {ddpm_path}, starting training...")
        ddpm, ddpm_hist = train_ddpm(
            unet, Z_train, C_train, Z_val, C_val, save_dir,
            sensor_type=sensor_type,
            z_mean=z_mean,
            z_std=z_std,
        )
        ddpm_hist_path = os.path.join(save_dir, f"ddpm_history_{sensor_type.lower()}.json")
        with open(ddpm_hist_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in ddpm_hist.items()}, f, indent=2)
    else:
        print(f"Found existing DDPM model at {ddpm_path}, skipping training...")
        unet.load_state_dict(torch.load(ddpm_path, map_location=DEVICE))
        ddpm = LatentDDPM(unet, T=T, beta_schedule=BETA_SCHEDULE, device=DEVICE)
        
        if isinstance(checkpoint, dict) and checkpoint.get("z_mean") is not None and checkpoint.get("z_std") is not None:
            ddpm.z_mean = torch.tensor(checkpoint["z_mean"], dtype=torch.float32, device=DEVICE)
            ddpm.z_std = torch.tensor(checkpoint["z_std"], dtype=torch.float32, device=DEVICE)
    
    unet.eval()
    return ddpm, unet

def run_eda_analysis(sensor_type: str, dm, save_dir: str, vae, ddpm, post_cfg: PostprocessConfig, fs_hz: float):
    print(f"\n{'='*60}")
    print(f"Running EDA for {sensor_type}")
    print(f"{'='*60}\n")
    
    real_windows, real_user_ids = load_windows_from_data(dm, sensor_type)
    
    synth_df = sample_synthetic(len(real_windows), ddpm, vae, dm)
    
    synth_windows = []
    synth_user_ids = []
    
    
    unique_real_user_ids = np.unique(real_user_ids)
    
    for user_id in unique_real_user_ids[:20]:
        
        user_id_str = str(user_id)
        user_synth = synth_df[synth_df["__user_id__"] == user_id_str]
        
        if len(user_synth) >= WINDOW_SIZE:
            user_data = user_synth[["Xvalue", "Yvalue", "Zvalue"]].values
            for i in range(0, len(user_data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                synth_windows.append(user_data[i:i+WINDOW_SIZE])
                synth_user_ids.append(user_id)
    
    
    if len(synth_windows) < len(real_windows) and len(synth_df) >= WINDOW_SIZE:
        print(f"Warning: Only found {len(synth_windows)} matching synthetic windows, using all available synthetic data")
        all_synth_data = synth_df[["Xvalue", "Yvalue", "Zvalue"]].values
        for i in range(0, len(all_synth_data) - WINDOW_SIZE + 1, WINDOW_SIZE):
            if len(synth_windows) >= len(real_windows):
                break
            synth_windows.append(all_synth_data[i:i+WINDOW_SIZE])
            
            synth_user_id = synth_df.iloc[i]["__user_id__"]
            
            try:
                
                synth_user_ids.append(int(synth_user_id) if synth_user_id.isdigit() else synth_user_id)
            except:
                synth_user_ids.append(synth_user_id)
    
    synth_windows = np.array(synth_windows[:len(real_windows)]) if len(synth_windows) > 0 else np.array([])
    synth_user_ids = np.array(synth_user_ids[:len(real_windows)]) if len(synth_user_ids) > 0 else np.array([])
    
    synth_windows_post = None
    if synth_windows.size > 0 and (post_cfg.enable_fft_lowpass or post_cfg.enable_savgol):
        synth_windows_post, _ = postprocess_windows(
            synth_windows, fs=fs_hz, config=post_cfg, log_prefix=f"[{sensor_type}][EDA]"
        )

    eda_save_dir = os.path.join(save_dir, "eda")
    run_eda(
        real_windows, real_user_ids,
        synth_windows, synth_user_ids,
        sensor_type, eda_save_dir,
        fs_hz=fs_hz,
        synth_windows_post=synth_windows_post,
    )

def run_realism_evaluation(sensor_type: str, real_windows, real_user_ids, synth_windows, synth_user_ids, save_dir: str, post_cfg: PostprocessConfig, fs_hz: float):
    print(f"\n{'='*60}")
    print(f"Running Realism Evaluation for {sensor_type}")
    print(f"{'='*60}\n")
    
    eval_save_dir = os.path.join(save_dir, "evaluation")
    os.makedirs(eval_save_dir, exist_ok=True)
    
    
    raw_dir = os.path.join(eval_save_dir, "raw")
    classifier_results = evaluate_real_vs_synthetic(real_windows, synth_windows, sensor_type, raw_dir)
    stats_results = run_statistical_tests(real_windows, synth_windows, sensor_type, raw_dir)


    if synth_windows is not None and len(synth_windows) > 0 and (post_cfg.enable_fft_lowpass or post_cfg.enable_savgol):
        synth_post, _ = postprocess_windows(
            synth_windows, fs=fs_hz, config=post_cfg, log_prefix=f"[{sensor_type}][Eval]"
        )
        post_dir = os.path.join(eval_save_dir, "postprocessed")
        _ = evaluate_real_vs_synthetic(real_windows, synth_post, sensor_type, post_dir)
        _ = run_statistical_tests(real_windows, synth_post, sensor_type, post_dir)
    
    return classifier_results, stats_results

def run_augmentation_experiment_wrapper(sensor_type: str, real_windows, real_user_ids, synth_windows, synth_user_ids, save_dir: str, post_cfg: PostprocessConfig, fs_hz: float):
    print(f"\n{'='*60}")
    print(f"Running Augmentation Experiment for {sensor_type}")
    print(f"{'='*60}\n")
    
    eval_save_dir = os.path.join(save_dir, "evaluation")
    
    raw_dir = os.path.join(eval_save_dir, "raw")
    augmentation_results = run_augmentation_experiment(
        real_windows, real_user_ids, synth_windows, synth_user_ids,
        sensor_type, raw_dir
    )

    if synth_windows is not None and len(synth_windows) > 0 and (post_cfg.enable_fft_lowpass or post_cfg.enable_savgol):
        synth_post, _ = postprocess_windows(
            synth_windows, fs=fs_hz, config=post_cfg, log_prefix=f"[{sensor_type}][Aug]"
        )
        post_dir = os.path.join(eval_save_dir, "postprocessed")
        _ = run_augmentation_experiment(
            real_windows, real_user_ids, synth_post, synth_user_ids,
            sensor_type, post_dir
        )
    
    return augmentation_results

def main(
    skip_eda: bool = False,
    post_fft: bool = False,
    post_savgol: bool = False,
    fs: float | None = None,
    fft_fc: float | None = None,
    savgol_window: int | None = None,
    savgol_poly: int | None = None,
):
    set_seed(SEED)
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    from .config import IMU_FS_HZ, POST_FFT_CUTOFF_HZ, POST_SAVGOL_WINDOW_LENGTH, POST_SAVGOL_POLYORDER
    fs_hz = float(IMU_FS_HZ if fs is None else fs)
    post_cfg = PostprocessConfig(
        enable_fft_lowpass=bool(post_fft),
        fft_cutoff_hz=float(POST_FFT_CUTOFF_HZ if fft_fc is None else fft_fc),
        enable_savgol=bool(post_savgol),
        savgol_window_length=int(POST_SAVGOL_WINDOW_LENGTH if savgol_window is None else savgol_window),
        savgol_polyorder=int(POST_SAVGOL_POLYORDER if savgol_poly is None else savgol_poly),
    )
    
    for sensor_type in SENSOR_TYPES:
        vae, dm, save_dir = train_windowed_vae(sensor_type, SAVE_DIR)
        ddpm, unet = train_ddpm_model(sensor_type, vae, dm, save_dir)
        
        real_windows, real_user_ids = load_windows_from_data(dm, sensor_type)
        
        synth_df = sample_synthetic(len(real_windows), ddpm, vae, dm)
        
        synth_windows = []
        synth_user_ids = []
        
        
        unique_real_user_ids = np.unique(real_user_ids)
        
        for user_id in unique_real_user_ids[:20]:
            user_id_str = str(user_id)
            user_synth = synth_df[synth_df["__user_id__"] == user_id_str]
            
            if len(user_synth) >= WINDOW_SIZE:
                user_data = user_synth[["Xvalue", "Yvalue", "Zvalue"]].values
                for i in range(0, len(user_data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    synth_windows.append(user_data[i:i+WINDOW_SIZE])
                    synth_user_ids.append(user_id)
        
        if len(synth_windows) < len(real_windows) and len(synth_df) >= WINDOW_SIZE:
            print(f"Warning: Only found {len(synth_windows)} matching synthetic windows, using all available synthetic data")
            all_synth_data = synth_df[["Xvalue", "Yvalue", "Zvalue"]].values
            for i in range(0, len(all_synth_data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                if len(synth_windows) >= len(real_windows):
                    break
                synth_windows.append(all_synth_data[i:i+WINDOW_SIZE])
                
                synth_user_id = synth_df.iloc[i]["__user_id__"]
                try:
                    synth_user_ids.append(int(synth_user_id) if synth_user_id.isdigit() else synth_user_id)
                except:
                    synth_user_ids.append(synth_user_id)
        
        synth_windows = np.array(synth_windows[:len(real_windows)]) if len(synth_windows) > 0 else np.array([])
        synth_user_ids = np.array(synth_user_ids[:len(real_windows)]) if len(synth_user_ids) > 0 else np.array([])
        
        if not skip_eda:
            run_eda_analysis(sensor_type, dm, save_dir, vae, ddpm, post_cfg=post_cfg, fs_hz=fs_hz)
        else:
            print(f"\n{'='*60}")
            print(f"Skipping EDA for {sensor_type} (skip_eda=True)")
            print(f"{'='*60}\n")
        
        run_realism_evaluation(sensor_type, real_windows, real_user_ids, synth_windows, synth_user_ids, save_dir, post_cfg=post_cfg, fs_hz=fs_hz)
        run_augmentation_experiment_wrapper(sensor_type, real_windows, real_user_ids, synth_windows, synth_user_ids, save_dir, post_cfg=post_cfg, fs_hz=fs_hz)
        
        print(f"\n{'='*60}")
        print(f"Pipeline complete for {sensor_type}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

