import os
import torch
import numpy as np
from .data import load_sensor_csvs, GaitDataModule
from .config import GAIT_BASE_DIR, WINDOW_SIZE, DEVICE, SAVE_DIR
from .vae import WindowVAE
from .eda.gait_signal_analysis import load_windows_from_data, run_eda

def main(sensor_type="Accelerometer", n_samples=500):
    # Load dataframe
    raw_df, _ = load_sensor_csvs(GAIT_BASE_DIR, sensor_type)
    dm = GaitDataModule(raw_df, save_dir=os.path.join(SAVE_DIR, sensor_type.lower()))
    dm.setup()

    # Load trained VAE
    vae_path = os.path.join(SAVE_DIR, sensor_type.lower(), f"vae_best_{sensor_type.lower()}.pt")
    vae = WindowVAE(input_dim=WINDOW_SIZE*3).to(DEVICE)
    vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
    vae.eval()

    # Load real windows
    real_windows, real_user_ids = load_windows_from_data(dm, sensor_type)
    real_windows = real_windows[:n_samples]
    real_user_ids = real_user_ids[:n_samples]

    # Compute reconstructions
    B = len(real_windows)
    x = torch.tensor(real_windows, dtype=torch.float32, device=DEVICE).view(B, -1)
    with torch.no_grad():
        recon_flat, _, _ = vae(x)
    recon_windows = recon_flat.view(B, WINDOW_SIZE, 3).cpu().numpy()

    # EDA save path
    save_dir = os.path.join(SAVE_DIR, sensor_type.lower(), "eda_vae_recon")
    os.makedirs(save_dir, exist_ok=True)

    # Run EDA comparing real vs reconstructed
    run_eda(real_windows, real_user_ids, recon_windows, real_user_ids, sensor_type, save_dir)

if __name__ == "__main__":
    main()
