import numpy as np
import torch
from .diffusion import LatentDDPM
from .vae import WindowVAE
from .data import GaitDataModule
from .config import WINDOW_SIZE

@torch.no_grad()
def sample_synthetic(n: int, ddpm: LatentDDPM, vae: WindowVAE, dm: GaitDataModule, steps: int = None):
    from .config import SAMPLE_STEPS, DEVICE
    if steps is None:
        steps = SAMPLE_STEPS
    
    cond_idx = None
    if hasattr(dm, 'train_ds') and dm.train_ds is not None:
        conds_list = []
        for i in range(min(100, len(dm.train_ds))):
            cond_val = dm.train_ds[i]["cond"]
            if isinstance(cond_val, torch.Tensor):
                cond_val = cond_val.item()
            conds_list.append(cond_val)
        unique_conds = np.unique(conds_list)
        if len(unique_conds) > 0:
            probs = np.ones(len(unique_conds)) / len(unique_conds)
            cond_idx = np.random.choice(unique_conds, size=n, p=probs)
    
    Z = ddpm.sample(n=n, cond_idx=cond_idx, steps=steps).to(DEVICE)
    if hasattr(ddpm, "z_mean") and hasattr(ddpm, "z_std") and ddpm.z_mean is not None and ddpm.z_std is not None:
        Z = Z * ddpm.z_std + ddpm.z_mean
    
    recon_flat = vae.decode(Z)
    recon_windows = recon_flat.view(n, WINDOW_SIZE, 3)
    
    recon_windows = recon_windows + 0.05 * torch.randn_like(recon_windows)
    
    windows_np = recon_windows.detach().cpu().numpy()
    
    synth_data = []
    for i, window in enumerate(windows_np):
        cond_val = cond_idx[i] if cond_idx is not None else 0
        user_id = dm.rev_cond_map.get(cond_val, f"synthetic_{cond_val}")
        for j, point in enumerate(window):
            synth_data.append({
                "EID": j,
                "Xvalue": float(point[0]),
                "Yvalue": float(point[1]),
                "Zvalue": float(point[2]),
                "__user_id__": str(user_id),
                "__sensor_type__": "synthetic"
            })
    
    import pandas as pd
    synth_df = pd.DataFrame(synth_data)
    return synth_df

