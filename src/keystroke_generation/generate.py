import numpy as np
import torch
from .diffusion import LatentDDPM
from .vae import VAE
from .data import KeystrokeDataModule
from .config import Config

@torch.no_grad()
def sample_synthetic(n: int, cfg: Config, ddpm: LatentDDPM, vae: VAE, dm: KeystrokeDataModule):
    cond_idx = None
    if dm.train_ds.cond_idx is not None:
        vals, counts = np.unique(dm.train_ds.cond_idx.numpy(), return_counts=True)
        probs = counts / counts.sum()
        cond_idx = np.random.choice(vals, size=n, p=probs)
    Z = ddpm.sample(n=n, cond_idx=cond_idx, steps=cfg.sample_steps)
    Z = Z.to(cfg.device)
    num_out, cat_logits = vae.decode(Z)
    num_np = num_out.detach().cpu().numpy() if num_out is not None else np.zeros((n, 0))
    cat_np_list = [cl.detach().cpu().numpy() for cl in cat_logits]
    synth_df = dm.pre.inverse_transform(num_np, cat_np_list)
    return synth_df

