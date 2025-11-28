import torch
from dataclasses import dataclass

def set_seed(seed: int = 42):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    csv_dir: str = "BB-MAS_Dataset/BB-MAS_Dataset/Keystroke_Features"
    rare_thresh: int = 10
    test_size: float = 0.15
    val_size: float = 0.15
    num_workers: int = 2
    cond_cols: tuple = ()
    
    latent_dim: int = 32
    hidden_dim: int = 256
    n_layers: int = 3
    cat_embed_dim: int = 16
    dropout: float = 0.0
    
    vae_epochs: int = 50
    vae_batch_size: int = 256
    vae_lr: float = 1e-3
    kl_max_beta: float = 1.0
    kl_warmup_epochs: int = 10
    vae_patience: int = 8
    
    T: int = 400
    beta_schedule: str = "cosine"
    ddpm_epochs: int = 100
    ddpm_batch_size: int = 512
    ddpm_lr: float = 2e-4
    ddpm_patience: int = 15
    sample_steps: int = 200
    num_cond_buckets: int = 128
    
    n_samples_demo: int = 2048
    out_csv: str = "synth_keystrokes.csv"
    
    seed: int = 42
    save_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

