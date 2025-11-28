import os
import sys
import torch

from .config import Config, set_seed
from .data import load_csv_folder, KeystrokeDataModule
from .vae import VAE
from .diffusion import UNetMLP
from .train import train_vae, train_ddpm, encode_dataset_to_latents
from .generate import sample_synthetic
from .evaluate import evaluate_utility, evaluate_privacy

def main():
    cfg = Config()
    set_seed(cfg.seed)
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("Loading data...")
    raw_df, csv_paths = load_csv_folder(cfg.csv_dir)
    print(f"Loaded {len(csv_paths)} CSVs. Combined shape: {raw_df.shape}")
    
    print("Setting up data module...")
    dm = KeystrokeDataModule(raw_df, cfg)
    dm.setup()
    print(f"Train/Val/Test sizes: {len(dm.train_ds)}, {len(dm.val_ds)}, {len(dm.test_ds)}")
    print(f"Detected conditioning columns: {dm.cond_cols}")
    print(f"Numeric/Categorical: {len(dm.pre.numeric_cols)}, {len(dm.pre.categorical_cols)}")
    
    print("Initializing VAE...")
    cat_vocab_sizes = [len(dm.pre.vocabs[c]) for c in dm.pre.categorical_cols]
    vae = VAE(cfg, num_features=len(dm.pre.numeric_cols), cat_vocab_sizes=cat_vocab_sizes).to(cfg.device)
    
    print("Training VAE...")
    vae_hist = train_vae(vae, dm, cfg)
    
    print("Loading best VAE weights...")
    vae.load_state_dict(torch.load(os.path.join(cfg.save_dir, "vae_best.pt"), map_location=cfg.device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    print("Encoding datasets to latents...")
    Z_train, C_train = encode_dataset_to_latents(dm.train_ds, vae, cfg)
    Z_val, C_val = encode_dataset_to_latents(dm.val_ds, vae, cfg)
    Z_test, C_test = encode_dataset_to_latents(dm.test_ds, vae, cfg)
    print(f"Latent shapes: {Z_train.shape}, {Z_val.shape}, {Z_test.shape}")
    
    print("Initializing UNet...")
    unet = UNetMLP(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        n_blocks=cfg.n_layers+1,
        cond_dim=(1 if dm.cond_cols else 0)
    ).to(cfg.device)
    
    print("Training DDPM...")
    ddpm = train_ddpm(unet, Z_train, C_train, Z_val, C_val, cfg)
    
    print("Loading best DDPM weights...")
    unet.load_state_dict(torch.load(os.path.join(cfg.save_dir, "ddpm_best.pt"), map_location=cfg.device))
    unet.eval()
    
    print("Generating synthetic samples...")
    synth = sample_synthetic(cfg.n_samples_demo, cfg, ddpm, vae, dm)
    print(f"Synthetic frame: {synth.shape}")
    print(synth.head(20))
    
    out_path = cfg.out_csv
    synth.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
    
    print("Evaluating utility and privacy...")
    real_test_df = dm.df.copy()
    evaluate_utility(real_test_df, synth, dm.pre)
    evaluate_privacy(dm.df.iloc[:len(dm.train_ds)], synth, dm.pre)
    
    pre_path = os.path.join(cfg.save_dir, "preprocessor.pkl")
    dm.pre.save(pre_path)
    print(f"Saved preprocessor to: {pre_path}")
    print(f"Best VAE: {os.path.join(cfg.save_dir, 'vae_best.pt')}")
    print(f"Best DDPM: {os.path.join(cfg.save_dir, 'ddpm_best.pt')}")
    print(f"Synthetic CSV: {cfg.out_csv}")

if __name__ == "__main__":
    main()

