import os
from .config import set_seed, SEED, SAVE_DIR
from .evaluation.ablation_vae import run_vae_ablation
from .evaluation.ablation_ddpm import run_ddpm_ablation

def main():
    set_seed(SEED)
    
    ablation_dir = os.path.join(SAVE_DIR, "ablations")
    os.makedirs(ablation_dir, exist_ok=True)
    
    sensor_type = "Accelerometer"
    
    print(f"\n{'='*60}")
    print(f"Running VAE Ablations for {sensor_type}")
    print(f"{'='*60}\n")
    
    vae_results = run_vae_ablation(sensor_type, os.path.join(ablation_dir, "vae"))
    
    print(f"\n{'='*60}")
    print(f"Running DDPM Ablations for {sensor_type}")
    print(f"{'='*60}\n")
    
    ddpm_results = run_ddpm_ablation(sensor_type, os.path.join(ablation_dir, "ddpm"))
    
    print(f"\n{'='*60}")
    print("All ablations complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

