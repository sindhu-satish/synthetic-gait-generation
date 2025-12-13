"""
Evaluation utilities for matched-per-user sampling and user-disjoint splits.
"""
import os
import json
import numpy as np
import torch
from typing import Tuple, Dict, List
from ..config import WINDOW_SIZE, SAMPLE_STEPS, DEVICE, set_seed
from ..diffusion import LatentDDPM
from ..vae import WindowVAE
from ..data import GaitDataModule


def generate_synthetic_windows_for_users(
    user_ids: np.ndarray,
    ddpm: LatentDDPM,
    vae: WindowVAE,
    dm: GaitDataModule,
    steps: int = None,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic windows conditioned on specific user IDs.
    
    Args:
        user_ids: Array of user IDs (raw user IDs from dataset, not cond indices)
        ddpm: Trained DDPM model
        vae: Trained VAE model
        dm: Data module with cond_map and rev_cond_map
        steps: Number of sampling steps (default: SAMPLE_STEPS)
        seed: Random seed for reproducibility
        
    Returns:
        synth_windows: (N, T, 3) array of synthetic windows
        synth_user_ids: (N,) array of user IDs matching the windows
    """
    if steps is None:
        steps = SAMPLE_STEPS
    
    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    
    cond_indices = []
    for uid in user_ids:
        if uid in dm.cond_map:
            cond_indices.append(dm.cond_map[uid])
        else:
            cond_indices.append(list(dm.cond_map.values())[0])
    
    cond_indices = np.array(cond_indices, dtype=np.int64)
    
    cond_tensor = torch.tensor(cond_indices, device=DEVICE, dtype=torch.long)
    Z = ddpm.sample(n=len(user_ids), cond_idx=cond_tensor, steps=steps).to(DEVICE)
    
    
    if hasattr(ddpm, "z_mean") and hasattr(ddpm, "z_std") and ddpm.z_mean is not None and ddpm.z_std is not None:
        Z = Z * ddpm.z_std + ddpm.z_mean
    
    vae.eval()
    with torch.no_grad():
        recon_flat = vae.decode(Z)
        recon_windows = recon_flat.view(len(user_ids), WINDOW_SIZE, 3)
        recon_windows = recon_windows + 0.05 * torch.randn_like(recon_windows)
        windows_np = recon_windows.detach().cpu().numpy()
    
    return windows_np, user_ids.copy()


@torch.no_grad()
def get_matched_real_and_synth_windows(
    sensor_type: str,
    split: str,
    per_user_k: int,
    ddpm: LatentDDPM,
    vae: WindowVAE,
    dm: GaitDataModule,
    eval_seed: int = 42,
    steps: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get matched real and synthetic windows using per-user sampling.
    
    For each user in the evaluation split, samples exactly K real windows
    and generates exactly K synthetic windows for that same user.
    
    Args:
        sensor_type: Sensor type (e.g., "Accelerometer")
        split: Data split to use ("test" or "val")
        per_user_k: Number of windows per user
        ddpm: Trained DDPM model
        vae: Trained VAE model
        dm: Data module
        eval_seed: Random seed for reproducibility
        steps: Number of sampling steps
        
    Returns:
        real_windows: (num_users * per_user_k, T, 3) array
        synth_windows: (num_users * per_user_k, T, 3) array
        real_user_ids: (num_users * per_user_k,) array
        synth_user_ids: (num_users * per_user_k,) array
    """
    if split == "test":
        eval_ds = dm.test_ds
    elif split == "val":
        eval_ds = dm.val_ds
    else:
        raise ValueError(f"split must be 'test' or 'val', got {split}")
    
    if eval_ds is None:
        raise ValueError(f"{split} dataset is None")
    
    real_windows_list = []
    real_user_ids_list = []
    
    for i in range(len(eval_ds)):
        item = eval_ds[i]
        window = item["window"].numpy()
        cond_idx = item["cond"]
        if isinstance(cond_idx, torch.Tensor):
            cond_idx = cond_idx.item()
        user_id = dm.rev_cond_map.get(cond_idx, cond_idx)
        real_windows_list.append(window)
        real_user_ids_list.append(user_id)
    
    real_windows_all = np.array(real_windows_list)
    real_user_ids_all = np.array(real_user_ids_list)
    
    unique_users = sorted(np.unique(real_user_ids_all))
    
    np.random.seed(eval_seed)
    
    real_windows_matched = []
    real_user_ids_matched = []
    synth_windows_matched = []
    synth_user_ids_matched = []
    
    user_real_counts = {}
    for uid in unique_users:
        user_mask = real_user_ids_all == uid
        user_windows = real_windows_all[user_mask]
        user_real_counts[uid] = len(user_windows)
        
        if len(user_windows) == 0:
            continue
        
        if len(user_windows) >= per_user_k:
            indices = np.random.choice(len(user_windows), size=per_user_k, replace=False)
        else:
            indices = np.random.choice(len(user_windows), size=per_user_k, replace=True)
            print(f"Warning: User {uid} has only {len(user_windows)} windows, sampling {per_user_k} with replacement")
        
        sampled_real = user_windows[indices]
        real_windows_matched.extend(sampled_real)
        real_user_ids_matched.extend([uid] * per_user_k)
    
    for uid in unique_users:
        if user_real_counts.get(uid, 0) == 0:
            continue
        
        
        user_ids_for_gen = np.array([uid] * per_user_k)
        synth_windows_user, _ = generate_synthetic_windows_for_users(
            user_ids_for_gen, ddpm, vae, dm, steps=steps, seed=eval_seed
        )
        synth_windows_matched.extend(synth_windows_user)
        synth_user_ids_matched.extend([uid] * per_user_k)
    
    real_windows = np.array(real_windows_matched)
    real_user_ids = np.array(real_user_ids_matched)
    synth_windows = np.array(synth_windows_matched)
    synth_user_ids = np.array(synth_user_ids_matched)
    
    unique_real_users = len(np.unique(real_user_ids))
    unique_synth_users = len(np.unique(synth_user_ids))
    expected_users = len([u for u in unique_users if user_real_counts.get(u, 0) > 0])
    
    assert unique_real_users == expected_users, f"unique_real_users ({unique_real_users}) != expected_users ({expected_users})"
    assert unique_synth_users == expected_users, f"unique_synth_users ({unique_synth_users}) != expected_users ({expected_users})"
    
    for uid in np.unique(real_user_ids):
        real_count = np.sum(real_user_ids == uid)
        synth_count = np.sum(synth_user_ids == uid)
        assert real_count == per_user_k, f"User {uid}: real_count ({real_count}) != per_user_k ({per_user_k})"
        assert synth_count == per_user_k, f"User {uid}: synth_count ({synth_count}) != per_user_k ({per_user_k})"
    
    print(f"\n[Matched Sampling] Summary:")
    print(f"  Users: {expected_users}")
    print(f"  Per-user K: {per_user_k}")
    print(f"  Total real windows: {len(real_windows)}")
    print(f"  Total synth windows: {len(synth_windows)}")
    print(f"  Unique real users: {unique_real_users}")
    print(f"  Unique synth users: {unique_synth_users}")
    
    return real_windows, real_user_ids, synth_windows, synth_user_ids


def save_matched_sampling_manifest(
    save_dir: str,
    sensor_type: str,
    users_count: int,
    per_user_k: int,
    eval_seed: int,
    real_user_ids: np.ndarray,
    synth_user_ids: np.ndarray,
    split: str = "test"
):
    """
    Save matched sampling manifest with metadata.
    
    Args:
        save_dir: Directory to save the manifest
        sensor_type: Sensor type
        users_count: Number of users
        per_user_k: Number of windows per user
        eval_seed: Evaluation seed used
        real_user_ids: Real user IDs array
        synth_user_ids: Synthetic user IDs array
        split: Data split used
    """
    os.makedirs(save_dir, exist_ok=True)
    
    real_counts_per_user = {}
    synth_counts_per_user = {}
    
    for uid in np.unique(real_user_ids):
        real_counts_per_user[int(uid)] = int(np.sum(real_user_ids == uid))
    
    for uid in np.unique(synth_user_ids):
        synth_counts_per_user[int(uid)] = int(np.sum(synth_user_ids == uid))
    
    real_counts = list(real_counts_per_user.values())
    synth_counts = list(synth_counts_per_user.values())
    
    manifest = {
        "sensor_type": sensor_type,
        "split": split,
        "users_count": users_count,
        "per_user_k": per_user_k,
        "eval_seed": eval_seed,
        "total_real": len(real_user_ids),
        "total_synth": len(synth_user_ids),
        "counts_per_user_real": {
            "min": int(np.min(real_counts)) if real_counts else 0,
            "median": float(np.median(real_counts)) if real_counts else 0.0,
            "max": int(np.max(real_counts)) if real_counts else 0
        },
        "counts_per_user_synth": {
            "min": int(np.min(synth_counts)) if synth_counts else 0,
            "median": float(np.median(synth_counts)) if synth_counts else 0.0,
            "max": int(np.max(synth_counts)) if synth_counts else 0
        }
    }
    
    manifest_path = os.path.join(save_dir, f"matched_sampling_manifest_{sensor_type.lower()}.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved matched sampling manifest to {manifest_path}")
    return manifest

