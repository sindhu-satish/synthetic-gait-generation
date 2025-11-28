"""
Inference script for Gait Generation V2.

This script:
1. Loads trained VAE + DDPM models for Accelerometer
2. Generates synthetic data for 10 users (similar amount to training)
3. Loads real data for 10 users
4. Trains a neural network classifier on 10 real + 10 synthetic users
5. Trains a neural network classifier on only 10 real users
6. Compares performance of both models
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import ConfigV2, set_seed
from .preprocessing import GaitPreprocessor
from .data import GaitWindowDataModule
from .conv_vae import ConvVAE
from .diffusion import LatentUNet, LatentDDPM, EMA
from .generate import generate_synthetic_windows


class UserClassifier(nn.Module):
    """
    Neural network classifier for user identification from gait windows.
    
    Architecture:
    - Input: (window_size, 4) window
    - 1D Convolutional layers
    - Global pooling
    - Fully connected layers
    - Output: num_users logits
    """
    
    def __init__(self, window_size: int = 256, in_channels: int = 4, num_users: int = 10, hidden_dim: int = 128):
        super().__init__()
        
        # 1D Convolutional feature extractor
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_users)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: (B, window_size, 4) -> (B, 4, window_size)
        if x.shape[-1] == 4:
            x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_models(sensor_type: str, cfg: ConfigV2):
    """Load trained VAE, DDPM, EMA, and preprocessor."""
    save_dir = os.path.join(cfg.save_dir, sensor_type.lower())
    
    pre_path = os.path.join(save_dir, "preprocessor.pkl")
    vae_path = os.path.join(save_dir, "vae_best.pt")
    ddpm_path = os.path.join(save_dir, "ddpm_best.pt")
    ema_path = os.path.join(save_dir, "ddpm_ema.pt")
    
    if not all(os.path.exists(p) for p in [pre_path, vae_path, ddpm_path, ema_path]):
        raise FileNotFoundError(f"Missing checkpoint files in {save_dir}")
    
    print(f"Loading models from {save_dir}...")
    
    # Load preprocessor
    preprocessor = GaitPreprocessor.load(pre_path)
    print(f"  Loaded preprocessor with {len(preprocessor.user_to_idx)} users")
    
    # Load VAE
    latent_dim = cfg.get_latent_dim(sensor_type)
    n_users = len(preprocessor.user_to_idx)
    vae = ConvVAE(cfg=cfg, latent_dim=latent_dim, n_users=n_users)
    vae.load_state_dict(torch.load(vae_path, map_location=cfg.device))
    vae = vae.to(cfg.device)
    vae.eval()
    print(f"  Loaded VAE (latent_dim={latent_dim}, n_users={n_users})")
    
    # Load DDPM
    unet = LatentUNet(
        latent_dim=latent_dim,
        hidden_dim=cfg.ddpm_hidden_dim,
        n_blocks=cfg.ddpm_n_blocks,
        n_users=n_users,
        user_embed_dim=cfg.user_embed_dim
    )
    unet.load_state_dict(torch.load(ddpm_path, map_location=cfg.device))
    unet = unet.to(cfg.device)
    unet.eval()
    
    ddpm = LatentDDPM(unet, T=cfg.T, beta_schedule=cfg.beta_schedule, device=cfg.device)
    
    # Load EMA
    ema = EMA(unet, decay=cfg.ema_decay)
    ema.load_state_dict(torch.load(ema_path, map_location=cfg.device))
    print(f"  Loaded DDPM and EMA")
    
    return vae, ddpm, ema, preprocessor


def get_real_windows_for_users(preprocessor: GaitPreprocessor, user_ids: list, cfg: ConfigV2):
    """
    Get real windows for specified users from the preprocessor.
    
    Returns:
        windows: (N, window_size, 4) array
        user_labels: (N,) array of user indices
    """
    all_windows = []
    all_labels = []
    users_found = []
    users_missing = []
    
    for user_id in user_ids:
        if user_id not in preprocessor.user_to_idx:
            users_missing.append(user_id)
            print(f"  Warning: User {user_id} not found in preprocessor.user_to_idx, skipping")
            continue
        
        user_idx = preprocessor.user_to_idx[user_id]
        
        # Get windows from train/val/test splits
        windows_list = []
        if preprocessor.train_windows is not None and user_id in preprocessor.train_user_ids:
            indices = [i for i, uid in enumerate(preprocessor.train_user_ids) if uid == user_id]
            windows_list.append(preprocessor.train_windows[indices])
        
        if preprocessor.val_windows is not None and user_id in preprocessor.val_user_ids:
            indices = [i for i, uid in enumerate(preprocessor.val_user_ids) if uid == user_id]
            windows_list.append(preprocessor.val_windows[indices])
        
        if preprocessor.test_windows is not None and user_id in preprocessor.test_user_ids:
            indices = [i for i, uid in enumerate(preprocessor.test_user_ids) if uid == user_id]
            windows_list.append(preprocessor.test_windows[indices])
        
        if windows_list:
            user_windows = np.concatenate(windows_list, axis=0)
            all_windows.append(user_windows)
            all_labels.extend([user_idx] * len(user_windows))
            users_found.append(user_id)
            print(f"  User {user_id} (idx {user_idx}): {len(user_windows)} windows")
        else:
            users_missing.append(user_id)
            print(f"  Warning: User {user_id} has no windows in train/val/test splits, skipping")
    
    if not all_windows:
        raise ValueError(
            f"No real windows found for specified users!\n"
            f"  Requested users: {user_ids}\n"
            f"  Users found: {users_found}\n"
            f"  Users missing: {users_missing}"
        )
    
    if users_missing:
        print(f"  Note: {len(users_missing)} users had no data: {users_missing}")
    
    windows = np.concatenate(all_windows, axis=0)
    labels = np.array(all_labels)
    
    return windows, labels


def extract_user_embeddings(vae: ConvVAE, ddpm: LatentDDPM, user_indices: list, device: str):
    """Extract user embeddings from VAE and DDPM models."""
    vae_embeddings = {}
    ddpm_embeddings = {}
    
    with torch.no_grad():
        for user_idx in user_indices:
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
            # Extract from VAE
            vae_emb = vae.decoder.user_embed(user_idx_tensor).squeeze(0)  # (user_embed_dim,)
            vae_embeddings[user_idx] = vae_emb.cpu()
            # Extract from DDPM
            ddpm_emb = ddpm.model.user_emb(user_idx_tensor).squeeze(0)  # (user_embed_dim,)
            ddpm_embeddings[user_idx] = ddpm_emb.cpu()
    
    return vae_embeddings, ddpm_embeddings


def interpolate_embeddings(emb1: torch.Tensor, emb2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate between two embeddings: alpha * emb1 + (1-alpha) * emb2."""
    return alpha * emb1 + (1 - alpha) * emb2


def create_fictional_user_embeddings(
    real_user_indices: list,
    n_fictional: int,
    interpolation_method: str = "varied",
    vae: ConvVAE = None,
    ddpm: LatentDDPM = None,
    device: str = "cpu"
):
    """Create fictional user embeddings by interpolating between real users.
    
    Args:
        real_user_indices: List of real user indices to interpolate between
        n_fictional: Number of fictional users to create
        interpolation_method: One of "varied", "extreme", "multi", "noisy", "mixed"
        vae: VAE model to extract embeddings from
        ddpm: DDPM model to extract embeddings from
        device: Device to use
        
    Returns:
        List of dicts with 'vae_embed', 'ddpm_embed', 'description'
    """
    if vae is None or ddpm is None:
        raise ValueError("VAE and DDPM models required to extract embeddings")
    
    # Extract embeddings from real users
    vae_embeddings, ddpm_embeddings = extract_user_embeddings(vae, ddpm, real_user_indices, device)
    
    fictional_users = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_fictional):
        if interpolation_method == "varied":
            # Varied alpha values (0.3-0.7)
            idx1, idx2 = np.random.choice(len(real_user_indices), size=2, replace=False)
            user1_idx = real_user_indices[idx1]
            user2_idx = real_user_indices[idx2]
            alpha = 0.3 + 0.4 * (i / n_fictional)
            vae_emb = interpolate_embeddings(vae_embeddings[user1_idx], vae_embeddings[user2_idx], alpha)
            ddpm_emb = interpolate_embeddings(ddpm_embeddings[user1_idx], ddpm_embeddings[user2_idx], alpha)
            desc = f"varied interpolation (alpha={alpha:.2f}) between users {user1_idx} and {user2_idx}"
            
        elif interpolation_method == "extreme":
            # Extreme alpha values (0.1, 0.9)
            idx1, idx2 = np.random.choice(len(real_user_indices), size=2, replace=False)
            user1_idx = real_user_indices[idx1]
            user2_idx = real_user_indices[idx2]
            alpha = 0.1 if i % 2 == 0 else 0.9
            vae_emb = interpolate_embeddings(vae_embeddings[user1_idx], vae_embeddings[user2_idx], alpha)
            ddpm_emb = interpolate_embeddings(ddpm_embeddings[user1_idx], ddpm_embeddings[user2_idx], alpha)
            desc = f"extreme interpolation (alpha={alpha:.2f}) between users {user1_idx} and {user2_idx}"
            
        elif interpolation_method == "multi":
            # Multi-user interpolation (3+ users)
            n_users = 3 + (i % 3)  # 3, 4, or 5 users
            selected_indices = np.random.choice(len(real_user_indices), size=n_users, replace=False)
            selected_user_indices = [real_user_indices[idx] for idx in selected_indices]
            weights = np.random.dirichlet(np.ones(n_users))
            
            vae_emb = sum(w * vae_embeddings[uidx] for w, uidx in zip(weights, selected_user_indices))
            ddpm_emb = sum(w * ddpm_embeddings[uidx] for w, uidx in zip(weights, selected_user_indices))
            desc = f"multi-user interpolation ({n_users} users, weights={weights.round(2).tolist()})"
            
        elif interpolation_method == "noisy":
            # Add noise to interpolated embedding
            idx1, idx2 = np.random.choice(len(real_user_indices), size=2, replace=False)
            user1_idx = real_user_indices[idx1]
            user2_idx = real_user_indices[idx2]
            alpha = 0.5
            base_vae_emb = interpolate_embeddings(vae_embeddings[user1_idx], vae_embeddings[user2_idx], alpha)
            base_ddpm_emb = interpolate_embeddings(ddpm_embeddings[user1_idx], ddpm_embeddings[user2_idx], alpha)
            
            # Add Gaussian noise (10% of embedding std)
            noise_scale = 0.1
            vae_emb = base_vae_emb + torch.randn_like(base_vae_emb) * noise_scale * base_vae_emb.std()
            ddpm_emb = base_ddpm_emb + torch.randn_like(base_ddpm_emb) * noise_scale * base_ddpm_emb.std()
            desc = f"noisy interpolation (alpha=0.5, noise={noise_scale}) between users {user1_idx} and {user2_idx}"
            
        elif interpolation_method == "mixed":
            # Mix of all methods
            method_idx = i % 4
            if method_idx == 0:  # Varied
                idx1, idx2 = np.random.choice(len(real_user_indices), size=2, replace=False)
                user1_idx = real_user_indices[idx1]
                user2_idx = real_user_indices[idx2]
                alpha = 0.3 + 0.4 * np.random.random()
                vae_emb = interpolate_embeddings(vae_embeddings[user1_idx], vae_embeddings[user2_idx], alpha)
                ddpm_emb = interpolate_embeddings(ddpm_embeddings[user1_idx], ddpm_embeddings[user2_idx], alpha)
                desc = f"varied: alpha={alpha:.2f} between users {user1_idx} and {user2_idx}"
            elif method_idx == 1:  # Extreme
                idx1, idx2 = np.random.choice(len(real_user_indices), size=2, replace=False)
                user1_idx = real_user_indices[idx1]
                user2_idx = real_user_indices[idx2]
                alpha = np.random.choice([0.1, 0.9])
                vae_emb = interpolate_embeddings(vae_embeddings[user1_idx], vae_embeddings[user2_idx], alpha)
                ddpm_emb = interpolate_embeddings(ddpm_embeddings[user1_idx], ddpm_embeddings[user2_idx], alpha)
                desc = f"extreme: alpha={alpha:.2f} between users {user1_idx} and {user2_idx}"
            elif method_idx == 2:  # Multi
                n_users = np.random.randint(3, 6)
                selected_indices = np.random.choice(len(real_user_indices), size=n_users, replace=False)
                selected_user_indices = [real_user_indices[idx] for idx in selected_indices]
                weights = np.random.dirichlet(np.ones(n_users))
                vae_emb = sum(w * vae_embeddings[uidx] for w, uidx in zip(weights, selected_user_indices))
                ddpm_emb = sum(w * ddpm_embeddings[uidx] for w, uidx in zip(weights, selected_user_indices))
                desc = f"multi: {n_users} users, weights={weights.round(2).tolist()}"
            else:  # Noisy
                idx1, idx2 = np.random.choice(len(real_user_indices), size=2, replace=False)
                user1_idx = real_user_indices[idx1]
                user2_idx = real_user_indices[idx2]
                alpha = 0.5
                base_vae_emb = interpolate_embeddings(vae_embeddings[user1_idx], vae_embeddings[user2_idx], alpha)
                base_ddpm_emb = interpolate_embeddings(ddpm_embeddings[user1_idx], ddpm_embeddings[user2_idx], alpha)
                noise_scale = 0.1
                vae_emb = base_vae_emb + torch.randn_like(base_vae_emb) * noise_scale * base_vae_emb.std()
                ddpm_emb = base_ddpm_emb + torch.randn_like(base_ddpm_emb) * noise_scale * base_ddpm_emb.std()
                desc = f"noisy: alpha=0.5, noise={noise_scale} between users {user1_idx} and {user2_idx}"
        else:
            raise ValueError(f"Unknown interpolation_method: {interpolation_method}")
        
        fictional_users.append({
            "fictional_id": f"fictional_{i+1}",
            "vae_embed": vae_emb.to(device),
            "ddpm_embed": ddpm_emb.to(device),
            "description": desc
        })
        print(f"  fictional_{i+1}: {desc}")
    
    return fictional_users


def generate_with_custom_embeddings(
    n_windows: int,
    vae_embed: torch.Tensor,
    ddpm_embed: torch.Tensor,
    vae: ConvVAE,
    ddpm: LatentDDPM,
    ema: EMA,
    preprocessor: GaitPreprocessor,
    cfg: ConfigV2,
    use_ema: bool = True,
    return_normalized: bool = False
) -> np.ndarray:
    """Generate synthetic windows using custom user embeddings (for fictional users).
    
    Args:
        n_windows: Number of windows to generate
        vae_embed: Custom VAE user embedding (user_embed_dim,)
        ddpm_embed: Custom DDPM user embedding (user_embed_dim,)
        vae: Trained VAE model
        ddpm: Trained DDPM model
        ema: EMA object
        preprocessor: Fitted preprocessor
        cfg: Configuration
        use_ema: Whether to use EMA weights
        return_normalized: If True, return normalized values
        
    Returns:
        windows: (n_windows, window_size, 4) synthetic windows
    """
    # Expand embeddings to batch size
    vae_embed_batch = vae_embed.unsqueeze(0).repeat(n_windows, 1)  # (n_windows, user_embed_dim)
    ddpm_embed_batch = ddpm_embed.unsqueeze(0).repeat(n_windows, 1)  # (n_windows, user_embed_dim)
    
    # Apply EMA if requested
    original_state = None
    if use_ema and ema is not None:
        original_state = {k: v.clone() for k, v in ddpm.model.state_dict().items()}
        ema.apply_shadow()
    
    try:
        # Sample latents from DDPM with custom embedding
        # We need to modify the sampling to use custom embeddings
        # For now, we'll use a dummy user_idx and replace the embedding in the forward pass
        # This requires modifying the model temporarily or creating a wrapper
        
        # Create a temporary wrapper that uses custom embeddings
        # We'll sample latents by temporarily modifying the model
        with torch.no_grad():
            # Sample noise
            z = torch.randn(n_windows, ddpm.model.latent_dim, device=cfg.device)
            
            # Reverse diffusion with custom embedding
            for i in reversed(range(cfg.sample_steps)):
                t = torch.full((n_windows,), i, device=cfg.device, dtype=torch.long)
                
                # Manually compute p_mean_variance with custom embedding
                # Predict noise using custom embedding
                t_emb = ddpm.model.time_emb(t)  # (n_windows, hidden_dim)
                
                # Use custom DDPM embedding instead of user_emb(user_idx)
                # Project: latent + custom_embed -> hidden
                h = torch.cat([z, ddpm_embed_batch], dim=-1)  # (n_windows, latent_dim + user_embed_dim)
                h = ddpm.model.inp(h)  # (n_windows, hidden_dim)
                
                # Create conditioning embedding (same as in LatentUNet.forward)
                cond_emb = ddpm.model.cond_proj(t_emb * ddpm.model.cond_weight)  # (n_windows, hidden_dim * 2)
                
                # Pass through blocks with custom conditioning
                for block in ddpm.model.blocks:
                    h = block(h, cond_emb)  # cond_emb is (hidden_dim * 2) = 512, matches fc1 output
                
                # Output projection
                eps_pred = ddpm.model.out(h)  # (n_windows, latent_dim)
                
                # Compute mean and variance
                sqrt_ac = ddpm.sqrt_alphas_cumprod[t][:, None]
                sqrt_om = ddpm.sqrt_one_minus_alphas_cumprod[t][:, None]
                x0_pred = (z - sqrt_om * eps_pred) / (sqrt_ac + 1e-8)
                
                beta_t = ddpm.betas[t][:, None]
                sqrt_recip_alpha_t = ddpm.sqrt_recip_alphas[t][:, None]
                ac = ddpm.alphas_cumprod[t][:, None]
                
                posterior_mean = sqrt_recip_alpha_t * (
                    z - beta_t * eps_pred / (torch.sqrt(1 - ac) + 1e-8)
                )
                posterior_variance = ddpm.posterior_variance[t][:, None]
                
                if i > 0:
                    noise = torch.randn_like(z)
                    z = posterior_mean + torch.sqrt(posterior_variance) * noise
                else:
                    z = posterior_mean
        
        # Decode with VAE using custom embedding
        vae.eval()
        with torch.no_grad():
            # Manually decode with custom VAE embedding
            # z: (n_windows, latent_dim)
            # Use custom VAE embedding instead of user_embed(user_idx)
            h = torch.cat([z, vae_embed_batch], dim=-1)  # (n_windows, latent_dim + user_embed_dim)
            h = vae.decoder.fc(h)  # (n_windows, hidden_channels[0] * init_t)
            h = h.view(-1, vae.decoder.hidden_channels[0], vae.decoder.init_t)  # (n_windows, 256, 16)
            windows_normalized = vae.decoder.deconvs(h)  # (n_windows, 4, 256)
            windows_normalized = windows_normalized.transpose(1, 2)  # (n_windows, 256, 4)
            windows_normalized = windows_normalized.cpu().numpy()
            
            # Clean NaN/Inf
            windows_normalized = np.nan_to_num(windows_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
            # Clamp to [-1, 1] range (normalized space)
            windows_normalized = np.clip(windows_normalized, -1.0, 1.0)
    
    finally:
        # Restore original weights
        if original_state is not None:
            ddpm.model.load_state_dict(original_state)
    
    if return_normalized:
        return windows_normalized
    
    # Inverse standardization to physical units
    windows_physical = preprocessor.inverse_standardize(windows_normalized)
    return windows_physical


def generate_synthetic_data_for_users(
    user_ids: list,
    n_windows_per_user: int,
    vae: ConvVAE,
    ddpm: LatentDDPM,
    ema: EMA,
    preprocessor: GaitPreprocessor,
    cfg: ConfigV2
):
    """
    Generate synthetic windows for specified users.
    
    Returns:
        windows: (N, window_size, 4) array
        user_labels: (N,) array of user indices
    """
    all_windows = []
    all_labels = []
    
    print(f"\nGenerating {n_windows_per_user} synthetic windows per user...")
    
    for user_id in user_ids:
        if user_id not in preprocessor.user_to_idx:
            print(f"  Warning: User {user_id} not found in preprocessor, skipping")
            continue
        
        user_idx = preprocessor.user_to_idx[user_id]
        
        try:
            windows = generate_synthetic_windows(
                n_windows=n_windows_per_user,
                user_id=user_id,
                vae=vae,
                ddpm=ddpm,
                preprocessor=preprocessor,
                cfg=cfg,
                use_ema=True,
                ema=ema,
                return_normalized=True  # Keep normalized for classifier
            )
            
            all_windows.append(windows)
            all_labels.extend([user_idx] * len(windows))
            print(f"  User {user_id}: {len(windows)} windows generated")
        except Exception as e:
            print(f"  Error generating for user {user_id}: {e}")
    
    if not all_windows:
        raise ValueError("No synthetic windows generated!")
    
    windows = np.concatenate(all_windows, axis=0)
    labels = np.array(all_labels)
    
    return windows, labels


def generate_fictional_users_data(
    fictional_users: list,
    n_windows_per_user: int,
    vae: ConvVAE,
    ddpm: LatentDDPM,
    ema: EMA,
    preprocessor: GaitPreprocessor,
    cfg: ConfigV2
):
    """
    Generate synthetic windows for fictional users (using interpolated embeddings).
    
    Returns:
        windows: (N, window_size, 4) array
        user_labels: (N,) array of fictional user IDs (as strings)
    """
    all_windows = []
    all_labels = []
    
    print(f"\nGenerating {n_windows_per_user} synthetic windows per fictional user...")
    
    for fictional_user in fictional_users:
        fictional_id = fictional_user["fictional_id"]
        vae_embed = fictional_user["vae_embed"]
        ddpm_embed = fictional_user["ddpm_embed"]
        
        try:
            windows = generate_with_custom_embeddings(
                n_windows=n_windows_per_user,
                vae_embed=vae_embed,
                ddpm_embed=ddpm_embed,
                vae=vae,
                ddpm=ddpm,
                ema=ema,
                preprocessor=preprocessor,
                cfg=cfg,
                use_ema=True,
                return_normalized=True
            )
            
            all_windows.append(windows)
            # Use fictional_id as label (will be remapped later)
            all_labels.extend([fictional_id] * len(windows))
            print(f"  {fictional_id}: {len(windows)} windows generated ({fictional_user['description']})")
        except Exception as e:
            print(f"  Error generating for {fictional_id}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_windows:
        raise ValueError("No fictional windows generated!")
    
    windows = np.concatenate(all_windows, axis=0)
    labels = np.array(all_labels)
    
    return windows, labels


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_users: int,
    window_size: int = 256,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu"
):
    """
    Train a neural network classifier.
    
    Returns:
        model: Trained classifier
        metrics: Dict with accuracy, classification report, etc.
    """
    # Check data statistics
    print(f"\nData statistics:")
    print(f"  Train: {len(X_train)} samples, labels: {np.unique(y_train, return_counts=True)}")
    print(f"  Val: {len(X_val)} samples, labels: {np.unique(y_val, return_counts=True)}")
    
    # Clean NaN/Inf values
    print(f"\nCleaning NaN/Inf values...")
    train_nan_count = np.isnan(X_train).sum() + np.isinf(X_train).sum()
    val_nan_count = np.isnan(X_val).sum() + np.isinf(X_val).sum()
    print(f"  Train: {train_nan_count} NaN/Inf values found")
    print(f"  Val: {val_nan_count} NaN/Inf values found")
    
    # Replace NaN/Inf with 0 (or could use mean, but 0 is safer for normalized data)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  X_train shape: {X_train.shape}, range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"  X_val shape: {X_val.shape}, range: [{X_val.min():.4f}, {X_val.max():.4f}]")
    print(f"  X_val mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = UserClassifier(window_size=window_size, num_users=num_users).to(device)
    
    # Compute class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Higher initial LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
    
    # Training loop
    best_val_acc = 0.0
    patience = 20  # More patience
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss at epoch {epoch+1}, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Print every epoch for first 20, then every 10
        if (epoch + 1) <= 20 or (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:03d}/{epochs} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | lr: {current_lr:.6f}")
        
        # Warn if validation accuracy is suspiciously high in early epochs
        if epoch == 0 and val_acc > 0.95:
            print(f"  ⚠️  WARNING: Very high validation accuracy ({val_acc:.4f}) in first epoch!")
            print(f"     This could indicate: easy task, data leakage, or overfitting.")
            print(f"     Monitor if validation accuracy continues to improve or plateaus.")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    report_str = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "classification_report_str": report_str,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "model": model
    }
    
    return model, metrics


def plot_confusion_matrix(cm, user_ids, title, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=user_ids, yticklabels=user_ids)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix to {save_path}")
    plt.close()


def main():
    """Main inference and classification script."""
    print("="*70)
    print("Gait Generation V2: Inference and Classification Comparison")
    print("="*70)
    
    # Configuration
    cfg = ConfigV2()
    set_seed(cfg.seed)
    sensor_type = "Accelerometer"
    
    # Number of users to use for classification
    n_real_users = 20  # Real users from dataset
    n_fictional_users = 20  # Fictional users to generate
    
    # Load models
    vae, ddpm, ema, preprocessor = load_models(sensor_type, cfg)
    
    # Find users that actually have data (windows) in train/val/test splits
    users_with_data = set()
    if preprocessor.train_user_ids is not None:
        users_with_data.update(preprocessor.train_user_ids)
    if preprocessor.val_user_ids is not None:
        users_with_data.update(preprocessor.val_user_ids)
    if preprocessor.test_user_ids is not None:
        users_with_data.update(preprocessor.test_user_ids)
    
    # Select real users (for real data)
    all_available_users = sorted(users_with_data, key=lambda x: int(x))
    real_users = all_available_users[:n_real_users]
    print(f"\nSelected {len(real_users)} real users: {real_users}")
    
    # Get real user indices for embedding extraction
    real_user_indices = [preprocessor.user_to_idx[uid] for uid in real_users]
    
    # Create fictional users by interpolating between real user embeddings
    print(f"\nCreating {n_fictional_users} fictional users by interpolating embeddings...")
    interpolation_method = "mixed"  # Options: "varied", "extreme", "multi", "noisy", "mixed"
    print(f"  Using interpolation method: {interpolation_method}")
    
    fictional_user_embeddings = create_fictional_user_embeddings(
        real_user_indices=real_user_indices,
        n_fictional=n_fictional_users,
        interpolation_method=interpolation_method,
        vae=vae,
        ddpm=ddpm,
        device=cfg.device
    )
    
    fictional_user_ids = [fu["fictional_id"] for fu in fictional_user_embeddings]
    print(f"\nCreated {len(fictional_user_ids)} fictional users: {fictional_user_ids}")
    
    # Estimate windows per user from training data
    # Average windows per user in training set
    if preprocessor.train_windows is not None:
        avg_windows_per_user = len(preprocessor.train_windows) // len(preprocessor.train_users)
        n_windows_per_user = max(100, avg_windows_per_user // 2)  # Use half of average, min 100
    else:
        n_windows_per_user = 200  # Default
    
    print(f"\nUsing {n_windows_per_user} windows per user (similar to training)")
    
    # ============================================================
    # Step 1: Generate synthetic data for FICTIONAL users (using interpolated embeddings)
    # ============================================================
    print("\n" + "="*70)
    print("STEP 1: Generating Synthetic Data for Fictional Users")
    print("="*70)
    
    fictional_windows, fictional_labels = generate_fictional_users_data(
        fictional_users=fictional_user_embeddings,
        n_windows_per_user=n_windows_per_user,
        vae=vae,
        ddpm=ddpm,
        ema=ema,
        preprocessor=preprocessor,
        cfg=cfg
    )
    
    print(f"\nTotal fictional windows: {len(fictional_windows)}")
    
    # ============================================================
    # Step 2: Load real data for REAL users
    # ============================================================
    print("\n" + "="*70)
    print("STEP 2: Loading Real Data for Real Users")
    print("="*70)
    
    real_windows, real_labels = get_real_windows_for_users(
        preprocessor=preprocessor,
        user_ids=real_users,
        cfg=cfg
    )
    
    print(f"\nTotal real windows: {len(real_windows)}")
    
    # Verify which users we actually got data for
    unique_real_labels = np.unique(real_labels)
    users_with_real_data = [preprocessor.idx_to_user[idx] for idx in unique_real_labels if idx in preprocessor.idx_to_user]
    print(f"Users with real data: {sorted(users_with_real_data, key=lambda x: int(x))}")
    
    # Verify fictional users (these are strings like "fictional_1", "fictional_2", etc.)
    unique_fictional_labels = np.unique(fictional_labels)
    print(f"Fictional users with data: {sorted(unique_fictional_labels)}")
    
    # Ensure no overlap (fictional labels are strings, real are user IDs, so no overlap possible)
    print(f"\n✓ Real and fictional users are disjoint (fictional users are interpolated, not from dataset)")
    
    # Clean NaN/Inf values from windows BEFORE label remapping
    print(f"\nCleaning NaN/Inf values from windows...")
    real_nan_count = np.isnan(real_windows).sum() + np.isinf(real_windows).sum()
    fictional_nan_count = np.isnan(fictional_windows).sum() + np.isinf(fictional_windows).sum()
    print(f"  Real windows: {real_nan_count} NaN/Inf values")
    print(f"  Fictional windows: {fictional_nan_count} NaN/Inf values")
    
    if real_nan_count > 0:
        real_windows = np.nan_to_num(real_windows, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  Cleaned real windows")
    if fictional_nan_count > 0:
        fictional_windows = np.nan_to_num(fictional_windows, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  Cleaned fictional windows")
    
    # Remap labels: Real users get classes 0 to (n_real-1), Fictional users get n_real to (n_real+n_fictional-1)
    unique_real_labels = sorted(np.unique(real_labels))
    unique_fictional_labels = sorted(np.unique(fictional_labels))
    
    # Create label mapping: real users first, then fictional users
    label_mapping = {}
    reverse_label_mapping = {}
    
    # Real users: 0 to n_real-1
    for new_idx, orig_idx in enumerate(unique_real_labels):
        label_mapping[orig_idx] = new_idx
        reverse_label_mapping[new_idx] = orig_idx
    
    # Fictional users: n_real to n_real+n_fictional-1
    n_real = len(unique_real_labels)
    for new_idx, orig_idx in enumerate(unique_fictional_labels):
        label_mapping[orig_idx] = n_real + new_idx
        reverse_label_mapping[n_real + new_idx] = orig_idx
    
    real_labels_remapped = np.array([label_mapping[l] for l in real_labels])
    fictional_labels_remapped = np.array([label_mapping[l] for l in fictional_labels])
    
    # Create user names for reporting
    num_users = len(unique_real_labels) + len(unique_fictional_labels)
    real_user_names = [f"Real_{preprocessor.idx_to_user[idx]}" for idx in unique_real_labels]
    fictional_user_names = [f"Fictional_{fid}" for fid in unique_fictional_labels]  # fictional_labels are strings
    user_names = real_user_names + fictional_user_names
    
    print(f"\nLabel mapping:")
    print(f"  REAL USERS (classes 0-{n_real-1}):")
    for orig_idx in unique_real_labels:
        user_id = preprocessor.idx_to_user.get(orig_idx, "?")
        print(f"    User {user_id} (idx {orig_idx}) -> class {label_mapping[orig_idx]}")
    print(f"  FICTIONAL USERS (classes {n_real}-{num_users-1}):")
    for fictional_id in unique_fictional_labels:  # These are strings like "fictional_1"
        print(f"    {fictional_id} -> class {label_mapping[fictional_id]}")
    print(f"\nUser names: {user_names[:5]}... (showing first 5)")
    print(f"Real labels range: {real_labels_remapped.min()} to {real_labels_remapped.max()}")
    print(f"Fictional labels range: {fictional_labels_remapped.min()} to {fictional_labels_remapped.max()}")
    print(f"Total number of classes: {num_users} ({n_real} real + {len(unique_fictional_labels)} fictional)")
    
    # ============================================================
    # Step 3: Train classifier on REAL users only
    # ============================================================
    print("\n" + "="*70)
    print("STEP 3: Training Classifier on Real Users Only")
    print("="*70)
    
    # Split real data
    X_real_train, X_real_val, y_real_train, y_real_val = train_test_split(
        real_windows, real_labels_remapped,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=real_labels_remapped
    )
    
    print(f"Real data split: {len(X_real_train)} train, {len(X_real_val)} val")
    
    model_real, metrics_real = train_classifier(
        X_train=X_real_train,
        y_train=y_real_train,
        X_val=X_real_val,
        y_val=y_real_val,
        num_users=n_real,  # Only real users
        window_size=cfg.window_size,
        epochs=50,
        device=cfg.device
    )
    
    # Get predictions for classification report
    model_real.eval()
    real_preds = []
    with torch.no_grad():
        val_dataset = TensorDataset(
            torch.tensor(X_real_val, dtype=torch.float32),
            torch.tensor(y_real_val, dtype=torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(cfg.device)
            outputs = model_real(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            real_preds.extend(predicted.cpu().numpy())
    
    print(f"\nReal-only Model Results:")
    print(f"  Accuracy: {metrics_real['accuracy']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(metrics_real['labels'], metrics_real['predictions'], 
                                target_names=real_user_names))
    
    # ============================================================
    # Step 4: Train classifier on FICTIONAL users only
    # ============================================================
    print("\n" + "="*70)
    print("STEP 4: Training Classifier on Fictional Users Only")
    print("="*70)
    
    # For fictional-only model, remap labels to start from 0
    fictional_label_mapping = {old: new for new, old in enumerate(unique_fictional_labels)}
    fictional_labels_remapped_for_model = np.array([fictional_label_mapping[l] for l in fictional_labels])
    
    # Split fictional data
    X_fictional_train, X_fictional_val, y_fictional_train, y_fictional_val = train_test_split(
        fictional_windows, fictional_labels_remapped_for_model,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=fictional_labels_remapped_for_model
    )
    
    print(f"Fictional data split: {len(X_fictional_train)} train, {len(X_fictional_val)} val")
    
    model_fictional, metrics_fictional = train_classifier(
        X_train=X_fictional_train,
        y_train=y_fictional_train,
        X_val=X_fictional_val,
        y_val=y_fictional_val,
        num_users=len(unique_fictional_labels),  # Only fictional users
        window_size=cfg.window_size,
        epochs=50,
        device=cfg.device
    )
    
    print(f"\nFictional-only Model Results:")
    print(f"  Accuracy: {metrics_fictional['accuracy']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(metrics_fictional['labels'], metrics_fictional['predictions'], 
                                target_names=fictional_user_names))
    
    # ============================================================
    # Step 5: Train classifier on Real + Fictional users
    # ============================================================
    print("\n" + "="*70)
    print("STEP 5: Training Classifier on Real + Fictional Users")
    print("="*70)
    
    # Combine real and fictional
    X_combined = np.concatenate([real_windows, fictional_windows], axis=0)
    y_combined = np.concatenate([real_labels_remapped, fictional_labels_remapped], axis=0)
    
    # Create a flag to track which samples are real vs fictional
    is_real = np.concatenate([
        np.ones(len(real_labels_remapped), dtype=bool),
        np.zeros(len(fictional_labels_remapped), dtype=bool)
    ])
    
    # Split combined data
    X_combined_train, X_combined_val, y_combined_train, y_combined_val, is_real_train, is_real_val = train_test_split(
        X_combined, y_combined, is_real,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=y_combined
    )
    
    print(f"Combined data split: {len(X_combined_train)} train, {len(X_combined_val)} val")
    print(f"  Real windows in train: {np.sum(is_real_train)}")
    print(f"  Fictional windows in train: {np.sum(~is_real_train)}")
    print(f"  Real windows in val: {np.sum(is_real_val)}")
    print(f"  Fictional windows in val: {np.sum(~is_real_val)}")
    
    # Use combined validation set (real + fictional) to match training distribution
    model_combined, metrics_combined = train_classifier(
        X_train=X_combined_train,
        y_train=y_combined_train,
        X_val=X_combined_val,  # Use combined validation set (real + fictional)
        y_val=y_combined_val,
        num_users=num_users,  # All users (real + fictional)
        window_size=cfg.window_size,
        epochs=50,
        device=cfg.device
    )
    
    print(f"\nReal+Fictional Model Results (tested on combined validation: real + fictional):")
    print(f"  Accuracy: {metrics_combined['accuracy']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(metrics_combined['labels'], metrics_combined['predictions'], 
                                target_names=user_names))
    
    # ============================================================
    # Step 6: Compare performance
    # ============================================================
    print("\n" + "="*70)
    print("STEP 6: Performance Comparison")
    print("="*70)
    
    comparison = {
        "Real-only": {
            "accuracy": metrics_real['accuracy'],
            "n_train": len(X_real_train),
            "n_val": len(X_real_val),
            "n_classes": n_real
        },
        "Fictional-only": {
            "accuracy": metrics_fictional['accuracy'],
            "n_train": len(X_fictional_train),
            "n_val": len(X_fictional_val),
            "n_classes": len(unique_fictional_labels)
        },
        "Real+Fictional": {
            "accuracy": metrics_combined['accuracy'],
            "n_train": len(X_combined_train),
            "n_val": len(X_combined_val),
            "n_classes": num_users
        }
    }
    
    print("\nComparison Summary:")
    print(f"{'Model':<20} {'Train Samples':<15} {'Val Samples':<15} {'Classes':<10} {'Accuracy':<10}")
    print("-" * 75)
    for model_name, metrics in comparison.items():
        print(f"{model_name:<20} {metrics['n_train']:<15} {metrics['n_val']:<15} {metrics['n_classes']:<10} {metrics['accuracy']:<10.4f}")
    
    improvement = metrics_combined['accuracy'] - metrics_real['accuracy']
    improvement_pct = (improvement / metrics_real['accuracy']) * 100 if metrics_real['accuracy'] > 0 else 0
    print(f"\nReal+Fictional vs Real-only: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    # Plot confusion matrices
    save_dir = os.path.join(cfg.save_dir, sensor_type.lower(), "classification_results")
    os.makedirs(save_dir, exist_ok=True)
    
    plot_confusion_matrix(
        metrics_real['confusion_matrix'],
        real_user_names,
        "Confusion Matrix: Real-only Model",
        os.path.join(save_dir, "confusion_matrix_real_only.png")
    )
    
    plot_confusion_matrix(
        metrics_fictional['confusion_matrix'],
        fictional_user_names,
        "Confusion Matrix: Fictional-only Model",
        os.path.join(save_dir, "confusion_matrix_fictional_only.png")
    )
    
    plot_confusion_matrix(
        metrics_combined['confusion_matrix'],
        user_names,
        "Confusion Matrix: Real+Fictional Model",
        os.path.join(save_dir, "confusion_matrix_real_fictional.png")
    )
    
    print(f"\nResults saved to: {save_dir}")
    
    return {
        "real_model": model_real,
        "fictional_model": model_fictional,
        "combined_model": model_combined,
        "metrics_real": metrics_real,
        "metrics_fictional": metrics_fictional,
        "metrics_combined": metrics_combined,
        "comparison": comparison
    }


if __name__ == "__main__":
    main()

