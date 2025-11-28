"""
1D Convolutional VAE for Gait Generation V2.

Architecture:
- Encoder: Conv1d stack with user conditioning, outputs μ and logσ²
- Decoder: Transposed Conv1d stack with user conditioning, outputs reconstructed window
- User conditioning via embedding concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
from .config import ConfigV2


class ConvEncoder(nn.Module):
    """
    1D Convolutional encoder for gait windows.
    
    Takes (B, 4, window_size) input and outputs (μ, logσ²) for VAE latent.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: Tuple[int, ...] = (64, 128, 256, 256),
        latent_dim: int = 64,
        user_embed_dim: int = 32,
        n_users: int = 117,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.user_embed_dim = user_embed_dim
        
        # User embedding
        self.user_embed = nn.Embedding(n_users, user_embed_dim)
        
        # Conv stack: (B, 4, 256) -> (B, 256, 16)
        # Each conv halves the temporal dimension
        layers = []
        ch_in = in_channels
        kernels = [7, 5, 5, 3]  # Decreasing kernel sizes
        
        for i, ch_out in enumerate(hidden_channels):
            k = kernels[i] if i < len(kernels) else 3
            p = k // 2
            layers.append(nn.Conv1d(ch_in, ch_out, kernel_size=k, stride=2, padding=p))
            layers.append(nn.BatchNorm1d(ch_out))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))  # Add dropout for regularization
            ch_in = ch_out
        
        self.convs = nn.Sequential(*layers)
        
        # After 4 downsamples: 256 -> 128 -> 64 -> 32 -> 16
        # Global average pool reduces to (B, hidden_channels[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Output heads for μ and logσ²
        fc_in = hidden_channels[-1] + user_embed_dim
        self.dropout_fc = nn.Dropout(dropout)  # Dropout before FC layers
        self.fc_mu = nn.Linear(fc_in, latent_dim)
        self.fc_logvar = nn.Linear(fc_in, latent_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        user_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, window_size) input windows
            user_idx: (B,) user indices
            
        Returns:
            mu: (B, latent_dim) mean of latent distribution
            logvar: (B, latent_dim) log variance of latent distribution
        """
        # Conv encoding
        h = self.convs(x)  # (B, hidden_channels[-1], T')
        h = self.pool(h).squeeze(-1)  # (B, hidden_channels[-1])
        
        # Concatenate user embedding
        u = self.user_embed(user_idx)  # (B, user_embed_dim)
        h = torch.cat([h, u], dim=-1)  # (B, hidden_channels[-1] + user_embed_dim)
        
        # Apply dropout before FC layers
        h = self.dropout_fc(h)
        
        # Output μ and logσ²
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    1D Convolutional decoder for gait windows.
    
    Takes latent z and user_idx, outputs reconstructed (B, 4, window_size).
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_channels: Tuple[int, ...] = (256, 256, 128, 64),
        out_channels: int = 4,
        user_embed_dim: int = 32,
        n_users: int = 117,
        window_size: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.user_embed_dim = user_embed_dim
        self.hidden_channels = hidden_channels
        self.window_size = window_size
        
        # User embedding
        self.user_embed = nn.Embedding(n_users, user_embed_dim)
        
        # Initial temporal size after encoder: window_size / 2^4 = 16
        self.init_t = window_size // 16
        
        # Project z + user_embed to initial feature map
        fc_in = latent_dim + user_embed_dim
        self.dropout_fc = nn.Dropout(dropout)  # Dropout before initial FC
        self.fc = nn.Linear(fc_in, hidden_channels[0] * self.init_t)
        
        # Transposed conv stack: (B, 256, 16) -> (B, 4, 256)
        layers = []
        ch_in = hidden_channels[0]
        
        for i, ch_out in enumerate(hidden_channels[1:]):
            layers.append(nn.ConvTranspose1d(ch_in, ch_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(ch_out))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))  # Add dropout for regularization
            ch_in = ch_out
        
        # Final layer to output channels
        layers.append(nn.ConvTranspose1d(ch_in, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.deconvs = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent codes
            user_idx: (B,) user indices
            
        Returns:
            x_recon: (B, out_channels, window_size) reconstructed windows
        """
        # Concatenate user embedding
        u = self.user_embed(user_idx)  # (B, user_embed_dim)
        h = torch.cat([z, u], dim=-1)  # (B, latent_dim + user_embed_dim)
        
        # Apply dropout before FC
        h = self.dropout_fc(h)
        
        # Project to initial feature map
        h = self.fc(h)  # (B, hidden_channels[0] * init_t)
        h = h.view(-1, self.hidden_channels[0], self.init_t)  # (B, 256, 16)
        
        # Transposed conv decoding
        x_recon = self.deconvs(h)  # (B, 4, 256)
        
        return x_recon


class ConvVAE(nn.Module):
    """
    Full 1D Convolutional VAE for gait windows.
    
    Combines encoder and decoder with user conditioning.
    """
    
    def __init__(
        self,
        cfg: ConfigV2,
        latent_dim: int,
        n_users: int
    ):
        super().__init__()
        
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.n_users = n_users
        
        # Encoder
        self.encoder = ConvEncoder(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            latent_dim=latent_dim,
            user_embed_dim=cfg.user_embed_dim,
            n_users=n_users,
            dropout=cfg.vae_dropout
        )
        
        # Decoder
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            hidden_channels=tuple(reversed(cfg.hidden_channels)),
            out_channels=cfg.in_channels,
            user_embed_dim=cfg.user_embed_dim,
            n_users=n_users,
            window_size=cfg.window_size,
            dropout=cfg.vae_dropout
        )
        
        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ConvVAE parameters: {total_params/1e6:.2f}M")
    
    def encode(
        self, 
        x: torch.Tensor, 
        user_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode windows to latent distribution parameters.
        
        Args:
            x: (B, window_size, in_channels) or (B, in_channels, window_size)
            user_idx: (B,) user indices
            
        Returns:
            mu, logvar: (B, latent_dim) each
        """
        # Ensure x is (B, C, T) for Conv1d
        if x.shape[-1] == self.cfg.in_channels:
            x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        
        return self.encoder(x, user_idx)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to windows.
        
        Args:
            z: (B, latent_dim) latent codes
            user_idx: (B,) user indices
            
        Returns:
            x_recon: (B, in_channels, window_size)
        """
        return self.decoder(z, user_idx)
    
    def forward(
        self, 
        x: torch.Tensor, 
        user_idx: torch.Tensor, 
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with loss computation.
        
        Args:
            x: (B, window_size, in_channels) input windows
            user_idx: (B,) user indices
            beta: KL loss weight
            
        Returns:
            loss: scalar total loss
            logs: dict with individual loss components
        """
        # Ensure x is (B, C, T) for Conv1d
        if x.shape[-1] == self.cfg.in_channels:
            x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        
        # Encode
        mu, logvar = self.encoder(x, user_idx)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z, user_idx)
        
        # Check for NaN/Inf in reconstruction
        if torch.isnan(x_recon).any() or torch.isinf(x_recon).any():
            # Clamp to reasonable range
            x_recon = torch.clamp(x_recon, min=-10.0, max=10.0)
            x_recon = torch.nan_to_num(x_recon, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        # Clamp logvar to prevent numerical instability
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)
        kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
        
        # Check for NaN
        if torch.isnan(recon_loss) or torch.isnan(kl_loss):
            print(f"  WARNING: NaN in loss computation!")
            print(f"    Recon loss: {recon_loss.item() if not torch.isnan(recon_loss) else 'NaN'}")
            print(f"    KL loss: {kl_loss.item() if not torch.isnan(kl_loss) else 'NaN'}")
            print(f"    mu stats: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")
            print(f"    logvar stats: min={logvar.min().item():.4f}, max={logvar.max().item():.4f}, mean={logvar.mean().item():.4f}")
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        logs = {
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "loss": loss.item()
        }
        
        return loss, logs
    
    def sample(
        self, 
        n: int, 
        user_idx: torch.Tensor, 
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Sample windows from prior.
        
        Args:
            n: number of samples
            user_idx: (n,) or scalar user index
            device: device to use
            
        Returns:
            samples: (n, in_channels, window_size) generated windows
        """
        if isinstance(user_idx, int):
            user_idx = torch.full((n,), user_idx, dtype=torch.long, device=device)
        elif user_idx.ndim == 0:
            user_idx = user_idx.repeat(n)
        
        z = torch.randn(n, self.latent_dim, device=device)
        
        with torch.no_grad():
            samples = self.decode(z, user_idx)
        
        return samples

