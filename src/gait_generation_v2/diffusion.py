"""
Latent DDPM for Gait Generation V2.

Implements:
- SinusoidalTimeEmbedding: Timestep encoding
- ResMLPBlock: Residual MLP block with conditioning
- LatentUNet: MLP-based denoising network for latent space
- LatentDDPM: Full DDPM implementation with forward/reverse processes
- EMA: Exponential Moving Average for model weights
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 2)
        self.lin2 = nn.Linear(dim * 2, dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timesteps
            
        Returns:
            emb: (B, dim) time embeddings
        """
        half = self.dim // 2
        device = t.device
        
        freqs = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1))
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=-1)
        
        emb = F.silu(self.lin1(emb))
        emb = self.lin2(emb)
        
        return emb


class ResMLPBlock(nn.Module):
    """Residual MLP block with optional conditioning injection."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        cond_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, dim) input
            cond_emb: (B, hidden_dim) optional conditioning
            
        Returns:
            out: (B, dim) output
        """
        h = self.fc1(x)
        
        # Inject conditioning
        if cond_emb is not None:
            h = h + cond_emb
        
        h = F.silu(self.ln1(h))
        h = self.fc2(h)
        h = self.ln2(h)
        
        return F.silu(x + h)


class LatentUNet(nn.Module):
    """
    MLP-based denoising network for latent space diffusion.
    
    Operates on flat latent vectors, conditioned on timestep and user.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 6,
        n_users: int = 117,
        user_embed_dim: int = 32
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        
        # User embedding
        self.user_emb = nn.Embedding(n_users, user_embed_dim)
        
        # Learnable conditioning weight
        self.cond_weight = nn.Parameter(torch.ones(1) * 2.0)
        
        # Input projection: latent + user_embed -> hidden
        self.inp = nn.Linear(latent_dim + user_embed_dim, hidden_dim)
        
        # Conditioning projection for blocks
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden_dim, hidden_dim * 2) for _ in range(n_blocks)
        ])
        
        # Output projection
        self.out = nn.Linear(hidden_dim, latent_dim)
        
        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LatentUNet parameters: {total_params/1e6:.2f}M")
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        user_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, latent_dim) noisy latent
            t: (B,) timesteps
            user_idx: (B,) user indices
            
        Returns:
            noise_pred: (B, latent_dim) predicted noise
        """
        # User embedding
        u = self.user_emb(user_idx)  # (B, user_embed_dim)
        
        # Concatenate latent and user
        h = torch.cat([x, u], dim=-1)  # (B, latent_dim + user_embed_dim)
        h = self.inp(h)  # (B, hidden_dim)
        
        # Time embedding
        t_emb = self.time_emb(t)  # (B, hidden_dim)
        h = h + t_emb
        
        # Conditioning for blocks (scaled)
        cond_emb = self.cond_proj(t_emb * self.cond_weight)  # (B, hidden_dim * 2)
        
        # Residual blocks with conditioning
        for block in self.blocks:
            h = block(h, cond_emb)
        
        # Output
        return self.out(h)


class LatentDDPM:
    """
    Denoising Diffusion Probabilistic Model for latent space.
    
    Implements forward process (adding noise) and reverse process (denoising).
    """
    
    def __init__(
        self,
        model: LatentUNet,
        T: int = 1000,
        beta_schedule: str = "cosine",
        device: str = "cpu"
    ):
        self.model = model
        self.T = T
        self.device = device
        
        # Compute beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T, device=device)
        elif beta_schedule == "cosine":
            steps = T + 1
            s = 0.008
            x = torch.linspace(0, T, steps, device=device)
            alphas_cumprod = torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 1e-5, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), 
            self.alphas_cumprod[:-1]
        ], dim=0)
        
        # Precompute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward process: add noise to x0 at timestep t.
        
        q(x_t | x_0) = N(x_t; sqrt(α̅_t) * x_0, (1 - α̅_t) * I)
        
        Args:
            x0: (B, latent_dim) clean latents
            t: (B,) timesteps
            noise: (B, latent_dim) optional noise (sampled if None)
            
        Returns:
            x_t: (B, latent_dim) noisy latents
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_ac * x0 + sqrt_om * noise
    
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        user_idx: torch.Tensor
    ):
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        
        Args:
            x_t: (B, latent_dim) noisy latents
            t: (B,) timesteps
            user_idx: (B,) user indices
            
        Returns:
            mean: (B, latent_dim) posterior mean
            var: (B, latent_dim) posterior variance
            x0_pred: (B, latent_dim) predicted x0
        """
        # Predict noise
        eps_pred = self.model(x_t, t, user_idx)
        
        # Predict x0
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        x0_pred = (x_t - sqrt_om * eps_pred) / (sqrt_ac + 1e-8)
        
        # Compute posterior mean
        beta_t = self.betas[t][:, None]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None]
        ac = self.alphas_cumprod[t][:, None]
        
        posterior_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * eps_pred / (torch.sqrt(1 - ac) + 1e-8)
        )
        posterior_variance = self.posterior_variance[t][:, None]
        
        return posterior_mean, posterior_variance, x0_pred
    
    @torch.no_grad()
    def sample(
        self,
        n: int,
        user_idx: torch.Tensor,
        steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample latents from the model.
        
        Args:
            n: number of samples
            user_idx: (n,) or scalar user index
            steps: number of denoising steps (default: T)
            
        Returns:
            samples: (n, latent_dim) generated latents
        """
        steps = steps or self.T
        self.model.eval()
        
        # Handle user_idx
        if isinstance(user_idx, int):
            user_idx = torch.full((n,), user_idx, dtype=torch.long, device=self.device)
        elif isinstance(user_idx, torch.Tensor):
            user_idx = user_idx.to(self.device)
            if user_idx.ndim == 0:
                user_idx = user_idx.repeat(n)
            if user_idx.shape[0] != n:
                user_idx = user_idx.repeat(int(math.ceil(n / user_idx.shape[0])))[:n]
        
        # Start from noise
        x = torch.randn(n, self.model.latent_dim, device=self.device)
        
        # Reverse diffusion
        for i in reversed(range(steps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            mean, var, x0_pred = self.p_mean_variance(x, t, user_idx)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
        
        return x.detach()


class EMA:
    """
    Exponential Moving Average for model weights.
    
    Maintains a shadow copy of weights for smoother sampling.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_shadow()
    
    def _init_shadow(self):
        """Initialize shadow weights from model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights (before apply_shadow)."""
        # This requires storing original weights, which we don't do by default
        # Use apply_shadow carefully
        pass
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get shadow weights as state dict."""
        return {k: v.clone() for k, v in self.shadow.items()}
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load shadow weights from state dict."""
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()

