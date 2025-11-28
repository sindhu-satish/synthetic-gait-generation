import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim*2)
        self.lin2 = nn.Linear(dim*2, dim)
    
    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / (half-1)))
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=-1)
        emb = F.silu(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class ResMLPBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        h = self.fc1(x)
        h = F.silu(self.ln1(h))
        h = self.fc2(h)
        h = self.ln2(h)
        return F.silu(x + h)

class UNetMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, n_blocks=4, cond_dim=0):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.cond_emb = nn.Embedding(1024, hidden_dim) if cond_dim > 0 else None
        self.inp = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim, hidden_dim*2) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden_dim, latent_dim)
        total = sum(p.numel() for p in self.parameters())
        print(f"UNetMLP parameters: {total/1e6:.2f}M")

    def forward(self, x, t, cond_idx=None):
        h = self.inp(x)
        emb = self.time_emb(t)
        h = h + emb
        if self.cond_emb is not None and cond_idx is not None:
            h = h + self.cond_emb(cond_idx % self.cond_emb.num_embeddings)
        for block in self.blocks:
            h = block(h)
        return self.out(h)

class LatentDDPM:
    def __init__(self, model: UNetMLP, T=400, beta_schedule="cosine", device="cpu"):
        self.model = model
        self.T = T
        self.device = device
        
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T, device=device)
        elif beta_schedule == "cosine":
            steps = T + 1
            s = 0.008
            x = torch.linspace(0, T, steps, device=device)
            alphas_cumprod = torch.cos(((x/T + s) / (1+s)) * math.pi/2)**2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 1e-5, 0.999)
        else:
            raise ValueError("unknown beta schedule")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0/self.alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_ac * x0 + sqrt_om * noise

    def p_mean_variance(self, x_t, t, cond_idx=None):
        eps_pred = self.model(x_t, t, cond_idx)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        x0_pred = (x_t - sqrt_om * eps_pred) / (sqrt_ac + 1e-8)
        
        beta_t = self.betas[t][:, None]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None]
        ac_prev = self.alphas_cumprod_prev[t][:, None]
        ac = self.alphas_cumprod[t][:, None]
        posterior_mean = sqrt_recip_alpha_t * (x_t - beta_t * eps_pred / (torch.sqrt(1 - ac) + 1e-8))
        posterior_variance = self.posterior_variance[t][:, None]
        return posterior_mean, posterior_variance, x0_pred

    @torch.no_grad()
    def sample(self, n, cond_idx=None, steps=None):
        steps = steps or self.T
        model = self.model
        model.eval()
        x = torch.randn(n, model.out.out_features, device=self.device)
        
        cond = None
        if cond_idx is not None:
            cond = torch.tensor(cond_idx, device=self.device, dtype=torch.long)
            if cond.ndim == 0:
                cond = cond.repeat(n)
            if cond.shape[0] != n:
                cond = cond.repeat(int(math.ceil(n/cond.shape[0])))[:n]
        
        for i in reversed(range(steps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            mean, var, x0_pred = self.p_mean_variance(x, t, cond)
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
        return x.detach()

class LatentDataset(Dataset):
    def __init__(self, Z, cond_idx=None):
        self.Z = torch.tensor(Z, dtype=torch.float32)
        self.cond = torch.tensor(cond_idx, dtype=torch.long) if cond_idx is not None else None
    
    def __len__(self):
        return self.Z.shape[0]
    
    def __getitem__(self, i):
        if self.cond is not None:
            return self.Z[i], self.cond[i]
        else:
            return self.Z[i], torch.tensor(0, dtype=torch.long)
