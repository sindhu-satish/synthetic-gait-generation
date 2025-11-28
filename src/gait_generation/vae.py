import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .config import LATENT_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT

def make_mlp(in_dim, hidden_dim, out_dim, n_layers=3, dropout=0.0):
    layers = []
    d = in_dim
    for i in range(n_layers - 1):
        layers += [nn.Linear(d, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

class WindowVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM, hidden_dim: int = HIDDEN_DIM, n_layers: int = N_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.encoder = make_mlp(input_dim, hidden_dim, 2*latent_dim, n_layers, dropout)
        self.decoder = make_mlp(latent_dim, hidden_dim, input_dim, n_layers, dropout)
        
        total = sum(p.numel() for p in self.parameters())
        print(f"WindowVAE parameters: {total/1e6:.2f}M")

    def encode(self, x_flat):
        stats = self.encoder(x_flat)
        mu, logvar = stats[..., :self.latent_dim], stats[..., self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_flat):
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decode(z)
        return recon_flat, mu, logvar

