import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        assert input_dim % 3 == 0, "input_dim must be divisible by 3 for (T, 3) windows"
        self.T = input_dim // 3
        self.C = 3

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(self.C, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.SiLU(),
        )

        cnn_out_dim = 64 * self.T

        self.encoder_mlp = make_mlp(
            cnn_out_dim,
            hidden_dim,
            2 * latent_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.decoder = make_mlp(
            latent_dim,
            hidden_dim,
            input_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"WindowVAE parameters: {total/1e6:.2f}M")

    def encode(self, x_flat):
        B = x_flat.shape[0]
        x = x_flat.view(B, self.T, self.C).permute(0, 2, 1)
        h = self.encoder_cnn(x)
        h = h.reshape(B, -1)

        stats = self.encoder_mlp(h)
        mu, logvar = stats[..., :self.latent_dim], stats[..., self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_flat):
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decode(z)
        return recon_flat, mu, logvar

