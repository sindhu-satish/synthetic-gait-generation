import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .config import Config

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

class VAE(nn.Module):
    def __init__(self, cfg: Config, num_features: int, cat_vocab_sizes: List[int]):
        super().__init__()
        self.cfg = cfg
        self.num_features = num_features
        self.cat_vocab_sizes = cat_vocab_sizes
        
        self.embeds = nn.ModuleList([nn.Embedding(v, min(cfg.cat_embed_dim, max(4, int(round(v**0.25)*2)))) for v in cat_vocab_sizes])
        embed_total = sum([emb.embedding_dim for emb in self.embeds])
        enc_in = num_features + embed_total
        self.encoder = make_mlp(enc_in, cfg.hidden_dim, 2*cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.decoder = make_mlp(cfg.latent_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
        
        self.num_head = nn.Linear(cfg.hidden_dim, num_features) if num_features > 0 else None
        self.cat_heads = nn.ModuleList([nn.Linear(cfg.hidden_dim, v) for v in cat_vocab_sizes])
        
        total = sum(p.numel() for p in self.parameters())
        print(f"VAE parameters: {total/1e6:.2f}M")

    def encode(self, num_x, cat_idx):
        embs = []
        for i, emb in enumerate(self.embeds):
            if cat_idx.shape[1] == 0:
                continue
            embs.append(emb(cat_idx[:, i]))
        if embs:
            cat_e = torch.cat(embs, dim=-1)
            h = torch.cat([num_x, cat_e], dim=-1) if self.num_features > 0 else cat_e
        else:
            h = num_x
        stats = self.encoder(h)
        mu, logvar = stats[..., :self.cfg.latent_dim], stats[..., self.cfg.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decoder(z)
        num_out = self.num_head(h) if self.num_head is not None else None
        cat_logits = [head(h) for head in self.cat_heads]
        return num_out, cat_logits

    def forward(self, num_x, cat_idx, beta=1.0):
        mu, logvar = self.encode(num_x, cat_idx)
        z = self.reparameterize(mu, logvar)
        num_out, cat_logits = self.decode(z)
        
        recon_num = F.mse_loss(num_out, num_x) if (num_out is not None and num_x.shape[1] > 0) else torch.tensor(0.0, device=num_x.device)
        recon_cat = torch.tensor(0.0, device=num_x.device)
        for i, logits in enumerate(cat_logits):
            if cat_idx.shape[1] == 0:
                continue
            recon_cat = recon_cat + F.cross_entropy(logits, cat_idx[:, i])
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_num + recon_cat + beta*kl
        return loss, {"recon_num": recon_num.item(), "recon_cat": recon_cat.item(), "kl": kl.item()}
