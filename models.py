import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFMConfig
from .config import _ensure_t_tensor 

# --- Time embedding ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), dim // 2)))

    def forward(self, t: torch.Tensor):
        t = _ensure_t_tensor(t).to(self.freqs.device)
        t = t.unsqueeze(-1)
        angles = t * self.freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, t], dim=-1)
        return emb

# --- Velocity Models ---
class MLPVelocity(nn.Module):
    """Suitable for low-dimensional or simple flow fields."""
    def __init__(self, x_dim, cond_dim, hidden_dim=512, num_layers=4, time_emb_dim=128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.cond_dim = cond_dim if cond_dim is not None else 0
        in_dim = x_dim + time_emb_dim + self.cond_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, y):
        te = self.time_emb(t)
        if y is None or y.numel() == 0:
            inp = torch.cat([x_t, te], dim=-1)
        else:
            inp = torch.cat([x_t, te, y], dim=-1)
        return self.net(inp)

class TransformerVelocity(nn.Module):
    """Suitable for high-dimensional or structured flow fields."""
    def __init__(self, x_dim, cond_dim, hidden_dim=512, num_layers=4, time_emb_dim=128, nhead=8, chunk_size=16):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.cond_dim = cond_dim if cond_dim is not None else 0
        self.chunk_size = chunk_size
        self.token_dim = min(self.chunk_size, x_dim)
        self.n_tokens = math.ceil(x_dim / self.token_dim)
        self.token_embed = nn.Linear(self.token_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, self.token_dim)
        if self.cond_dim > 0:
            self.cond_proj = nn.Linear(self.cond_dim + time_emb_dim, hidden_dim)
        else:
            self.cond_proj = None

    def forward(self, x_t, t, y):
        B, D = x_t.shape
        pad = self.n_tokens * self.token_dim - D
        if pad > 0:
            x_p = F.pad(x_t, (0, pad))
        else:
            x_p = x_t
        x_tok = x_p.view(B, self.n_tokens, self.token_dim)
        h = self.token_embed(x_tok)
        te = self.time_emb(t)
        if y is None or y.numel() == 0:
            cond_input = te
        else:
            cond_input = torch.cat([te, y], dim=-1)
        if self.cond_proj is not None:
            cond_proj = self.cond_proj(cond_input).unsqueeze(1)
            h = h + cond_proj
        h = h.permute(1, 0, 2)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)
        out_tok = self.out(h)
        out = out_tok.contiguous().view(B, -1)[:, :D]
        return out

class SimpleAutoencoder(nn.Module):
    def __init__(self, x_dim, latent_dim=128, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, x_dim))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

def build_velocity_model(x_dim: int, cond_dim: int, cfg: CFMConfig, ae: Optional[SimpleAutoencoder]=None) -> nn.Module:
    typ = cfg.model_type
    if typ == "mlp" or typ == "autoencoder_latent":
        # MLP preferred for low D
        return MLPVelocity(x_dim, cond_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, time_emb_dim=cfg.time_emb_dim)
    elif typ == "transformer":
        # Transformer preferred for high D
        return TransformerVelocity(x_dim, cond_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, time_emb_dim=cfg.time_emb_dim)
    else:
        raise NotImplementedError(f"Unknown model type {typ}")
