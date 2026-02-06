import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .config import CFMConfig, _ensure_t_tensor

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = np.log(10000) / (self.half_dim - 1)
        self.emb = torch.exp(torch.arange(self.half_dim) * -self.emb)

    def forward(self, t):
        t = _ensure_t_tensor(t).to(self.emb.device)
        emb = t[:, None] * self.emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class VelocityMLP(nn.Module):
    def __init__(self, x_dim, cond_dim, hidden_dim, num_layers, time_dim):
        super().__init__()
        self.time_mlp = SinusoidalTimeEmbeddings(time_dim)
        input_dim = x_dim + time_dim + cond_dim
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, c=None):
        t_emb = self.time_mlp(t)
        if c is not None and c.shape[1] > 0:
            inp = torch.cat([x, t_emb, c], dim=-1)
        else:
            inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

class VelocityTransformer(nn.Module):
    def __init__(self, x_dim, cond_dim, hidden_dim, num_layers, time_dim, chunk_size=16):
        super().__init__()
        self.time_mlp = SinusoidalTimeEmbeddings(time_dim)
        self.chunk_size = chunk_size
        self.token_dim = min(chunk_size, x_dim)
        
        # Calculate number of tokens needed
        self.n_tokens = math.ceil(x_dim / self.token_dim)
        self.pad_len = (self.n_tokens * self.token_dim) - x_dim
        
        self.feat_emb = nn.Linear(self.token_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens + 1, hidden_dim)) # +1 for cond
        
        # Condition embedding
        self.cond_emb = nn.Linear(cond_dim, hidden_dim) if cond_dim > 0 else None
        
        # Time integration
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_head = nn.Linear(hidden_dim, self.token_dim)

    def forward(self, x, t, c=None):
        B, D = x.shape
        
        # 1. Padding if needed
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
            
        # 2. Tokenize: (B, N_Tokens, Token_Dim)
        x_tok = x.view(B, self.n_tokens, self.token_dim)
        h = self.feat_emb(x_tok)
        
        # 3. Add Time (global to all tokens)
        t_emb = self.time_mlp(t)
        t_emb = self.time_proj(t_emb).unsqueeze(1) # (B, 1, H)
        h = h + t_emb
        
        # 4. Prepend Condition Token
        if c is not None and self.cond_emb is not None:
            c_emb = self.cond_emb(c).unsqueeze(1) # (B, 1, H)
            h = torch.cat([c_emb, h], dim=1)
            
        # 5. Add Position Embedding
        seq_len = h.shape[1]
        h = h + self.pos_emb[:, :seq_len, :]
        
        # 6. Transformer Pass
        h = self.transformer(h)
        
        # 7. Decode
        if c is not None and self.cond_emb is not None:
            h = h[:, 1:, :] # Remove cond token
            
        out_tok = self.out_head(h)
        out = out_tok.view(B, -1)
        
        # 8. Remove Padding
        return out[:, :D]

class SimpleAutoencoder(nn.Module):
    def __init__(self, x_dim, latent_dim=128, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, x_dim))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)

def build_velocity_model(x_dim, cond_dim, cfg, ae=None):
    if cfg.model_type == "mlp":
        return VelocityMLP(x_dim, cond_dim, cfg.hidden_dim, cfg.num_layers, cfg.time_emb_dim)
    elif cfg.model_type == "transformer":
        return VelocityTransformer(x_dim, cond_dim, cfg.hidden_dim, cfg.num_layers, cfg.time_emb_dim)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
