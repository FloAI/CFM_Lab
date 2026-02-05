from typing import Dict, Callable
import torch
import torch.nn.functional as F
import numpy as np

from .coupling import sinkhorn_matrix 

class LossRegistry:
    """Central registry for all loss functions."""
    def __init__(self):
        self.loss_fns: Dict[str, Callable] = {
            "l2": self.l2, "l1": self.l1, "huber": self.huber,
            "smooth_l1": self.smooth_l1, "charbonnier": self.charbonnier,
            "cauchy": self.cauchy, "tukey": self.tukey_biweight,
            "cosine": self.cosine, "kl": self.kl, "js": self.js,
            "poisson": self.poisson, "zero_inflated": self.zero_inflated,
            "ot_sinkhorn": self.ot_sinkhorn_approx,
            "vfm_poisson_nll": self.vfm_poisson_nll,
        }

    # --- Basic L-p and Robust Losses ---
    def l2(self, pred, target, **_):
        return ((pred - target) ** 2).mean()

    def l1(self, pred, target, **_):
        return (pred - target).abs().mean()

    def huber(self, pred, target, delta=1.0, **_):
        return F.huber_loss(pred, target, delta=delta, reduction="mean")

    def smooth_l1(self, pred, target, **_):
        return F.smooth_l1_loss(pred, target, reduction="mean")

    def charbonnier(self, pred, target, eps=1e-3, **_):
        return torch.sqrt((pred - target)**2 + eps**2).mean()

    def cauchy(self, pred, target, c=2.0, **_):
        r = pred - target
        return (c**2 * torch.log(1 + (r / c)**2)).mean()

    def tukey_biweight(self, pred, target, c=4.685, **_):
        r = pred - target
        r_norm = r / c
        mask = (r_norm.abs() <= 1.0).float()
        loss = torch.zeros_like(r)
        loss = ((c**2)/6.0) * (1 - (1 - r_norm**2)**3) * mask + (r_norm.abs() > 1.0).float() * ((c**2)/6.0)
        return loss.mean()

    # --- Divergence and Specialized Losses ---
    def cosine(self, pred, target, **_):
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-9)
        targ_n = target / (target.norm(dim=-1, keepdim=True) + 1e-9)
        return (1 - (pred_n * targ_n).sum(dim=-1)).mean()

    def kl(self, pred, target, eps=1e-9, **_):
        p = F.softmax(target, dim=-1)
        q = F.softmax(pred, dim=-1) + eps
        return (p * (p.log() - q.log())).sum(dim=-1).mean()

    def js(self, pred, target, eps=1e-9, **_):
        p = F.softmax(pred, dim=-1)
        q = F.softmax(target, dim=-1)
        m = 0.5 * (p + q)
        kl_pm = (p * (p + eps).log() - p * (m + eps).log()).sum(dim=-1)
        kl_qm = (q * (q + eps).log() - q * (m + eps).log()).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm).mean()

    def poisson(self, pred, target, eps=1e-9, **_):
        lam = F.softplus(pred)
        return (lam - target * torch.log(lam + eps)).mean()

    def zero_inflated(self, pred, target, zero_weight=2.0, base_loss="l2", **_):
        if base_loss == "l1":
            base = (pred - target).abs()
        else:
            base = (pred - target).pow(2)
        zero_mask = (target == 0).float()
        zero_penalty = zero_weight * (zero_mask * pred.abs())
        return (base + zero_penalty).mean()

    def ot_sinkhorn_approx(self, pred, target, eps=0.1, iters=10, **_):
        p = F.softplus(pred).detach().cpu().numpy()
        q = F.softplus(target).detach().cpu().numpy()
        B, D = p.shape
        losses = []
        for i in range(B):
            pi = p[i] / (p[i].sum() + 1e-9)
            qi = q[i] / (q[i].sum() + 1e-9)
            M = ((np.arange(D)[:, None] - np.arange(D)[None, :])**2).astype(np.float64)
            P = sinkhorn_matrix(pi, qi, M, eps=eps, niter=iters)
            losses.append((P * M).sum())
        return torch.tensor(losses, device=pred.device).mean()
        
    def vfm_poisson_nll(self, pred, target, eps=1e-9, **_):
        """Proxy loss for EF-VFM (Poisson case: L2 loss on velocities)."""
        l2_loss = ((pred - target) ** 2)
        return l2_loss.mean()
