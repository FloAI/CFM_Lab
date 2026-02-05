import math
from typing import Dict, Tuple, Callable, Union
import torch
import torch.nn.functional as F

from .config import _ensure_t_tensor
from .geometric_flows import alpha_flow_interpolant, alpha_flow_derivative

# --- Interpolants & derivatives (PATH_REGISTRY) ---

def linear_interpolant(x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], **_):
    t = _ensure_t_tensor(t).to(x0.device)
    return (1.0 - t[:, None]) * x0 + t[:, None] * x1

def linear_derivative(x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], **_):
    return x1 - x0

def vp_interpolant(x0, x1, t, alpha_fn=None, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    if alpha_fn is None:
        alpha = 1.0 - t
    else:
        alpha = alpha_fn(t)
    alpha = alpha.clamp(0.0, 1.0)
    return torch.sqrt(alpha)[:, None] * x0 + torch.sqrt(1.0 - alpha)[:, None] * x1

def vp_derivative(x0, x1, t, alpha_fn=None, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    eps = 1e-6
    if alpha_fn is None:
        t_safe = torch.clamp(t, eps, 1-eps)
        dv0 = -0.5 / torch.sqrt(1 - t_safe)
        dv1 = 0.5 / torch.sqrt(t_safe)
        return dv0[:, None] * x0 + dv1[:, None] * x1
    else:
        dt = 1e-5
        return (vp_interpolant(x0, x1, t + dt, alpha_fn=alpha_fn) - vp_interpolant(x0, x1, t, alpha_fn=alpha_fn))/dt

def log_interpolant(x0, x1, t, eps=1e-8, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    x0p = x0 + eps
    x1p = x1 + eps
    logx = (1.0 - t[:, None]) * torch.log(x0p) + t[:, None] * torch.log(x1p)
    return torch.exp(logx)

def log_derivative(x0, x1, t, eps=1e-8, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    xt = log_interpolant(x0, x1, t, eps=eps)
    return xt * (torch.log(x1 + eps) - torch.log(x0 + eps))

def spherical_interpolant(x0, x1, t, eps=1e-8, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    x0n = x0 / (x0.norm(dim=-1, keepdim=True) + eps)
    x1n = x1 / (x1.norm(dim=-1, keepdim=True) + eps)
    cos_theta = (x0n * x1n).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta) + 1e-6
    t = t[:, None]
    part0 = torch.sin((1.0 - t) * theta) / sin_theta
    part1 = torch.sin(t * theta) / sin_theta
    return part0[:, None] * x0n + part1[:, None] * x1n

def spherical_derivative(x0, x1, t, **_):
    dt = 1e-5
    return (spherical_interpolant(x0, x1, t + dt) - spherical_interpolant(x0, x1, t)) / dt

def softmax_interpolant(x0, x1, t, eps=1e-8, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    logx = (1.0 - t[:, None]) * torch.log(x0 + eps) + t[:, None] * torch.log(x1 + eps)
    return F.softmax(logx, dim=-1)

def softmax_derivative(x0, x1, t, eps=1e-8, **_):
    dt = 1e-5
    return (softmax_interpolant(x0, x1, t + dt, eps=eps) - softmax_interpolant(x0, x1, t, eps=eps)) / dt

def ot_displacement_interpolant(x0, x1, t, normalize=True, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    if normalize:
        x0n = x0 / (x0.sum(dim=-1, keepdim=True) + 1e-9)
        x1n = x1 / (x1.sum(dim=-1, keepdim=True) + 1e-9)
    else:
        x0n = x0; x1n = x1
    return (1.0 - t[:, None]) * x0n + t[:, None] * x1n

def ot_displacement_derivative(x0, x1, t, normalize=True, **_):
    if normalize:
        return (x1 / (x1.sum(dim=-1, keepdim=True) + 1e-9)) - (x0 / (x0.sum(dim=-1, keepdim=True) + 1e-9))
    else:
        return x1 - x0

def expfam_interpolant(x0, x1, t, family="poisson", eps=1e-8, **_):
    t = _ensure_t_tensor(t).to(x0.device)
    if family == "poisson":
        theta0 = torch.log(x0 + eps)
        theta1 = torch.log(x1 + eps)
        thetat = (1.0 - t[:, None]) * theta0 + t[:, None] * theta1
        return torch.exp(thetat)
    else:
        return linear_interpolant(x0, x1, t)

def expfam_derivative(x0, x1, t, family="poisson", eps=1e-8, **_):
    if family == "poisson":
        t = _ensure_t_tensor(t).to(x0.device)
        theta0 = torch.log(x0 + eps)
        theta1 = torch.log(x1 + eps)
        thetat = (1.0 - t[:, None]) * theta0 + t[:, None] * theta1
        return torch.exp(thetat) * (theta1 - theta0)
    else:
        return linear_derivative(x0, x1, t)

def latent_interpolant(x0, x1, t, encoder: Callable = None, decoder: Callable = None, **_):
    assert encoder is not None and decoder is not None
    t = _ensure_t_tensor(t).to(x0.device)
    z0 = encoder(x0)
    z1 = encoder(x1)
    zt = (1.0 - t[:, None]) * z0 + t[:, None] * z1
    return decoder(zt)

def latent_derivative(x0, x1, t, encoder: Callable = None, decoder: Callable = None, **_):
    dt = 1e-5
    return (latent_interpolant(x0, x1, t + dt, encoder=encoder, decoder=decoder) - latent_interpolant(x0, x1, t, encoder=encoder, decoder=decoder)) / dt

def hybrid_interpolant(x0, x1, t, gamma=0.5, eps=1e-8, **_):
    lin = linear_interpolant(x0, x1, t)
    logp = log_interpolant(x0, x1, t, eps=eps)
    return (1.0 - gamma) * lin + gamma * logp

def hybrid_derivative(x0, x1, t, gamma=0.5, eps=1e-8, **_):
    return (1.0 - gamma) * linear_derivative(x0, x1, t) + gamma * log_derivative(x0, x1, t, eps=eps)

def sparseaware_interpolant(x0, x1, t, delta=1e-3, **_):
    lin = linear_interpolant(x0, x1, t)
    mask_zero = ((x0 == 0.0) | (x1 == 0.0)).float()
    smooth = delta * (_ensure_t_tensor(t) * (1.0 - _ensure_t_tensor(t)))[:, None].to(x0.device)
    return lin + smooth * mask_zero

def sparseaware_derivative(x0, x1, t, delta=1e-3, **_):
    dt = 1e-5
    return (sparseaware_interpolant(x0, x1, t + dt, delta=delta) - sparseaware_interpolant(x0, x1, t, delta=delta)) / dt


PATH_REGISTRY: Dict[str, Tuple[Callable, Callable]] = {
    "linear": (linear_interpolant, linear_derivative),
    "vp": (vp_interpolant, vp_derivative),
    "log": (log_interpolant, log_derivative),
    "spherical": (spherical_interpolant, spherical_derivative),
    "softmax": (softmax_interpolant, softmax_derivative),
    "ot_displacement": (ot_displacement_interpolant, ot_displacement_derivative),
    "expfam": (expfam_interpolant, expfam_derivative),
    "latent": (latent_interpolant, latent_derivative),
    "hybrid": (hybrid_interpolant, hybrid_derivative),
    "sparseaware": (sparseaware_interpolant, sparseaware_derivative),
    
    "alpha_flow": (alpha_flow_interpolant, alpha_flow_derivative),
    "variance_preserving_cfm": (vp_interpolant, vp_derivative),
}
