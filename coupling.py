import numpy as np
import torch
from typing import Dict, Any, Union, Tuple, Optional

from .config import CFMConfig, CFMLabDataset # Use new dataset name

# --- Priors ---
def sample_prior(batch_size: int, x_dim: int, dataset: Union[CFMLabDataset, None], cfg: CFMConfig) -> torch.Tensor:
    method = cfg.prior_sampler
    p = cfg.prior_params
    device = cfg.device
    
    if method == "gaussian":
        std = p.get("std", 1.0)
        return torch.randn(batch_size, x_dim, device=device) * std
        
    elif method == "empirical_shuffle":
        assert dataset is not None and hasattr(dataset, 'x'), "Dataset required for empirical_shuffle prior."
        idx = np.random.choice(len(dataset), size=batch_size, replace=False)
        return torch.from_numpy(dataset.x[idx]).to(device)
        
    elif method == "dirichlet":
        alpha = p.get("dirichlet_alpha", 0.5)
        samples = np.random.dirichlet([alpha] * x_dim, size=batch_size).astype(np.float32)
        return torch.from_numpy(samples).to(device)
        
    elif method == "custom":
        prior_callable = cfg.get("prior_callable", None)
        assert callable(prior_callable), "prior_callable must be provided for custom prior"
        return prior_callable(batch_size, x_dim)
        
    else:
        raise NotImplementedError(f"Unknown prior sampler {method}")

# --- Optimal Transport (OT) Utilities ---

def sinkhorn_matrix(a: np.ndarray, b: np.ndarray, M: np.ndarray, eps: float = 0.05, niter: int = 50) -> np.ndarray:
    """Implements the Sinkhorn-Knopp algorithm (CPU-based)."""
    K = np.exp(-M / eps)
    u = np.ones_like(a)
    v = np.ones_like(b)
    
    for _ in range(niter):
        v = b / (K.T.dot(u) + 1e-16)
        u = a / (K.dot(v) + 1e-16)
        
    P = np.diag(u).dot(K).dot(np.diag(v))
    return P

def batch_sinkhorn_coupling(x0_batch: np.ndarray, x1_pool: np.ndarray, eps: float = 0.05, iters: int = 50) -> np.ndarray:
    """Computes an approximate OT coupling index."""
    n = x0_batch.shape[0]
    m = x1_pool.shape[0]
    
    a = np.ones(n) / n
    b = np.ones(m) / m
    
    M = np.sum((x0_batch[:, None, :] - x1_pool[None, :, :]) ** 2, axis=-1)
    
    P = sinkhorn_matrix(a, b, M, eps=eps, niter=iters)
    
    idx = np.argmax(P, axis=1)
    return idx

# --- CFM Variant Dispatcher ---

def get_coupling_x0_x1(
    x0_batch: torch.Tensor, 
    dataset: CFMLabDataset, # Use new dataset name
    cfg: CFMConfig,
    x_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Determines the target endpoint x1 based on the CFM variant's required coupling π(x0, x1).
    """
    device = cfg.device
    B = x0_batch.shape[0]
    coupling_mode = cfg.coupling
    
    # 1. Independent Endpoint CFM (q(x0)q(x1)) - Default
    if coupling_mode in ["independent", "variance_preserving_cfm", "conditional_flow_matcher"]:
        x1 = sample_prior(B, x_dim, dataset, cfg).to(device)
        return x0_batch, x1
    
    # 2. Exact OT-CFM / Schrödinger Bridge CFM (Minibatch Approximation)
    elif coupling_mode in ["schrodinger_bridge_cfm", "exact_optimal_transport_cfm"]:
        
        sinkhorn_params = {"eps": cfg.sinkhorn_eps, "iters": cfg.sinkhorn_iters}
        pool_size = min(len(dataset), max(2 * B, 512))
        pool_idx = np.random.choice(len(dataset.x), size=pool_size, replace=False)
        pool = dataset.x[pool_idx]
        
        x0_np = x0_batch.cpu().numpy()
        idx_map = batch_sinkhorn_coupling(x0_np, pool, **sinkhorn_params)
        
        x1 = torch.from_numpy(pool[idx_map]).to(device)
        
        return x0_batch, x1
        
    # 3. Target Conditional Flow Matcher (z=x1)
    elif coupling_mode == "target_cfm":
        x1 = x0_batch 
        x0 = sample_prior(B, x_dim, None, cfg).to(device) 
        
        return x0, x1
        
    else:
        raise NotImplementedError(f"Unknown CFM coupling mode: {coupling_mode}")
