import numpy as np
import torch
from typing import Tuple, Union, Optional
from .config import CFMConfig, CFMLabDataset

def sample_prior(batch_size: int, x_dim: int, dataset: Optional[CFMLabDataset], cfg: CFMConfig) -> torch.Tensor:
    """Samples from the prior distribution p(x0)."""
    method = cfg.prior_sampler
    device = cfg.device
    
    if method == "gaussian":
        std = cfg.prior_params.get("std", 1.0)
        return torch.randn(batch_size, x_dim, device=device) * std
        
    elif method == "empirical_shuffle":
        assert dataset is not None, "Dataset required for empirical_shuffle prior."
        idx = np.random.choice(len(dataset), size=batch_size, replace=False)
        return torch.from_numpy(dataset.x[idx]).to(device)
        
    else:
        raise NotImplementedError(f"Unknown prior sampler: {method}")

def sinkhorn_matrix(a: np.ndarray, b: np.ndarray, M: np.ndarray, eps: float, niter: int) -> np.ndarray:
    """Computes the Sinkhorn-Knopp transport matrix."""
    K = np.exp(-M / eps)
    u = np.ones_like(a)
    v = np.ones_like(b)
    
    for _ in range(niter):
        v = b / (K.T.dot(u) + 1e-16)
        u = a / (K.dot(v) + 1e-16)
        
    return np.diag(u).dot(K).dot(np.diag(v))

def batch_sinkhorn_coupling(
    x0_batch: np.ndarray, 
    x1_pool: np.ndarray, 
    eps: float = 0.05, 
    iters: int = 50,
    stochastic: bool = False
) -> np.ndarray:
    """
    Computes OT coupling indices.
    - stochastic=True: Samples x1 based on transport plan probabilities (Schrödinger Bridge).
    - stochastic=False: Selects x1 with highest probability (Exact OT Approx).
    """
    n = x0_batch.shape[0]
    m = x1_pool.shape[0]
    
    # Uniform marginals
    a = np.ones(n) / n
    b = np.ones(m) / m
    
    # Squared Euclidean Distance Cost
    M = np.sum((x0_batch[:, None, :] - x1_pool[None, :, :]) ** 2, axis=-1)
    
    P = sinkhorn_matrix(a, b, M, eps=eps, niter=iters)
    
    if stochastic:
        # Normalize rows to sum to 1 to treat as probabilities
        row_sums = P.sum(axis=1, keepdims=True) + 1e-16
        probs = P / row_sums
        # Sample index for each row i based on probs[i]
        idx = np.array([np.random.choice(m, p=probs[i]) for i in range(n)])
    else:
        # Hard assignment
        idx = np.argmax(P, axis=1)
        
    return idx

def get_coupling_x0_x1(
    x0_batch: torch.Tensor, 
    dataset: CFMLabDataset, 
    cfg: CFMConfig,
    x_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determines the target endpoint x1 based on coupling strategy."""
    device = cfg.device
    B = x0_batch.shape[0]
    coupling_mode = cfg.coupling
    
    # 1. Independent Coupling (Standard Flow Matching)
    if coupling_mode in ["independent", "variance_preserving_cfm", "conditional_flow_matcher"]:
        x1 = sample_prior(B, x_dim, dataset, cfg).to(device)
        return x0_batch, x1
    
    # 2. Optimal Transport / Schrödinger Bridge
    elif coupling_mode in ["schrodinger_bridge_cfm", "exact_optimal_transport_cfm"]:
        
        sinkhorn_params = {"eps": cfg.sinkhorn_eps, "iters": cfg.sinkhorn_iters}
        
        # Sample a pool of real data to match against
        pool_size = min(len(dataset), max(2 * B, 512))
        pool_idx = np.random.choice(len(dataset.x), size=pool_size, replace=False)
        pool = dataset.x[pool_idx]
        
        x0_np = x0_batch.cpu().numpy()
        
        # FIX: Distinguish between Soft (SB) and Hard (OT) matching
        is_stochastic = (coupling_mode == "schrodinger_bridge_cfm")
        
        idx_map = batch_sinkhorn_coupling(x0_np, pool, stochastic=is_stochastic, **sinkhorn_params)
        
        x1 = torch.from_numpy(pool[idx_map]).to(device)
        return x0_batch, x1
        
    # 3. Target Conditional (Fixed x1, random x0)
    elif coupling_mode == "target_cfm":
        x1 = x0_batch # The batch itself is x1
        x0 = sample_prior(B, x_dim, None, cfg).to(device)
        return x0, x1
        
    else:
        raise NotImplementedError(f"Unknown coupling mode: {coupling_mode}")
