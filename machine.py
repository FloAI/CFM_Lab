import os
import numpy as np
import math
import torch
from typing import Optional, Tuple, Callable, Union, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

# Updated Imports
from .config import CFMConfig, CFMLabDataset 
from .models import SimpleAutoencoder, build_velocity_model
from .paths import PATH_REGISTRY
from .coupling import sample_prior, batch_sinkhorn_coupling, get_coupling_x0_x1
from .losses import LossRegistry

# Define the mapping from user-facing flow names to internal coupling strategies
CFM_COUPLING_REGISTRY = {
    # Independent Endpoints (q(x0)q(x1))
    "conditional_flow_matcher": "independent",
    "variance_preserving_cfm": "independent",
    "alpha_flow": "independent",
    "linear": "independent",
    
    # OT/Schrodinger Bridge Coupling (π(x0, x1))
    "exact_optimal_transport_cfm": "schrodinger_bridge_cfm", 
    "schrodinger_bridge_cfm": "schrodinger_bridge_cfm",
    
    # Target Conditional Flow Matcher (q(x0)=N(0,I), q(x1)=P_data)
    "target_conditional_flow_matcher": "target_cfm", 
}


class CFMMachine:
    """
    Central machine for Conditional Flow Matching (CFM) for CFM_Lab.
    Handles low and high-dimensional data flows.
    """
    def __init__(self, cfg: CFMConfig, x_dim: int, cond_dim: int, ae: Optional[SimpleAutoencoder]=None):
        self.cfg = cfg
        self.device = cfg.device
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.ae = ae.to(self.device) if ae else None
        
        # Velocity model automatically selects MLP/Transformer based on cfg.model_type
        self.model = build_velocity_model(x_dim, cond_dim, cfg, self.ae).to(self.device)
        self.loss_registry = LossRegistry()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
        # Use torch.amp.GradScaler
        self.scaler = torch.amp.GradScaler(self.cfg.device, enabled=cfg.use_amp)
        
        # Path Setup: Select interpolant/derivative based on configured flow/path name
        self.interp_fn, self.deriv_fn = PATH_REGISTRY.get(cfg.interpolant, PATH_REGISTRY["linear"])
        self.interp_params = cfg.interpolant_params.copy()
        
        if cfg.interpolant == "alpha_flow":
            self.interp_params["alpha"] = cfg.alpha_flow_alpha
            
        if cfg.interpolant == "latent":
            assert ae is not None, "Autoencoder required for latent interpolant"
            self.interp_params["encoder"] = self.ae.encode
            self.interp_params["decoder"] = self.ae.decode

        # ODE/SDE Solvers
        self.SOLVER_STEP = {
            "euler": self._euler_ode_step,
            "heun": self._heun_step,
            "rk4": self._rk4_step,
            "euler_maruyama": self._euler_maruyama_step,
        }

    # --- Private ODE/SDE Step Implementations (Unchanged) ---
    
    def _euler_ode_step(self, f: Callable, x: torch.Tensor, t: torch.Tensor, dt: float, y: Optional[torch.Tensor]):
        return x + dt * f(x, t, y)

    def _heun_step(self, f: Callable, x: torch.Tensor, t: torch.Tensor, dt: float, y: Optional[torch.Tensor]):
        k1 = f(x, t, y)
        mid = x + dt * k1
        k2 = f(mid, t + dt, y)
        return x + dt * 0.5 * (k1 + k2)

    def _rk4_step(self, f: Callable, x: torch.Tensor, t: torch.Tensor, dt: float, y: Optional[torch.Tensor]):
        k1 = f(x, t, y)
        k2 = f(x + 0.5*dt*k1, t + 0.5*dt, y)
        k3 = f(x + 0.5*dt*k2, t + 0.5*dt, y)
        k4 = f(x + dt*k3, t + dt, y)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def _euler_maruyama_step(self, f: Callable, x: torch.Tensor, t: torch.Tensor, dt: float, y: Optional[torch.Tensor]):
        """SDE Solver: Euler-Maruyama step (reverse sampling direction)."""
        v = f(x, t, y)
        
        if self.cfg.flow_variant != "stochastic":
            diffusion_scale = 0.0
        else:
            diffusion_scale = self.cfg.stochastic_noise_scale
        
        dw = torch.randn_like(x) * math.sqrt(abs(dt)) 
        
        return x + v * dt + diffusion_scale * dw

    # --- Core Velocity/Loss Utilities ---
    
    def _compute_target_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self.deriv_fn(x0, x1, t, **self.interp_params)
        
        if self.cfg.flow_variant == "stochastic":
            noise = torch.randn_like(v) * self.cfg.stochastic_noise_scale
            v = v + noise
        return v
        
    # --- Coupling Utility (Uses external dispatcher) ---

    def _get_coupling_x1(self, x0: torch.Tensor, dataset: CFMLabDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines the paired (x0, x1) using the CFM coupling dispatcher.
        Returns the potentially re-sampled/re-paired (x0, x1) tensors.
        """
        coupling_key = CFM_COUPLING_REGISTRY.get(self.cfg.interpolant, "independent")
        
        original_coupling = self.cfg.coupling
        self.cfg.coupling = coupling_key
        
        # Pass x_dim to coupling dispatcher
        x0_paired, x1_paired = get_coupling_x0_x1(x0, dataset, self.cfg, self.x_dim)
        
        self.cfg.coupling = original_coupling
        
        return x0_paired, x1_paired

    # --- Public Methods (train, evaluate, sample) ---

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader]):
        """Runs the training loop."""
        best_val = float("inf")
        global_step = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for xb_np, yb_np in pbar:
                xb = xb_np.to(self.device)
                yb = yb_np.to(self.device) if yb_np.numel() > 0 else None
                
                x0, x1 = self._get_coupling_x1(xb, train_loader.dataset)
                
                t = torch.rand(xb.shape[0], device=self.device)
                x_t = self.interp_fn(x0, x1, t, **self.interp_params)
                v_target = self._compute_target_velocity(x0, x1, t)

                v_pred = self.model(x_t, t, yb)
                loss_fn = self.loss_registry.loss_fns[self.cfg.loss_type]
                loss = loss_fn(v_pred, v_target, **self.cfg.loss_params)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                global_step += 1
                if global_step % self.cfg.log_every == 0:
                    pbar.set_postfix({"loss": f"{loss.detach().cpu().item():.4f}"})

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"\nEpoch {epoch} validation loss: {val_loss:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.cfg.save_dir, "best.pth"))
                    print("-> Saved best checkpoint.")

        torch.save(self.model.state_dict(), os.path.join(self.cfg.save_dir, "last.pth"))
        return self.model

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluates the model on a data loader."""
        self.model.eval()
        loss_fn = self.loss_registry.loss_fns[self.cfg.loss_type]
        total = 0.0
        n = 0
        
        original_coupling = self.cfg.coupling
        self.cfg.coupling = "independent" # Use independent coupling for consistent evaluation
        
        for xb_np, yb_np in loader:
            xb = xb_np.to(self.device)
            yb = yb_np.to(self.device) if yb_np.numel() > 0 else None
            
            x0, x1 = get_coupling_x0_x1(xb, loader.dataset, self.cfg, self.x_dim)
            t = torch.rand(xb.shape[0], device=self.device)
            
            x_t = self.interp_fn(x0, x1, t, **self.interp_params)
            v_target = self._compute_target_velocity(x0, x1, t)
            v_pred = self.model(x_t, t, yb)
            
            loss = loss_fn(v_pred, v_target, **self.cfg.loss_params)
            total += float(loss.cpu().item()) * xb.shape[0]
            n += xb.shape[0]

        self.cfg.coupling = original_coupling
        return total / n if n > 0 else float("inf")

    @torch.no_grad()
    def sample(self, y_cond: Optional[torch.Tensor], num_samples: int) -> torch.Tensor:
        """Generates samples by integrating the flow backward."""
        self.model.eval()
        steps = self.cfg.sample_steps
        solver = self.cfg.solver
        
        if self.cfg.flow_variant == "stochastic" and solver in ["euler", "heun", "rk4"]:
             solver = "euler_maruyama"
             
        step_fn = self.SOLVER_STEP.get(solver, self.SOLVER_STEP["rk4"])
        
        x = sample_prior(num_samples, self.x_dim, None, self.cfg).to(self.device)
        
        ts = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        dt = ts[0] - ts[1]
        
        yb = y_cond.to(self.device) if y_cond is not None else None
        if yb is not None and yb.shape[0] == 1 and num_samples > 1:
             yb = yb.repeat(num_samples, 1)

        for i in tqdm(range(steps), desc="Sampling"):
            t_cur = ts[i].expand(num_samples)
            x = step_fn(lambda xx, tt, yy: self.model(xx, tt, yy), x, t_cur, -dt, yb)

        return x
