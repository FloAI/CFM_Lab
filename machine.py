import torch
import torch.optim as optim
from tqdm import tqdm
from .config import CFMConfig
from .models import build_velocity_model
from .coupling import get_coupling_x0_x1
from .paths import PATH_REGISTRY
from .losses import LossRegistry

class CFMMachine:
    def __init__(self, cfg: CFMConfig, x_dim: int, cond_dim: int, ae=None):
        self.cfg = cfg
        self.device = cfg.device
        self.x_dim = x_dim
        
        self.model = build_velocity_model(x_dim, cond_dim, cfg, ae).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.loss_fn = LossRegistry().loss_fns[cfg.loss_type]
        
        # Path/Interpolant Setup
        self.interp_fn, self.deriv_fn = PATH_REGISTRY.get(cfg.interpolant, PATH_REGISTRY["linear"])
        self.interp_params = cfg.interpolant_params.copy()
        
        if cfg.interpolant == "latent":
            assert ae is not None
            self.interp_params.update({"encoder": ae.encode, "decoder": ae.decode})

    def train(self, loader, val_loader=None):
        self.model.train()
        print(f" Training on {self.device}...")
        
        for epoch in range(self.cfg.epochs):
            epoch_loss = 0
            for x1, c in loader:
                x1 = x1.to(self.device)
                c = c.to(self.device) if c.shape[1] > 0 else None
                B = x1.shape[0]
                
                # 1. Sample Noise/Prior
                x0 = torch.randn_like(x1)
                
                # 2. Coupling (Handles Independent, Exact OT, and Soft SB)
                # x0 is re-paired with x1 (or x1 is swapped) based on strategy
                if self.cfg.coupling != "independent":
                    x0, x1 = get_coupling_x0_x1(x0, loader.dataset, self.cfg, self.x_dim)
                
                # 3. Sample Time
                t = torch.rand(B, device=self.device)
                
                # 4. Interpolate Flow
                x_t = self.interp_fn(x0, x1, t, **self.interp_params)
                v_target = self.deriv_fn(x0, x1, t, **self.interp_params)
                
                # 5. Predict & Loss
                v_pred = self.model(x_t, t, c)
                loss = self.loss_fn(v_pred, v_target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

    @torch.no_grad()
    def sample(self, y_cond, num_samples):
        self.model.eval()
        x = torch.randn(num_samples, self.x_dim, device=self.device)
        steps = self.cfg.sample_steps
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        
        if y_cond is not None and y_cond.shape[0] == 1:
            y_cond = y_cond.repeat(num_samples, 1)
            
        # RK4 Solver
        for i in range(steps):
            t = times[i].repeat(num_samples)
            
            if self.cfg.solver == "rk4":
                k1 = self.model(x, t, y_cond)
                k2 = self.model(x + 0.5*dt*k1, t + 0.5*dt, y_cond)
                k3 = self.model(x + 0.5*dt*k2, t + 0.5*dt, y_cond)
                k4 = self.model(x + dt*k3, t + dt, y_cond)
                x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                # Euler fallback
                v = self.model(x, t, y_cond)
                x = x + v * dt
                
        return x
