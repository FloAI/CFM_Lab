import random
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset

# --- Configuration Class ---
class CFMConfig:
    """Configuration for the CFM pipeline in CFM_Lab."""
    def __init__(self, **kwargs):
        self.data_path: str = kwargs.pop("data_path", "data/features.csv")
        self.cond_path: Optional[str] = kwargs.pop("cond_path", None) 
        self.condition_column_name: Optional[str] = kwargs.pop("condition_column_name", None)
        self.batch_size: int = kwargs.pop("batch_size", 128)
        self.num_workers: int = kwargs.pop("num_workers", 4)
        self.device: str = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and Dimensionality Settings
        self.model_type: str = kwargs.pop("model_type", "mlp") # 'mlp' or 'transformer'
        self.hidden_dim: int = kwargs.pop("hidden_dim", 512)
        self.num_layers: int = kwargs.pop("num_layers", 4)
        self.time_emb_dim: int = kwargs.pop("time_emb_dim", 128)
        self.cond_dim: Optional[int] = kwargs.pop("cond_dim", None)
        
        # Interpolants / Paths
        self.interpolant: str = kwargs.pop("interpolant", "linear")
        self.interpolant_params: Dict[str, Any] = kwargs.pop("interpolant_params", {})
        self.alpha_flow_alpha: float = kwargs.pop("alpha_flow_alpha", 1.0)
        
        # Coupling
        self.coupling: str = kwargs.pop("coupling", "independent")
        self.prior_sampler: str = kwargs.pop("prior_sampler", "gaussian")
        self.prior_params: Dict[str, Any] = kwargs.pop("prior_params", {"std": 1.0, "dirichlet_alpha": 0.5})
        self.sinkhorn_eps: float = kwargs.pop("sinkhorn_eps", 0.05)
        self.sinkhorn_iters: int = kwargs.pop("sinkhorn_iters", 50)
        
        # SDE/Flow parameters
        self.flow_variant: str = kwargs.pop("flow_variant", "deterministic")
        self.stochastic_noise_scale: float = kwargs.pop("stochastic_noise_scale", 0.1)
        
        # Loss
        self.loss_type: str = kwargs.pop("loss_type", "l2")
        self.loss_params: Dict[str, Any] = kwargs.pop("loss_params", {})
        self.ef_vfm_family: Optional[str] = kwargs.pop("ef_vfm_family", None)
        
        self.lr: float = kwargs.pop("lr", 1e-4)
        self.weight_decay: float = kwargs.pop("weight_decay", 0.0)
        self.epochs: int = kwargs.pop("epochs", 50)
        self.grad_clip: Optional[float] = kwargs.pop("grad_clip", 5.0)
        self.log_every: int = kwargs.pop("log_every", 100)
        self.save_dir: str = kwargs.pop("save_dir", "./cfmlab_checkpoints")
        self.seed: int = kwargs.pop("seed", 42)
        self.use_amp: bool = kwargs.pop("use_amp", False)
        
        # Sampling parameters
        self.solver: str = kwargs.pop("solver", "rk4")
        self.sample_steps: int = kwargs.pop("sample_steps", 50)
        self.prior_std: float = kwargs.pop("prior_std", 1.0)
        self.latent_dim: int = kwargs.pop("latent_dim", 128)
        self.ae_hidden: int = kwargs.pop("ae_hidden", 512)
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

# --- Utilities ---
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _ensure_t_tensor(t: Union[float, int, torch.Tensor]) -> torch.Tensor:
    if isinstance(t, (float, int)):
        t = torch.tensor([t], dtype=torch.float32)
    return t.float()

# --- Dataset ---
class CFMLabDataset(Dataset):
    """Simple dataset wrapper. Returns: x (float32, D), y (float32, C or empty)"""
    def __init__(self, data: np.ndarray, cond: Optional[np.ndarray] = None):
        assert isinstance(data, np.ndarray)
        self.x = data.astype(np.float32)
        self.cond = None if cond is None else cond.astype(np.float32)
        if self.cond is not None:
            assert len(self.cond) == len(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.cond[idx] if self.cond is not None else np.zeros((0,), dtype=np.float32)
        return x, y
