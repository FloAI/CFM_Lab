import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union

from .config import CFMConfig, seed_everything, CFMLabDataset
from .models import SimpleAutoencoder
from .data_utils import load_data_from_config
from .machine import CFMMachine
from .evaluation import CFMEvaluator

def generate_samples_from_csv(
    data_path: str,
    num_samples: int,
    condition_column_name: Optional[str] = None,
    cond_path: Optional[str] = None,
    evaluate: bool = False,
    **kwargs
) -> np.ndarray:
    
    # 1. Config
    cfg = CFMConfig(
        data_path=data_path, 
        condition_column_name=condition_column_name, 
        cond_path=cond_path, 
        **kwargs
    )
    os.makedirs(cfg.save_dir, exist_ok=True)
    seed_everything(cfg.seed)
    
    # 2. Load
    print(f"Loading data from {data_path}...")
    x, c, x_dim, cond_dim, col_names = load_data_from_config(cfg)
    cfg.cond_dim = cond_dim
    
    # 3. Prepare Data
    if c is not None:
        x_train, _, c_train, _ = train_test_split(x, c, test_size=0.1, random_state=cfg.seed)
    else:
        x_train, _ = train_test_split(x, test_size=0.1, random_state=cfg.seed)
        c_train = None
        
    ds = CFMLabDataset(x_train, c_train)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, 
                        num_workers=cfg.num_workers, drop_last=True)
    
    # 4. Train
    print(f"Initializing {cfg.model_type.upper()}...")
    ae = SimpleAutoencoder(x_dim, 128) if cfg.interpolant == "latent" else None
    machine = CFMMachine(cfg, x_dim, cond_dim, ae=ae)
    machine.train(loader, None)
    
    # 5. Sample
    print(f"Generating {num_samples} samples...")
    y_cond = None
    if cond_dim > 0:
        # Use first row of real data as template for condition
        y_cond = torch.from_numpy(c_train[0:1]).to(cfg.device).repeat(1, 1)
        
    samples = machine.sample(y_cond, num_samples).cpu().numpy()
    
    # 6. Evaluate
    if evaluate:
        print("Running Evaluation...")
        import pandas as pd
        real_df = pd.DataFrame(x_train, columns=col_names)
        syn_df = pd.DataFrame(samples, columns=col_names)
        
        evaluator = CFMEvaluator(real_df, syn_df)
        metrics = evaluator.generate_report(os.path.join(cfg.save_dir, "report"))
        print(f"   [+] Correlation Distance: {metrics.get('correlation_distance', 0):.4f}")
        print(f"   [+] Report saved to {cfg.save_dir}/report")
        
    return samples

__all__ = ["CFMConfig", "CFMMachine", "generate_samples_from_csv"]
