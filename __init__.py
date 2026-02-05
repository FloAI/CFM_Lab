import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union

# Import machine components using the new names
from .config import CFMConfig, seed_everything, CFMLabDataset
from .models import SimpleAutoencoder
from .data_utils import load_data_from_config
from .machine import CFMMachine

# --- Primary Public Function ---

def generate_samples_from_csv(
    data_path: str,
    num_samples: int,
    condition_column_name: Optional[str] = None,
    cond_template_vector: Optional[Union[np.ndarray, torch.Tensor]] = None,
    **cfm_params: Dict[str, Any]
) -> np.ndarray:
    """
    A simplified, one-shot function to load data from CSV, train a CFM_Lab model, 
    and generate new samples immediately.
    """
    # 1. Initialize Configuration
    cfg = CFMConfig(
        data_path=data_path,
        condition_column_name=condition_column_name,
        **cfm_params
    )
    os.makedirs(cfg.save_dir, exist_ok=True)
    seed_everything(cfg.seed)

    # 2. Load Data and Dimensions
    print(f"Loading data from {cfg.data_path}...")
    data, cond, x_dim, cond_dim = load_data_from_config(cfg)
    cfg.cond_dim = cond_dim

    # --- Optional Preprocessing (e.g., CLR) ---
    # NOTE: If CLR is needed for compositional data, it should be applied here.
    # if cfg.interpolant in ["log", "alpha_flow"] and x_dim > 0:
    #     from .data_utils import apply_clr_transform
    #     print("Applying Centered Log-Ratio (CLR) transformation...")
    #     data = apply_clr_transform(data) 
    #     print(f"CLR successful. Feature dimension remains ({data.shape[1]}).")
    # ---------------------------------------------

    # 3. Prepare Datasets 
    if cond is not None:
        split_data = train_test_split(data, cond, test_size=0.1, random_state=cfg.seed)
        train_x, _, train_c, _ = split_data
    else:
        train_x, _ = train_test_split(data, test_size=0.1, random_state=cfg.seed)
        train_c = None

    train_ds = CFMLabDataset(train_x, train_c) # Use new dataset name
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    
    # 4. Initialize CFMMachine
    ae = SimpleAutoencoder(x_dim=x_dim, latent_dim=cfg.latent_dim, hidden=cfg.ae_hidden) if cfg.interpolant == "latent" else None
    cfm_machine = CFMMachine(cfg, x_dim, cfg.cond_dim, ae=ae)
    
    print(f"Training model ({cfm_machine.model.__class__.__name__})...")

    # 5. Train the Model
    cfm_machine.train(train_loader, None)

    # 6. Prepare Conditioning Vector for Sampling
    final_cond_vector = None
    if cfg.cond_dim > 0:
        if cond_template_vector is not None:
            if isinstance(cond_template_vector, np.ndarray):
                cond_template_vector = torch.from_numpy(cond_template_vector).float()
            final_cond_vector = cond_template_vector.to(cfg.device).view(1, -1)
        else:
            # Default: Use the first processed vector in the training set as the template
            if train_ds.cond is not None:
                y_template_np = train_ds.cond[0:1]
                final_cond_vector = torch.from_numpy(y_template_np).to(cfg.device).view(1, -1)
            else:
                 print("Warning: Conditional column requested, but no conditional data found.")


    # 7. Generate Samples
    print(f"Generating {num_samples} samples...")
    generated_samples = cfm_machine.sample(
        y_cond=final_cond_vector, 
        num_samples=num_samples
    )

    return generated_samples.cpu().numpy()


__all__ = [
    "CFMConfig", "CFMLabDataset", "seed_everything", "CFMMachine", 
    "generate_samples_from_csv"
]
