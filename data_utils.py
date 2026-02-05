import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os

# Import the new configuration object and dataset
from .config import CFMConfig, CFMLabDataset 

# Define the structure for the output data
DataTuple = Tuple[np.ndarray, Optional[np.ndarray], int, int]

# --- Internal Loaders ---

def _load_two_files(feature_file: str, metadata_file: str) -> DataTuple:
    """Handles loading features (x) and separate metadata (y) from two files."""
    try:
        df_features = pd.read_csv(feature_file, index_col=0).dropna()
        df_metadata = pd.read_csv(metadata_file, index_col=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found in two-file mode: {e.filename}")

    common_index = df_features.index.intersection(df_metadata.index)
    df_features = df_features.reindex(common_index).dropna()
    df_metadata = df_metadata.reindex(common_index).dropna()

    data_x = df_features.select_dtypes(include=[np.number]).values.astype(np.float32)
    
    df_metadata_processed = pd.get_dummies(df_metadata)
    data_cond = df_metadata_processed.values.astype(np.float32)

    if data_x.shape[0] == 0:
        raise ValueError("No matching samples found after aligning and cleaning two files.")
        
    print(f"Loaded in Two-File Mode. Samples: {data_x.shape[0]}")
    return data_x, data_cond, data_x.shape[1], data_cond.shape[1]


def _load_single_csv(data_file: str, condition_column_name: Optional[str]) -> DataTuple:
    """Handles loading data from one file, separating features and condition column."""
    try:
        df_full = pd.read_csv(data_file, index_col=0).dropna()
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {data_file}")

    df_full = df_full.dropna() 
    
    data_cond = None
    cond_dim = 0
    df_features = df_full
    
    if condition_column_name and condition_column_name in df_full.columns:
        df_cond = df_full[[condition_column_name]]
        
        df_cond_processed = pd.get_dummies(df_cond, columns=[condition_column_name], drop_first=True)
        
        data_cond = df_cond_processed.values.astype(np.float32)
        cond_dim = data_cond.shape[1]
        
        df_features = df_full.drop(columns=[condition_column_name])
        print(f"Loaded in Single-File Mode. Condition column: {condition_column_name} (Dim: {cond_dim})")
        
    else:
        print("Loaded in Single-File Mode (Unconditional).")

    data_x = df_features.select_dtypes(include=[np.number]).values.astype(np.float32)
    x_dim = data_x.shape[1]

    if data_x.size == 0:
        raise ValueError("No numerical feature columns found after processing.")

    return data_x, data_cond, x_dim, cond_dim


# --- Public Dispatcher ---

def load_data_from_config(cfg: CFMConfig) -> DataTuple:
    """Determines loading mode based on config and loads data."""
    
    if cfg.cond_path is not None:
        if not os.path.exists(cfg.cond_path):
             raise FileNotFoundError(f"Conditioning file not found at: {cfg.cond_path}")
        return _load_two_files(cfg.data_path, cfg.cond_path)
    
    elif cfg.condition_column_name is not None:
        return _load_single_csv(cfg.data_path, cfg.condition_column_name)
    
    else:
        return _load_single_csv(cfg.data_path, None)

# --- Optional Preprocessing ---
def apply_clr_transform(data_counts: np.ndarray) -> np.ndarray:
    """Applies CLR transformation to count data (assumes all values >= 0)."""
    data_positive = data_counts + 1e-6 
    log_data = np.log(data_positive)
    geometric_mean = np.mean(log_data, axis=1, keepdims=True)
    return log_data - geometric_mean
