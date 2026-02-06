import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List
from .config import CFMConfig

# Output type: (X, Cond, X_dim, Cond_dim, Column_Names)
DataTuple = Tuple[np.ndarray, Optional[np.ndarray], int, int, List[str]]

def _load_two_files(feature_file: str, metadata_file: str) -> DataTuple:
    """Handles loading features (x) and separate metadata (y) from two files."""
    try:
        df_features = pd.read_csv(feature_file, index_col=0).dropna()
        df_metadata = pd.read_csv(metadata_file, index_col=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e.filename}")

    # Align indices
    common_index = df_features.index.intersection(df_metadata.index)
    if len(common_index) == 0:
        raise ValueError("No matching indices found between feature and metadata files.")

    df_features = df_features.loc[common_index]
    df_metadata = df_metadata.loc[common_index]

    data_x = df_features.select_dtypes(include=[np.number]).values.astype(np.float32)
    
    # One-Hot Encode all metadata columns (Keep all categories)
    df_metadata_processed = pd.get_dummies(df_metadata)
    data_cond = df_metadata_processed.values.astype(np.float32)
        
    print(f"📚 Two-File Mode: {data_x.shape[0]} samples. Cond Dim: {data_cond.shape[1]}")
    return data_x, data_cond, data_x.shape[1], data_cond.shape[1], list(df_features.columns)


def _load_single_csv(data_file: str, condition_column_name: Optional[str]) -> DataTuple:
    """Handles loading data from one file, separating features and condition column."""
    try:
        df_full = pd.read_csv(data_file, index_col=0).dropna()
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {data_file}")

    # Handle unnamed index column if present
    if "Unnamed: 0" in df_full.columns:
        df_full = df_full.drop(columns=["Unnamed: 0"])
    
    df_full = df_full.dropna() 
    
    data_cond = None
    cond_dim = 0
    
    if condition_column_name:
        if condition_column_name not in df_full.columns:
            raise ValueError(f"Condition column '{condition_column_name}' not found in CSV.")

        df_cond = df_full[[condition_column_name]]
        
        # FIX: drop_first=False to ensure explicit encoding (A=[1,0], B=[0,1])
        # This matches the behavior of _load_two_files and avoids implicit "Control" bias
        df_cond_processed = pd.get_dummies(df_cond, columns=[condition_column_name], drop_first=False)
        
        data_cond = df_cond_processed.values.astype(np.float32)
        cond_dim = data_cond.shape[1]
        
        df_features = df_full.drop(columns=[condition_column_name])
        print(f"📄 Single-File Mode: Conditioning on '{condition_column_name}' (Dim: {cond_dim})")
    else:
        df_features = df_full
        print("📄 Single-File Mode (Unconditional).")

    # Select only numerical features for X
    df_features = df_features.select_dtypes(include=[np.number])
    data_x = df_features.values.astype(np.float32)

    if data_x.size == 0:
        raise ValueError("No numerical feature columns found after processing.")

    return data_x, data_cond, data_x.shape[1], cond_dim, list(df_features.columns)


def load_data_from_config(cfg: CFMConfig) -> DataTuple:
    """Dispatcher: Determines loading mode based on config."""
    if cfg.cond_path is not None:
        return _load_two_files(cfg.data_path, cfg.cond_path)
    else:
        return _load_single_csv(cfg.data_path, cfg.condition_column_name)
