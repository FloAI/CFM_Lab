import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any

# Import the core, streamlined function
from CFM_Lab import generate_samples_from_csv, seed_everything

# --- 1. DATA SIMULATION (High Dimensional) ---
def create_high_dim_ot_data(N=800, D=100):
    """Generates a CSV simulating 100 features and a categorical condition."""
    seed_everything(42)
    
    # Simulate high-dimensional data (e.g., gene expression or high-res images)
    # Data is complex, suitable for a Transformer
    features = np.random.randn(N, D) * 10 + np.sin(np.linspace(0, 10, N))[:, None]
    features = features.astype(np.float32)
    
    # Create a categorical condition
    status = np.random.choice(['Group_A', 'Group_B'], size=N, p=[0.5, 0.5])
    
    df = pd.DataFrame(features, columns=[f'Feature_{i}' for i in range(D)])
    df['Condition_Type'] = status
    
    os.makedirs("data_ot_high_dim", exist_ok=True)
    csv_path = "data_ot_high_dim/high_dim_data.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Dummy high-dimensional data created at: {csv_path}")
    return csv_path

# --- 2. MAIN EXECUTION ---

if __name__ == "__main__":
    
    DATA_CSV_PATH = create_high_dim_ot_data()
    NUM_GENERATE = 15
    
    # --- Configuration Overrides for Advanced OT-CFM ---
    
    CONFIG_OVERRIDES: Dict[str, Any] = {
        # --- FIX: Ensure 'save_dir' is present ---
        "save_dir": "./cfmlab_ot_transformer_output", 
        
        # Data settings
        "data_path": DATA_CSV_PATH,
        "condition_column_name": "Condition_Type", 
        
        # Model: Use Transformer for high dimensions
        "model_type": "transformer",
        
        # Flow settings: Use OT-CFM coupling (Schrödinger Bridge approx)
        "interpolant": "schrodinger_bridge_cfm", # Sets the coupling mode internally
        "sinkhorn_eps": 0.05,                     # Regularization for Sinkhorn OT
        "sinkhorn_iters": 50,                     # Iterations for Sinkhorn OT
        
        # Sampling settings
        "solver": "rk4",                          # High-accuracy deterministic solver
        "epochs": 10,                             # Reduced for demo speed
        "device": "cpu",                     
    }

    print("\n--- Starting Transformer OT-CFM Generation ---")
    
    try:
        # The function handles all data loading, model building, and training
        generated_data_np = generate_samples_from_csv(
            **CONFIG_OVERRIDES,
            num_samples=NUM_GENERATE,
        )
        
        # --- 3. SAVING RESULTS ---
        
        SAVE_DIR_PATH = CONFIG_OVERRIDES["save_dir"]
        os.makedirs(SAVE_DIR_PATH, exist_ok=True)
        
        output_path = os.path.join(SAVE_DIR_PATH, "synthetic_ot_transformer_data.npy")
        np.save(output_path, generated_data_np)
        
        # --- Output Metrics ---
        print("\n--- Generation Complete ---")
        print(f"1. Model Used: {CONFIG_OVERRIDES['model_type']}")
        print(f"2. Coupling: {CONFIG_OVERRIDES['interpolant']}")
        print(f"3. Final shape of generated data: {generated_data_np.shape}")
        print(f"4. Average sample mean (all features): {generated_data_np.mean():.4f}")
        print(f"5. Samples saved to: {output_path}")
        
    except Exception as e:
        print(f"\nFATAL ERROR during execution: {e}")
        print("Ensure the library files and dependencies are correctly set up, and check the 'save_dir' path.")
