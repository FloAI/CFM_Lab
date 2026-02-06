import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class CFMEvaluator:
    """
    Generates comparison reports between Real and Synthetic data.
    """
    def __init__(self, real_df: pd.DataFrame, syn_df: pd.DataFrame):
        self.real = real_df
        self.syn = syn_df
        # Align columns to ensure we are comparing the same features
        cols = self.real.columns.intersection(self.syn.columns)
        self.real = self.real[cols]
        self.syn = self.syn[cols]

    def generate_report(self, save_dir: str) -> Dict[str, float]:
        """
        Orchestrates the creation of plots and calculation of metrics.
        """
        os.makedirs(save_dir, exist_ok=True)
        stats = {}

        # 1. Correlation Matrix Comparison
        # Checks if the model captured the relationships between variables
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(self.real.corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Real Correlation")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(self.syn.corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Synthetic Correlation")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "correlation.png"))
        plt.close()

        # 2. Distribution Overlays (First 6 features)
        # Checks if the model captured the range and shape of variables
        features = self.real.columns[:6]
        n_feats = len(features)
        rows = (n_feats + 2) // 3
        
        plt.figure(figsize=(15, 3 * rows))
        for i, col in enumerate(features):
            plt.subplot(rows, 3, i+1)
            try:
                sns.kdeplot(self.real[col], label='Real', fill=True, color='blue', alpha=0.3)
                sns.kdeplot(self.syn[col], label='Syn', fill=True, color='orange', alpha=0.3)
            except Exception: 
                # Fallback for constant values or errors in KDE
                pass
            plt.title(col)
            if i == 0: plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distributions.png"))
        plt.close()

        # 3. Statistical Metrics
        # Correlation Matrix Distance (Frobenius Norm)
        corr_dist = np.linalg.norm(self.real.corr() - self.syn.corr())
        stats['correlation_distance'] = float(corr_dist)
        
        # Mean Absolute Error of Means (Did we get the centers right?)
        mae_means = (self.real.mean() - self.syn.mean()).abs().mean()
        stats['mae_means'] = float(mae_means)
        
        # Save metrics to text file
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v:.4f}\n")
        
        return stats
