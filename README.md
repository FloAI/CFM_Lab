
# CFM_Lab

**Conditional Flow Matching for Science & Tabular Data**

CFM_Lab is a modular library for high-dimensional generative modeling using Conditional Flow Matching (CFM). Designed for complex datasets such as metagenomics and clinical records, it supports various path interpolants, optimal transport (OT) based couplings, and both deterministic (ODE) and stochastic (SDE) integration strategies.

## 1. Setup and Dependencies

### Prerequisites

Ensure the following external libraries are installed in your environment:

```bash
pip install numpy pandas torch scikit-learn tqdm matplotlib seaborn

```

### Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/yourusername/CFM_Lab.git
cd CFM_Lab
pip install -e .

```

## 2. Quick Start

The library provides a single entry point, `generate_samples_from_csv`, which handles data loading, model training, and sampling in a unified pipeline.

### Python API Example

```python
import torch
from CFM_Lab import generate_samples_from_csv

# Define configuration
DATA_FILE = "data/clinical_metrics.csv"
OUTPUT_DIR = "./results"

# Execute the pipeline
synthetic_data = generate_samples_from_csv(
    data_path=DATA_FILE,
    num_samples=100,
    
    # Data Configuration
    condition_column_name="Diagnosis", 
    
    # Model Configuration
    model_type="mlp",
    interpolant="schrodinger_bridge_cfm",
    prior_sampler="gaussian",
    
    # Training Configuration
    loss_type="huber",  # Robust loss for noisy data
    epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir=OUTPUT_DIR,
    evaluate=True
)

```

## 3. Data Loading Modes

The library supports three distinct data loading strategies:

| Mode | Configuration Parameters | Description |
| --- | --- | --- |
| **Single File (Conditional)** | `data_path`, `condition_column_name` | Features and conditional labels reside in the same CSV file. Labels are automatically one-hot encoded. |
| **Two Files (Conditional)** | `data_path`, `cond_path` | Features and metadata are stored in separate files with aligned indices. Suitable for high-dimensional omics data. |
| **Unconditional** | `data_path`, `condition_column_name=None` | Learns the raw data distribution without conditional dependencies. |

## 4. Model Architectures

The velocity field approximation () can be parameterized using different neural network architectures depending on the data dimensionality ().

| Architecture | Key | Recommended Usage | Description |
| --- | --- | --- | --- |
| **Multi-Layer Perceptron** | `"mlp"` | Low Dimensions () | Standard feed-forward network. Efficient for tabular data with local feature interactions. |
| **Transformer Encoder** | `"transformer"` | High Dimensions () | Utilizes self-attention to model long-range dependencies between features. Essential for structured data. |
| **Latent Autoencoder** | `"autoencoder_latent"` | Manifold Data | Learns the flow dynamics within a compressed latent space. |

## 5. Flow Matching Configuration

### Coupling Strategies & Priors

The coupling strategy determines how source samples (noise) are paired with target samples (data).

| Variant | Key | Description |
| --- | --- | --- |
| **Independent CFM** | `"independent"` | Standard linear path where initial and final states are independent. |
| **Optimal Transport** | `"exact_optimal_transport_cfm"` | Deterministic coupling minimizing transport cost for straight trajectories. |
| **Schrödinger Bridge** | `"schrodinger_bridge_cfm"` | Stochastic OT using Sinkhorn approximation for entropic regularization. |
| **Target CFM** | `"target_cfm"` | Fixes the data endpoint and samples the noise endpoint from the prior. |

**Prior Distributions:**

* `"gaussian"`: Standard normal distribution (Default).
* `"dirichlet"`: Useful for compositional data (simplex).
* `"empirical_shuffle"`: Samples from the empirical data distribution.

### Path Geometries

| Interpolant | Key | Use Case |
| --- | --- | --- |
| **Linear** | `"linear"` | General purpose Euclidean paths. |
| **Spherical** | `"spherical"` | Data lying on a hypersphere (normalized data). |
| **Log-Euclidean** | `"log"` | Compositional or count data. Ensures positivity constraints. |
| **Variance Preserving** | `"vp"` | Mimics the noise schedules found in diffusion models. |
| **Sparse-Aware** | `"sparseaware"` | Introduces smoothing for sparse datasets to improve stability near zero. |

### Loss Functions

The library includes specialized losses for robust regression and scientific data types.

| Loss Type | Key | Application |
| --- | --- | --- |
| **Standard** | `"l2"`, `"l1"` | General purpose regression. |
| **Robust** | `"huber"`, `"tukey"` | Datasets containing outliers or heavy tails. |
| **Count Data** | `"poisson"`, `"zero_inflated"` | Biological count data (e.g., RNA-seq, microbiome). |
| **Divergence** | `"kl"`, `"js"`, `"cosine"` | Distribution matching and directional alignment. |

## 6. Library Architecture

The `generate_samples_from_csv` function automates the following workflow stages:

1. **Configuration:** Parses user overrides and initializes the `CFMConfig` object.
2. **Data Ingestion:** Loads CSV files, performs one-hot encoding, and prepares the `CFMLabDataset`.
3. **Training:**
* **Coupling:** Pairs batch noise () with data () via the selected strategy.
* **Loss Calculation:** Computes the regression loss between the predicted velocity  and the target vector field.


4. **Sampling:** Integrates the learned velocity field backward in time from  (prior) to  (data) using the specified numerical solver (`rk4`, `euler`, `heun`).
5. **Evaluation:** (Optional) Generates correlation matrices and distribution overlap plots to assess synthesis quality.

## License

This project is licensed under the MIT License.
