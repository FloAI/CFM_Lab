
# 🧬 CFM_Lab

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

A comprehensive and modular library for high-dimensional generative modeling using **Conditional Flow Matching (CFM)**. This tool is designed to handle complex datasets like those found in metagenomics, supporting various path interpolants, advanced **OT-based coupling**, and both **ODE** and **SDE** integration for sampling.

-----

## 1\. 🛠️ Setup and Dependencies

### Prerequisites

Ensure you have the required external libraries installed:

```bash
pip install numpy pandas torch scikit-learn tqdm
```

### Library Structure

The library requires a specific package structure. Ensure your repository contains the `metagenomics_cfm` directory with all the core Python files (e.g., `machine.py`, `config.py`, `coupling.py`).

-----

## 2\.  Data Preparation and Loading Modes

The library loads data directly from **CSV files** via the `data_path` parameter. It automatically handles feature selection, categorical **One-Hot Encoding**, and train/validation splitting.

### Data Loading Modes

You must choose one of the following modes by setting the appropriate parameters in your configuration:

| Mode | Configuration Parameters | Description |
| :--- | :--- | :--- |
| **Mode 1: Single File + Column** | `data_path='file.csv'`, **`condition_column_name='Age_Group'`** | Features ($\mathbf{X}$) and condition ($\mathbf{Y}$) are in one CSV file. **Recommended method.** |
| **Mode 2: Two Separate Files** | `data_path='features.csv'`, **`cond_path='metadata.csv'`** | $\mathbf{X}$ and $\mathbf{Y}$ are split across two aligned CSV files. |
| **Mode 3: Unconditional** | `data_path='file.csv'`, **`condition_column_name=None`** | Training a generative model using only features ($\mathbf{X}$), ignoring any conditional columns. |

-----

## 3\.  Core Workflow: One-Shot Generation

Use the `generate_samples_from_csv` function for the fastest end-to-end workflow: loading, training (with a split), and sampling in one call.

### Conditional Example (Mode 1)

```python
from metagenomics_cfm import generate_samples_from_csv

# 1. Define input and desired output
DATA_FILE = "data/user_metrics.csv" 
NUM_SAMPLES = 50

# 2. Run the complete pipeline
synthetic_data = generate_samples_from_csv(
    data_path=DATA_FILE,
    num_samples=NUM_SAMPLES,
    
    # Mode 1 Setup
    condition_column_name="Device_Type", 
    
    # Training Overrides
    epochs=40, 
    device='cuda',
)
```

### Conditional Sampling Control

To control the generated output, you must provide the exact **one-hot template vector** ($\mathbf{y}_{template}$) that represents the desired category.

```python
# Assuming the vector [0.0, 1.0, 0.0] represents the target category ('Mobile')
template_np = np.array([[0.0, 1.0, 0.0]]) 

generated_samples = cfm_machine.sample(
    y_cond=torch.from_numpy(template_np).to(cfg.device), 
    num_samples=100
)
```

-----

## 4\. 🔬 Supported Flow Matching Variants


***

### A. Advanced Conditional Flow Matching (CFM) Variants

These flow matching strategies utilize specific endpoint couplings ($q(\mathbf{z})$) and path definitions to achieve optimal transport, stochastic flows, or geometrically specific generation.

| CFM Variant Name | Target Joint $q(\mathbf{z})$ | Path Strategy / Geometry | Implementation Key |
| :--- | :--- | :--- | :--- |
| **ConditionalFlowMatcher** | $q(\mathbf{x}_0)q(\mathbf{x}_1)$ (Independent) | Typically **Linear Path** | `"independent"` coupling |
| **VariancePreservingCFM** | $q(\mathbf{x}_0)q(\mathbf{x}_1)$ (Independent) | Conditional Gaussian path (**VP path**) | `"variance_preserving_cfm"` interpolant |
| **TargetConditionalFlowMatcher** | Implicit $\pi(\mathbf{x}_0 \mid \mathbf{x}_1)$ | Linear path from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ to data $\mathbf{x}_1$ | `"target_cfm"` coupling |
| **ExactOT-CFM / SchrodingerBridgeCFM** | $pi(x0, x1), pi_epsilon(x0, x1)$ | Linear path | `"schrodinger_bridge_cfm"` coupling (uses **Sinkhorn approx.**) |

***
### B. Path Geometries and Solvers

These define the specific nature of the interpolation $\mathbf{x}(t)$.

| Key | Geometry / Dynamics | Use Case |
| :--- | :--- | :--- |
| **`"linear"`** | Standard straight-line path. | General data, simplest flow. |
| **`"alpha_flow"`** | **Generalized Geodesic Path.** | Compositional data, flows on statistical manifolds. |
| **`"log"`** | Log-Euclidean path. | **Compositional data** (counts, relative abundances), ensuring positivity. |
| **`"sparseaware"`** | Zero-aware path. | Improves stability for **sparse data** by smoothing paths near zero. |
| **`"vp"`** / **`"variance_preserving_cfm"`** | Maintains variance structure over time. | Flows built on diffusion model principles. |

#### SDE vs. ODE Sampling

The sampling method depends on the **training method** (`flow_variant`).

| Mode | `flow_variant` (Training) | `solver` (Sampling) | Behavior |
| :--- | :--- | :--- | :--- |
| **Deterministic (ODE)** | `"deterministic"` (Default) | `"rk4"` (Default), `"euler"`, `"heun"` | Integration is exact and repeatable. |
| **Stochastic (SDE)** | **`"stochastic"`** | **`"euler_maruyama"`** (Auto-selected) | Integration includes a diffusion term, yielding diverse, noisy samples. |


## Model Architectures for Velocity Field

The choice of model architecture is controlled by the `model_type` parameter in your configuration. These models are designed to map the noisy input state ($\mathbf{x}_t$), time ($t$), and condition ($\mathbf{y}$) to the target velocity vector ($\mathbf{v}_\theta$).

| Model Key | Architecture | Description | Use Case |
| :--- | :--- | :--- | :--- |
| **`"mlp"`** | Multi-Layer Perceptron (MLP) | The standard feed-forward network baseline. Suitable for low- to moderate-dimensional, unstructured numerical data. | General tabular data, quick iteration. |
| **`"transformer"`** | Transformer Encoder | Uses a self-attention mechanism to model long-range feature dependencies, treating features as tokens. | High-dimensional data (like metagenomics) where relationships between far-apart features are critical. |
| **`"autoencoder_latent"`** | Uses an MLP, but the path geometry (`interpolant="latent"`) is defined in a learned Autoencoder space. | Generation on a learned manifold; used when the intrinsic dimensionality of the data is much lower than the observation space. | Dimensionality reduction, non-linear manifold learning. |

---

## 5. Adapting to Data Dimensionality

The **`CFM_Lab`** library is designed to perform optimally across various data scales by allowing the user to select specialized neural network architectures for the velocity field ($\mathbf{v}_\theta$). This choice is controlled by the `model_type` parameter.

### Model Architectures for Velocity Field

The optimal model depends on the dimensionality ($D$) of your feature space.

| Model Key | Architecture | Best Dimensionality | Description |
| :--- | :--- | :--- | :--- |
| **`"mlp"`** | Multi-Layer Perceptron (MLP) | **Low to Moderate $D$** ($D \approx 1$ to $100$) | The standard feed-forward network. Highly efficient for simpler flow fields where feature interactions are local or linear.  |
| **`"transformer"`** | Transformer Encoder | **High $D$** ($D > 100$) | Uses self-attention to model complex, long-range dependencies between features (treating each feature as a token). Essential for structured or high-volume datasets like genomics. |
| **`"autoencoder_latent"`** | MLP on Latent Space | **High $D$** with Manifold Structure | Used when the true information lies on a low-dimensional manifold; the flow is learned in the compressed latent space. |

### Suggested Best Practice

When configuring your model, choose the architecture based on your feature count:

* **Low-D (e.g., Nutritional Data, $D<50$):** Set `"model_type": "mlp"`. This minimizes computational cost and avoids overfitting to small feature sets.
* **High-D (e.g., Metagenomics, $D>100$):** Set `"model_type": "transformer"`. This leverages attention mechanisms to capture global correlations essential for complex scientific data.

### E. Specialized Loss Functions

While the **L2 loss** (`"l2"`) on the velocity field is the default for all CFM variants, the library includes specialized losses for robustness and incorporating specific probabilistic assumptions (like those needed for Exponential Family Variational Flow Matching).

| Loss Key | Underlying Principle | Use Case |
| :--- | :--- | :--- |
| **`"l2"`** | Euclidean Distance on velocity: $E[\|v_\theta - v^*\|^2]$ | Default for all standard/OT/SB/VP flows. |
| **`"huber"`, `"l1"`, `"charbonnier"`** | Robust Regression | Use when training data or velocity fields are noisy, prone to outliers. |
| **`"poisson"`** | Poisson Negative Log-Likelihood (NLL) | Count data regression (predicting rates); suitable for exponential family assumptions. |
| **`"vfm_poisson_nll"`** | **EF-VFM Proxy Loss** | Specifically designed for **Exponential Family Variational Flow Matching** (EF-VFM) on count-like data (e.g., metagenomics). |
| **`"ot_sinkhorn"`** | Optimal Transport (OT) Distance | Used as a **metric for evaluation** or as an advanced loss for comparing predicted flow density to target density. |

---

##  CFM Library Workflow

The pipeline is primarily driven by the single entry point, `generate_samples_from_csv(...)`, which encapsulates four sequential phases: Configuration, Data Loading, Training, and Sampling.

---

## 1.  Phase I: Configuration and Initialization

This phase sets the stage by defining all parameters, including the CFM variant, network structure, and training hyper-parameters.

1.  **Input:** The user calls `generate_samples_from_csv(..., **CONFIG_OVERRIDES)`.
2.  **Configuration:** The function internally creates a `CFMConfig` object using the user's overrides.
    * **Determines Flow:** Sets the `interpolant` (e.g., `"alpha_flow"`, `"linear"`) and `coupling` mode (e.g., `"target_cfm"`, `"schrodinger_bridge_cfm"`).
    * **Sets Dynamics:** Defines SDE/ODE parameters (`flow_variant`, `stochastic_noise_scale`, `solver`).
3.  **Model Setup:** A `CFMMachine` instance is initialized, which builds the core velocity model ($\mathbf{v}_\theta$) using the specified `model_type` (`"mlp"` or `"transformer"`).

---

## 2. 💾 Phase II: Data Loading and Preprocessing

The library handles data preparation, transforming raw CSV inputs into processed PyTorch data loaders.

1.  **Data Loading:** The `load_data_from_config` function reads the CSV(s) based on the user's chosen mode (Mode 1, 2, or 3).
    * **Feature Extraction ($\mathbf{X}$):** Selects numerical columns for the generative features.
    * **Conditioning ($\mathbf{Y}$):** If a `condition_column_name` is provided, the column is extracted and **One-Hot Encoded** to create the conditioning vector $\mathbf{Y}$.
2.  **Optional Preprocessing:** (If manually enabled in the calling script) The raw feature data is transformed, e.g., using the **CLR transformation** (`apply_clr_transform`) for compositional data.
3.  **Data Setup:** The processed $\mathbf{X}$ and $\mathbf{Y}$ matrices are converted into a `MetagenomicsDataset` object, which is then wrapped in a PyTorch `DataLoader`.

---

## 3.  Phase III: Training and Flow Matching

This is the core learning phase, where the velocity model $\mathbf{v}_\theta$ is trained to match the target velocity $\mathbf{v}^*$ defined by the chosen CFM variant.

1.  **Batch Iteration:** The `CFMMachine.train()` method iterates through batches ($\mathbf{x}_B$, $\mathbf{y}_B$).
2.  **Endpoint Coupling (Critical Step):** The library executes the specific coupling strategy:
    * It uses the `get_coupling_x0_x1` dispatcher function (in `coupling.py`) based on the chosen CFM variant (e.g., `"target_cfm"`, `"schrodinger_bridge_cfm"`).
    * This pairs the batch data $\mathbf{x}_0$ (or sampled prior) with the target endpoint $\mathbf{x}_1$ (prior or matched data) to form the joint $\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1)$.
3.  **Path and Target Velocity:**
    * A random time $t \sim U(0, 1)$ is sampled.
    * The instantaneous state $\mathbf{x}_t = \mathbf{x}(t)$ is calculated using the configured **interpolant** (`"linear"`, `"alpha_flow"`, etc.).
    * The **Target Velocity** $\mathbf{v}^*$ is calculated as the derivative of the path $d\mathbf{x}(t)/dt$.
4.  **Loss Calculation:** The model predicts the velocity $\mathbf{v}_\theta = \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{y}_B)$, and the loss is computed (default is $\text{L}_2$ on velocities, or a specialized loss like `"vfm_poisson_nll"`).
5.  **Optimization:** The model parameters are updated via backpropagation and the configured optimizer.

---

## 4.  Phase IV: Sampling and Generation

Once trained, the model is used as a vector field to generate new, synthetic data by integrating the flow backward in time.

1.  **Initialization:** The sampling process begins at $t=1$ by sampling $\mathbf{x}_1$ from the **prior distribution** (usually Gaussian noise).
2.  **Conditional Input:** The desired conditioning vector $\mathbf{y}_{template}$ (e.g., the one-hot vector for 'Healthy') is prepared.
3.  **Integration:** The `CFMMachine.sample()` method uses the configured **solver** (`"rk4"` for deterministic, `"euler_maruyama"` for stochastic) to integrate the flow backward from $t=1$ to $t=0$:
    * $\mathbf{x}_{t - \Delta t} \approx \mathbf{x}_t - \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{y}_{template}) \cdot \Delta t$
    * 
4.  **Output:** The final state $\mathbf{x}_0$ is the generated synthetic data, which is returned to the user as a NumPy array.
