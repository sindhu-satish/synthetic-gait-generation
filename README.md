# Synthetic Gait Generation

A deep learning system for generating synthetic gait sensor data using a Variational Autoencoder (VAE) and Denoising Diffusion Probabilistic Model (DDPM). The system learns to generate realistic accelerometer and gyroscope signals from the BB-MAS dataset while preserving user-specific gait characteristics.

## Table of Contents

- [Overview](#overview)
- [What This Application Does](#what-this-application-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Full Pipeline](#full-pipeline)
  - [Training](#training)
  - [Generation](#generation)
  - [Evaluation](#evaluation)
  - [Ablation Studies](#ablation-studies)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Model Details](#model-details)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a two-stage generative model for synthesizing gait sensor data. The model first compresses gait signal windows into a latent space using a VAE, then learns to generate new latent representations using a DDPM. The generated data maintains statistical properties and user-specific characteristics of real gait patterns.

### Key Features

- Dual sensor support: Generates data for both Accelerometer and Gyroscope sensors
- Physics-informed losses: Incorporates smoothness, distribution, and spectral losses to ensure realistic signal properties
- User identity preservation: Conditions generation on user embeddings to maintain individual gait characteristics
- Comprehensive evaluation: Includes realism tests, statistical comparisons, and augmentation effectiveness studies
- Ablation studies: Systematic analysis of model components and hyperparameters

## What This Application Does

1. **Data Loading and Preprocessing**:
   - Loads accelerometer and gyroscope CSV files from the BB-MAS dataset
   - Extracts X, Y, Z sensor values from PocketPhone recordings
   - Creates sliding windows of fixed length (128 timesteps) with configurable stride
   - Normalizes sensor data and splits by user ID to prevent data leakage

2. **Model Training**:
   - Trains a VAE encoder/decoder to compress gait windows into latent representations
   - Uses KL divergence with cosine warm-up schedule
   - Applies physics-informed losses (smoothness, distribution matching, spectral similarity)
   - Trains a DDPM model to generate new latent representations
   - Conditions generation on user identity embeddings
   - Supports Exponential Moving Average (EMA) for model stability

3. **Synthetic Data Generation**:
   - Samples new latent representations from the trained DDPM
   - Decodes latents back to sensor windows using the VAE decoder
   - Generates data for specified users and sensor types
   - Outputs synthetic data in the same format as the original dataset

4. **Evaluation**:
   - Real vs Synthetic Classifier: Tests if a classifier can distinguish real from synthetic data
   - Statistical Tests: Compares distributions using Kolmogorov-Smirnov tests and other statistical measures
   - Augmentation Effectiveness: Evaluates whether synthetic data improves downstream task performance
   - Exploratory Data Analysis: Visual comparisons of real and synthetic signal characteristics

## Architecture

The model consists of two main components:

1. **VAE Encoder/Decoder**:
   - Encodes gait windows (128 × 3) into latent space (256 dimensions)
   - Uses 1D convolutions with residual blocks for temporal processing
   - Implements GroupNorm and SiLU activations
   - Applies physics-informed regularization during training

2. **DDPM (Denoising Diffusion Probabilistic Model)**:
   - Generates latent representations through iterative denoising
   - Uses cosine noise schedule over 400 diffusion steps
   - Conditions on user identity embeddings
   - Implements variance matching loss for better distribution alignment
   - Supports EMA for improved generation quality

## Prerequisites

- Python: 3.8 or higher
- CUDA: Optional, but recommended for GPU acceleration (CUDA 11.8+)
- Dataset: BB-MAS dataset with gait sensor CSV files organized by user folders

## Installation

1. **Navigate to the project directory**:
   ```bash
   cd /path/to/synthetic-gait-generation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify dataset structure**:
   Ensure the BB-MAS dataset is located at:
   ```
   synthetic-gait-generation/BB-MAS_Dataset/BB-MAS_Dataset/{user_id}/*PocketPhone*{sensor_type}*.csv
   ```

## Usage

### Full Pipeline

Run the complete training and evaluation pipeline:

```bash
cd src/gait_generation
python -m gait_generation.main
```

Or with options:

```bash
# Skip EDA analysis
python -m gait_generation.main --skip-eda

# Run ablation studies instead
python -m gait_generation.main ablations
```

The full pipeline will:
- Train VAE models for both Accelerometer and Gyroscope sensors
- Train DDPM models for both sensors
- Generate synthetic data
- Run EDA analysis (unless skipped)
- Evaluate realism and augmentation effectiveness

### Training

Training is handled automatically by the full pipeline, but you can also train models programmatically:

```bash
cd src/gait_generation
python -c "from gait_generation.run_full_pipeline import main; main()"
```

**Training process**:
- Loads sensor data from the BB-MAS dataset
- Creates train/validation/test splits (70%/15%/15%) by user ID
- Trains VAE with KL warm-up schedule and physics losses
- Encodes training data to latent space
- Trains DDPM on latent representations
- Saves checkpoints to `checkpoints/gait/{sensor_type}/`

**Training parameters** (configurable in `config.py`):
- VAE batch size: 256
- VAE learning rate: 1e-3
- VAE epochs: 50
- KL warm-up epochs: 20
- DDPM batch size: 512
- DDPM learning rate: 2e-4
- DDPM epochs: 200
- Diffusion steps: 400

### Generation

Generate synthetic gait data using trained models:

```python
from gait_generation.generate import sample_synthetic
from gait_generation.diffusion import LatentDDPM
from gait_generation.vae import WindowVAE
from gait_generation.data import GaitDataModule

# Load trained models and data module
# ... (see generate.py for details)

# Generate synthetic data
synth_df = sample_synthetic(n=2048, ddpm=ddpm, vae=vae, dm=dm)
```

The generated data is returned as a pandas DataFrame with columns:
- `EID`: Event ID (timestep index)
- `Xvalue`, `Yvalue`, `Zvalue`: Sensor values
- `__user_id__`: User identifier
- `__sensor_type__`: Sensor type (Accelerometer or Gyroscope)

### Evaluation

Evaluation runs automatically as part of the full pipeline. It includes:

1. **Real vs Synthetic Classifier**:
   - Trains a classifier to distinguish real from synthetic data
   - Lower accuracy indicates more realistic synthetic data

2. **Statistical Tests**:
   - Kolmogorov-Smirnov tests on feature distributions
   - Mean and variance comparisons
   - Correlation analysis

3. **Augmentation Effectiveness**:
   - Tests whether synthetic data improves downstream task performance
   - Compares models trained on real data alone vs real + synthetic data

Results are saved to `checkpoints/gait/{sensor_type}/evaluation/`

### Ablation Studies

Run systematic ablation studies to analyze model components:

```bash
cd src/gait_generation
python -m gait_generation.main ablations
```

Ablation studies examine:
- VAE architecture variations
- DDPM hyperparameter sensitivity
- Loss function contributions
- Conditioning strategies

## Configuration

Edit `src/gait_generation/config.py` to customize:

### Data Parameters
```python
GAIT_BASE_DIR = "BB-MAS_Dataset/BB-MAS_Dataset"
SENSOR_TYPES = ("Accelerometer", "Gyroscope")
WINDOW_SIZE = 128
WINDOW_STRIDE = 64
TEST_SIZE = 0.15
VAL_SIZE = 0.15
```

### Model Architecture
```python
LATENT_DIM = 256
HIDDEN_DIM = 1024
N_LAYERS = 4
CAT_EMBED_DIM = 16
DROPOUT = 0.05
```

### Training Hyperparameters
```python
VAE_EPOCHS = 50
VAE_BATCH_SIZE = 256
VAE_LR = 1e-3
KL_MAX_BETA = 1.0
KL_WARMUP_EPOCHS = 20

DDPM_EPOCHS = 200
DDPM_BATCH_SIZE = 512
DDPM_LR = 2e-4
DDPM_PATIENCE = 60
```

### Diffusion Parameters
```python
T = 400
BETA_SCHEDULE = "cosine"
SAMPLE_STEPS = 300
DDPM_USE_EMA = True
DDPM_EMA_DECAY = 0.9999
```

### Physics Losses
```python
USE_SMOOTHNESS_LOSS = True
USE_DISTRIBUTION_LOSS = True
LAMBDA_SMOOTH = 1e-3
LAMBDA_PHYS = 5e-3
LAMBDA_SPECTRAL = 1e-2
```

## Output Files

### Checkpoints
- `checkpoints/gait/{sensor_type}/vae_best_{sensor_type}.pt`: Best VAE model
- `checkpoints/gait/{sensor_type}/ddpm_best_{sensor_type}.pt`: Best DDPM model
- `checkpoints/gait/{sensor_type}/vae_history_{sensor_type}.json`: VAE training history
- `checkpoints/gait/{sensor_type}/ddpm_history_{sensor_type}.json`: DDPM training history

### Evaluation Results
- `checkpoints/gait/{sensor_type}/evaluation/`: Evaluation metrics and visualizations
- Classification results, statistical test outputs, and augmentation study results

### EDA Outputs
- `checkpoints/gait/{sensor_type}/eda/`: Exploratory data analysis plots and comparisons

## Model Details

### Input Format
- **Gait Windows**: Shape `(128, 3)` where:
  - 128 = window length (number of timesteps)
  - 3 = sensor channels (X, Y, Z values)

### Normalization
Sensor data is normalized using z-score normalization per channel, computed across the training set.

### VAE Architecture
- Encoder: 1D convolutions with residual blocks, downsampling to latent dimension
- Decoder: 1D transposed convolutions with residual blocks, upsampling from latent to original dimensions
- Latent space: 256-dimensional continuous representation

### DDPM Process
- Forward process: Gradually adds Gaussian noise over 400 steps using cosine schedule
- Reverse process: Neural network learns to denoise latent representations
- Conditioning: User identity embeddings guide generation
- Sampling: Uses DDIM-style sampling with configurable number of steps (default 300)

### Physics-Informed Losses
- **Smoothness Loss**: Encourages temporal smoothness in reconstructed signals
- **Distribution Loss**: Matches statistical distribution of real data
- **Spectral Loss**: Preserves frequency domain characteristics

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `VAE_BATCH_SIZE` or `DDPM_BATCH_SIZE` in `config.py`
   - Use CPU by setting `DEVICE = "cpu"` in `config.py`

2. **No Checkpoint Found**:
   - Ensure you've run the training pipeline first
   - Check that `checkpoints/gait/{sensor_type}/` directory exists and contains model files

3. **Dataset Not Found**:
   - Verify dataset path in `config.py`: `GAIT_BASE_DIR = "BB-MAS_Dataset/BB-MAS_Dataset"`
   - Ensure user folders (1-117) exist with PocketPhone CSV files
   - Check file naming pattern: `*PocketPhone*{sensor_type}*.csv`

4. **Import Errors**:
   - Activate virtual environment: `source venv/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Ensure you're running from the correct directory

5. **Generation Produces Unrealistic Values**:
   - Check that models were trained on the same dataset
   - Verify normalization statistics match between training and generation
   - Ensure sufficient training epochs and proper convergence

### Performance Tips

- **GPU Acceleration**: Set `DEVICE = "cuda"` in `config.py` (requires CUDA-compatible GPU)
- **Faster Training**: Increase batch sizes if you have more GPU memory
- **Memory Efficiency**: Reduce `WINDOW_SIZE` or `LATENT_DIM` if needed
- **Faster Sampling**: Reduce `SAMPLE_STEPS` for quicker generation (may reduce quality)

## Dependencies

See `requirements.txt` for full list:
- `torch>=2.0.0`: PyTorch for deep learning
- `pytorch-lightning>=2.0.0`: Training utilities
- `numpy>=1.24.0`: Numerical operations
- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: Machine learning utilities
- `scipy>=1.10.0`: Statistical functions
- `tqdm>=4.65.0`: Progress bars
- `einops>=0.7.0`: Tensor operations
- `matplotlib>=3.7.0`: Plotting
- `seaborn>=0.13.2`: Statistical visualization
- `umap-learn>=0.5.9.post2`: Dimensionality reduction for visualization
