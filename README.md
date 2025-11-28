# Keystroke Dynamics Synthetic Data Generation

A deep learning system for generating synthetic keystroke dynamics data using a 1D Latent Diffusion Model, with identity preservation across multiple devices (Desktop, Phone, Tablet).

## 📋 Table of Contents

- [Overview](#overview)
- [What This Application Does](#what-this-application-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generation](#generation)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Model Details](#model-details)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a **Stable Diffusion-style architecture** adapted for 1D time-series keystroke data. The model learns to generate realistic keystroke timing patterns (Flight times and Keyhold durations) while preserving user identity characteristics across different input devices.

### Key Features

- **Identity Preservation**: Maintains user-specific keystroke patterns across devices
- **Multi-Device Support**: Generates data for Desktop, Phone, and Tablet
- **Latent Diffusion**: Uses VAE + U-Net architecture for high-quality generation
- **Comprehensive Evaluation**: Includes authentication performance, statistical similarity, and identity preservation metrics

## What This Application Does

1. **Data Loading & Preprocessing**:
   - Loads keystroke feature files (Flight1-4, Keyhold) from the BB-MAS dataset
   - Creates sliding windows of keystroke sequences
   - Normalizes timing data using log transform + z-score + min-max scaling
   - Splits data by user ID to prevent data leakage

2. **Model Training**:
   - Trains a Variational Autoencoder (VAE) to compress keystroke windows into latent space
   - Trains a 1D U-Net diffusion model to generate latent representations
   - Uses identity loss with warm-up schedule to preserve user characteristics
   - Conditions generation on user and device embeddings

3. **Synthetic Data Generation**:
   - Generates synthetic keystroke windows for specified users and devices
   - Reconstructs Flight and Keyhold features from generated windows
   - Properly denormalizes timing values back to milliseconds
   - Saves synthetic data in BB-MAS dataset format (CSV files)

4. **Evaluation**:
   - **Authentication Performance**: Tests if synthetic data improves user authentication
   - **Statistical Similarity**: Compares distributions using KS-tests and t-tests
   - **Identity Preservation**: Measures how well user identity is maintained in synthetic data

## Architecture

The model consists of three main components:

1. **VAE Encoder/Decoder**:
   - Encodes keystroke windows (256 × 5) → latent space (64 × 64)
   - Decodes latent representations back to keystroke windows
   - Uses 1D convolutions for temporal processing

2. **1D U-Net Diffusion Model**:
   - Denoises latent representations over 1000 diffusion steps
   - Conditions on user embeddings (128-dim) and device embeddings (32-dim)
   - Uses attention mechanisms and residual connections

3. **Identity Classifier**:
   - Auxiliary classifier to enforce identity preservation
   - Trained with warm-up schedule (starts at epoch 25)
   - Helps maintain user-specific patterns in generated data

## Project Structure

```
CS228/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── BB-MAS_Dataset/                    # Dataset directory
│   └── BB-MAS_Dataset/
│       ├── Keystroke_Features/        # Input keystroke feature files
│       │   ├── {user_id}_Flight1_{device}.csv
│       │   ├── {user_id}_Flight2_{device}.csv
│       │   ├── {user_id}_Flight3_{device}.csv
│       │   ├── {user_id}_Flight4_{device}.csv
│       │   └── {user_id}_Keyhold_{device}.csv
│       └── Demographics.csv           # User demographic information
│
└── src/
    └── keystroke_generation/          # Main application code
        ├── config.py                  # Configuration parameters
        ├── data_loader.py             # Data loading and preprocessing
        ├── model.py                   # Model architecture (VAE + U-Net)
        ├── train.py                   # Training script
        ├── generate.py                # Synthetic data generation
        ├── evaluate.py                # Evaluation metrics
        ├── checkpoints/               # Saved model checkpoints
        │   ├── best_model.pt
        │   └── final_model.pt
        ├── logs/                      # Training history
        │   └── training_history.json
        ├── synthetic_data/            # Generated synthetic data
        │   ├── {user_id}_Flight1_{device}.csv
        │   ├── {user_id}_Flight2_{device}.csv
        │   ├── {user_id}_Flight3_{device}.csv
        │   ├── {user_id}_Flight4_{device}.csv
        │   └── {user_id}_Keyhold_{device}.csv
        └── evaluation_results.json   # Evaluation metrics output
```

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Optional, but recommended for GPU acceleration (CUDA 11.8+)
- **Dataset**: BB-MAS dataset with keystroke feature files

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/CS228
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify dataset structure**:
   Ensure the BB-MAS dataset is located at:
   ```
   CS228/BB-MAS_Dataset/BB-MAS_Dataset/Keystroke_Features/
   ```

## Usage

### Training

Train the model on the BB-MAS keystroke dataset:

```bash
cd src/keystroke_generation
python train.py
```

**What happens during training**:
- Loads all keystroke feature files from the dataset
- Creates train/validation/test splits (70%/15%/15%) by user ID
- Trains the VAE encoder/decoder and diffusion model
- Implements identity loss warm-up (starts at epoch 25)
- Saves checkpoints to `checkpoints/best_model.pt` and `checkpoints/final_model.pt`
- Logs training history to `logs/training_history.json`

**Training parameters** (configurable in `config.py`):
- Batch size: 32
- Learning rate: 5e-5
- Number of epochs: 100
- Identity loss warm-up: 25 epochs
- Early stopping: Enabled

### Generation

Generate synthetic keystroke data using a trained model:

```bash
cd src/keystroke_generation
python generate.py [num_windows] [user_ids] [devices]
```

**Examples**:

```bash
# Generate 10 windows per user-device pair for all users and devices
python generate.py 10

# Generate 20 windows for specific users
python generate.py 20 1,2,3,10,20

# Generate for specific users and devices
python generate.py 15 1,2,3 Desktop,Phone

# Generate for all users but only Desktop device
python generate.py 10 "" Desktop
```

**Arguments**:
- `num_windows` (optional): Number of windows to generate per user-device pair (default: 10)
- `user_ids` (optional): Comma-separated list of user IDs (default: all users)
- `devices` (optional): Comma-separated list of device names (default: all devices)

**Output**: Synthetic CSV files saved to `synthetic_data/` directory in BB-MAS format.

### Evaluation

Evaluate the quality of generated synthetic data:

```bash
cd src/keystroke_generation
python evaluate.py
```

**What the evaluation does**:

1. **Authentication Performance**:
   - Trains a Random Forest classifier on real data
   - Tests on real + synthetic data
   - Computes Accuracy, ROC-AUC, and Equal Error Rate (EER)

2. **Statistical Similarity**:
   - Performs Kolmogorov-Smirnov (KS) tests on feature distributions
   - Performs t-tests on mean values
   - Compares real vs synthetic data distributions

3. **Identity Preservation**:
   - Trains an identity classifier on real data
   - Tests on synthetic data
   - Measures user identification accuracy

**Output**: Results saved to `evaluation_results.json`

## Configuration

Edit `src/keystroke_generation/config.py` to customize:

### Data Parameters
```python
WINDOW_LENGTH = 256        # Length of keystroke windows
NUM_FEATURES = 5           # Flight1, Flight2, Flight3, Flight4, Keyhold
NUM_USERS = 117            # Number of users in dataset
DEVICES = ["Desktop", "Phone", "Tablet"]
```

### Model Architecture
```python
LATENT_DIM = 64            # Latent space dimension
EMBED_DIM = 128            # User embedding dimension
DEVICE_EMBED_DIM = 32      # Device embedding dimension
VAE_HIDDEN_DIM = 256       # VAE hidden layer dimension
```

### Training Hyperparameters
```python
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
IDENTITY_WARMUP_EPOCHS = 25
FINAL_IDENTITY_LOSS_WEIGHT = 0.001
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

### Diffusion Parameters
```python
NUM_DIFFUSION_STEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
NOISE_SCHEDULE = "linear"
```

## Output Files

### Checkpoints
- `checkpoints/best_model.pt`: Best model based on validation loss
- `checkpoints/final_model.pt`: Final model after training completes

### Synthetic Data
- `synthetic_data/{user_id}_Flight1_{device}.csv`
- `synthetic_data/{user_id}_Flight2_{device}.csv`
- `synthetic_data/{user_id}_Flight3_{device}.csv`
- `synthetic_data/{user_id}_Flight4_{device}.csv`
- `synthetic_data/{user_id}_Keyhold_{device}.csv`

Each CSV file contains:
- **Flight files**: `key1`, `key2`, `time` columns
- **Keyhold files**: `key`, `keyhold` columns

### Logs
- `logs/training_history.json`: Training metrics per epoch (loss, validation loss, etc.)

### Evaluation
- `evaluation_results.json`: Comprehensive evaluation metrics

## Model Details

### Input Format
- **Keystroke Windows**: Shape `(256, 5)` where:
  - 256 = window length (number of keystrokes)
  - 5 = features (Flight1, Flight2, Flight3, Flight4, Keyhold)

### Normalization Process
1. **Log transform**: `log(1 + time_ms)`
2. **Z-score normalization**: `(log_times - mean) / std`
3. **Min-max scaling**: `(z_scores - z_min) / (z_max - z_min)` → [0, 1]

### Denormalization Process (Generation)
1. **Reverse min-max**: `z_scores = normalized * z_range + z_min`
2. **Reverse z-score**: `log_times = z_scores * std + mean`
3. **Reverse log**: `times = exp(log_times) - 1`

### Identity Loss Warm-up
- **Epochs 1-25**: Identity loss weight = 0.0 (model learns basic generation)
- **Epochs 26-35**: Linear ramp from 0.0 to 0.001
- **Epochs 36+**: Identity loss weight = 0.001 (identity preservation enforced)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE` in `config.py`
   - Use CPU by setting `DEVICE = "cpu"` in `config.py`

2. **No Checkpoint Found**:
   - Ensure you've run `train.py` first
   - Check that `checkpoints/` directory exists and contains model files

3. **Dataset Not Found**:
   - Verify dataset path in `config.py`: `DATASET_DIR = PROJECT_ROOT / "BB-MAS_Dataset" / "BB-MAS_Dataset"`
   - Ensure `Keystroke_Features/` directory exists with CSV files

4. **Import Errors**:
   - Activate virtual environment: `source venv/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`

5. **Generation Produces Unrealistic Values**:
   - Check that denormalization is working correctly
   - Verify normalization statistics in `generate.py` match training data
   - Ensure model was trained on the same dataset

### Performance Tips

- **GPU Acceleration**: Set `DEVICE = "cuda"` in `config.py` (requires CUDA-compatible GPU)
- **Faster Training**: Increase `BATCH_SIZE` if you have more GPU memory
- **Memory Efficiency**: Reduce `WINDOW_LENGTH` or `NUM_FEATURES` if needed

## Dependencies

See `requirements.txt` for full list:
- `torch>=2.0.0`: PyTorch for deep learning
- `numpy>=1.24.0`: Numerical operations
- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: Machine learning utilities
- `scipy>=1.10.0`: Statistical functions
- `tqdm>=4.65.0`: Progress bars