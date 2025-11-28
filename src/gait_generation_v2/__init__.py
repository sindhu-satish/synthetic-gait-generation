"""
Gait Generation V2: Temporal VAE + Latent DDPM Pipeline

This module implements window-based (256 timesteps) 1D convolutional VAE 
with latent space DDPM for generating synthetic gait data from 
Accelerometer and Gyroscope sensors.

Key features:
- Window-based processing (256 timesteps at 100Hz = 2.56s)
- 1D Convolutional VAE with user conditioning
- Latent space DDPM with EMA for sampling
- Sensor-specific preprocessing (segmentation, resampling, clipping)
- User-based train/val/test splits
- Checkpoint/resume support
"""

from .config import ConfigV2, set_seed
from .preprocessing import GaitPreprocessor
from .data import GaitWindowDataset, GaitWindowDataModule
from .conv_vae import ConvVAE
from .diffusion import LatentUNet, LatentDDPM, EMA
from .train import train_vae, train_ddpm, encode_to_latents
from .generate import generate_synthetic_windows
from .main import main, train_sensor_model

__all__ = [
    'ConfigV2',
    'set_seed',
    'GaitPreprocessor',
    'GaitWindowDataset',
    'GaitWindowDataModule',
    'ConvVAE',
    'LatentUNet',
    'LatentDDPM',
    'EMA',
    'train_vae',
    'train_ddpm',
    'encode_to_latents',
    'generate_synthetic_windows',
    'main',
    'train_sensor_model',
]

