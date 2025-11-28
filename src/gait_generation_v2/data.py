"""
Data module for Gait Generation V2.

Provides PyTorch Dataset and DataModule for window-based gait data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from .config import ConfigV2
from .preprocessing import GaitPreprocessor


class GaitWindowDataset(Dataset):
    """
    Dataset for gait windows.
    
    Each sample is a window of shape (window_size, 4) containing
    [Xvalue, Yvalue, Zvalue, magnitude] in normalized space.
    """
    
    def __init__(
        self, 
        windows: np.ndarray, 
        user_ids: List[str], 
        user_to_idx: Dict[str, int]
    ):
        """
        Args:
            windows: (N, window_size, 4) array of normalized windows
            user_ids: List of user IDs for each window
            user_to_idx: Mapping from user_id to integer index
        """
        # Clean NaN/Inf values
        windows = np.nan_to_num(windows, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.user_idx = torch.tensor(
            [user_to_idx[u] for u in user_ids], 
            dtype=torch.long
        )
        self.user_ids = user_ids
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "window": self.windows[i],      # (window_size, 4)
            "user_idx": self.user_idx[i],   # scalar
        }


class LatentDataset(Dataset):
    """
    Dataset for latent codes (used for DDPM training).
    """
    
    def __init__(self, Z: np.ndarray, user_idx: np.ndarray):
        """
        Args:
            Z: (N, latent_dim) array of latent codes
            user_idx: (N,) array of user indices
        """
        self.Z = torch.tensor(Z, dtype=torch.float32)
        self.user_idx = torch.tensor(user_idx, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.Z)
    
    def __getitem__(self, i: int):
        return self.Z[i], self.user_idx[i]


class GaitWindowDataModule:
    """
    Data module that wraps the preprocessor and provides dataloaders.
    """
    
    def __init__(self, preprocessor: GaitPreprocessor, cfg: ConfigV2):
        """
        Args:
            preprocessor: Fitted GaitPreprocessor with windows
            cfg: Configuration
        """
        self.preprocessor = preprocessor
        self.cfg = cfg
        
        self.train_ds: Optional[GaitWindowDataset] = None
        self.val_ds: Optional[GaitWindowDataset] = None
        self.test_ds: Optional[GaitWindowDataset] = None
        
        self._setup_done = False
    
    @property
    def n_users(self) -> int:
        """Total number of unique users."""
        return len(self.preprocessor.user_to_idx)
    
    @property
    def user_to_idx(self) -> Dict[str, int]:
        """User ID to index mapping."""
        return self.preprocessor.user_to_idx
    
    @property
    def idx_to_user(self) -> Dict[int, str]:
        """Index to user ID mapping."""
        return self.preprocessor.idx_to_user
    
    def setup(self):
        """Create datasets from preprocessed windows."""
        if self._setup_done:
            return
        
        pre = self.preprocessor
        
        # Create train dataset
        if pre.train_windows is not None and len(pre.train_windows) > 0:
            self.train_ds = GaitWindowDataset(
                pre.train_windows,
                pre.train_user_ids,
                pre.user_to_idx
            )
        
        # Create val dataset
        if pre.val_windows is not None and len(pre.val_windows) > 0:
            self.val_ds = GaitWindowDataset(
                pre.val_windows,
                pre.val_user_ids,
                pre.user_to_idx
            )
        
        # Create test dataset
        if pre.test_windows is not None and len(pre.test_windows) > 0:
            self.test_ds = GaitWindowDataset(
                pre.test_windows,
                pre.test_user_ids,
                pre.user_to_idx
            )
        
        self._setup_done = True
        
        print(f"\nDataModule setup complete:")
        print(f"  Train samples: {len(self.train_ds) if self.train_ds else 0}")
        print(f"  Val samples: {len(self.val_ds) if self.val_ds else 0}")
        print(f"  Test samples: {len(self.test_ds) if self.test_ds else 0}")
        print(f"  Total users: {self.n_users}")
    
    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Get training dataloader."""
        if self.train_ds is None:
            raise ValueError("Train dataset not available. Call setup() first.")
        
        bs = batch_size or self.cfg.vae_batch_size
        return DataLoader(
            self.train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Get validation dataloader."""
        if self.val_ds is None:
            raise ValueError("Val dataset not available. Call setup() first.")
        
        bs = batch_size or self.cfg.vae_batch_size
        return DataLoader(
            self.val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Get test dataloader."""
        if self.test_ds is None:
            raise ValueError("Test dataset not available. Call setup() first.")
        
        bs = batch_size or self.cfg.vae_batch_size
        return DataLoader(
            self.test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

