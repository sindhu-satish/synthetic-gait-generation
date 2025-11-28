import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from .preprocessor import Preprocessor
from .config import (
    RARE_THRESH, TEST_SIZE, VAL_SIZE, SEED, WINDOW_SIZE, WINDOW_STRIDE,
    VAE_BATCH_SIZE, NUM_WORKERS, COND_COLS, PIN_MEMORY
)

def load_sensor_csvs(gait_base_dir: str, sensor_type: str):
    original_path = gait_base_dir
    if not os.path.isabs(gait_base_dir):
        if not os.path.exists(gait_base_dir):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_file_dir))
            gait_base_dir = os.path.join(project_root, gait_base_dir)
    
    base_dataset_dir = gait_base_dir.rstrip("/")
    if not os.path.exists(base_dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dataset_dir}\nTried: {original_path} -> {gait_base_dir}")
    
    all_paths = []
    for folder_num in range(1, 118):
        folder_path = os.path.join(base_dataset_dir, str(folder_num))
        if os.path.exists(folder_path):
            pattern = f"*PocketPhone*{sensor_type}*.csv"
            sensor_files = glob.glob(os.path.join(folder_path, pattern))
            all_paths.extend(sensor_files)
    
    print(f"Scanning folders 1-117 in {base_dataset_dir} for {sensor_type}: found {len(all_paths)} CSV files")
    
    if not all_paths:
        raise FileNotFoundError(
            f"No PocketPhone {sensor_type} CSV files found in folders 1-117 under {base_dataset_dir}\n"
            f"Checked directory: {base_dataset_dir}\n"
            f"Pattern: *PocketPhone*{sensor_type}*.csv"
        )
    
    dfs = []
    for p in sorted(all_paths):
        try:
            df = pd.read_csv(p)
            if len(df) == 0 or len(df.columns) == 0:
                print(f"Skipping empty file: {os.path.basename(p)}")
                continue
            df["__source_file__"] = os.path.basename(p)
            df["__user_id__"] = os.path.basename(os.path.dirname(p))
            df["__sensor_type__"] = sensor_type
            dfs.append(df)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
    
    if not dfs:
        raise ValueError(f"No valid PocketPhone {sensor_type} CSV files found in folders 1-117")
    
    data = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Successfully loaded {len(dfs)} {sensor_type} files with {len(data)} total rows")
    return data, all_paths

def create_windows(df: pd.DataFrame, window_size: int, stride: int, sensor_cols: list = ["Xvalue", "Yvalue", "Zvalue"]):
    windows = []
    user_ids = []
    
    for (user_id, source_file), group in df.groupby(["__user_id__", "__source_file__"]):
        group = group.sort_values("EID").reset_index(drop=True)
        
        if len(group) < window_size:
            continue
        
        sensor_data = group[sensor_cols].values.astype(np.float32)
        
        for start_idx in range(0, len(sensor_data) - window_size + 1, stride):
            window = sensor_data[start_idx:start_idx + window_size]
            windows.append(window)
            user_ids.append(user_id)
    
    if len(windows) == 0:
        raise ValueError("No windows created. Check window_size and data length.")
    
    return np.array(windows), np.array(user_ids)

class WindowedGaitDataset(Dataset):
    def __init__(self, windows, user_ids):
        self.windows = windows
        self.user_ids = user_ids

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return {
            "window": torch.tensor(self.windows[i], dtype=torch.float32),
            "cond": int(self.user_ids[i])
        }

class GaitDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, preprocessor: Preprocessor = None, save_dir: str = None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.save_dir = save_dir
        self.pre = preprocessor if preprocessor is not None else Preprocessor(rare_thresh=RARE_THRESH)
        self.cond_cols = list(COND_COLS) if COND_COLS else ["__user_id__"]
        self.cond_map = {}
        self.rev_cond_map = {}
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage=None):
        df = self.df.copy()
        
        unique_files = df.groupby(["__user_id__", "__source_file__"]).size().reset_index()
        unique_files = unique_files[["__user_id__", "__source_file__"]]
        
        user_file_counts = unique_files["__user_id__"].value_counts()
        can_stratify = (user_file_counts >= 2).all() and len(user_file_counts) > 1
        
        idx = np.arange(len(unique_files))
        stratify_labels = unique_files["__user_id__"].values if can_stratify else None
        
        idx_train, idx_tmp = train_test_split(
            idx, 
            test_size=VAL_SIZE + TEST_SIZE, 
            random_state=SEED, 
            stratify=stratify_labels
        )
        
        if can_stratify:
            tmp_stratify = unique_files.iloc[idx_tmp]["__user_id__"].values
            tmp_counts = pd.Series(tmp_stratify).value_counts()
            can_stratify_tmp = (tmp_counts >= 2).all() and len(tmp_counts) > 1
            tmp_stratify = tmp_stratify if can_stratify_tmp else None
        else:
            tmp_stratify = None
        
        idx_val, idx_test = train_test_split(
            idx_tmp, 
            test_size=TEST_SIZE/(VAL_SIZE + TEST_SIZE), 
            random_state=SEED,
            stratify=tmp_stratify
        )
        
        train_files = unique_files.iloc[idx_train]
        val_files = unique_files.iloc[idx_val]
        test_files = unique_files.iloc[idx_test]
        
        train_df = df.merge(train_files, on=["__user_id__", "__source_file__"], how="inner")
        val_df = df.merge(val_files, on=["__user_id__", "__source_file__"], how="inner")
        test_df = df.merge(test_files, on=["__user_id__", "__source_file__"], how="inner")
        
        train_windows, train_user_ids = create_windows(train_df, WINDOW_SIZE, WINDOW_STRIDE)
        val_windows, val_user_ids = create_windows(val_df, WINDOW_SIZE, WINDOW_SIZE)
        test_windows, test_user_ids = create_windows(test_df, WINDOW_SIZE, WINDOW_SIZE)
        
        unique_users = sorted(np.unique(np.concatenate([train_user_ids, val_user_ids, test_user_ids])))
        self.cond_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.rev_cond_map = {idx: user_id for user_id, idx in self.cond_map.items()}
        
        train_cond_idx = np.array([self.cond_map[uid] for uid in train_user_ids])
        val_cond_idx = np.array([self.cond_map[uid] for uid in val_user_ids])
        test_cond_idx = np.array([self.cond_map[uid] for uid in test_user_ids])
        
        self.train_ds = WindowedGaitDataset(train_windows, train_cond_idx)
        self.val_ds = WindowedGaitDataset(val_windows, val_cond_idx)
        self.test_ds = WindowedGaitDataset(test_windows, test_cond_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=VAE_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=VAE_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=VAE_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

