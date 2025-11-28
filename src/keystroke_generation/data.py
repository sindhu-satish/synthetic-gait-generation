import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from .preprocessor import Preprocessor
from .config import Config

def load_csv_folder(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_file__"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
    data = pd.concat(dfs, ignore_index=True, sort=False)
    return data, paths

def auto_cond_columns(df: pd.DataFrame):
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["user", "session", "device", "platform", "type", "digraph", "unigraph", "pair"])]
    return candidates[:3]

class KeystrokeDataset(Dataset):
    def __init__(self, num_x, cat_idx, cond_idx=None, labels=None):
        self.num_x = torch.tensor(num_x, dtype=torch.float32)
        self.cat_idx = torch.tensor(cat_idx, dtype=torch.long) if cat_idx.size > 0 else torch.zeros((len(num_x), 0), dtype=torch.long)
        self.cond_idx = torch.tensor(cond_idx, dtype=torch.long) if cond_idx is not None else None
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
    
    def __len__(self):
        return self.num_x.shape[0]
    
    def __getitem__(self, i):
        out = {"num": self.num_x[i]}
        out["cat"] = self.cat_idx[i] if self.cat_idx.shape[1] > 0 else torch.zeros((0,), dtype=torch.long)
        if self.cond_idx is not None:
            out["cond"] = self.cond_idx[i]
        if self.labels is not None:
            out["label"] = self.labels[i]
        return out

class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, cfg: Config):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.pre = Preprocessor(rare_thresh=cfg.rare_thresh)
        self.cond_cols = list(cfg.cond_cols) if cfg.cond_cols else auto_cond_columns(self.df)
        self.cond_map = {}
        self.rev_cond_map = {}
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage=None):
        df = self.df.copy()
        
        cond_idx = None
        if self.cond_cols:
            cond_str = df[self.cond_cols].astype(str).agg("|".join, axis=1)
            cond_id = cond_str.apply(lambda s: hash(s) % self.cfg.num_cond_buckets)
            cond_idx = cond_id.values.astype(np.int64)
        
        label_col = None
        for cand in ["digraph", "pair", "type", "device", "Device", "User", "user", "session"]:
            if cand in df.columns:
                label_col = cand
                break
        labels = df[label_col].factorize()[0] if label_col is not None else None

        idx = np.arange(len(df))
        strat = labels if labels is not None else cond_idx
        idx_train, idx_tmp = train_test_split(idx, test_size=self.cfg.val_size + self.cfg.test_size, random_state=self.cfg.seed, stratify=strat if strat is not None else None)
        idx_val, idx_test = train_test_split(idx_tmp, test_size=self.cfg.test_size/(self.cfg.val_size + self.cfg.test_size), random_state=self.cfg.seed, stratify=strat[idx_tmp] if strat is not None else None)

        self.pre.fit(df.iloc[idx_train])

        def make_ds(rows):
            num_x, cat_idx = self.pre.transform(df.iloc[rows])
            cond = cond_idx[rows] if cond_idx is not None else None
            y = labels[rows] if labels is not None else None
            return KeystrokeDataset(num_x, cat_idx, cond, y)

        self.train_ds = make_ds(idx_train)
        self.val_ds = make_ds(idx_val)
        self.test_ds = make_ds(idx_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.vae_batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.vae_batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.cfg.vae_batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)

