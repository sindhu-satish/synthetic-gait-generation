import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self, rare_thresh: int = 10, numeric_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None):
        self.rare_thresh = rare_thresh
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.imputer = None
        self.scaler = None
        self.clip_lo = None
        self.clip_hi = None
        self.vocabs: Dict[str, List[str]] = {}
        self.token2idx: Dict[str, Dict[str, int]] = {}
        self.idx2token: Dict[str, Dict[int, str]] = {}
        self.fitted = False

    def auto_detect(self, df: pd.DataFrame):
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
        self.categorical_cols = [c for c in self.categorical_cols if c != "__source_file__"]
        return self.numeric_cols, self.categorical_cols

    def fit(self, df: pd.DataFrame):
        self.auto_detect(df)
        num_df = df[self.numeric_cols].copy() if self.numeric_cols else pd.DataFrame(index=df.index)
        
        if self.numeric_cols:
            imp = SimpleImputer(strategy="median")
            num_imp = imp.fit_transform(num_df)
            self.imputer = imp
            
            q_lo = np.nanpercentile(num_imp, 1, axis=0)
            q_hi = np.nanpercentile(num_imp, 99, axis=0)
            self.clip_lo, self.clip_hi = q_lo, q_hi
            num_clip = np.clip(num_imp, q_lo, q_hi)
            scaler = StandardScaler()
            scaler.fit(num_clip)
            self.scaler = scaler

        self.vocabs, self.token2idx, self.idx2token = {}, {}, {}
        for col in self.categorical_cols:
            vc = df[col].astype(str).fillna("__NA__").value_counts()
            vocab = vc[vc >= self.rare_thresh].index.tolist()
            if "__RARE__" not in vocab:
                vocab.append("__RARE__")
            self.vocabs[col] = vocab
            t2i = {tok: i for i, tok in enumerate(vocab)}
            self.token2idx[col] = t2i
            self.idx2token[col] = {i: t for t, i in t2i.items()}
        self.fitted = True

    def transform(self, df: pd.DataFrame):
        assert self.fitted, "call fit first"
        
        if self.numeric_cols:
            num = df[self.numeric_cols].copy()
            num_imp = self.imputer.transform(num)
            num_clip = np.clip(num_imp, self.clip_lo, self.clip_hi)
            num_scaled = self.scaler.transform(num_clip).astype(np.float32)
        else:
            num_scaled = np.zeros((len(df), 0), dtype=np.float32)

        cats_idx = []
        for col in self.categorical_cols:
            vals = df[col].astype(str).fillna("__NA__").values
            t2i = self.token2idx[col]
            idx = np.array([t2i.get(v, t2i.get("__RARE__")) for v in vals], dtype=np.int64)
            cats_idx.append(idx)
        cat_matrix = np.stack(cats_idx, axis=1) if cats_idx else np.zeros((len(df), 0), dtype=np.int64)
        return num_scaled, cat_matrix

    def inverse_transform(self, num_scaled: np.ndarray, cat_logits_list: List[np.ndarray]):
        if num_scaled.shape[1] > 0:
            num_clip = self.scaler.inverse_transform(num_scaled)
            num_inv = num_clip
        else:
            num_inv = np.zeros((num_scaled.shape[0], 0))
        
        cat_dfs = []
        for col, logits in zip(self.categorical_cols, cat_logits_list):
            idx = logits.argmax(axis=1)
            tok_map = self.idx2token[col]
            tokens = [tok_map.get(int(i), "__UNK__") for i in idx]
            cat_dfs.append(pd.Series(tokens, name=col))
        cat_df = pd.concat(cat_dfs, axis=1) if cat_dfs else pd.DataFrame(index=range(num_scaled.shape[0]))
        
        out = pd.DataFrame(num_inv, columns=self.numeric_cols)
        for col in self.categorical_cols:
            out[col] = cat_df[col]
        return out

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "rare_thresh": self.rare_thresh,
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
                "scaler": self.scaler,
                "imputer": self.imputer,
                "clip_lo": self.clip_lo,
                "clip_hi": self.clip_hi,
                "vocabs": self.vocabs,
                "token2idx": self.token2idx,
                "idx2token": self.idx2token,
                "fitted": self.fitted
            }, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        p = Preprocessor(d["rare_thresh"], d["numeric_cols"], d["categorical_cols"])
        p.scaler = d["scaler"]
        p.imputer = d["imputer"]
        p.clip_lo = d["clip_lo"]
        p.clip_hi = d["clip_hi"]
        p.vocabs = d["vocabs"]
        p.token2idx = d["token2idx"]
        p.idx2token = d["idx2token"]
        p.fitted = d["fitted"]
        return p

