import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from .preprocessor import Preprocessor

def vectorize_for_sklearn(df: pd.DataFrame, pre: Preprocessor):
    num, cat_idx = pre.transform(df)
    parts = [num]
    for j, col in enumerate(pre.categorical_cols):
        V = len(pre.vocabs[col])
        onehot = np.zeros((len(df), V), dtype=np.float32)
        onehot[np.arange(len(df)), cat_idx[:, j]] = 1.0
        parts.append(onehot)
    X = np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=np.float32)
    return X

def evaluate_utility(real_df, synth_df, pre: Preprocessor):
    if pre.numeric_cols:
        real_num = real_df[pre.numeric_cols]
        synth_num = synth_df[pre.numeric_cols]
        stats = pd.DataFrame({
            "real_mean": real_num.mean(),
            "synth_mean": synth_num.mean(),
            "real_std": real_num.std(),
            "synth_std": synth_num.std()
        })
        print("Numeric mean/std comparison (first 10 rows):")
        print(stats.head(10))
        
        r_corr = real_num.corr().fillna(0)
        s_corr = synth_num.corr().fillna(0)
        corr_diff = (r_corr - s_corr).abs().values.mean()
        print(f"Avg absolute difference in pairwise correlations: {corr_diff:.4f}")
    else:
        print("No numeric columns for stats.")
    
    label_col = None
    for cand in ["digraph", "pair", "type", "device", "Device", "User", "user", "session"]:
        if cand in real_df.columns:
            label_col = cand
            break
    
    if label_col is not None:
        X_tr = vectorize_for_sklearn(synth_df, pre)
        y_te = real_df[label_col].factorize()[0]
        X_te = vectorize_for_sklearn(real_df, pre)
        try:
            clf = LogisticRegression(max_iter=1000, n_jobs=None)
            if label_col in synth_df.columns:
                y_tr = synth_df[label_col].factorize()[0]
            else:
                y_tr = np.random.randint(0, len(np.unique(y_te)), size=len(synth_df))
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_te, y_te)
            print(f"TSTR accuracy (LogReg on '{label_col}'): {acc:.3f}")
        except Exception as e:
            print(f"TSTR failed: {e}")
    else:
        print("No obvious label column; skipping TSTR.")

def evaluate_privacy(real_train_df, synth_df, pre: Preprocessor):
    X_real = vectorize_for_sklearn(real_train_df, pre)
    X_syn = vectorize_for_sklearn(synth_df, pre)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_real)
    dists, _ = nbrs.kneighbors(X_syn)
    dists = dists[:, 0]
    print("Nearest-neighbor distances (synth -> real):")
    print(pd.Series(dists).describe())

    X = np.vstack([X_real, X_syn])
    y = np.array([1]*len(X_real) + [0]*len(X_syn))
    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, prob)
        print(f"Real-vs-Synthetic AUC (0.5 ideal): {auc:.3f}")
    except Exception as e:
        print(f"AUC computation failed: {e}")

