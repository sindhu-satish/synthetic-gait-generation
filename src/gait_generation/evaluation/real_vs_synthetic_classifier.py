import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from ..eda.gait_signal_analysis import compute_features

sns.set_style("whitegrid")

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

def evaluate_real_vs_synthetic(
    real_windows,
    synth_windows,
    sensor_type: str,
    save_dir: str,
    seed=42,
    split_mode="window",
    real_user_ids=None,
    synth_user_ids=None
):
    """
    Evaluate real vs synthetic classifier.
    
    Args:
        real_windows: Real windows array
        synth_windows: Synthetic windows array
        sensor_type: Sensor type
        save_dir: Save directory
        seed: Random seed
        split_mode: "window" (window-level split) or "user_disjoint" (user-level split)
        real_user_ids: User IDs for real windows (required for user_disjoint split)
        synth_user_ids: User IDs for synthetic windows (required for user_disjoint split)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    
    if len(synth_windows) == 0:
        print(f"Warning: No synthetic data available for {sensor_type}. Skipping realism evaluation.")
        return {
            "sensor_type": sensor_type,
            "accuracy": None,
            "auc": None,
            "eer": None,
            "confusion_matrix": None,
            "error": "No synthetic data available"
        }
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    real_features = compute_features(real_windows)
    synth_features = compute_features(synth_windows)
    
    feature_cols = ["mean_mag", "std_mag", "rms_mag", "jerk_std", "dominant_freq", "corr_xy", "corr_xz", "corr_yz"]
    
    
    if len(synth_features) == 0 or not all(col in synth_features.columns for col in feature_cols):
        print(f"Warning: Synthetic features are empty or incomplete for {sensor_type}. Skipping realism evaluation.")
        return {
            "sensor_type": sensor_type,
            "accuracy": None,
            "auc": None,
            "eer": None,
            "confusion_matrix": None,
            "error": "Synthetic features incomplete"
        }
    
    X_real = real_features[feature_cols].values.astype(np.float32)
    X_synth = synth_features[feature_cols].values.astype(np.float32)
    
    min_samples = min(len(X_real), len(X_synth))
    X_real = X_real[:min_samples]
    X_synth = X_synth[:min_samples]
    
    if real_user_ids is not None:
        real_user_ids = real_user_ids[:min_samples]
    if synth_user_ids is not None:
        synth_user_ids = synth_user_ids[:min_samples]
    
    X = np.vstack([X_real, X_synth])
    y = np.array([1] * len(X_real) + [0] * len(X_synth))
    
    if split_mode == "user_disjoint":
        if real_user_ids is None or synth_user_ids is None:
            raise ValueError("real_user_ids and synth_user_ids are required for user_disjoint split")
        
        all_user_ids = np.concatenate([real_user_ids, synth_user_ids])
        unique_users = sorted(np.unique(all_user_ids))
        
        np.random.seed(seed)
        np.random.shuffle(unique_users)
        
        n_train_users = int(len(unique_users) * 0.8)
        train_users = set(unique_users[:n_train_users])
        test_users = set(unique_users[n_train_users:])
        
        assert train_users.isdisjoint(test_users), "Train and test users must be disjoint"
        
        user_ids_all = np.concatenate([real_user_ids, synth_user_ids])
        
        train_mask = np.array([uid in train_users for uid in user_ids_all])
        test_mask = np.array([uid in test_users for uid in user_ids_all])
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        num_train_users = len(train_users)
        num_test_users = len(test_users)
        num_train_samples = len(X_train)
        num_test_samples = len(X_test)
        
        print(f"\n[Classifier - User-Disjoint Split]")
        print(f"  Train users: {num_train_users}, Train samples: {num_train_samples}")
        print(f"  Test users: {num_test_users}, Test samples: {num_test_samples}")
        
    else:  # split_mode == "window"
        indices = np.arange(len(X))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, indices, test_size=0.2, random_state=seed, stratify=y
        )
        num_train_users = None
        num_test_users = None
        num_train_samples = len(X_train)
        num_test_samples = len(X_test)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    model = BinaryClassifier(input_dim=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    eer, _ = compute_eer(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Real vs Synthetic Classifier Results ({sensor_type}, split_mode={split_mode}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  EER: {eer:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xticklabels(["Synthetic", "Real"])
    axes[1].set_yticklabels(["Synthetic", "Real"])
    
    plt.tight_layout()
    filename = f"real_vs_synth_classifier_{sensor_type.lower()}_{split_mode}_split.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    
    results = {
        "sensor_type": sensor_type,
        "split_mode": split_mode,
        "accuracy": accuracy,
        "auc": auc,
        "eer": eer,
        "confusion_matrix": cm.tolist(),
        "num_train_users": num_train_users,
        "num_test_users": num_test_users,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
        "seed": seed
    }
    
    json_path = os.path.join(save_dir, f"classifier_{split_mode}_split_{sensor_type.lower()}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

