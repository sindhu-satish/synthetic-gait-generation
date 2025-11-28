import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from ..eda.gait_signal_analysis import compute_features

sns.set_style("whitegrid")

class UserClassifier(nn.Module):
    def __init__(self, input_dim=8, num_users=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_users)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_and_evaluate(X_train, y_train, X_test, y_test, num_users, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    model = UserClassifier(input_dim=X_train.shape[1], num_users=num_users)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average='macro')
    
    return accuracy, f1

def run_augmentation_experiment(real_windows, real_user_ids, synth_windows, synth_user_ids, sensor_type: str, save_dir: str, n_seeds=5, low_data_ratio=0.25):
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if synthetic windows are empty
    if len(synth_windows) == 0:
        print(f"Warning: No synthetic data available for {sensor_type}. Skipping augmentation experiment.")
        return {}
    
    real_features = compute_features(real_windows)
    synth_features = compute_features(synth_windows)
    
    feature_cols = ["mean_mag", "std_mag", "rms_mag", "jerk_std", "dominant_freq", "corr_xy", "corr_xz", "corr_yz"]
    
    # Check if synth_features has the required columns
    if len(synth_features) == 0 or not all(col in synth_features.columns for col in feature_cols):
        print(f"Warning: Synthetic features are empty or incomplete for {sensor_type}. Skipping augmentation experiment.")
        return {}
    
    X_real = real_features[feature_cols].values.astype(np.float32)
    unique_users = np.unique(real_user_ids)
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    y_real = np.array([user_to_idx[uid] for uid in real_user_ids])
    
    X_synth = synth_features[feature_cols].values.astype(np.float32)
    y_synth = np.array([user_to_idx.get(uid, 0) for uid in synth_user_ids])
    
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    
    num_users = len(unique_users)
    N_full = len(X_real_train)
    N_low = int(N_full * low_data_ratio)
    
    results_full = {"real_only": [], "real_synth": [], "real_bootstrap": []}
    results_low = {"real_only": [], "real_synth": [], "real_bootstrap": []}
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        indices_full = np.random.choice(len(X_real_train), N_full, replace=False)
        indices_low = np.random.choice(len(X_real_train), N_low, replace=False)
        
        for regime, N, results_dict in [("full", N_full, results_full), ("low", N_low, results_low)]:
            if regime == "full":
                indices = indices_full
            else:
                indices = indices_low
            
            X_train_subset = X_real_train[indices]
            y_train_subset = y_real_train[indices]
            
            acc_real, f1_real = train_and_evaluate(
                X_train_subset, y_train_subset, X_real_test, y_real_test, num_users, seed=seed
            )
            results_dict["real_only"].append(f1_real)
            
            N_half = N // 2
            X_train_half = X_train_subset[:N_half]
            y_train_half = y_train_subset[:N_half]
            
            synth_indices = np.random.choice(len(X_synth), N_half, replace=False)
            X_train_synth = np.vstack([X_train_half, X_synth[synth_indices]])
            y_train_synth = np.concatenate([y_train_half, y_synth[synth_indices]])
            
            acc_synth, f1_synth = train_and_evaluate(
                X_train_synth, y_train_synth, X_real_test, y_real_test, num_users, seed=seed
            )
            results_dict["real_synth"].append(f1_synth)
            
            bootstrap_indices = np.random.choice(len(X_train_subset), N_half, replace=True)
            X_train_bootstrap = np.vstack([X_train_half, X_train_subset[bootstrap_indices]])
            y_train_bootstrap = np.concatenate([y_train_half, y_train_subset[bootstrap_indices]])
            
            acc_bootstrap, f1_bootstrap = train_and_evaluate(
                X_train_bootstrap, y_train_bootstrap, X_real_test, y_real_test, num_users, seed=seed
            )
            results_dict["real_bootstrap"].append(f1_bootstrap)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (regime, results_dict) in enumerate([("Full Data", results_full), ("Low Data (25%)", results_low)]):
        ax = axes[idx]
        data = [results_dict["real_only"], results_dict["real_synth"], results_dict["real_bootstrap"]]
        labels = ["Real Only", "Real + Synthetic", "Real + Bootstrap"]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        means = [np.mean(d) for d in data]
        stds = [np.std(d) for d in data]
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i+1, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        real_only = np.array(results_dict["real_only"])
        real_synth = np.array(results_dict["real_synth"])
        real_bootstrap = np.array(results_dict["real_bootstrap"])
        
        t_stat_synth, p_val_synth = ttest_rel(real_only, real_synth)
        t_stat_bootstrap, p_val_bootstrap = ttest_rel(real_only, real_bootstrap)
        
        y_max = max([np.max(d) for d in data]) * 1.15
        if p_val_synth < 0.05:
            ax.plot([1, 2], [y_max*0.95, y_max*0.95], 'k-', linewidth=1)
            ax.text(1.5, y_max*0.97, f'p={p_val_synth:.3f}', ha='center', fontsize=8)
        if p_val_bootstrap < 0.05:
            ax.plot([1, 3], [y_max*0.90, y_max*0.90], 'k-', linewidth=1)
            ax.text(2, y_max*0.92, f'p={p_val_bootstrap:.3f}', ha='center', fontsize=8)
        
        ax.set_ylabel("Macro F1 Score")
        ax.set_title(f"{regime}")
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"augmentation_effectiveness_{sensor_type.lower()}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Augmentation experiment complete. Results saved to {save_dir}")
    
    return {
        "full_data": results_full,
        "low_data": results_low
    }

