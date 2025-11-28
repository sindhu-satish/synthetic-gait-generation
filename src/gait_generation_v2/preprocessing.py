"""
Preprocessing pipeline for Gait Generation V2.

Implements:
1. Load CSVs from numbered folders
2. Segment on gaps (dt > 0.2s)
3. Resample to 100Hz
4. Compute magnitude
5. Sensor-specific clipping
6. Standardization (fit on train users only)
7. Windowing with stride
8. User balancing (cap datapoints per user)
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
from .config import ConfigV2


class GaitPreprocessor:
    """
    Preprocessor for gait sensor data.
    
    Handles the full pipeline from raw CSVs to normalized windows.
    """
    
    def __init__(self, cfg: ConfigV2, sensor_type: str):
        self.cfg = cfg
        self.sensor_type = sensor_type
        self.clip_values = cfg.get_clip_values(sensor_type)
        
        # Standardization stats (fit on train users)
        self.stats: Optional[Dict[str, Tuple[float, float]]] = None  # {col: (mean, std)}
        
        # User mappings
        self.all_users: List[str] = []
        self.train_users: List[str] = []
        self.val_users: List[str] = []
        self.test_users: List[str] = []
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        
        # Processed data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Dict[str, pd.DataFrame] = {}  # user_id -> processed df
        
        # Windows storage
        self.train_windows: Optional[np.ndarray] = None
        self.train_user_ids: Optional[List[str]] = None
        self.val_windows: Optional[np.ndarray] = None
        self.val_user_ids: Optional[List[str]] = None
        self.test_windows: Optional[np.ndarray] = None
        self.test_user_ids: Optional[List[str]] = None
        
        self.fitted = False
    
    def load_and_process(self):
        """Main entry point: load data and run full preprocessing pipeline."""
        print(f"\n{'='*60}")
        print(f"Preprocessing {self.sensor_type} data")
        print(f"{'='*60}")
        
        # Step 1: Load raw data
        self._load_raw_data()
        
        # Step 2: Process each user's data FIRST (before splitting)
        # This ensures we only split users that have valid processed data
        self._process_all_users()
        
        # Step 3: Split users (only those with valid processed data)
        self._split_users()
        
        # Step 4: Fit standardization on train users
        self._fit_standardization()
        
        # Step 5: Apply standardization and create windows
        self._create_all_windows()
        
        self.fitted = True
        print(f"\nPreprocessing complete for {self.sensor_type}!")
        print(f"  Train windows: {len(self.train_windows) if self.train_windows is not None else 0}")
        print(f"  Val windows: {len(self.val_windows) if self.val_windows is not None else 0}")
        print(f"  Test windows: {len(self.test_windows) if self.test_windows is not None else 0}")
    
    def _load_raw_data(self):
        """Load CSVs from numbered folders (1-117)."""
        gait_base_dir = self.cfg.gait_base_dir
        
        # Resolve path
        if not os.path.isabs(gait_base_dir):
            if not os.path.exists(gait_base_dir):
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_file_dir))
                gait_base_dir = os.path.join(project_root, gait_base_dir)
        
        base_dataset_dir = gait_base_dir.rstrip("/")
        if not os.path.exists(base_dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {base_dataset_dir}")
        
        print(f"Loading {self.sensor_type} data from {base_dataset_dir}...")
        
        all_dfs = []
        users_with_data = set()
        
        for folder_num in range(1, 118):
            folder_path = os.path.join(base_dataset_dir, str(folder_num))
            if not os.path.exists(folder_path):
                continue
            
            pattern = f"*PocketPhone*{self.sensor_type}*.csv"
            sensor_files = glob.glob(os.path.join(folder_path, pattern))
            
            for file_path in sensor_files:
                try:
                    df = pd.read_csv(file_path)
                    if len(df) == 0:
                        continue
                    
                    df["__user_id__"] = str(folder_num)
                    df["__source_file__"] = os.path.basename(file_path)
                    all_dfs.append(df)
                    users_with_data.add(str(folder_num))
                except Exception as e:
                    print(f"  Warning: Failed to read {file_path}: {e}")
        
        if not all_dfs:
            raise ValueError(f"No {self.sensor_type} data found!")
        
        self.raw_data = pd.concat(all_dfs, ignore_index=True)
        self.all_users = sorted(list(users_with_data), key=lambda x: int(x))
        
        print(f"  Loaded {len(self.raw_data):,} rows from {len(users_with_data)} users")
        print(f"  Columns: {list(self.raw_data.columns)}")
    
    def _split_users(self):
        """Split users into train/val/test sets with fixed seed.
        
        Only splits users that have valid processed data.
        """
        np.random.seed(self.cfg.seed)
        
        # Only use users that have valid processed data
        users_with_valid_data = sorted(list(self.processed_data.keys()), key=lambda x: int(x))
        
        if len(users_with_valid_data) == 0:
            raise ValueError(
                f"No users with valid processed data for {self.sensor_type}! "
                f"This may be due to:\n"
                f"  - All segments being too short (< {self.cfg.min_segment_length} samples)\n"
                f"  - Resampling failures\n"
                f"  - Missing required columns (Timestamp, Xvalue, Yvalue, Zvalue)"
            )
        
        print(f"\nUsers with valid processed data: {len(users_with_valid_data)}")
        print(f"  Sample users: {users_with_valid_data[:10]}")
        
        users = users_with_valid_data.copy()
        np.random.shuffle(users)
        
        n_train = min(self.cfg.n_train_users, len(users))
        n_val = min(self.cfg.n_val_users, len(users) - n_train)
        
        self.train_users = users[:n_train]
        self.val_users = users[n_train:n_train + n_val]
        self.test_users = users[n_train + n_val:]
        
        # Create user index mapping (only for users that will be used in training)
        all_split_users = self.train_users + self.val_users + self.test_users
        self.user_to_idx = {u: i for i, u in enumerate(all_split_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        
        print(f"\nUser split (seed={self.cfg.seed}):")
        print(f"  Train: {len(self.train_users)} users")
        print(f"  Val: {len(self.val_users)} users")
        print(f"  Test: {len(self.test_users)} users")
        
        if len(self.train_users) == 0:
            raise ValueError(
                f"No training users available for {self.sensor_type}! "
                f"Need at least 1 user with valid processed data."
            )
    
    def _process_all_users(self):
        """Process data for each user: segment, resample, compute magnitude, clip."""
        print("\nProcessing user data...")
        
        users_processed = 0
        users_failed = 0
        failure_reasons = {"no_data": 0, "column_mapping": 0, "timestamp_convert": 0, "resampling": 0, "other": 0}
        
        # Debug: process first user in detail
        debug_user = self.all_users[0] if self.all_users else None
        
        for user_id in self.all_users:
            user_df = self.raw_data[self.raw_data["__user_id__"] == user_id].copy()
            
            if len(user_df) == 0:
                failure_reasons["no_data"] += 1
                continue
            
            # Cap datapoints per user
            if len(user_df) > self.cfg.max_datapoints_per_user:
                user_df = user_df.iloc[:self.cfg.max_datapoints_per_user]
            
            # Process the user's data
            processed, reason = self._process_user_data(user_df, debug=(user_id == debug_user))
            
            if processed is not None and len(processed) > 0:
                self.processed_data[user_id] = processed
                users_processed += 1
            else:
                users_failed += 1
                if reason:
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        print(f"  Successfully processed: {users_processed} users")
        print(f"  Failed (no valid data): {users_failed} users")
        if users_failed > 0:
            print(f"  Failure reasons: {failure_reasons}")
        
        if users_processed == 0:
            raise ValueError(
                f"No users with valid processed data for {self.sensor_type}! "
                f"All {len(self.all_users)} users failed processing. "
                f"Failure breakdown: {failure_reasons}\n"
                f"Check that data has required columns (Timestamp, Xvalue, Yvalue, Zvalue) "
                f"and segments are long enough (>= {self.cfg.min_segment_length} samples at {self.cfg.target_hz}Hz)."
            )
    
    def _process_user_data(self, df: pd.DataFrame, debug: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Process a single user's data: segment, resample, compute magnitude, clip.
        
        Returns:
            (processed_df, failure_reason) where failure_reason is None if successful
        """
        if debug:
            print(f"\n  [DEBUG] Processing user data: {len(df)} rows")
            print(f"  [DEBUG] Columns: {list(df.columns)}")
        
        # Map column names (handle variations)
        col_mapping = {}
        
        # Timestamp column
        timestamp_cols = ["Timestamp", "timestamp", "Time", "time", "t"]
        timestamp_col = None
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        if timestamp_col is None:
            if debug:
                print(f"  [DEBUG] Failed: No timestamp column found")
            return None, "column_mapping"
        col_mapping["Timestamp"] = timestamp_col
        
        # X, Y, Z columns
        x_cols = ["Xvalue", "X", "x", "AccX", "accX", "Xvalue (m/s²)", "Xvalue (rad/s)"]
        y_cols = ["Yvalue", "Y", "y", "AccY", "accY", "Yvalue (m/s²)", "Yvalue (rad/s)"]
        z_cols = ["Zvalue", "Z", "z", "AccZ", "accZ", "Zvalue (m/s²)", "Zvalue (rad/s)"]
        
        x_col = None
        y_col = None
        z_col = None
        
        for col in x_cols:
            if col in df.columns:
                x_col = col
                break
        for col in y_cols:
            if col in df.columns:
                y_col = col
                break
        for col in z_cols:
            if col in df.columns:
                z_col = col
                break
        
        if x_col is None or y_col is None or z_col is None:
            if debug:
                print(f"  [DEBUG] Failed: Missing sensor columns (X={x_col}, Y={y_col}, Z={z_col})")
            return None, "column_mapping"
        
        col_mapping["Xvalue"] = x_col
        col_mapping["Yvalue"] = y_col
        col_mapping["Zvalue"] = z_col
        
        if debug:
            print(f"  [DEBUG] Column mapping: {col_mapping}")
        
        # Rename columns to standard names
        df = df.rename(columns={v: k for k, v in col_mapping.items()})
        
        if debug:
            print(f"  [DEBUG] Sample timestamp values (first 5): {df['Timestamp'].head().tolist()}")
            print(f"  [DEBUG] Sample X values (first 5): {df['Xvalue'].head().tolist()}")
            print(f"  [DEBUG] Timestamp dtype: {df['Timestamp'].dtype}")
            print(f"  [DEBUG] Xvalue dtype: {df['Xvalue'].dtype}")
        
        # Convert timestamp to numeric (handle datetime strings)
        try:
            timestamp_before = df["Timestamp"].copy()
            
            # First try to parse as datetime if it's a string
            if df["Timestamp"].dtype == 'object':
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                # Convert to numeric (seconds since epoch, or use relative time)
                # Use relative time (seconds from first timestamp) for better numerical stability
                if df["Timestamp"].notna().any():
                    first_timestamp = df["Timestamp"].min()
                    df["Timestamp"] = (df["Timestamp"] - first_timestamp).dt.total_seconds()
                else:
                    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
            else:
                # Already numeric, just ensure it's float
                df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
            
            timestamp_nan_count = df["Timestamp"].isna().sum()
            if debug:
                print(f"  [DEBUG] Timestamp conversion: {timestamp_nan_count}/{len(df)} became NaN")
                if len(df) > 0 and df["Timestamp"].notna().any():
                    print(f"  [DEBUG] Timestamp range: {df['Timestamp'].min():.3f} to {df['Timestamp'].max():.3f} seconds")
        except Exception as e:
            if debug:
                print(f"  [DEBUG] Failed: Timestamp conversion error: {e}")
                import traceback
                traceback.print_exc()
            return None, "timestamp_convert"
        
        # Convert X, Y, Z to numeric as well
        for col in ["Xvalue", "Yvalue", "Zvalue"]:
            if col in df.columns:
                before_nan = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after_nan = df[col].isna().sum()
                if debug and after_nan > before_nan:
                    print(f"  [DEBUG] {col} conversion: {after_nan - before_nan} additional NaN values")
        
        # Drop rows with invalid timestamps or sensor values
        initial_len = len(df)
        df = df.dropna(subset=["Timestamp", "Xvalue", "Yvalue", "Zvalue"])
        
        if debug:
            print(f"  [DEBUG] After conversion: {len(df)} rows (dropped {initial_len - len(df)} invalid)")
            if len(df) == 0:
                print(f"  [DEBUG] Breakdown of NaN values:")
                print(f"    Timestamp NaN: {timestamp_before.isna().sum() if 'timestamp_before' in locals() else 'N/A'}")
                print(f"    After conversion - Timestamp NaN: {df['Timestamp'].isna().sum() if len(df) > 0 else 'all rows dropped'}")
        
        if len(df) == 0:
            if debug:
                print(f"  [DEBUG] Failed: No valid rows after conversion")
            return None, "timestamp_convert"
        
        # Sort by timestamp
        df = df.sort_values("Timestamp").reset_index(drop=True)
        
        if debug:
            print(f"  [DEBUG] Timestamp range: {df['Timestamp'].min():.3f} to {df['Timestamp'].max():.3f}")
            print(f"  [DEBUG] Duration: {df['Timestamp'].max() - df['Timestamp'].min():.3f} seconds")
        
        # Segment on gaps
        df = self._segment_by_gaps(df)
        
        n_segments = df["__segment_id__"].nunique()
        if debug:
            print(f"  [DEBUG] Created {n_segments} segments")
        
        # Resample each segment to target Hz
        resampled_segments = []
        for seg_id in df["__segment_id__"].unique():
            seg_df = df[df["__segment_id__"] == seg_id]
            resampled = self._resample_segment(seg_df)
            if resampled is not None:
                resampled_segments.append(resampled)
        
        if debug:
            print(f"  [DEBUG] Resampled segments: {len(resampled_segments)}/{n_segments}")
        
        if not resampled_segments:
            if debug:
                print(f"  [DEBUG] Failed: No valid resampled segments")
            return None, "resampling"
        
        df = pd.concat(resampled_segments, ignore_index=True)
        
        if debug:
            print(f"  [DEBUG] Final processed data: {len(df)} rows")
        
        # Compute magnitude
        df["magnitude"] = np.sqrt(
            df["Xvalue"]**2 + df["Yvalue"]**2 + df["Zvalue"]**2
        )
        
        # Apply clipping (use standard column names after renaming)
        for col, (lo, hi) in self.clip_values.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        
        # Final check: remove any remaining NaN/Inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) == 0:
            if debug:
                print(f"  [DEBUG] Failed: All rows dropped after NaN removal")
            return None, "resampling"
        
        return df, None
    
    def _segment_by_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split data into segments where dt > gap_threshold."""
        df = df.copy()
        dt = df["Timestamp"].diff()
        segment_starts = (dt > self.cfg.gap_threshold_sec) | (dt.isna())
        df["__segment_id__"] = segment_starts.cumsum()
        return df
    
    def _resample_segment(self, seg_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample a segment to target Hz using linear interpolation."""
        if len(seg_df) < 2:
            return None
        
        t = seg_df["Timestamp"].values
        t_start, t_end = t.min(), t.max()
        duration = t_end - t_start
        
        if duration < 0.01:  # Skip very short segments (< 10ms)
            return None
        
        # Remove duplicate timestamps (keep first occurrence)
        seg_df = seg_df.drop_duplicates(subset=["Timestamp"], keep="first").sort_values("Timestamp").reset_index(drop=True)
        t = seg_df["Timestamp"].values
        
        if len(seg_df) < 2:
            return None
        
        # Recalculate after deduplication
        t_start, t_end = t.min(), t.max()
        duration = t_end - t_start
        
        if duration < 0.01:
            return None
        
        n_samples = int(duration * self.cfg.target_hz) + 1
        
        # Allow segments shorter than min_segment_length, but they won't create windows
        # This is less strict - we'll filter during windowing instead
        if n_samples < 10:  # At least 10 samples (0.1s at 100Hz)
            return None
        
        t_new = np.linspace(t_start, t_end, n_samples)
        
        resampled_data = {"Timestamp": t_new}
        
        for col in ["Xvalue", "Yvalue", "Zvalue"]:
            if col in seg_df.columns:
                # Linear interpolation
                try:
                    # Only interpolate if we have unique timestamps
                    if len(np.unique(t)) < len(t):
                        # Handle duplicates by taking mean value for duplicate timestamps
                        unique_df = seg_df.groupby("Timestamp")[col].mean().reset_index()
                        t_unique = unique_df["Timestamp"].values
                        values_unique = unique_df[col].values
                    else:
                        t_unique = t
                        values_unique = seg_df[col].values
                    
                    if len(t_unique) < 2:
                        return None
                    
                    f = interpolate.interp1d(t_unique, values_unique, kind="linear", fill_value="extrapolate", bounds_error=False)
                    resampled_values = f(t_new)
                    
                    # Check for NaN/Inf from interpolation
                    if np.isnan(resampled_values).any() or np.isinf(resampled_values).any():
                        # Fill NaN/Inf with nearest valid value
                        mask = ~(np.isnan(resampled_values) | np.isinf(resampled_values))
                        if mask.any():
                            resampled_values = np.interp(t_new, t_new[mask], resampled_values[mask])
                        else:
                            return None
                    
                    resampled_data[col] = resampled_values
                except Exception as e:
                    # If interpolation fails, skip this segment
                    return None
        
        # Preserve user_id and segment_id
        resampled_data["__user_id__"] = seg_df["__user_id__"].iloc[0]
        resampled_data["__segment_id__"] = seg_df["__segment_id__"].iloc[0]
        
        return pd.DataFrame(resampled_data)
    
    def _fit_standardization(self):
        """Fit standardization stats on train users only."""
        print("\nFitting standardization on train users...")
        
        train_dfs = []
        missing_users = []
        for user_id in self.train_users:
            if user_id in self.processed_data:
                train_dfs.append(self.processed_data[user_id])
            else:
                missing_users.append(user_id)
        
        if not train_dfs:
            raise ValueError(
                f"No training data available for {self.sensor_type}!\n"
                f"  Train users requested: {len(self.train_users)}\n"
                f"  Users with valid data: {len(train_dfs)}\n"
                f"  Missing users: {missing_users[:10] if missing_users else 'None'}\n"
                f"  Total users with processed data: {len(self.processed_data)}\n"
                f"  Available user IDs: {sorted(list(self.processed_data.keys()), key=lambda x: int(x))[:20]}"
            )
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        self.stats = {}
        for col in ["Xvalue", "Yvalue", "Zvalue", "magnitude"]:
            if col in train_df.columns:
                mean = train_df[col].mean()
                std = train_df[col].std()
                if std < 1e-8:
                    std = 1.0
                self.stats[col] = (float(mean), float(std))
                print(f"  {col}: μ={mean:.4f}, σ={std:.4f}")
    
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standardization and scale to approximately [-1, 1]."""
        df = df.copy()
        
        for col, (mean, std) in self.stats.items():
            if col in df.columns:
                # Standardize: (x - μ) / σ
                df[col] = (df[col] - mean) / std
                # Scale to [-1, 1]: clip(x / 4, -1, 1)
                # Changed from /3.0 to /4.0 to be less aggressive and preserve more information
                df[col] = np.clip(df[col] / 4.0, -1.0, 1.0)
        
        return df
    
    def _create_all_windows(self):
        """Create windows for train/val/test splits."""
        print("\nCreating windows...")
        
        # Train windows
        self.train_windows, self.train_user_ids = self._create_windows_for_users(
            self.train_users, "train"
        )
        
        # Val windows
        self.val_windows, self.val_user_ids = self._create_windows_for_users(
            self.val_users, "val"
        )
        
        # Test windows
        self.test_windows, self.test_user_ids = self._create_windows_for_users(
            self.test_users, "test"
        )
    
    def _create_windows_for_users(
        self, 
        users: List[str], 
        split_name: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Create windows for a set of users."""
        all_windows = []
        all_user_ids = []
        
        for user_id in users:
            if user_id not in self.processed_data:
                continue
            
            df = self.processed_data[user_id]
            
            # Standardize
            df = self._standardize(df)
            
            # Create windows per segment
            for seg_id in df["__segment_id__"].unique():
                seg_df = df[df["__segment_id__"] == seg_id]
                
                if len(seg_df) < self.cfg.window_size:
                    continue
                
                # Extract feature columns
                features = seg_df[["Xvalue", "Yvalue", "Zvalue", "magnitude"]].values
                
                # Create windows with stride
                for start in range(0, len(features) - self.cfg.window_size + 1, self.cfg.stride):
                    window = features[start:start + self.cfg.window_size]
                    all_windows.append(window)
                    all_user_ids.append(user_id)
        
        if all_windows:
            windows_array = np.array(all_windows, dtype=np.float32)
        else:
            windows_array = np.zeros((0, self.cfg.window_size, 4), dtype=np.float32)
        
        print(f"  {split_name}: {len(all_windows)} windows from {len(set(all_user_ids))} users")
        
        return windows_array, all_user_ids
    
    def inverse_standardize(self, windows: np.ndarray) -> np.ndarray:
        """
        Inverse standardization: convert from [-1, 1] back to physical units.
        
        Args:
            windows: (N, window_size, 4) array in normalized space
            
        Returns:
            (N, window_size, 4) array in physical units
        """
        windows = windows.copy()
        cols = ["Xvalue", "Yvalue", "Zvalue", "magnitude"]
        
        for i, col in enumerate(cols):
            if col in self.stats:
                mean, std = self.stats[col]
                # Reverse: x_norm -> x_std -> x_raw
                # x_norm = clip(x_std / 4, -1, 1)
                # x_std = (x_raw - mean) / std
                # Approximate inverse (some information lost due to clipping)
                windows[..., i] = windows[..., i] * 4.0  # Undo /4
                windows[..., i] = windows[..., i] * std + mean  # Undo standardization
        
        return windows
    
    def save(self, path: str):
        """Save preprocessor state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "sensor_type": self.sensor_type,
            "stats": self.stats,
            "all_users": self.all_users,
            "train_users": self.train_users,
            "val_users": self.val_users,
            "test_users": self.test_users,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "clip_values": self.clip_values,
            "train_windows": self.train_windows,
            "train_user_ids": self.train_user_ids,
            "val_windows": self.val_windows,
            "val_user_ids": self.val_user_ids,
            "test_windows": self.test_windows,
            "test_user_ids": self.test_user_ids,
            "fitted": self.fitted,
            "cfg_params": {
                "window_size": self.cfg.window_size,
                "stride": self.cfg.stride,
                "target_hz": self.cfg.target_hz,
                "gap_threshold_sec": self.cfg.gap_threshold_sec,
                "max_datapoints_per_user": self.cfg.max_datapoints_per_user,
            }
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        print(f"Saved preprocessor to {path}")
    
    @staticmethod
    def load(path: str) -> "GaitPreprocessor":
        """Load preprocessor state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        # Create a minimal config for loading
        cfg = ConfigV2()
        if "cfg_params" in state:
            cfg.window_size = state["cfg_params"]["window_size"]
            cfg.stride = state["cfg_params"]["stride"]
            cfg.target_hz = state["cfg_params"]["target_hz"]
            cfg.gap_threshold_sec = state["cfg_params"]["gap_threshold_sec"]
            cfg.max_datapoints_per_user = state["cfg_params"]["max_datapoints_per_user"]
        
        preprocessor = GaitPreprocessor(cfg, state["sensor_type"])
        preprocessor.stats = state["stats"]
        preprocessor.all_users = state["all_users"]
        preprocessor.train_users = state["train_users"]
        preprocessor.val_users = state["val_users"]
        preprocessor.test_users = state["test_users"]
        preprocessor.user_to_idx = state["user_to_idx"]
        preprocessor.idx_to_user = state["idx_to_user"]
        preprocessor.clip_values = state["clip_values"]
        preprocessor.train_windows = state["train_windows"]
        preprocessor.train_user_ids = state["train_user_ids"]
        preprocessor.val_windows = state["val_windows"]
        preprocessor.val_user_ids = state["val_user_ids"]
        preprocessor.test_windows = state["test_windows"]
        preprocessor.test_user_ids = state["test_user_ids"]
        preprocessor.fitted = state["fitted"]
        
        return preprocessor

