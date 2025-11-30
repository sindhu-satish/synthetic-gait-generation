import numpy as np
import torch
from typing import Dict, Tuple, Optional


def compute_hf_power_ratio(
    windows: np.ndarray,
    fs_hz: float = 100.0,
    cutoff_hz: float = 10.0,
    exclude_dc: bool = True
) -> Tuple[float, float]:
    """
    Compute high frequency band power ratio for each window.
    
    Args:
        windows: (B, T, 3) array of windows
        fs_hz: Sampling rate in Hz
        cutoff_hz: Frequency cutoff (default 10 Hz)
        exclude_dc: Whether to exclude DC component from total power
        
    Returns:
        (mean_ratio, p95_ratio) across windows
    """
    if windows.size == 0 or windows.shape[0] == 0:
        return 0.0, 0.0
    
    ratios = []
    nyquist = fs_hz / 2.0
    
    for window in windows:
        window_ratios = []
        for axis in range(3):
            x = window[:, axis]
            n = len(x)
            if n < 2:
                continue
            
            X = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
            power = np.abs(X) ** 2
            
            start_idx = 1 if exclude_dc else 0
            total_power = float(power[start_idx:].sum()) + 1e-12
            
            hf_mask = (freqs > cutoff_hz) & (freqs <= nyquist)
            hf_power = float(power[hf_mask].sum())
            
            ratio = hf_power / total_power
            window_ratios.append(ratio)
        
        if window_ratios:
            ratios.append(np.mean(window_ratios))
    
    if not ratios:
        return 0.0, 0.0
    
    ratios = np.array(ratios)
    return float(np.mean(ratios)), float(np.percentile(ratios, 95))


def compute_jerk_tail_statistic(
    windows: np.ndarray,
    percentile: float = 95.0
) -> Tuple[float, float]:
    """
    Compute jerk tail statistic (percentile of jerk magnitude).
    
    Jerk is computed as first difference of magnitude:
    jerk[t] = ||x[t] - x[t-1]||_2
    
    Args:
        windows: (B, T, 3) array of windows
        percentile: Percentile to compute (95 or 99)
        
    Returns:
        (mean_jerk_tail, p95_jerk_tail) across windows
    """
    if windows.size == 0 or windows.shape[0] == 0:
        return 0.0, 0.0
    
    window_tails = []
    
    for window in windows:
        if window.shape[0] < 2:
            continue
        
        mag = np.sqrt((window ** 2).sum(axis=1))
        
        jerk = np.diff(mag)
        jerk_abs = np.abs(jerk)
        
        if len(jerk_abs) == 0:
            continue
        
        tail_val = np.percentile(jerk_abs, percentile)
        window_tails.append(float(tail_val))
    
    if not window_tails:
        return 0.0, 0.0
    
    window_tails = np.array(window_tails)
    return float(np.mean(window_tails)), float(np.percentile(window_tails, 95))


def compute_dominant_freq_tail_mass(
    windows: np.ndarray,
    fs_hz: float = 100.0,
    cutoff_hz: float = 10.0
) -> Tuple[float, float]:
    """
    Compute dominant frequency tail mass above cutoff.
    
    For each window:
    1. Compute magnitude signal m[t] = ||x[t]||_2
    2. Compute power spectrum of m[t]
    3. Find dominant frequency (argmax power excluding DC)
    4. Compute tail mass: sum(power[f>cutoff]) / sum(power[f>0])
    
    Args:
        windows: (B, T, 3) array of windows
        fs_hz: Sampling rate in Hz
        cutoff_hz: Frequency cutoff (default 10 Hz)
        
    Returns:
        (mean_tail_mass, p95_tail_mass) across windows
    """
    if windows.size == 0 or windows.shape[0] == 0:
        return 0.0, 0.0
    
    tail_masses = []
    
    for window in windows:
        n = window.shape[0]
        if n < 2:
            continue
        
        mag = np.sqrt((window ** 2).sum(axis=1))
        
        mag_detrended = mag - np.mean(mag)
        hann = np.hanning(n)
        mag_windowed = mag_detrended * hann
        
        fft_vals = np.fft.rfft(mag_windowed)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
        power = np.abs(fft_vals) ** 2
        
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]
        
        if len(power_no_dc) == 0:
            continue
        
        total_power = float(power_no_dc.sum()) + 1e-12
        
        tail_mask = freqs_no_dc > cutoff_hz
        tail_power = float(power_no_dc[tail_mask].sum())
        
        tail_mass = tail_power / total_power
        tail_masses.append(float(tail_mass))
    
    if not tail_masses:
        return 0.0, 0.0
    
    tail_masses = np.array(tail_masses)
    return float(np.mean(tail_masses)), float(np.percentile(tail_masses, 95))


def compute_realism_metrics(
    windows: np.ndarray,
    fs_hz: float = 100.0,
    hf_cutoff_hz: float = 10.0,
    jerk_percentile: float = 95.0
) -> Dict[str, float]:
    """
    Compute all three realism metrics on a batch of windows.
    
    Args:
        windows: (B, T, 3) array of windows
        fs_hz: Sampling rate in Hz
        hf_cutoff_hz: High frequency cutoff (default 10 Hz)
        jerk_percentile: Jerk percentile to compute (95 or 99)
        
    Returns:
        Dictionary with all metrics:
        - hf_power_ratio_mean, hf_power_ratio_p95
        - jerk_tail_mean, jerk_tail_p95
        - dom_tail_mass_mean, dom_tail_mass_p95
    """
    windows = np.asarray(windows)
    if windows.ndim != 3 or windows.shape[2] != 3:
        raise ValueError(f"windows must have shape (B, T, 3), got {windows.shape}")
    
    hf_mean, hf_p95 = compute_hf_power_ratio(windows, fs_hz, hf_cutoff_hz)
    jerk_mean, jerk_p95 = compute_jerk_tail_statistic(windows, jerk_percentile)
    tail_mean, tail_p95 = compute_dominant_freq_tail_mass(windows, fs_hz, hf_cutoff_hz)
    
    return {
        "hf_power_ratio_mean": hf_mean,
        "hf_power_ratio_p95": hf_p95,
        "jerk_tail_mean": jerk_mean,
        "jerk_tail_p95": jerk_p95,
        "dom_tail_mass_mean": tail_mean,
        "dom_tail_mass_p95": tail_p95,
    }

