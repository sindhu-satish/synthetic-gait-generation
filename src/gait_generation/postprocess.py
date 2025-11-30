import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

try:
    from scipy.signal import savgol_filter
except Exception:                  
    savgol_filter = None

        
@dataclass
class PostprocessConfig:
    enable_fft_lowpass: bool = False
    fft_cutoff_hz: float = 12.0

    enable_savgol: bool = False
    savgol_window_length: int = 11
    savgol_polyorder: int = 3


def _lowpass_fft_1d(x: np.ndarray, fs: float, fc: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Low-pass filter a 1D signal using rFFT by zeroing bins above fc.
    Returns filtered signal and stats about removed power.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x.astype(np.float32), {"removed_power_frac": 0.0}

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    power = (np.abs(X) ** 2)
    total_power = float(power.sum()) + 1e-12

    mask_hi = freqs > float(fc)
    removed_power = float(power[mask_hi].sum())
    removed_frac = removed_power / total_power

    X_f = X.copy()
    X_f[mask_hi] = 0.0
    y = np.fft.irfft(X_f, n=n)
    return y.astype(np.float32), {"removed_power_frac": removed_frac}


def _savgol_1d(x: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    if savgol_filter is None:
        raise ImportError("scipy is required for Savitzky–Golay smoothing (scipy.signal.savgol_filter).")
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return x

    wl = int(window_length)
    if wl < 3:
        return x
    if wl % 2 == 0:
        wl += 1
    if wl > n:
        wl = n if (n % 2 == 1) else max(1, n - 1)
    if wl < 3:
        return x

    po = int(polyorder)
    po = max(0, min(po, wl - 1))
    if po == 0:
        return x
    return savgol_filter(x, window_length=wl, polyorder=po, axis=0, mode="interp").astype(np.float32)


def _band_energy_stats(x: np.ndarray, fs: float, fc: float) -> Dict[str, float]:
    """Compute fraction of FFT power above fc and below/equal fc."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return {"power_hi_frac": 0.0, "power_lo_frac": 0.0}
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    power = (np.abs(X) ** 2)
    total = float(power.sum()) + 1e-12
    hi = float(power[freqs > float(fc)].sum()) / total
    lo = float(power[freqs <= float(fc)].sum()) / total
    return {"power_hi_frac": hi, "power_lo_frac": lo}


def postprocess_synthetic(window_xyz: np.ndarray, fs: float, config: PostprocessConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Postprocess one synthetic window (T, 3) in XYZ space.

    - Applies optional FFT low-pass per axis (rFFT, zero bins above fc, irFFT).
    - Applies optional Savitzky–Golay per axis as an alternative smoothing step.
    - Magnitude should be recomputed downstream from filtered axes; we never filter magnitude directly.

    Returns: (filtered_window_xyz, stats)
    """
    w = np.asarray(window_xyz, dtype=np.float32)
    if w.ndim != 2 or w.shape[-1] != 3:
        raise ValueError(f"window_xyz must have shape (T, 3), got {w.shape}")
    T = w.shape[0]
    if T == 0:
        return w, {"fs": fs, "fc": getattr(config, "fft_cutoff_hz", None)}

    fc = float(getattr(config, "fft_cutoff_hz", 12.0))
    stats: Dict[str, Any] = {"fs_hz": float(fs), "fc_hz": fc}

    out = w.copy()


    pre = [_band_energy_stats(out[:, ax], fs, fc) for ax in range(3)]
    stats["pre_power_hi_frac_mean"] = float(np.mean([d["power_hi_frac"] for d in pre]))
    stats["pre_power_lo_frac_mean"] = float(np.mean([d["power_lo_frac"] for d in pre]))

    removed_fracs = []
    if getattr(config, "enable_fft_lowpass", False):
        for ax in range(3):
            out[:, ax], s = _lowpass_fft_1d(out[:, ax], fs=fs, fc=fc)
            removed_fracs.append(float(s["removed_power_frac"]))
        stats["fft_removed_power_frac_mean"] = float(np.mean(removed_fracs)) if removed_fracs else 0.0

    if getattr(config, "enable_savgol", False):
        wl = int(getattr(config, "savgol_window_length", 11))
        po = int(getattr(config, "savgol_polyorder", 3))
        for ax in range(3):
            out[:, ax] = _savgol_1d(out[:, ax], window_length=wl, polyorder=po)
        stats["savgol_window_length"] = wl
        stats["savgol_polyorder"] = po


    post = [_band_energy_stats(out[:, ax], fs, fc) for ax in range(3)]
    stats["post_power_hi_frac_mean"] = float(np.mean([d["power_hi_frac"] for d in post]))
    stats["post_power_lo_frac_mean"] = float(np.mean([d["power_lo_frac"] for d in post]))


    pre_hi = stats["pre_power_hi_frac_mean"]
    post_hi = stats["post_power_hi_frac_mean"]
    if pre_hi > 1e-12:
        stats["pct_power_above_fc_removed"] = float(max(0.0, (pre_hi - post_hi) / pre_hi) * 100.0)
    else:
        stats["pct_power_above_fc_removed"] = 0.0


    if out.shape != w.shape:
        raise RuntimeError(f"Postprocess changed shape from {w.shape} to {out.shape}")

    return out.astype(np.float32), stats


def postprocess_windows(
    windows_xyz: np.ndarray,
    fs: float,
    config: PostprocessConfig,
    log_prefix: str = "",
    max_log: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Postprocess a batch of windows (N, T, 3). Returns (processed_windows, aggregated_stats).
    Prints aggregate sanity stats once.
    """
    W = np.asarray(windows_xyz, dtype=np.float32)
    if W.ndim != 3 or W.shape[-1] != 3:
        raise ValueError(f"windows_xyz must have shape (N, T, 3), got {W.shape}")

    out = np.empty_like(W)
    stats_list = []
    for i in range(W.shape[0]):
        out[i], s = postprocess_synthetic(W[i], fs=fs, config=config)
        stats_list.append(s)

    agg = {
        "n_windows": int(W.shape[0]),
        "fs_hz": float(fs),
        "fc_hz": float(getattr(config, "fft_cutoff_hz", 12.0)),
        "enable_fft_lowpass": bool(getattr(config, "enable_fft_lowpass", False)),
        "enable_savgol": bool(getattr(config, "enable_savgol", False)),
        "pre_power_hi_frac_mean": float(np.mean([s.get("pre_power_hi_frac_mean", 0.0) for s in stats_list])) if stats_list else 0.0,
        "post_power_hi_frac_mean": float(np.mean([s.get("post_power_hi_frac_mean", 0.0) for s in stats_list])) if stats_list else 0.0,
        "pct_power_above_fc_removed_mean": float(np.mean([s.get("pct_power_above_fc_removed", 0.0) for s in stats_list])) if stats_list else 0.0,
    }

    if getattr(config, "enable_fft_lowpass", False):
        agg["fft_removed_power_frac_mean"] = float(np.mean([s.get("fft_removed_power_frac_mean", 0.0) for s in stats_list])) if stats_list else 0.0

    prefix = (log_prefix + " ") if log_prefix else ""
    print(
        f"{prefix}Postprocess sanity: "
        f"fc={agg['fc_hz']:.2f}Hz fs={agg['fs_hz']:.1f}Hz | "
        f"hi-band power frac {agg['pre_power_hi_frac_mean']:.4f} → {agg['post_power_hi_frac_mean']:.4f} | "
        f"pct removed above fc (mean) {agg['pct_power_above_fc_removed_mean']:.2f}%"
    )
    if "fft_removed_power_frac_mean" in agg:
        print(f"{prefix}FFT removed power frac (mean across axes/windows): {agg['fft_removed_power_frac_mean']:.4f}")

    return out, agg


