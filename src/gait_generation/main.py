import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gait generation pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA analysis")
    parser.add_argument("--post-fft", action="store_true", help="Apply FFT low-pass postprocess to synthetic windows")
    parser.add_argument("--post-savgol", action="store_true", help="Apply Savitzky–Golay smoothing postprocess to synthetic windows")
    parser.add_argument("--fs", type=float, default=None, help="Sampling rate in Hz (default: config.IMU_FS_HZ)")
    parser.add_argument("--fft-fc", type=float, default=None, help="FFT low-pass cutoff frequency in Hz (default: config.POST_FFT_CUTOFF_HZ)")
    parser.add_argument("--savgol-window", type=int, default=None, help="Savitzky–Golay window length (odd; default: config.POST_SAVGOL_WINDOW_LENGTH)")
    parser.add_argument("--savgol-poly", type=int, default=None, help="Savitzky–Golay polynomial order (default: config.POST_SAVGOL_POLYORDER)")
    parser.add_argument("--split-mode", type=str, default="window", choices=["window", "user_disjoint"], help="Classifier split mode: 'window' (window-level) or 'user_disjoint' (user-level)")
    parser.add_argument("--per-user-k", type=int, default=50, help="Number of windows per user for matched sampling (default: 50)")
    parser.add_argument("--eval-seed", type=int, default=42, help="Random seed for evaluation reproducibility (default: 42)")
    parser.add_argument("mode", nargs="?", default="full", 
                       help="Pipeline mode: 'full' (default) or 'ablations'")
    
    args = parser.parse_args()
    if len(sys.argv) > 1 and sys.argv[1] == "ablations" and args.mode == "full":
        args.mode = "ablations"
    
    if args.mode == "ablations":
        from .run_ablations import main
        main()
    else:
        from .run_full_pipeline import main
        main(
            skip_eda=args.skip_eda,
            post_fft=args.post_fft,
            post_savgol=args.post_savgol,
            fs=args.fs,
            fft_fc=args.fft_fc,
            savgol_window=args.savgol_window,
            savgol_poly=args.savgol_poly,
            split_mode=args.split_mode,
            per_user_k=args.per_user_k,
            eval_seed=args.eval_seed,
        )

