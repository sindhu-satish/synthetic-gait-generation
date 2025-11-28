import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gait generation pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA analysis")
    parser.add_argument("mode", nargs="?", default="full", 
                       help="Pipeline mode: 'full' (default) or 'ablations'")
    
    args = parser.parse_args()
    
    # Handle legacy positional argument format
    if len(sys.argv) > 1 and sys.argv[1] == "ablations" and args.mode == "full":
        args.mode = "ablations"
    
    if args.mode == "ablations":
        from .run_ablations import main
        main()
    else:
        from .run_full_pipeline import main
        main(skip_eda=args.skip_eda)

