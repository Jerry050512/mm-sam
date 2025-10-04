#!/usr/bin/env python
"""
Training script for NEU-RSSDDS-AUG defect detection.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pyscripts.launch import Launcher
from utilbox.global_config import OUTPUT_ROOT, IS_DEVELOPMENT, DEVICE_TYPE


def main():
    parser = argparse.ArgumentParser(description="Train NEU-RSSDDS defect detection model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1.6e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=DEVICE_TYPE, choices=["cuda", "cpu"], 
                       help="Device to use for training")
    parser.add_argument("--dev", action="store_true", help="Use development mode (./hy-tmp, CPU)")
    
    args = parser.parse_args()
    
    # Set environment variables based on mode
    if args.dev:
        os.environ['MM_SAM_DEV'] = 'true'
        device = 'cpu'
    else:
        os.environ['MM_SAM_DEV'] = 'false'
        device = args.device
    
    print("=" * 60)
    print("NEU-RSSDDS Training")
    print("=" * 60)
    print(f"Mode: {'Development' if args.dev else 'Production'}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 60)
    
    # Prepare arguments for launcher
    sys.argv = [
        'train_neu_rssdds.py',
        '--config_name', 'cm_transfer/neu_rssdds_1gpu',
        '--train_epoch_num', str(args.epochs),
        '--train_bs', str(args.batch_size),
        '--gpu_num', '1' if device == 'cuda' else '0',
        '--train_only', 'False'  # Enable both training and testing
    ]
    
    try:
        # Launch training
        Launcher.launch()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()