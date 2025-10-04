#!/usr/bin/env python
"""
Testing script for NEU-RSSDDS-AUG defect detection.
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
    parser = argparse.ArgumentParser(description="Test NEU-RSSDDS defect detection model")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to checkpoint file (default: hy-tmp/output/checkpoint.pth)")
    parser.add_argument("--batch-size", type=int, default=1, help="Testing batch size")
    parser.add_argument("--device", type=str, default=DEVICE_TYPE, choices=["cuda", "cpu"], 
                       help="Device to use for testing")
    parser.add_argument("--dev", action="store_true", help="Use development mode (./hy-tmp, CPU)")
    
    args = parser.parse_args()
    
    # Set environment variables based on mode
    if args.dev:
        os.environ['MM_SAM_DEV'] = 'true'
        device = 'cpu'
    else:
        os.environ['MM_SAM_DEV'] = 'false'
        device = args.device
    
    # Set checkpoint path
    if args.checkpoint is None:
        checkpoint_path = os.path.join(OUTPUT_ROOT, 'output', 'checkpoint.pth')
    else:
        checkpoint_path = args.checkpoint
    
    print("=" * 60)
    print("NEU-RSSDDS Testing")
    print("=" * 60)
    print(f"Mode: {'Development' if args.dev else 'Production'}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 60)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run training first or specify a valid checkpoint path.")
        sys.exit(1)
    
    # Prepare arguments for launcher
    sys.argv = [
        'test_neu_rssdds.py',
        '--config_name', 'cm_transfer/neu_rssdds_1gpu',
        '--test_bs', str(args.batch_size),
        '--gpu_num', '1' if device == 'cuda' else '0',
        '--test_only', 'True',
        '--ckpt_path', checkpoint_path
    ]
    
    try:
        # Launch testing
        Launcher.launch()
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()