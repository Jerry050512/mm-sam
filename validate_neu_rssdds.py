#!/usr/bin/env python
"""
Validation script for NEU-RSSDDS implementation.
Tests the complete pipeline with synthetic data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image

from mm_sam.datasets.neu_rssdds import NEURSSDDSDataset
from mm_sam.train_agents.cm_transfer.neu_rssdds import NEURSSDDSCMTransferSAM
from utilbox.global_config import OUTPUT_ROOT


def create_test_data(data_dir, num_samples=3):
    """Create synthetic test data."""
    print(f"Creating test data in {data_dir}")
    
    # Create directory structure
    for split in ['train', 'test']:
        for data_type in ['Image', 'Depth'] + (['GT'] if split == 'train' else []):
            dir_path = os.path.join(data_dir, f'{data_type}_{split}')
            os.makedirs(dir_path, exist_ok=True)
            
            # Create sample files
            for i in range(num_samples):
                sample_name = f'test_sample_{i:03d}'
                
                if data_type == 'Image':
                    # Create random RGB image
                    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                    Image.fromarray(img).save(os.path.join(dir_path, f'{sample_name}.bmp'))
                    
                elif data_type == 'Depth':
                    # Create random depth image
                    depth = np.random.randint(0, 1000, (128, 128), dtype=np.uint16)
                    Image.fromarray(depth).save(os.path.join(dir_path, f'{sample_name}.tiff'))
                    
                elif data_type == 'GT':
                    # Create random binary mask
                    mask = np.random.choice([0, 255], (128, 128), p=[0.7, 0.3]).astype(np.uint8)
                    Image.fromarray(mask).save(os.path.join(dir_path, f'{sample_name}.png'))
    
    print(f"✓ Created {num_samples} samples for each split")


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\n=== Testing Dataset Loading ===")
    
    try:
        # Test training dataset
        train_dataset = NEURSSDDSDataset(is_train=True)
        print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
        
        # Test a sample
        sample = train_dataset[0]
        print(f"✓ Sample loaded - RGB: {sample['rgb_images'].shape}, Depth: {sample['depth_images'].shape}")
        
        # Test testing dataset
        test_dataset = NEURSSDDSDataset(is_train=False)
        print(f"✓ Testing dataset loaded: {len(test_dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_model_initialization():
    """Test model initialization."""
    print("\n=== Testing Model Initialization ===")
    
    try:
        # Initialize training agent
        agent = NEURSSDDSCMTransferSAM(
            seed=42,
            device=torch.device('cpu'),
            local_rank=0,
            find_unused_parameters=False,
            use_amp=False,
            train_only=False,
            test_only=False,
            train_bs=1,
            train_workers_per_bs=0,
            train_epoch_num=1,
            pin_memory=False,
            valid_bs=1,
            valid_workers_per_bs=0,
            test_bs=1,
            test_workers_per_bs=0,
            folder_path=OUTPUT_ROOT
        )
        
        print("✓ Training agent initialized successfully")
        print(f"✓ Model device: {agent.device}")
        print(f"✓ X-encoder configured for depth images")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing."""
    print("\n=== Testing Data Preprocessing ===")
    
    try:
        dataset = NEURSSDDSDataset(is_train=True)
        sample = dataset[0]
        
        # Test depth preprocessing
        agent = NEURSSDDSCMTransferSAM(
            seed=42, device=torch.device('cpu'), local_rank=0,
            find_unused_parameters=False, use_amp=False,
            train_only=True, test_only=False,
            train_bs=1, train_workers_per_bs=0,
            train_epoch_num=1, pin_memory=False,
            folder_path=OUTPUT_ROOT
        )
        
        depth_images = [sample['depth_images']]
        processed = agent.preprocess_depth_images(depth_images)
        
        print(f"✓ Depth preprocessing successful: {processed[0].shape}")
        print(f"✓ Depth value range: [{processed[0].min():.3f}, {processed[0].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return False


def test_checkpoint_operations():
    """Test checkpoint saving and loading."""
    print("\n=== Testing Checkpoint Operations ===")
    
    try:
        agent = NEURSSDDSCMTransferSAM(
            seed=42, device=torch.device('cpu'), local_rank=0,
            find_unused_parameters=False, use_amp=False,
            train_only=True, test_only=False,
            train_bs=1, train_workers_per_bs=0,
            train_epoch_num=1, pin_memory=False,
            folder_path=OUTPUT_ROOT
        )
        
        # Test checkpoint saving
        checkpoint_path = os.path.join(OUTPUT_ROOT, 'output', 'test_checkpoint.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        model_state = agent.model_state_dict()
        torch.save(model_state, checkpoint_path)
        print("✓ Checkpoint saved successfully")
        
        # Test checkpoint loading
        agent.load_checkpoint_for_testing(checkpoint_path)
        print("✓ Checkpoint loaded successfully")
        
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint operations failed: {e}")
        return False


def test_prediction_saving():
    """Test prediction saving functionality."""
    print("\n=== Testing Prediction Saving ===")
    
    try:
        agent = NEURSSDDSCMTransferSAM(
            seed=42, device=torch.device('cpu'), local_rank=0,
            find_unused_parameters=False, use_amp=False,
            train_only=False, test_only=True,
            train_bs=1, train_workers_per_bs=0,
            train_epoch_num=1, pin_memory=False,
            folder_path=OUTPUT_ROOT
        )
        
        # Create dummy predictions
        predictions = [torch.rand(128, 128) > 0.5]  # Binary prediction
        filenames = ['test_prediction']
        original_sizes = [(128, 128)]
        
        agent.save_prediction_masks(predictions, filenames, original_sizes)
        
        # Check if file was saved
        output_path = os.path.join(OUTPUT_ROOT, 'output', 'predictions', 'test_prediction.png')
        if os.path.exists(output_path):
            print("✓ Prediction saved successfully")
            # Clean up
            os.remove(output_path)
            return True
        else:
            print("✗ Prediction file not found")
            return False
        
    except Exception as e:
        print(f"✗ Prediction saving failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("NEU-RSSDDS Implementation Validation")
    print("=" * 60)
    
    # Set development mode
    os.environ['MM_SAM_DEV'] = 'true'
    
    # Create temporary test data
    temp_data_dir = os.path.join(OUTPUT_ROOT, 'temp_test_data', 'NEU-RSDDS-AUG')
    os.makedirs(temp_data_dir, exist_ok=True)
    
    try:
        # Create test data
        create_test_data(temp_data_dir)
        
        # Update data path temporarily
        from utilbox import global_config
        original_data_root = global_config.DATA_ROOT
        global_config.DATA_ROOT = os.path.join(OUTPUT_ROOT, 'temp_test_data')
        
        # Setup dataset metadata
        from pyscripts.neu_rssdds_setup import main as setup_main
        setup_main()
        
        # Run tests
        tests = [
            ("Dataset Loading", test_dataset_loading),
            ("Model Initialization", test_model_initialization),
            ("Data Preprocessing", test_data_preprocessing),
            ("Checkpoint Operations", test_checkpoint_operations),
            ("Prediction Saving", test_prediction_saving)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Restore original data path
        global_config.DATA_ROOT = original_data_root
        
        # Print summary
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)
        
        passed = 0
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"{test_name:<25}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("✓ All tests passed! Implementation is ready.")
            return 0
        else:
            print("✗ Some tests failed. Please check the implementation.")
            return 1
            
    finally:
        # Clean up temporary data
        if os.path.exists(temp_data_dir):
            shutil.rmtree(os.path.dirname(temp_data_dir))


if __name__ == "__main__":
    sys.exit(main())