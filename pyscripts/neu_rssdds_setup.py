#!/usr/bin/env python
"""
Setup script for NEU-RSSDDS-AUG dataset.
This script scans the dataset directories and creates metadata files for training and testing.
"""

import os
import json
from pathlib import Path
from utilbox.global_config import DATA_ROOT

def scan_dataset_files(base_path, image_dir, depth_dir, gt_dir=None):
    """
    Scan dataset directories and create file mappings.
    
    Args:
        base_path: Base dataset path
        image_dir: RGB image directory name
        depth_dir: Depth image directory name  
        gt_dir: Ground truth directory name (None for test data)
    
    Returns:
        Dictionary mapping sample names to file paths
    """
    data_dict = {}
    
    image_path = os.path.join(base_path, image_dir)
    depth_path = os.path.join(base_path, depth_dir)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image directory {image_path} does not exist")
        return data_dict
        
    if not os.path.exists(depth_path):
        print(f"Warning: Depth directory {depth_path} does not exist")
        return data_dict
    
    # Get all .bmp files from image directory
    image_files = [f for f in os.listdir(image_path) if f.endswith('.bmp')]
    
    for image_file in image_files:
        # Extract base name without extension
        base_name = os.path.splitext(image_file)[0]
        
        # Check for corresponding depth file (.tiff)
        depth_file = base_name + '.tiff'
        depth_file_path = os.path.join(depth_path, depth_file)
        
        if not os.path.exists(depth_file_path):
            print(f"Warning: Missing depth file for {image_file}")
            continue
            
        # Create file mapping
        file_mapping = {
            'image_path': os.path.join(image_path, image_file),
            'depth_path': depth_file_path
        }
        
        # Add ground truth path if provided
        if gt_dir:
            gt_path = os.path.join(base_path, gt_dir)
            gt_file = base_name + '.png'
            gt_file_path = os.path.join(gt_path, gt_file)
            
            if os.path.exists(gt_file_path):
                file_mapping['gt_mask_path'] = gt_file_path
            else:
                print(f"Warning: Missing GT file for {image_file}")
                continue
        
        data_dict[base_name] = file_mapping
    
    return data_dict

def create_random_test_data():
    """Create random test data for development testing."""
    import numpy as np
    from PIL import Image
    
    # Create test data directories
    base_path = os.path.join(DATA_ROOT, 'NEU-RSDDS-AUG')
    
    for split in ['train', 'test']:
        for data_type in ['Image', 'Depth'] + (['GT'] if split == 'train' else []):
            dir_path = os.path.join(base_path, f'{data_type}_{split}')
            os.makedirs(dir_path, exist_ok=True)
            
            # Create 5 sample files for testing
            for i in range(5):
                sample_name = f'sample_{i:03d}'
                
                if data_type == 'Image':
                    # Create random RGB image
                    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    Image.fromarray(img).save(os.path.join(dir_path, f'{sample_name}.bmp'))
                    
                elif data_type == 'Depth':
                    # Create random depth image
                    depth = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)
                    Image.fromarray(depth).save(os.path.join(dir_path, f'{sample_name}.tiff'))
                    
                elif data_type == 'GT':
                    # Create random binary mask
                    mask = np.random.choice([0, 255], (256, 256), p=[0.8, 0.2]).astype(np.uint8)
                    Image.fromarray(mask).save(os.path.join(dir_path, f'{sample_name}.png'))
    
    print("Created random test data for development")

def main():
    """Main setup function."""
    dataset_path = os.path.join(DATA_ROOT, 'NEU-RSDDS-AUG')
    
    # Check if dataset exists, if not create random test data
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Creating random test data for development...")
        create_random_test_data()
    
    # Scan training data
    print("Scanning training data...")
    train_data = scan_dataset_files(
        dataset_path, 'Image_train', 'Depth_train', 'GT_train'
    )
    
    # Scan testing data  
    print("Scanning testing data...")
    test_data = scan_dataset_files(
        dataset_path, 'Image_test', 'Depth_test'
    )
    
    # Save metadata files
    metadata_dir = os.path.join(dataset_path, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    with open(os.path.join(metadata_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
        
    with open(os.path.join(metadata_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Setup complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Metadata saved to: {metadata_dir}")

if __name__ == '__main__':
    main()