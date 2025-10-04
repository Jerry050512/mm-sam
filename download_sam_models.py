#!/usr/bin/env python
"""
Download SAM pretrained models.
"""

import os
import urllib.request
import os
from pathlib import Path

# Use ./pretrained directory
PRETRAINED_ROOT = os.path.join(Path(__file__).parent, 'pretrained')

def download_sam_models():
    """Download SAM pretrained models."""
    print("Downloading SAM pretrained models...")
    
    os.makedirs(PRETRAINED_ROOT, exist_ok=True)
    
    # Only download the base model for testing
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    model_path = os.path.join(PRETRAINED_ROOT, "sam_vit_b_01ec64.pth")
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return
    
    print(f"Downloading to: {model_path}")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("✓ Download completed successfully")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("Please download manually from:")
        print(model_url)

if __name__ == "__main__":
    download_sam_models()