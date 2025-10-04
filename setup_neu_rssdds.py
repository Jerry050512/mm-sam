#!/usr/bin/env python
"""
Environment setup script for NEU-RSSDDS-AUG training and testing.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"Error: Python 3.8+ required, but found Python {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")


def setup_conda_environment(env_name="mm_sam_neu", python_version="3.10"):
    """Setup conda environment."""
    print(f"\n=== Setting up conda environment: {env_name} ===")
    
    # Check if conda is available
    result = run_command("conda --version", check=False)
    if result.returncode != 0:
        print("Error: conda not found. Please install Anaconda or Miniconda first.")
        sys.exit(1)
    
    # Create conda environment
    run_command(f"conda create -n {env_name} python={python_version} -y")
    
    print(f"✓ Conda environment '{env_name}' created successfully")
    print(f"To activate: conda activate {env_name}")


def install_dependencies(cuda_version="11.8"):
    """Install required dependencies."""
    print(f"\n=== Installing dependencies ===")
    
    # Install PyTorch with CUDA support
    if cuda_version == "11.8":
        torch_command = "pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118"
    elif cuda_version == "12.1":
        torch_command = "pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121"
    else:
        torch_command = "pip install torch==2.1.2 torchvision==0.16.2"
    
    run_command(torch_command)
    
    # Install the package in development mode
    run_command("pip install -e .")
    
    # Install additional dependencies
    additional_deps = [
        "gdal==3.8.3",  # Will be installed via conda
        "Pillow>=8.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.60.0",
        "humanfriendly>=10.0",
        "packaging>=21.0"
    ]
    
    # Install GDAL via conda (more reliable)
    run_command("conda install -c conda-forge gdal==3.8.3 -y")
    
    # Install other dependencies via pip
    for dep in additional_deps[1:]:  # Skip GDAL as it's installed via conda
        run_command(f"pip install {dep}")
    
    print("✓ All dependencies installed successfully")


def setup_directories():
    """Setup required directories."""
    print(f"\n=== Setting up directories ===")
    
    # Create output directories
    output_dirs = [
        "hy-tmp/output",
        "hy-tmp/output/predictions", 
        "hy-tmp/experiments",
        "hy-tmp/pretrained"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def download_sam_pretrained():
    """Download SAM pretrained models."""
    print(f"\n=== Downloading SAM pretrained models ===")
    
    pretrained_dir = "pretrained"
    os.makedirs(pretrained_dir, exist_ok=True)
    
    sam_models = {
        "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    for model_name, url in sam_models.items():
        model_path = os.path.join(pretrained_dir, model_name)
        if os.path.exists(model_path):
            print(f"✓ {model_name} already exists")
            continue
            
        print(f"Downloading {model_name}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ Downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            print(f"Please download manually from: {url}")


def setup_dataset():
    """Setup NEU-RSSDDS dataset."""
    print(f"\n=== Setting up NEU-RSSDDS dataset ===")
    
    # Run dataset setup script
    run_command("python -m pyscripts.neu_rssdds_setup")
    print("✓ Dataset setup completed")


def main():
    parser = argparse.ArgumentParser(description="Setup environment for NEU-RSSDDS training")
    parser.add_argument("--env-name", default="mm_sam_neu", help="Conda environment name")
    parser.add_argument("--python-version", default="3.10", help="Python version for conda env")
    parser.add_argument("--cuda-version", default="11.8", choices=["11.8", "12.1", "cpu"], 
                       help="CUDA version")
    parser.add_argument("--skip-conda", action="store_true", help="Skip conda environment creation")
    parser.add_argument("--skip-download", action="store_true", help="Skip SAM model download")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NEU-RSSDDS Environment Setup")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Setup conda environment
    if not args.skip_conda:
        setup_conda_environment(args.env_name, args.python_version)
        print(f"\nIMPORTANT: Please run 'conda activate {args.env_name}' before continuing!")
        print("Then run this script again with --skip-conda flag.")
        return
    
    # Install dependencies
    install_dependencies(args.cuda_version)
    
    # Setup directories
    setup_directories()
    
    # Download SAM models
    if not args.skip_download:
        download_sam_pretrained()
    
    # Setup dataset
    setup_dataset()
    
    print("\n" + "=" * 60)
    print("✓ Environment setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. For training: python train_neu_rssdds.py")
    print("2. For testing: python test_neu_rssdds.py")
    print("3. Check logs at: hy-tmp/output/result.log")
    print("4. Find predictions at: hy-tmp/output/predictions/")


if __name__ == "__main__":
    main()