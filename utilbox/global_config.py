"""
This file contains all the general configurations for your project.

Please register the information based on your local machine here before making your project.

"""
import os
from os.path import exists, abspath, dirname

# the root directory of the project source codes
PROJECT_ROOT = abspath(dirname(dirname(__file__)))

# Flexible path configuration for development and production environments
# For development: use ./hy-tmp and CPU
# For production: use /hy-tmp and CUDA
IS_DEVELOPMENT = os.environ.get('MM_SAM_DEV', 'true').lower() == 'true'

if IS_DEVELOPMENT:
    # Development environment paths
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'hy-tmp')
    DEVICE_TYPE = 'cpu'
else:
    # Production environment paths  
    OUTPUT_ROOT = '/hy-tmp'
    DEVICE_TYPE = 'cuda'

# Create output directories if they don't exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'output'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'output', 'predictions'), exist_ok=True)

# the root directory that places the experiment log data
EXP_ROOT = os.path.join(OUTPUT_ROOT, 'experiments')
# the root directory of the local datasets  
DATA_ROOT = os.path.join(PROJECT_ROOT, '..', 'datasets')
# the root directory of the pretrained models
PRETRAINED_ROOT = os.path.join(PROJECT_ROOT, 'pretrained')

# Create required directories
for path in [EXP_ROOT, PRETRAINED_ROOT]:
    os.makedirs(path, exist_ok=True)

# runtime checking for mandatory configurations
config_dict = dict(
    PROJECT_ROOT=PROJECT_ROOT, EXP_ROOT=EXP_ROOT, DATA_ROOT=DATA_ROOT, 
    PRETRAINED_ROOT=PRETRAINED_ROOT, OUTPUT_ROOT=OUTPUT_ROOT
)
for k, v in config_dict.items():
    if v is None or v == '':
        raise RuntimeError(f"Please register {k} in utilbox.global_config!")
    # Skip existence check for DATA_ROOT as it may not exist yet
    elif k != 'DATA_ROOT' and not exists(v):
        raise RuntimeError(f"Your registered {k} ({v}) does not exist! Please create it manually!")
