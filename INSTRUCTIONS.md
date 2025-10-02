# Instructions for NEU-RSSDDS-AUG Experiment

This document provides instructions on how to set up the environment and run the training and testing processes for the NEU-RSSDDS-AUG dataset.

## 1. Environment Setup

It is recommended to use a virtual environment to manage the dependencies.

### 1.1. Create and Activate a Virtual Environment (Optional)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.2. Install Dependencies

Install the required packages using pip:

```bash
pip install torch==2.3.0 torchvision==0.18.0
pip install -e .
```

## 2. Data Preparation

Place the NEU-RSSDDS-AUG dataset in the `../datasets/` directory relative to the project root. The expected directory structure is as follows:

```
../datasets/
└── NEU-RSDDS-AUG/
    ├── Image_train/
    ├── Depth_train/
    ├── GT_train/
    ├── Image_test/
    └── Depth_test/
```

## 3. Training

To start the training process, run the following command from the project root:

```bash
python pyscripts/train_single.py --config_name neu_rssdds_aug
```

This will start the training process using the settings defined in `config/neu_rssdds_aug.yaml`. Checkpoints will be saved to `/hy-tmp/output/checkpoint.pth` and logs will be written to `/hy-tmp/output/result.log`.

## 4. Testing

To run the testing process, use the same command as for training, but add the `--test_only` flag:

```bash
python pyscripts/train_single.py --config_name neu_rssdds_aug --test_only
```

This will load the trained checkpoint from `/hy-tmp/output/checkpoint.pth` and generate prediction masks in the `/hy-tmp/output/predictions/` directory.