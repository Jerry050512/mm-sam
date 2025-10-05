# MM-SAM for RGB-D Salient Object Detection on NEU-RSSDDS-AUG

This project is a simplified, minimum viable product (MVP) version of the MM-SAM repository, specifically tailored for training and testing on the NEU-RSSDDS-AUG dataset using a single GPU.

## 1. Environment Setup

### Dependencies
Install the required Python libraries using pip. It is recommended to use a virtual environment.

```bash
pip install torch torchvision opencv-python-headless numpy Pillow tqdm
```

### Pre-trained SAM Weights
For training to be effective, you must download the pre-trained SAM weights. This implementation uses the ViT-Base model.

1.  **Download the weights**: Download the `sam_vit_b_01ec64.pth` file from the official [Segment Anything Model repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).
2.  **Place the weights**: Create a directory (e.g., `pretrained_weights/`) in the root of this MVP project and place the downloaded `.pth` file inside it.

Your directory structure should look like this:
```
.
├── neu_mmsam/
│   ├── main.py
│   ├── model.py
│   └── ...
├── pretrained_weights/
│   └── sam_vit_b_01ec64.pth
└── ...
```

## 2. Dataset Preparation

The script expects the NEU-RSSDDS-AUG dataset to be organized in the following structure:

```
../datasets/NEU-RSDDS-AUG/
├── Image_train/
│   ├── 0001.bmp
│   └── ...
├── Depth_train/
│   ├── 0001.tiff
│   └── ...
├── GT_train/
│   ├── 0001.png
│   └── ...
├── Image_test/
│   ├── 0001.bmp
│   └── ...
└── Depth_test/
    ├── 0001.tiff
    └── ...
```
Make sure your dataset is located at a path like `../datasets/NEU-RSDDS-AUG` relative to where you run the script, or provide the correct path using the `--data_root` argument.

## 3. Training

To train the model, run `main.py` with `train` mode. You must provide the path to the pre-trained SAM checkpoint.

-   `--data_root`: Path to the NEU-RSSDDS-AUG dataset.
-   `--sam_checkpoint`: Path to the downloaded `sam_vit_b_01ec64.pth` file.
-   `--checkpoint_path`: Path where the final trained checkpoint will be saved.
-   `--log_file`: Path to the log file.
-   `--device`: `cuda` for GPU training or `cpu` for CPU training.

**Example command:**
```bash
python -m neu_mmsam.main \
    --mode train \
    --data_root ../datasets/NEU-RSDDS-AUG \
    --sam_checkpoint ./pretrained_weights/sam_vit_b_01ec64.pth \
    --checkpoint_path /hy-tmp/output/checkpoint.pth \
    --log_file /hy-tmp/output/result.log \
    --device cuda \
    --epochs 50 \
    --batch_size 2
```

The script will save the latest checkpoint to `/hy-tmp/output/checkpoint.pth` after each epoch, overwriting the previous one.

## 4. Testing

To test the model, run `main.py` with `test` mode. The script will automatically load the checkpoint saved during training.

-   `--output_dir`: Directory where the predicted segmentation masks will be saved.
-   The `--checkpoint_path` and `--log_file` should point to the same paths used during training.

**Example command:**
```bash
python -m neu_mmsam.main \
    --mode test \
    --data_root ../datasets/NEU-RSDDS-AUG \
    --checkpoint_path /hy-tmp/output/checkpoint.pth \
    --output_dir /hy-tmp/output/predictions/ \
    --log_file /hy-tmp/output/result.log \
    --device cuda
```

The script will generate segmentation masks and save them as PNG files in the specified output directory, resized to match the original image dimensions.