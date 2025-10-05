import json
import os
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image

from mm_sam.datasets import DATASETS
TRANSFER_DATASETS = DATASETS["CMTransfer"]
FUSION_DATASETS = DATASETS["MMFusion"]

from mm_sam.datasets.base import BaseSAMDataset
from utilbox.data_load.read_utils import read_image_as_rgb_from_disk, read_greyscale_mask_from_disk
from utilbox.demo_vis.vis_utils import nonrgb_to_rgb
from utilbox.global_config import DATA_ROOT
from utilbox.transforms import init_transforms_by_config
from utilbox.transforms.img_segm import Compose


class NEURSSDDSDataset(BaseSAMDataset):
    """
    Dataset class for NEU-RSSDDS-AUG defect detection dataset.
    Supports RGB-D data with binary defect segmentation.
    """
    
    semantic_classes = ['background', 'defect']

    def __init__(
            self,
            is_train: bool,
            data_dir: str = f"{DATA_ROOT}/NEU-RSDDS-AUG",
            image_type: str = 'depth_images',
            transforms: Union[Dict, Compose, None] = None,
            **prompt_args
    ):
        """
        Initialize NEU-RSSDDS dataset.
        
        Args:
            is_train: Whether this is training data
            data_dir: Root directory of the dataset
            image_type: Type of images to use ('rgb_images', 'depth_images', 'depth_rgb_images')
            transforms: Data transforms to apply
            **prompt_args: Additional arguments for prompt generation
        """
        if image_type not in ['rgb_images', 'depth_images', 'depth_rgb_images']:
            raise ValueError("Invalid image_type! Must be one of 'rgb_images', 'depth_images', 'depth_rgb_images'.")
        self.image_type = image_type
        self.data_dir = data_dir
        
        # Load metadata
        metadata_file = 'train.json' if is_train else 'test.json'
        metadata_path = os.path.join(data_dir, 'metadata', metadata_file)
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Please run neu_rssdds_setup.py first.")
        
        with open(metadata_path, 'r') as f:
            data_dict = json.load(f)
        
        # For training data, we need ground truth masks
        if is_train:
            # Filter out samples without ground truth
            data_dict = {k: v for k, v in data_dict.items() if 'gt_mask_path' in v}
        
        super(NEURSSDDSDataset, self).__init__(
            data_dict=data_dict, is_train=is_train, label_threshold=128, **prompt_args
        )

        # Register the aligned transforms for RGB-Depth-MASK triples
        if isinstance(transforms, Dict):
            self.transforms = init_transforms_by_config(
                transform_config=transforms, tgt_package="utilbox.transforms.img_segm",
                default_args=self.transforms_default_args
            )
        elif isinstance(transforms, Compose):
            self.transforms = transforms
        else:
            self.transforms = None

    def get_image_by_path(self, image_path: str):
        """Load RGB image from .bmp file."""
        return read_image_as_rgb_from_disk(image_path)

    def get_depth_by_path(self, depth_path: str):
        """
        Load depth image from .tiff file.
        
        Args:
            depth_path: Path to .tiff depth file
            
        Returns:
            Depth image as numpy array with shape (H, W, 1)
        """
        try:
            # Load TIFF depth image
            depth_img = Image.open(depth_path)
            depth_array = np.array(depth_img)
            
            # Ensure depth is single channel
            if len(depth_array.shape) == 2:
                depth_array = np.expand_dims(depth_array, axis=2)
            elif len(depth_array.shape) == 3 and depth_array.shape[2] > 1:
                # Take first channel if multi-channel
                depth_array = depth_array[:, :, 0:1]
            
            # Normalize depth values to [0, 1] range
            if depth_array.max() > 1.0:
                depth_array = depth_array.astype(np.float32) / depth_array.max()
            
            return depth_array.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load depth image from {depth_path}: {e}")

    def get_gt_by_path(self, gt_path: str):
        """Load ground truth mask from .png file."""
        return read_greyscale_mask_from_disk(gt_path, self.label_threshold)

    def __getitem__(self, index):
        """Get a single sample from the dataset."""
        index_name = self.index_name_list[index]
        index_dict = self.data_dict[index_name]
        
        # Load RGB image
        rgb_image = self.get_image_by_path(index_dict['image_path'])
        
        # Load depth image
        depth_image = self.get_depth_by_path(index_dict['depth_path'])
        
        # Load ground truth mask (if available)
        if 'gt_mask_path' in index_dict:
            gt_mask = self.get_gt_by_path(index_dict['gt_mask_path'])
        else:
            # Create dummy mask for test data
            gt_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)

        # Apply aligned transforms for the triple data
        if self.transforms is not None:
            transformed = self.transforms(image=rgb_image, depth=depth_image, mask=gt_mask)
            rgb_image, depth_image, gt_mask = transformed["image"], transformed['depth'], transformed['mask']

        # Determine image shape based on image type
        if self.image_type == 'rgb_images':
            image_shape = (rgb_image.shape[0], rgb_image.shape[1])
        elif self.image_type in ['depth_images', 'depth_rgb_images']:
            image_shape = (depth_image.shape[0], depth_image.shape[1])
        else:
            raise RuntimeError(f"Invalid image type: {self.image_type}")

        # Generate prompts from ground truth masks (only for training)
        if self.is_train and 'gt_mask_path' in index_dict:
            point_coords, box_coords, noisy_object_masks, object_masks, object_classes = (
                self.get_prompts_from_gt_masks(gt_mask=gt_mask, image_shape=image_shape)
            )
        else:
            # For test data, create empty prompts
            point_coords = None
            box_coords = None
            noisy_object_masks = None
            object_masks = None
            object_classes = None

        # Convert depth to RGB visualization
        depth_rgb_image = nonrgb_to_rgb(depth_image)

        # Prepare return dictionary
        ret_dict = {
            'rgb_images': torch.from_numpy(rgb_image).to(torch.float32).permute(2, 0, 1),
            'depth_images': torch.from_numpy(depth_image).to(torch.float32).permute(2, 0, 1),
            'depth_rgb_images': torch.from_numpy(depth_rgb_image).to(torch.float32).permute(2, 0, 1),
            'gt_masks': torch.from_numpy(gt_mask),
            'point_coords': point_coords,
            'box_coords': box_coords,
            'noisy_object_masks': noisy_object_masks,
            'object_masks': object_masks,
            'object_classes': object_classes,
            'index_name': index_name
        }
        
        # Set the main image based on image_type
        ret_dict['images'] = ret_dict[self.image_type]
        
        return ret_dict


@TRANSFER_DATASETS.register("neu_rssdds")
class NEURSSDDSTransfer(NEURSSDDSDataset):
    """NEU-RSSDDS dataset for cross-modal transfer (depth images)."""
    def __init__(self, **init_kwargs):
        super().__init__(image_type="depth_images", **init_kwargs)


@FUSION_DATASETS.register("neu_rssdds")
class NEURSSDDSFusion(NEURSSDDSDataset):
    """NEU-RSSDDS dataset for multi-modal fusion (RGB images)."""
    def __init__(self, **init_kwargs):
        super().__init__(image_type="rgb_images", **init_kwargs)