import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class NEUDataset(Dataset):
    """
    A PyTorch dataset for the NEU-RSSDDS-AUG dataset.
    """
    def __init__(self, data_root, mode='train', target_size=(1024, 1024)):
        """
        Args:
            data_root (str): The root directory of the dataset.
            mode (str): 'train' or 'test'.
            target_size (tuple): The target size (height, width) to resize images to.
        """
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'"
        self.data_root = data_root
        self.mode = mode
        self.target_size = target_size

        self.image_dir = os.path.join(data_root, f'Image_{self.mode}')
        self.depth_dir = os.path.join(data_root, f'Depth_{self.mode}')
        if self.mode == 'train':
            self.gt_dir = os.path.join(data_root, f'GT_{self.mode}')

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.bmp')])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Paths
        rgb_path = os.path.join(self.image_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name.replace('.bmp', '.tiff'))

        # Load RGB image (for potential visualization and to get original size)
        rgb_image = Image.open(rgb_path).convert('RGB')
        original_size = (rgb_image.height, rgb_image.width)

        # Load depth image
        # Using cv2 to load tiff image as it's more reliable
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise IOError(f"Could not load depth image: {depth_path}")

        # Ensure depth image is single channel
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        # Resize images
        depth_image_resized = cv2.resize(depth_image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        # Convert to tensor
        depth_tensor = self.transform(depth_image_resized.astype(np.float32))

        sample = {
            'depth': depth_tensor,
            'original_size': original_size,
            'name': img_name
        }

        # Load ground truth for training mode
        if self.mode == 'train':
            gt_path = os.path.join(self.gt_dir, img_name.replace('.bmp', '.png'))
            gt_image = Image.open(gt_path).convert('L') # Ensure single channel
            gt_image_resized = gt_image.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

            gt_tensor = self.transform(gt_image_resized)
            sample['gt'] = (gt_tensor > 0.5).float() # Binarize the mask

        return sample