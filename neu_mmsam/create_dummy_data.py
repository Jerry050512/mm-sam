import os
import numpy as np
from PIL import Image
import cv2
import argparse

def create_dummy_data(root_dir, num_train=3, num_test=2, img_size=(224, 224)):
    """
    Creates a dummy dataset with the same structure as NEU-RSSDDS-AUG.
    """
    print(f"Creating dummy dataset at: {root_dir}")

    # Define paths
    train_img_dir = os.path.join(root_dir, 'Image_train')
    train_depth_dir = os.path.join(root_dir, 'Depth_train')
    train_gt_dir = os.path.join(root_dir, 'GT_train')
    test_img_dir = os.path.join(root_dir, 'Image_test')
    test_depth_dir = os.path.join(root_dir, 'Depth_test')

    # Create directories
    for d in [train_img_dir, train_depth_dir, train_gt_dir, test_img_dir, test_depth_dir]:
        os.makedirs(d, exist_ok=True)

    # --- Create Training Data ---
    for i in range(num_train):
        filename = f"dummy_train_{i:04d}"

        # RGB Image (.bmp)
        rgb_img = Image.new('RGB', img_size, color = 'black')
        rgb_img.save(os.path.join(train_img_dir, f"{filename}.bmp"))

        # Depth Image (.tiff)
        depth_array = np.zeros(img_size, dtype=np.uint16)
        depth_array[:, :] = np.linspace(0, 65535, img_size[0], dtype=np.uint16)
        cv2.imwrite(os.path.join(train_depth_dir, f"{filename}.tiff"), depth_array)

        # Ground Truth Mask (.png)
        gt_array = np.zeros(img_size, dtype=np.uint8)
        # Create a circle in the middle
        center = (img_size[0] // 2, img_size[1] // 2)
        radius = img_size[0] // 4
        cv2.circle(gt_array, center, radius, 255, -1)
        gt_img = Image.fromarray(gt_array, 'L')
        gt_img.save(os.path.join(train_gt_dir, f"{filename}.png"))

    print(f"Created {num_train} training samples.")

    # --- Create Testing Data ---
    for i in range(num_test):
        filename = f"dummy_test_{i:04d}"

        # RGB Image (.bmp)
        rgb_img = Image.new('RGB', img_size, color = 'darkgrey')
        rgb_img.save(os.path.join(test_img_dir, f"{filename}.bmp"))

        # Depth Image (.tiff)
        depth_array = np.zeros(img_size, dtype=np.uint16)
        depth_array[i*50:(i+1)*50, :] = 30000 # Different pattern for test
        cv2.imwrite(os.path.join(test_depth_dir, f"{filename}.tiff"), depth_array)

    print(f"Created {num_test} testing samples.")
    print("Dummy data creation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dummy data for testing.")
    parser.add_argument('--data_root', type=str, default='../datasets/NEU-RSDDS-AUG', help="Root directory for the dummy dataset.")
    args = parser.parse_args()
    create_dummy_data(args.data_root)