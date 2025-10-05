import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from .model import SAMbyUCMT
from .dataset import NEUDataset

# -- Loss Function --

class DiceBCELoss(nn.Module):
    """A combination of Dice Loss and Binary Cross-Entropy Loss."""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        return bce + dice_loss

# -- Helper Functions --

def get_bounding_box_from_mask(masks):
    """
    Calculates bounding boxes from binary masks.
    Args:
        masks (torch.Tensor): A tensor of shape (B, 1, H, W).
    Returns:
        torch.Tensor: A tensor of shape (B, 1, 4) with boxes in [x1, y1, x2, y2] format.
    """
    batch_size = masks.shape[0]
    boxes = torch.zeros((batch_size, 1, 4), device=masks.device)
    for i in range(batch_size):
        mask = masks[i, 0]
        y_indices, x_indices = torch.where(mask > 0)
        if len(y_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            boxes[i, 0] = torch.tensor([x_min, y_min, x_max, y_max])
    return boxes.float()

def setup_logging(log_path):
    """Sets up logging to file and console."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

# -- Training and Testing Logic --

def train(args):
    """Main training loop."""
    logging.info("Starting training process...")
    device = torch.device(args.device)

    # Dataset and DataLoader
    logging.info(f"Loading training data from: {args.data_root}")
    train_dataset = NEUDataset(data_root=args.data_root, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    logging.info(f"Initializing model (SAM: {args.model_type}).")
    logging.info(f"SAM checkpoint path: {args.sam_checkpoint}")
    model = SAMbyUCMT(
        model_type=args.model_type,
        sam_checkpoint=args.sam_checkpoint
    ).to(device)

    # Optimizer (only train the depth encoder)
    optimizer = torch.optim.AdamW(model.depth_encoder.parameters(), lr=args.lr)
    criterion = DiceBCELoss()

    logging.info(f"Training for {args.epochs} epochs on {device}.")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            depth_images = batch['depth'].to(device)
            gt_masks = batch['gt'].to(device)

            # Generate bounding box prompts from ground truth masks
            boxes = get_bounding_box_from_mask(gt_masks)

            # Forward pass
            optimizer.zero_grad()
            predicted_masks, _ = model(depth_images, boxes=boxes)

            loss = criterion(predicted_masks, gt_masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint, overwriting previous one
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint_path)
        logging.info(f"Checkpoint saved to {args.checkpoint_path}")

    logging.info("Training complete.")

def test(args):
    """Main testing loop."""
    logging.info("Starting testing process...")
    device = torch.device(args.device)

    # Dataset and DataLoader
    logging.info(f"Loading test data from: {args.data_root}")
    test_dataset = NEUDataset(data_root=args.data_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    logging.info(f"Initializing model (SAM: {args.model_type}).")
    model = SAMbyUCMT(
        model_type=args.model_type,
        sam_checkpoint=None
    ).to(device)

    # Load trained checkpoint
    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint not found at {args.checkpoint_path}. Please train the model first.")
        return
    logging.info(f"Loading trained checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Saving predictions to: {args.output_dir}")

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            depth_image = batch['depth'].to(device)
            original_size = (batch['original_size'][0].item(), batch['original_size'][1].item())
            filename = batch['name'][0]

            predicted_mask = model.infer(depth_image)

            # Resize to original dimensions
            mask_resized = F.interpolate(
                predicted_mask.unsqueeze(0).unsqueeze(0).float(),
                size=original_size,
                mode='nearest'
            ).squeeze().cpu().numpy().astype(np.uint8) * 255

            # Save prediction
            mask_img = Image.fromarray(mask_resized, mode='L')
            output_path = os.path.join(args.output_dir, filename.replace('.bmp', '.png'))
            mask_img.save(output_path)

    logging.info("Testing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test MM-SAM on the NEU-RSSDDS-AUG dataset.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Run mode.")
    parser.add_argument('--data_root', type=str, default='../datasets/NEU-RSDDS-AUG', help="Root directory of the dataset.")
    parser.add_argument('--sam_checkpoint', type=str, default=None, help="Path to the pre-trained SAM checkpoint (e.g., sam_vit_b_01ec64.pth).")
    parser.add_argument('--log_file', type=str, default='/hy-tmp/output/result.log', help="Path to save the log file.")
    parser.add_argument('--checkpoint_path', type=str, default='/hy-tmp/output/checkpoint.pth', help="Path to save or load the model checkpoint.")
    parser.add_argument('--output_dir', type=str, default='/hy-tmp/output/predictions/', help="Directory to save test predictions.")
    parser.add_argument('--model_type', type=str, default='vit_b', help="SAM model type (e.g., vit_b, vit_l, vit_h).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for training/testing.")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    if args.mode == 'train':
        if not args.sam_checkpoint:
            logging.error("A pre-trained SAM checkpoint must be provided for training.")
        else:
            train(args)
    elif args.mode == 'test':
        test(args)