import os
import time
from typing import Dict, List
import torch
import torch.nn.functional as F

from mm_sam.train_agents.cm_transfer.base import CMTransferSAM
from utilbox.global_config import OUTPUT_ROOT
from utilbox.neu_rssdds_logger import (
    setup_neu_rssdds_logger, log_training_start, log_training_epoch, 
    log_training_end, log_testing_start, log_testing_progress, 
    log_testing_end, log_error, log_checkpoint_save
)


class NEURSSDDSCMTransferSAM(CMTransferSAM):
    """
    Training agent for NEU-RSSDDS-AUG defect detection dataset using Cross-Modal Transfer.
    Extends the base CM Transfer SAM to work with RGB-D defect detection data.
    """
    
    def __init__(self, train_data: str = "neu_rssdds", **kwargs):
        """
        Initialize NEU-RSSDDS CM Transfer training agent.
        
        Args:
            train_data: Dataset name (should be "neu_rssdds")
            **kwargs: Additional arguments passed to base class
        """
        super(NEURSSDDSCMTransferSAM, self).__init__(
            train_data=train_data, 
            x_data_field='depth_images', 
            x_channel_num=1, 
            **kwargs
        )
        
        # Setup logger
        self.logger = setup_neu_rssdds_logger()
        self._training_start_time = None
        self._testing_start_time = None
        self._num_processed = 0

    def preprocess_depth_images(self, depth_images: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Preprocess depth images for the X-encoder.
        
        Args:
            depth_images: List of depth image tensors
            
        Returns:
            List of preprocessed depth image tensors
        """
        processed_images = []
        
        for depth_img in depth_images:
            # Ensure depth image is in the correct format (C, H, W)
            if len(depth_img.shape) == 2:
                depth_img = depth_img.unsqueeze(0)  # Add channel dimension
            elif len(depth_img.shape) == 3 and depth_img.shape[0] != 1:
                # If multiple channels, take the first one
                depth_img = depth_img[0:1, :, :]
            
            # Normalize depth values to [0, 1] if needed
            if depth_img.max() > 1.0:
                depth_img = depth_img / depth_img.max()
            
            # Ensure float32 type
            depth_img = depth_img.to(torch.float32)
            
            processed_images.append(depth_img)
        
        return processed_images

    def preprocess_x_images(self, x_images: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Override base class method to use depth-specific preprocessing.
        
        Args:
            x_images: List of depth image tensors
            
        Returns:
            List of preprocessed depth image tensors
        """
        return self.preprocess_depth_images(x_images)

    def train_step(self, batch: Dict, epoch: int, step: int) -> Dict[str, torch.Tensor]:
        """
        Single training step for NEU-RSSDDS defect detection.
        
        Args:
            batch: Batch of training data
            epoch: Current epoch number
            step: Current step number
            
        Returns:
            Dictionary containing loss values and metrics
        """
        # Extract data from batch
        images = batch['images']  # List of depth images
        gt_masks = batch['gt_masks']  # List of ground truth masks
        point_coords = batch.get('point_coords', None)
        box_coords = batch.get('box_coords', None)
        
        batch_size = len(images)
        device = images[0].device if len(images) > 0 else torch.device('cpu')
        
        # Set images for SAM encoder using the base class method
        self.set_infer_img(data_dict=batch)
        
        # Prepare prompts for the entire batch
        batch_point_coords = []
        batch_box_coords = []
        valid_indices = []
        
        for i in range(batch_size):
            sample_point_coords = point_coords[i] if point_coords is not None else None
            sample_box_coords = box_coords[i] if box_coords is not None else None
            
            # Skip samples without valid prompts
            if sample_point_coords is None and sample_box_coords is None:
                batch_point_coords.append(None)
                batch_box_coords.append(None)
                continue
            
            # Process point coordinates
            if sample_point_coords is not None:
                # Filter out placeholder points (label = -1)
                if hasattr(sample_point_coords, 'shape') and len(sample_point_coords.shape) == 2:
                    valid_points = sample_point_coords[:, 2] != -1  # Assuming format (x, y, label)
                    if valid_points.any():
                        batch_point_coords.append(sample_point_coords[valid_points, :2])  # (N, 2)
                    else:
                        batch_point_coords.append(None)
                else:
                    batch_point_coords.append(None)
            else:
                batch_point_coords.append(None)
            
            # Process box coordinates
            if sample_box_coords is not None:
                batch_box_coords.append(sample_box_coords)
            else:
                batch_box_coords.append(None)
            
            valid_indices.append(i)
        
        # Skip if no valid samples
        if not valid_indices:
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'valid_samples': torch.tensor(0, device=device),
                'batch_size': torch.tensor(batch_size, device=device)
            }
        
        try:
            # Forward pass through SAM for the entire batch
            pred_masks, pred_ious = self.sam.infer(
                point_coords=batch_point_coords,
                box_coords=batch_box_coords
            )
            
            total_loss = 0.0
            valid_samples = 0
            
            # Compute loss for each valid sample
            for idx, i in enumerate(valid_indices):
                if batch_point_coords[i] is None and batch_box_coords[i] is None:
                    continue
                
                pred_mask = pred_masks[idx]  # Get prediction for this sample
                sample_gt_mask = gt_masks[i]  # Ground truth mask
                
                # Resize ground truth to match prediction if needed
                if pred_mask.shape != sample_gt_mask.shape:
                    sample_gt_mask = F.interpolate(
                        sample_gt_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=pred_mask.shape,
                        mode='nearest'
                    ).squeeze().long()
                
                # Convert to binary (0 or 1)
                sample_gt_mask = (sample_gt_mask > 0).long()
                
                # Compute binary cross-entropy loss
                pred_mask_logits = torch.logit(pred_mask.clamp(1e-7, 1-1e-7))
                loss = F.binary_cross_entropy_with_logits(
                    pred_mask_logits, sample_gt_mask.float()
                )
                
                total_loss += loss
                valid_samples += 1
                
        except Exception as e:
            print(f"Error during SAM inference: {e}")
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'valid_samples': torch.tensor(0, device=device),
                'batch_size': torch.tensor(batch_size, device=device)
            }
        
        # Average loss over valid samples
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {
            'loss': avg_loss,
            'valid_samples': torch.tensor(valid_samples, device=device),
            'batch_size': torch.tensor(batch_size, device=device)
        }

    def before_train_epoch(self):
        """Hook called before each training epoch."""
        super().before_train_epoch()
        if self._training_start_time is None:
            self._training_start_time = time.time()
            config = {
                'dataset': 'NEU-RSSDDS-AUG',
                'model': 'CM Transfer SAM',
                'x_data_field': self.x_data_field,
                'x_channel_num': 1
            }
            log_training_start(self.logger, config)

    def after_train_epoch(self):
        """Hook called after each training epoch to save checkpoint."""
        super().after_train_epoch()
        
        # Save checkpoint to the specified path (overwrite each time)
        checkpoint_path = os.path.join(OUTPUT_ROOT, 'output', 'checkpoint.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Save model state dict
        model_state = self.model_state_dict()
        torch.save(model_state, checkpoint_path)
        
        # Log checkpoint save
        current_epoch = getattr(self, '_current_epoch', None)
        log_checkpoint_save(self.logger, checkpoint_path, current_epoch)

    def before_test(self):
        """Hook called before testing starts."""
        super().before_test()
        self._testing_start_time = time.time()
        self._num_processed = 0
        checkpoint_path = os.path.join(OUTPUT_ROOT, 'output', 'checkpoint.pth')
        log_testing_start(self.logger, checkpoint_path)

    def after_test(self):
        """Hook called after testing ends."""
        super().after_test()
        if self._testing_start_time is not None:
            total_time = time.time() - self._testing_start_time
            output_dir = os.path.join(OUTPUT_ROOT, 'output', 'predictions')
            log_testing_end(self.logger, self._num_processed, total_time, output_dir)

    def load_checkpoint_for_testing(self, checkpoint_path: str = None):
        """
        Load checkpoint for testing.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, uses default path.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(OUTPUT_ROOT, 'output', 'checkpoint.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_model_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    def save_prediction_masks(self, predictions: List[torch.Tensor], filenames: List[str], original_sizes: List[tuple]):
        """
        Save prediction masks to the output directory.
        
        Args:
            predictions: List of prediction tensors
            filenames: List of original filenames (without extension)
            original_sizes: List of (height, width) tuples for original image sizes
        """
        import numpy as np
        from PIL import Image
        
        output_dir = os.path.join(OUTPUT_ROOT, 'output', 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        
        for pred, filename, orig_size in zip(predictions, filenames, original_sizes):
            # Convert prediction to numpy
            if isinstance(pred, torch.Tensor):
                pred_np = pred.detach().cpu().numpy()
            else:
                pred_np = pred
            
            # Ensure binary values (0 or 255)
            pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
            
            # Resize to original size if needed
            if pred_binary.shape != orig_size:
                pred_img = Image.fromarray(pred_binary)
                pred_img = pred_img.resize((orig_size[1], orig_size[0]), Image.NEAREST)
                pred_binary = np.array(pred_img)
            
            # Save as PNG
            output_path = os.path.join(output_dir, f"{filename}.png")
            Image.fromarray(pred_binary).save(output_path)
        
        print(f"Saved {len(predictions)} prediction masks to {output_dir}")

    def test_step(self, batch: Dict, iter_name: str = None):
        """
        Custom test step that generates and saves predictions.
        
        Args:
            batch: Batch of test data
            iter_name: Iterator name (unused)
        """
        # Get data from batch
        images = batch['images']  # List of depth images
        index_names = batch['index_name']
        
        # Set images for SAM encoder
        self.set_infer_img(data_dict=batch)
        
        predictions = []
        filenames = []
        original_sizes = []
        
        batch_size = len(index_names)
        
        # Prepare bounding box prompts for all samples
        batch_box_coords = []
        batch_output_sizes = []
        
        for i in range(batch_size):
            # For test data, use the entire image as a bounding box prompt
            h, w = images[i].shape[-2:]
            box_coords = torch.tensor([0, 0, w-1, h-1], dtype=torch.float32, device=images[i].device)
            batch_box_coords.append(box_coords)
            batch_output_sizes.append((h, w))
        
        try:
            # Generate predictions for the entire batch
            pred_masks, pred_ious = self.sam.infer(
                box_coords=batch_box_coords,
                output_mask_size=batch_output_sizes
            )
            
            # Process each prediction
            for i in range(batch_size):
                pred_mask = pred_masks[i]  # Get prediction for this sample
                h, w = batch_output_sizes[i]
                
                # Store prediction info
                predictions.append(pred_mask)
                filenames.append(index_names[i])
                original_sizes.append((h, w))
                
        except Exception as e:
            print(f"Error during batch inference: {e}")
            # Create empty predictions as fallback
            for i in range(batch_size):
                h, w = images[i].shape[-2:]
                empty_pred = torch.zeros((h, w), device=images[i].device)
                predictions.append(empty_pred)
                filenames.append(index_names[i])
                original_sizes.append((h, w))
        
        # Save predictions
        if predictions:
            self.save_prediction_masks(predictions, filenames, original_sizes)
            self._num_processed += len(predictions)
            log_testing_progress(self.logger, self._num_processed, self._num_processed)

    def get_test_results(self) -> Dict:
        """
        Return test results. For inference-only testing, we return empty metrics.
        """
        return {
            'message': 'Inference completed. Predictions saved to output directory.',
            'num_processed': getattr(self, '_num_processed', 0)
        }