"""
Logging configuration for NEU-RSSDDS training and testing.
"""

import logging
import os
from datetime import datetime
from utilbox.global_config import OUTPUT_ROOT


def setup_neu_rssdds_logger(name: str = 'neu_rssdds') -> logging.Logger:
    """
    Setup unified logger for NEU-RSSDDS training and testing.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log file path
    log_file = os.path.join(OUTPUT_ROOT, 'output', 'result.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_training_start(logger: logging.Logger, config: dict):
    """Log training start information."""
    logger.info("=" * 50)
    logger.info("NEU-RSSDDS TRAINING STARTED")
    logger.info("=" * 50)
    logger.info(f"Configuration: {config}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def log_training_epoch(logger: logging.Logger, epoch: int, total_epochs: int, loss: float, metrics: dict = None):
    """Log training epoch information."""
    msg = f"Epoch [{epoch}/{total_epochs}] - Loss: {loss:.4f}"
    if metrics:
        for key, value in metrics.items():
            msg += f" - {key}: {value:.4f}"
    logger.info(msg)


def log_training_end(logger: logging.Logger, total_time: float):
    """Log training end information."""
    logger.info("=" * 50)
    logger.info("NEU-RSSDDS TRAINING COMPLETED")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info("=" * 50)


def log_testing_start(logger: logging.Logger, checkpoint_path: str):
    """Log testing start information."""
    logger.info("=" * 50)
    logger.info("NEU-RSSDDS TESTING STARTED")
    logger.info("=" * 50)
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def log_testing_progress(logger: logging.Logger, processed: int, total: int):
    """Log testing progress."""
    if processed % 10 == 0 or processed == total:  # Log every 10 samples or at the end
        logger.info(f"Processed {processed}/{total} test samples ({processed/total*100:.1f}%)")


def log_testing_end(logger: logging.Logger, total_samples: int, total_time: float, output_dir: str):
    """Log testing end information."""
    logger.info("=" * 50)
    logger.info("NEU-RSSDDS TESTING COMPLETED")
    logger.info(f"Processed {total_samples} test samples")
    logger.info(f"Total testing time: {total_time:.2f} seconds")
    logger.info(f"Predictions saved to: {output_dir}")
    logger.info("=" * 50)


def log_error(logger: logging.Logger, error_msg: str, exception: Exception = None):
    """Log error information."""
    logger.error(f"ERROR: {error_msg}")
    if exception:
        logger.error(f"Exception details: {str(exception)}")


def log_checkpoint_save(logger: logging.Logger, checkpoint_path: str, epoch: int = None):
    """Log checkpoint save information."""
    msg = f"Checkpoint saved to: {checkpoint_path}"
    if epoch is not None:
        msg += f" (Epoch {epoch})"
    logger.info(msg)