import torch
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import os
import secrets


def set_seed(logger, seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logger.debug(f"Set random seed to {seed}")


def get_device(logger, device=None):
    if device is None:
        # auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.debug(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.debug("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.debug("Using CPU")
    else:
        device = torch.device(device)

    return device


def set_device(logger, model, device=None):
    """Move model to device"""
    device = get_device(logger, device)
    model = model.to(device)
    logger.info(f"Set model device to: {device}")

    return model, device


def generate_unique_model_id(logger, model_type, save_dir):
    """Generates unique ID for training run"""
    date = datetime.now().strftime("%Y%m%d")
    short_hash = secrets.token_hex(3) # 6 character hex
    run_id = f"{model_type}_{date}_{short_hash}"
    model_dir = os.path.join(save_dir, run_id)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Saving to: {model_dir}")
    return model_dir, run_id


def save_training_config(logger, config, save_dir):
    """Save training config to JSON"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "config.json"

    # convert non-serializable objects to strings
    serializable = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable[key] = value
        else:
            serializable[key] = str(value)

    with open(config_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    logger.debug(f"Saved training config to {config_path}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(logger, state, save_dir, filename="checkpoint.pth", is_best=False):
    """Save model checkpoint"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(state, best_path)
        logger.debug(f"Saved best model to {best_path}")


def load_checkpoint(logger, checkpoint_path, model, optimizer=None, device=None):
    """Load model checkpoint"""
    device = get_device(logger, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.debug(f"Loaded checkpoint from {checkpoint_path}")
    logger.debug(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    metric_value = checkpoint['best_si_snr']
    logger.debug(f"  Best SI-SNR: {metric_value:.4f} dB")

    start_epoch = checkpoint.get("epoch", 1) + 1

    return checkpoint, start_epoch, metric_value


class AverageMeter:
    """Track average values during training"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.values = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(val)

    def std(self):
        """Compute standard deviation of tracked values"""
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values))

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"

