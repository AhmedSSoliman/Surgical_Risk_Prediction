# utils/utils.py
"""
Utility functions
"""

import torch
import numpy as np
import random
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from config import LOGS_DIR

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen: {frozen:,} ({100*frozen/total:.1f}%)")
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen
    }


def save_config(config: Dict[str, Any], filepath: Path):
    """Save configuration to YAML file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to: {filepath}")


def load_config(filepath: Path) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from: {filepath}")
    return config


def create_logger(name: str, log_file: str = None) -> logging.Logger:
    """Create logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(LOGS_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if should stop"""
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is improvement"""
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)


class MetricTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key: str, last_n: int = None) -> float:
        """Get average of metric"""
        if key not in self.metrics:
            return 0.0
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values) if values else 0.0
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}