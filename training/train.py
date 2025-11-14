# training/train.py
"""
Training pipeline for multimodal surgical risk prediction model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast  # GradScaler imported conditionally later
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
from datetime import datetime

from config import TRAINING_CONFIG, MODEL_CONFIG, COMPLICATIONS, LOGS_DIR, MODEL_DIR
from models.model import MultimodalSurgicalRiskModel


def calculate_class_weights(data_loader: DataLoader, device: str = 'cpu') -> Dict[str, float]:
    """
    Calculate class weights for each task based on positive/negative ratio
    
    Args:
        data_loader: DataLoader for the training data
        device: Device to use for computation
    
    Returns:
        Dictionary of class weights (pos_weight) for each task
    """
    print("Calculating class weights for imbalanced complications...")
    
    # Count positive and negative cases for each task
    task_counts = {comp: {'pos': 0, 'neg': 0} for comp in COMPLICATIONS.keys()}
    
    for batch in data_loader:
        outcomes = batch['outcomes']
        for task_name in COMPLICATIONS.keys():
            targets = outcomes[task_name]
            task_counts[task_name]['pos'] += targets.sum().item()
            task_counts[task_name]['neg'] += (1 - targets).sum().item()
    
    # Calculate pos_weight = n_neg / n_pos (higher weight for rare positive cases)
    class_weights = {}
    for task_name, counts in task_counts.items():
        n_pos = max(counts['pos'], 1)  # Avoid division by zero
        n_neg = max(counts['neg'], 1)
        pos_weight = n_neg / n_pos
        
        # Cap at 10 to avoid extreme weights for very rare complications
        pos_weight = min(pos_weight, 10.0)
        
        class_weights[task_name] = pos_weight
        
        prevalence = n_pos / (n_pos + n_neg) * 100
        print(f"  {task_name}: {prevalence:.1f}% positive, pos_weight={pos_weight:.2f}")
    
    return class_weights

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean', pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight  # Additional weight for positive class
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [batch_size]
            targets: Ground truth [batch_size]
        """
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Apply positive class weight if specified
        if self.pos_weight != 1.0:
            bce_loss = bce_loss * (targets * self.pos_weight + (1 - targets))
        
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with task-specific weights and class balancing
    """
    
    def __init__(self, 
                 loss_type: str = 'focal',
                 task_weights: Dict[str, float] = None,
                 class_weights: Dict[str, float] = None,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25):
        super().__init__()
        
        self.loss_type = loss_type
        
        # Task weights from config
        if task_weights is None:
            task_weights = {comp: COMPLICATIONS[comp]['weight'] 
                          for comp in COMPLICATIONS.keys()}
        self.task_weights = task_weights
        
        # Class weights for imbalanced data (pos_weight for each task)
        self.class_weights = class_weights or {}
        
        # Loss function per task (each task gets its own focal loss with appropriate pos_weight)
        self.criterions = {}
        for task_name in COMPLICATIONS.keys():
            pos_weight = self.class_weights.get(task_name, 1.0)
            if loss_type == 'focal':
                self.criterions[task_name] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
            elif loss_type == 'weighted_bce':
                self.criterions[task_name] = nn.BCELoss()
            else:
                self.criterions[task_name] = nn.BCELoss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss
        
        Returns:
            total_loss, individual_losses
        """
        total_loss = 0.0
        individual_losses = {}
        
        for task_name in sorted(COMPLICATIONS.keys()):
            pred = predictions[task_name]
            target = targets[task_name]
            weight = self.task_weights.get(task_name, 1.0)
            
            task_loss = self.criterions[task_name](pred, target)
            weighted_loss = weight * task_loss
            
            total_loss += weighted_loss
            individual_losses[task_name] = task_loss.item()
        
        return total_loss, individual_losses


class Trainer:
    """
    Trainer for multimodal surgical risk prediction model
    
    Features:
    - Multi-task learning
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """
    
    def __init__(self,
                 model: MultimodalSurgicalRiskModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TRAINING_CONFIG
        
        # Device - prioritize MPS for Apple Silicon
        # NOTE: PyTorch 2.9.1 has stability issues with large models on MPS
        # Using CPU for now until PyTorch MPS support improves
        use_mps = self.config.get('use_mps', False)  # Disabled by default
        
        if use_mps and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"✓ Using Apple Silicon GPU (MPS) for training")
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using NVIDIA GPU (CUDA) for training")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            print(f"Using CPU for training (MPS disabled due to stability issues with PyTorch 2.9.1)")
            print(f"  To enable MPS: set 'use_mps': True in config (may cause segfaults)")
        
        self.model.to(self.device)
        
        # Multi-GPU (only for CUDA)
        if torch.cuda.device_count() > 1 and self.config.get('num_gpus', 1) > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Calculate class weights from training data
        class_weights = calculate_class_weights(train_loader, device=str(self.device))
        
        # Loss function with class weights
        self.criterion = MultiTaskLoss(
            loss_type=self.config['loss_function'],
            class_weights=class_weights,
            focal_gamma=self.config.get('focal_loss_gamma', 2.0),
            focal_alpha=self.config.get('focal_loss_alpha', 0.25)
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision (only for CUDA)
        self.use_amp = self.config.get('use_mixed_precision', True) and torch.cuda.is_available()
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✓ Mixed precision training enabled (CUDA)")
        else:
            self.scaler = None
            if self.config.get('use_mixed_precision', True):
                print("Mixed precision disabled (requires CUDA)")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'task_losses': {task: [] for task in COMPLICATIONS.keys()}
        }
        
        # Directories
        self.checkpoint_dir = Path(MODEL_DIR)
        self.log_dir = Path(LOGS_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        if self.config.get('use_tensorboard', True):
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        else:
            self.writer = None
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=self.config.get('betas', (0.9, 0.999)),
                eps=self.config.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine_with_warmup')
        
        if scheduler_name == 'cosine_with_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            warmup_epochs = self.config.get('warmup_epochs', 5)
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=warmup_epochs,
                T_mult=2
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('reduce_lr_factor', 0.5),
                patience=self.config.get('reduce_lr_patience', 5),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        num_epochs = self.config['num_epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_metrics = self._train_epoch()
            
            # Validate
            val_loss, val_metrics = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(current_lr)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                for task_name, task_loss in train_metrics['task_losses'].items():
                    self.writer.add_scalar(f'Task_Loss_Train/{task_name}', task_loss, epoch)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config['early_stopping_patience']}")
            
            # Regular checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self._save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "="*60)
        print("Training Completed")
        print("="*60 + "\n")
        
        if self.writer:
            self.writer.close()
        
        return self.training_history
    
    def _train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        task_losses = {task: 0.0 for task in COMPLICATIONS.keys()}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        time_series=batch['time_series_full'],
                        phase_markers=batch['phase_markers'],
                        ts_attention_mask=batch['mask_full'],
                        text_embedding=batch['text_combined'],
                        static_features=batch['static']
                    )
                    
                    # Prepare targets
                    targets = {task: batch['outcomes'][:, i] 
                             for i, task in enumerate(sorted(COMPLICATIONS.keys()))}
                    
                    # Compute loss
                    loss, individual_losses = self.criterion(
                        outputs['predictions'],
                        targets
                    )
                    
                    # Gradient accumulation
                    loss = loss / self.config.get('gradient_accumulation_steps', 1)
            else:
                outputs = self.model(
                    time_series=batch['time_series_full'],
                    phase_markers=batch['phase_markers'],
                    ts_attention_mask=batch['mask_full'],
                    text_embedding=batch['text_combined'],
                    static_features=batch['static']
                )
                
                targets = {task: batch['outcomes'][:, i] 
                         for i, task in enumerate(sorted(COMPLICATIONS.keys()))}
                
                loss, individual_losses = self.criterion(
                    outputs['predictions'],
                    targets
                )
                
                loss = loss / self.config.get('gradient_accumulation_steps', 1)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate losses
            total_loss += loss.item() * self.config.get('gradient_accumulation_steps', 1)
            for task, task_loss in individual_losses.items():
                task_losses[task] += task_loss
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        metrics = {
            'task_losses': avg_task_losses
        }
        
        return avg_loss, metrics
    
    def _validate_epoch(self) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        task_losses = {task: 0.0 for task in COMPLICATIONS.keys()}
        
        all_predictions = {task: [] for task in COMPLICATIONS.keys()}
        all_targets = {task: [] for task in COMPLICATIONS.keys()}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            
            for batch in pbar:
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    time_series=batch['time_series_full'],
                    phase_markers=batch['phase_markers'],
                    ts_attention_mask=batch['mask_full'],
                    text_embedding=batch['text_combined'],
                    static_features=batch['static']
                )
                
                # Prepare targets
                targets = {task: batch['outcomes'][:, i] 
                         for i, task in enumerate(sorted(COMPLICATIONS.keys()))}
                
                # Compute loss
                loss, individual_losses = self.criterion(
                    outputs['predictions'],
                    targets
                )
                
                # Accumulate
                total_loss += loss.item()
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss
                
                # Store predictions and targets
                for task in COMPLICATIONS.keys():
                    all_predictions[task].append(outputs['predictions'][task].cpu())
                    all_targets[task].append(targets[task].cpu())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        # Concatenate all predictions
        all_predictions = {task: torch.cat(preds) for task, preds in all_predictions.items()}
        all_targets = {task: torch.cat(targets) for task, targets in all_targets.items()}
        
        metrics = {
            'task_losses': avg_task_losses,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch+1}.pt'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")