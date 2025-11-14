# visualization/plot_results.py
"""
Comprehensive result visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.calibration import calibration_curve

from config import COMPLICATIONS, FIGURES_DIR, VISUALIZATION_CONFIG

class ResultsPlotter:
    """
    Create comprehensive visualizations for model results
    
    Plots:
    - Training curves
    - ROC curves
    - PR curves
    - Confusion matrices
    - Calibration plots
    - Risk stratification
    """
    
    def __init__(self, save_dir: Path = None):
        self.save_dir = Path(save_dir or FIGURES_DIR) / 'results'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = VISUALIZATION_CONFIG
        
        # Set style
        sns.set_style(self.config['style'])
        plt.rcParams['figure.dpi'] = self.config['dpi']
        plt.rcParams['savefig.dpi'] = self.config['dpi']
    
    def plot_training_curves(self, history: Dict):
        """Plot training and validation curves"""
        print("Creating training curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[0, 1]
        ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, pad=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Task losses (subset)
        ax = axes[1, 0]
        for i, (task, losses) in enumerate(list(history['task_losses'].items())[:4]):
            if losses:
                ax.plot(epochs[:len(losses)], losses, label=COMPLICATIONS[task]['name'], linewidth=1.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Task Loss', fontsize=12)
        ax.set_title('Individual Task Losses', fontsize=14, pad=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Min/Max/Mean loss
        ax = axes[1, 1]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.fill_between(epochs, 
                        np.array(history['train_loss']) * 0.9,
                        np.array(history['train_loss']) * 1.1,
                        alpha=0.2, color='blue')
        ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax.fill_between(epochs,
                        np.array(history['val_loss']) * 0.9,
                        np.array(history['val_loss']) * 1.1,
                        alpha=0.2, color='red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss with Uncertainty', fontsize=14, pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'training_curves.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_roc_curves(self, results: Dict):
        """Plot ROC curves for all tasks"""
        print("Creating ROC curves...")
        
        n_tasks = len(COMPLICATIONS)
        n_cols = 3
        n_rows = (n_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, (task_name, task_results) in enumerate(sorted(results['per_task'].items())):
            ax = axes[idx]
            
            # Get ROC curve data
            fpr = task_results['roc_curve']['fpr']
            tpr = task_results['roc_curve']['tpr']
            auroc = task_results['auroc']
            
            # Plot
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {auroc:.3f}')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.5, label='Random')
            
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(COMPLICATIONS[task_name]['name'], fontsize=12, pad=10)
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for idx in range(n_tasks, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('ROC Curves for All Complications', fontsize=16, y=1.00)
        plt.tight_layout()
        
        save_path = self.save_dir / 'roc_curves_all.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_pr_curves(self, results: Dict):
        """Plot Precision-Recall curves"""
        print("Creating PR curves...")
        
        n_tasks = len(COMPLICATIONS)
        n_cols = 3
        n_rows = (n_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, (task_name, task_results) in enumerate(sorted(results['per_task'].items())):
            ax = axes[idx]
            
            # Get PR curve data
            precision = task_results['pr_curve']['precision']
            recall = task_results['pr_curve']['recall']
            auprc = task_results['auprc']
            
            # Plot
            ax.plot(recall, precision, 'b-', linewidth=2, label=f'AUPRC = {auprc:.3f}')
            
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_title(COMPLICATIONS[task_name]['name'], fontsize=12, pad=10)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for idx in range(n_tasks, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Precision-Recall Curves for All Complications', fontsize=16, y=1.00)
        plt.tight_layout()
        
        save_path = self.save_dir / 'pr_curves_all.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_confusion_matrices(self, results: Dict):
        """Plot confusion matrices"""
        print("Creating confusion matrices...")
        
        n_tasks = len(COMPLICATIONS)
        n_cols = 3
        n_rows = (n_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten()
        
        for idx, (task_name, task_results) in enumerate(sorted(results['per_task'].items())):
            ax = axes[idx]
            
            cm = task_results['confusion_matrix']
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=ax, cbar_kws={'format': '%.0f%%'})
            
            ax.set_title(COMPLICATIONS[task_name]['name'], fontsize=11, pad=10)
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_tasks, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices (Normalized)', fontsize=16, y=1.00)
        plt.tight_layout()
        
        save_path = self.save_dir / 'confusion_matrices.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_calibration_curves(self, 
                               predictions: Dict[str, np.ndarray],
                               targets: Dict[str, np.ndarray]):
        """Plot calibration curves"""
        print("Creating calibration plots...")
        
        n_tasks = len(COMPLICATIONS)
        n_cols = 3
        n_rows = (n_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, task_name in enumerate(sorted(COMPLICATIONS.keys())):
            ax = axes[idx]
            
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                target, pred, n_bins=10, strategy='uniform'
            )
            
            # Plot
            ax.plot(mean_predicted_value, fraction_of_positives, 'bs-', 
                   linewidth=2, label='Model')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect Calibration')
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=10)
            ax.set_ylabel('Fraction of Positives', fontsize=10)
            ax.set_title(COMPLICATIONS[task_name]['name'], fontsize=12, pad=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for idx in range(n_tasks, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Calibration Curves', fontsize=16, y=1.00)
        plt.tight_layout()
        
        save_path = self.save_dir / 'calibration_curves.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_performance_summary(self, results: Dict):
        """Create comprehensive performance summary plot"""
        print("Creating performance summary...")
        
        # Prepare data
        metrics_data = []
        for task_name, task_results in sorted(results['per_task'].items()):
            metrics_data.append({
                'Task': COMPLICATIONS[task_name]['name'],
                'AUROC': task_results['auroc'],
                'AUPRC': task_results['auprc'],
                'F1': task_results['f1_score'],
                'Precision': task_results['precision'],
                'Recall': task_results['recall']
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(df))
        width = 0.15
        
        metrics = ['AUROC', 'AUPRC', 'F1', 'Precision', 'Recall']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 2)
            ax.bar(x + offset, df[metric], width, label=metric, color=color, alpha=0.8)
        
        ax.set_xlabel('Complication', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Summary Across All Complications', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Task'], rotation=45, ha='right')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add horizontal line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'performance_summary.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")