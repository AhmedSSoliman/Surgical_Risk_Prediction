# training/evaluate.py
"""
Evaluation and metrics for surgical risk prediction model
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from scipy.special import softmax
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

from config import COMPLICATIONS, EVALUATION_METRICS, RESULTS_DIR

class Evaluator:
    """
    Comprehensive evaluator for surgical risk prediction
    
    Computes:
    - AUROC, AUPRC
    - Accuracy, Precision, Recall, F1
    - Calibration metrics (Brier score, ECE)
    - Confusion matrices
    - Per-task and overall metrics
    """
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                uncertainties: Dict[str, torch.Tensor] = None) -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            predictions: Dictionary of predictions for each task
            targets: Dictionary of ground truth for each task
            uncertainties: Optional uncertainties
            
        Returns:
            Dictionary with all metrics
        """
        results = {
            'per_task': {},
            'overall': {},
            'curves': {}
        }
        
        all_aurocs = []
        all_auprcs = []
        
        # Evaluate each task
        for task_name in sorted(COMPLICATIONS.keys()):
            pred = predictions[task_name].numpy()
            target = targets[task_name].numpy()
            
            task_metrics = self._compute_task_metrics(pred, target, task_name)
            results['per_task'][task_name] = task_metrics
            
            all_aurocs.append(task_metrics['auroc'])
            all_auprcs.append(task_metrics['auprc'])
            
            # Store curves
            results['curves'][task_name] = {
                'roc': task_metrics['roc_curve'],
                'pr': task_metrics['pr_curve']
            }
        
        # Overall metrics
        results['overall'] = {
            'mean_auroc': np.mean(all_aurocs),
            'std_auroc': np.std(all_aurocs),
            'mean_auprc': np.mean(all_auprcs),
            'std_auprc': np.std(all_auprcs)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _compute_task_metrics(self, 
                              pred: np.ndarray,
                              target: np.ndarray,
                              task_name: str) -> Dict:
        """Compute metrics for single task"""
        metrics = {}
        
        # Threshold predictions
        pred_binary = (pred >= 0.5).astype(int)
        
        # AUROC
        try:
            metrics['auroc'] = roc_auc_score(target, pred)
        except:
            metrics['auroc'] = 0.5
        
        # AUPRC
        try:
            metrics['auprc'] = average_precision_score(target, pred)
        except:
            metrics['auprc'] = 0.0
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(target, pred_binary)
        metrics['precision'] = precision_score(target, pred_binary, zero_division=0)
        metrics['recall'] = recall_score(target, pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(target, pred_binary, zero_division=0)
        
        # Specificity and NPV
        tn, fp, fn, tp = confusion_matrix(target, pred_binary, labels=[0, 1]).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(target, pred_binary)
        
        # Brier score
        metrics['brier_score'] = brier_score_loss(target, pred)
        
        # Calibration
        metrics['ece'] = self._compute_ece(pred, target)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(target, pred)
        metrics['roc_curve'] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        # PR curve
        precision, recall, thresholds = precision_recall_curve(target, pred)
        metrics['pr_curve'] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
        
        return metrics
    
    def _compute_ece(self, pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(pred > bin_lower, pred <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(target[in_bin])
                avg_confidence_in_bin = np.mean(pred[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _save_results(self, results: Dict):
        """Save evaluation results"""
        # Create summary DataFrame
        summary_data = []
        for task_name, metrics in results['per_task'].items():
            row = {
                'Task': COMPLICATIONS[task_name]['name'],
                'AUROC': metrics['auroc'],
                'AUPRC': metrics['auprc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1_score'],
                'Specificity': metrics['specificity'],
                'Brier Score': metrics['brier_score'],
                'ECE': metrics['ece']
            }
            summary_data.append(row)
        
        # Add overall row
        summary_data.append({
            'Task': 'Overall (Mean)',
            'AUROC': results['overall']['mean_auroc'],
            'AUPRC': results['overall']['mean_auprc'],
            'Accuracy': np.mean([m['accuracy'] for m in results['per_task'].values()]),
            'Precision': np.mean([m['precision'] for m in results['per_task'].values()]),
            'Recall': np.mean([m['recall'] for m in results['per_task'].values()]),
            'F1': np.mean([m['f1_score'] for m in results['per_task'].values()]),
            'Specificity': np.mean([m['specificity'] for m in results['per_task'].values()]),
            'Brier Score': np.mean([m['brier_score'] for m in results['per_task'].values()]),
            'ECE': np.mean([m['ece'] for m in results['per_task'].values()])
        })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save
        summary_path = self.results_dir / 'evaluation_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nEvaluation Results saved to: {summary_path}")
        print("\n" + "="*80)
        print(summary_df.to_string(index=False))
        print("="*80 + "\n")
    
    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80 + "\n")
        
        for task_name in sorted(COMPLICATIONS.keys()):
            metrics = results['per_task'][task_name]
            
            print(f"{COMPLICATIONS[task_name]['name']}:")
            print(f"  AUROC: {metrics['auroc']:.4f}")
            print(f"  AUPRC: {metrics['auprc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print()
        
        print("Overall Performance:")
        print(f"  Mean AUROC: {results['overall']['mean_auroc']:.4f} ± {results['overall']['std_auroc']:.4f}")
        print(f"  Mean AUPRC: {results['overall']['mean_auprc']:.4f} ± {results['overall']['std_auprc']:.4f}")
        print("\n" + "="*80 + "\n")