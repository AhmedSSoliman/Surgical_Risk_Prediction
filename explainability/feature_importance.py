# explainability/feature_importance.py
"""
Feature importance analysis via permutation and ablation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import EXPLAINABILITY_CONFIG, COMPLICATIONS, FIGURES_DIR

class FeatureImportanceAnalyzer:
    """
    Analyze feature importance through:
    - Permutation importance
    - Ablation studies
    - Gradient-based importance
    """
    
    def __init__(self, model, dataloader, device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.config = EXPLAINABILITY_CONFIG['feature_importance']
        self.figures_dir = Path(FIGURES_DIR) / 'feature_importance'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_permutation_importance(self,
                                     feature_groups: Dict[str, List[int]] = None) -> Dict:
        """
        Compute permutation importance for feature groups
        
        Args:
            feature_groups: Dictionary mapping group names to feature indices
            
        Returns:
            Dictionary with importance scores
        """
        print("Computing permutation importance...")
        
        self.model.eval()
        
        # Get baseline performance
        baseline_metrics = self._evaluate_model()
        
        # Default feature groups if not provided
        if feature_groups is None:
            feature_groups = {
                'time_series': 'time_series_full',
                'text': 'text_combined',
                'static': 'static',
                'medications': 'medications'
            }
        
        importance_scores = {}
        
        for group_name, feature_key in feature_groups.items():
            print(f"  Permuting {group_name}...")
            
            # Permute this feature group
            permuted_metrics = self._evaluate_with_permutation(feature_key)
            
            # Calculate importance as performance drop
            importance = {}
            for task in COMPLICATIONS.keys():
                baseline_auroc = baseline_metrics[task]['auroc']
                permuted_auroc = permuted_metrics[task]['auroc']
                importance[task] = baseline_auroc - permuted_auroc
            
            importance_scores[group_name] = importance
        
        # Visualize
        self._plot_importance_scores(importance_scores)
        
        return importance_scores
    
    def _evaluate_model(self) -> Dict:
        """Evaluate model performance"""
        all_predictions = {task: [] for task in COMPLICATIONS.keys()}
        all_targets = {task: [] for task in COMPLICATIONS.keys()}
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    time_series=batch['time_series_full'],
                    phase_markers=batch['phase_markers'],
                    ts_attention_mask=batch['mask_full'],
                    text_embedding=batch['text_combined'],
                    static_features=batch['static']
                )
                
                targets = {task: batch['outcomes'][:, i] 
                         for i, task in enumerate(sorted(COMPLICATIONS.keys()))}
                
                for task in COMPLICATIONS.keys():
                    all_predictions[task].append(outputs['predictions'][task].cpu())
                    all_targets[task].append(targets[task].cpu())
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score
        
        metrics = {}
        for task in COMPLICATIONS.keys():
            pred = torch.cat(all_predictions[task]).numpy()
            target = torch.cat(all_targets[task]).numpy()
            
            try:
                auroc = roc_auc_score(target, pred)
            except:
                auroc = 0.5
            
            metrics[task] = {'auroc': auroc}
        
        return metrics
    
    def _evaluate_with_permutation(self, feature_key: str) -> Dict:
        """Evaluate with permuted features"""
        all_predictions = {task: [] for task in COMPLICATIONS.keys()}
        all_targets = {task: [] for task in COMPLICATIONS.keys()}
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Permuting {feature_key}", leave=False):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Permute the specified feature
                if feature_key in batch:
                    perm_idx = torch.randperm(batch[feature_key].shape[0])
                    batch[feature_key] = batch[feature_key][perm_idx]
                
                outputs = self.model(
                    time_series=batch['time_series_full'],
                    phase_markers=batch['phase_markers'],
                    ts_attention_mask=batch['mask_full'],
                    text_embedding=batch['text_combined'],
                    static_features=batch['static']
                )
                
                targets = {task: batch['outcomes'][:, i] 
                         for i, task in enumerate(sorted(COMPLICATIONS.keys()))}
                
                for task in COMPLICATIONS.keys():
                    all_predictions[task].append(outputs['predictions'][task].cpu())
                    all_targets[task].append(targets[task].cpu())
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score
        
        metrics = {}
        for task in COMPLICATIONS.keys():
            pred = torch.cat(all_predictions[task]).numpy()
            target = torch.cat(all_targets[task]).numpy()
            
            try:
                auroc = roc_auc_score(target, pred)
            except:
                auroc = 0.5
            
            metrics[task] = {'auroc': auroc}
        
        return metrics
    
    def _plot_importance_scores(self, importance_scores: Dict):
        """Plot feature importance scores"""
        print("Creating feature importance plots...")
        
        # Prepare data
        feature_groups = list(importance_scores.keys())
        tasks = sorted(COMPLICATIONS.keys())
        
        # Create matrix
        importance_matrix = np.zeros((len(feature_groups), len(tasks)))
        
        for i, group in enumerate(feature_groups):
            for j, task in enumerate(tasks):
                importance_matrix[i, j] = importance_scores[group][task]
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            importance_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=[COMPLICATIONS[t]['name'] for t in tasks],
            yticklabels=feature_groups,
            cbar_kws={'label': 'Importance (AUROC Drop)'},
            ax=ax
        )
        
        ax.set_title('Permutation Feature Importance', fontsize=14, pad=20)
        ax.set_xlabel('Complication', fontsize=12)
        ax.set_ylabel('Feature Group', fontsize=12)
        
        plt.tight_layout()
        
        save_path = self.figures_dir / 'permutation_importance_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
        
        # Bar plot for mean importance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_importance = importance_matrix.mean(axis=1)
        std_importance = importance_matrix.std(axis=1)
        
        ax.barh(feature_groups, mean_importance, xerr=std_importance, 
               color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Mean Importance (AUROC Drop)', fontsize=12)
        ax.set_ylabel('Feature Group', fontsize=12)
        ax.set_title('Mean Feature Importance Across All Complications', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.figures_dir / 'mean_importance_bars.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")