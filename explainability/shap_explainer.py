# explainability/shap_explainer.py
"""
SHAP (SHapley Additive exPlanations) for model interpretability
"""

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import EXPLAINABILITY_CONFIG, COMPLICATIONS, FIGURES_DIR

class SHAPExplainer:
    """
    SHAP-based explainer for multimodal surgical risk model
    
    Provides:
    - Feature importance via SHAP values
    - Summary plots
    - Dependence plots
    - Waterfall plots for individual predictions
    - Force plots
    """
    
    def __init__(self, 
                 model,
                 background_data: torch.utils.data.DataLoader,
                 device: str = 'cuda'):
        
        self.model = model
        self.device = device
        self.config = EXPLAINABILITY_CONFIG['shap']
        self.figures_dir = Path(FIGURES_DIR) / 'shap'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Get background samples for SHAP
        self.background_samples = self._prepare_background_data(
            background_data,
            n_samples=self.config['num_background_samples']
        )
        
        # Initialize SHAP explainer
        self.explainer = self._create_explainer()
    
    def _prepare_background_data(self, 
                                 dataloader: torch.utils.data.DataLoader,
                                 n_samples: int) -> Dict[str, torch.Tensor]:
        """Prepare background data for SHAP"""
        background = {
            'time_series_full': [],
            'phase_markers': [],
            'mask_full': [],
            'text_combined': [],
            'static': []
        }
        
        count = 0
        for batch in dataloader:
            for key in background.keys():
                background[key].append(batch[key])
            
            count += batch['static'].shape[0]
            if count >= n_samples:
                break
        
        # Concatenate and truncate
        for key in background.keys():
            background[key] = torch.cat(background[key], dim=0)[:n_samples]
        
        return background
    
    def _create_explainer(self):
        """Create SHAP explainer"""
        
        def model_wrapper(inputs):
            """Wrapper function for SHAP"""
            # inputs is numpy array of shape [n_samples, n_features]
            # Need to reconstruct the multimodal input
            
            batch_size = inputs.shape[0]
            
            # This is simplified - in practice, need to properly separate modalities
            # For now, use background data structure
            batch = {
                'time_series_full': self.background_samples['time_series_full'][:batch_size].to(self.device),
                'phase_markers': self.background_samples['phase_markers'][:batch_size].to(self.device),
                'mask_full': self.background_samples['mask_full'][:batch_size].to(self.device),
                'text_combined': self.background_samples['text_combined'][:batch_size].to(self.device),
                'static': torch.FloatTensor(inputs).to(self.device)
            }
            
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # Return predictions for all tasks
            preds = []
            for task in sorted(COMPLICATIONS.keys()):
                preds.append(outputs['predictions'][task].cpu().numpy())
            
            return np.stack(preds, axis=1)
        
        # Use DeepExplainer for neural networks
        if self.config['explainer_type'] == 'deep':
            background_static = self.background_samples['static'].numpy()
            explainer = shap.DeepExplainer(
                model_wrapper,
                background_static
            )
        elif self.config['explainer_type'] == 'kernel':
            background_static = self.background_samples['static'].numpy()
            explainer = shap.KernelExplainer(
                model_wrapper,
                background_static
            )
        else:
            # GradientExplainer
            explainer = shap.GradientExplainer(
                model_wrapper,
                self.background_samples['static'].numpy()
            )
        
        return explainer
    
    def explain_predictions(self, 
                          test_samples: Dict[str, torch.Tensor],
                          feature_names: List[str] = None) -> Dict:
        """
        Generate SHAP explanations for test samples
        
        Args:
            test_samples: Dictionary with test data
            feature_names: Names of features
            
        Returns:
            Dictionary with SHAP values and visualizations
        """
        print("Computing SHAP values...")
        
        # Compute SHAP values for static features
        static_data = test_samples['static'].numpy()
        
        shap_values = self.explainer.shap_values(static_data)
        
        # shap_values is list of arrays, one per task
        # Each array is [n_samples, n_features]
        
        results = {
            'shap_values': shap_values,
            'base_values': self.explainer.expected_value,
            'data': static_data,
            'feature_names': feature_names
        }
        
        # Generate visualizations
        if self.config.get('save_plots', True):
            self._create_summary_plot(results)
            self._create_waterfall_plots(results)
            self._create_dependence_plots(results)
        
        return results
    
    def _create_summary_plot(self, results: Dict):
        """Create SHAP summary plot"""
        print("Creating SHAP summary plots...")
        
        for task_idx, task_name in enumerate(sorted(COMPLICATIONS.keys())):
            plt.figure(figsize=(12, 8))
            
            shap.summary_plot(
                results['shap_values'][task_idx],
                results['data'],
                feature_names=results['feature_names'],
                show=False,
                plot_type='dot'
            )
            
            plt.title(f'SHAP Summary - {COMPLICATIONS[task_name]["name"]}', fontsize=14, pad=20)
            plt.tight_layout()
            
            save_path = self.figures_dir / f'shap_summary_{task_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {save_path}")
        
        # Combined summary plot (mean across tasks)
        plt.figure(figsize=(14, 10))
        
        mean_shap_values = np.mean([results['shap_values'][i] for i in range(len(COMPLICATIONS))], axis=0)
        
        shap.summary_plot(
            mean_shap_values,
            results['data'],
            feature_names=results['feature_names'],
            show=False,
            plot_type='bar'
        )
        
        plt.title('SHAP Summary - Mean Across All Complications', fontsize=14, pad=20)
        plt.tight_layout()
        
        save_path = self.figures_dir / 'shap_summary_combined.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def _create_waterfall_plots(self, results: Dict, n_samples: int = 5):
        """Create waterfall plots for individual predictions"""
        print("Creating SHAP waterfall plots...")
        
        for sample_idx in range(min(n_samples, results['data'].shape[0])):
            for task_idx, task_name in enumerate(sorted(COMPLICATIONS.keys())):
                
                # Create waterfall plot
                plt.figure(figsize=(10, 6))
                
                shap_values_sample = results['shap_values'][task_idx][sample_idx]
                base_value = results['base_values'][task_idx]
                
                # Create explanation object
                explanation = shap.Explanation(
                    values=shap_values_sample,
                    base_values=base_value,
                    data=results['data'][sample_idx],
                    feature_names=results['feature_names']
                )
                
                shap.plots.waterfall(explanation, show=False)
                
                plt.title(f'SHAP Waterfall - {COMPLICATIONS[task_name]["name"]} (Sample {sample_idx+1})', 
                         fontsize=12)
                plt.tight_layout()
                
                save_path = self.figures_dir / f'waterfall_{task_name}_sample{sample_idx+1}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"  Saved waterfall plots for {n_samples} samples")
    
    def _create_dependence_plots(self, results: Dict, top_k: int = 5):
        """Create SHAP dependence plots for top features"""
        print("Creating SHAP dependence plots...")
        
        for task_idx, task_name in enumerate(sorted(COMPLICATIONS.keys())):
            
            # Get feature importance
            mean_abs_shap = np.abs(results['shap_values'][task_idx]).mean(axis=0)
            top_features = np.argsort(mean_abs_shap)[-top_k:][::-1]
            
            # Create dependence plots for top features
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, feature_idx in enumerate(top_features):
                if i >= 6:
                    break
                
                ax = axes[i]
                
                # Dependence plot
                shap.dependence_plot(
                    feature_idx,
                    results['shap_values'][task_idx],
                    results['data'],
                    feature_names=results['feature_names'],
                    ax=ax,
                    show=False
                )
                
                ax.set_title(f'Feature: {results["feature_names"][feature_idx]}', fontsize=10)
            
            # Hide unused subplots
            for i in range(len(top_features), 6):
                axes[i].axis('off')
            
            plt.suptitle(f'SHAP Dependence Plots - {COMPLICATIONS[task_name]["name"]}', 
                        fontsize=14, y=1.00)
            plt.tight_layout()
            
            save_path = self.figures_dir / f'dependence_{task_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {save_path}")