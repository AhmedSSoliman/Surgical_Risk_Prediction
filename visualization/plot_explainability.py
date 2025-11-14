# visualization/plot_explainability.py
"""
Explainability-specific visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

from config import COMPLICATIONS, FIGURES_DIR

class ExplainabilityPlotter:
    """
    Create explainability visualizations
    
    Plots:
    - SHAP summary plots
    - Feature importance rankings
    - Attention patterns
    - Temporal dynamics
    - Risk stratification
    """
    
    def __init__(self, save_dir: Path = None):
        self.save_dir = Path(save_dir or FIGURES_DIR) / 'explainability'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_temporal_dynamics(self,
                             time_series: np.ndarray,
                             predictions: Dict[str, np.ndarray],
                             feature_names: List[str]):
        """Plot temporal dynamics of predictions"""
        print("Creating temporal dynamics plot...")
        
        # Assuming time_series is [seq_len, num_features]
        seq_len = time_series.shape[0]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Time series features
        ax = axes[0]
        time_points = np.arange(seq_len)
        
        # Plot first 5 features
        for i in range(min(5, time_series.shape[1])):
            ax.plot(time_points, time_series[:, i], 
                   label=feature_names[i] if i < len(feature_names) else f'Feature {i}',
                   alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Normalized Value', fontsize=11)
        ax.set_title('Time Series Features Over Time', fontsize=13, pad=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=seq_len//2, color='red', linestyle='--', linewidth=2, 
                  label='Surgery Time', alpha=0.6)
        
        # Plot 2: Risk predictions over time (if dynamic)
        ax = axes[1]
        
        # Plot top 3 risk complications
        task_items = sorted(predictions.items(), 
                          key=lambda x: x[1].mean(), reverse=True)[:3]
        
        for task_name, pred in task_items:
            # If predictions are scalar, create constant line
            if isinstance(pred, (float, np.floating)):
                ax.axhline(y=pred, label=COMPLICATIONS[task_name]['name'], linewidth=2)
            else:
                ax.plot(pred, label=COMPLICATIONS[task_name]['name'], linewidth=2)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Risk Score', fontsize=11)
        ax.set_title('Top 3 Complication Risks', fontsize=13, pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 3: Risk heatmap
        ax = axes[2]
        
        # Create risk matrix
        risk_matrix = []
        for task_name in sorted(COMPLICATIONS.keys()):
            if task_name in predictions:
                pred = predictions[task_name]
                if isinstance(pred, (float, np.floating)):
                    risk_matrix.append([pred] * seq_len)
                else:
                    risk_matrix.append(pred)
        
        risk_matrix = np.array(risk_matrix)
        
        im = ax.imshow(risk_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_yticks(range(len(COMPLICATIONS)))
        ax.set_yticklabels([COMPLICATIONS[t]['name'] for t in sorted(COMPLICATIONS.keys())], 
                          fontsize=9)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_title('Risk Heatmap Across All Complications', fontsize=13, pad=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Risk Score', fontsize=10)
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'temporal_dynamics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_risk_stratification(self,
                                predictions: Dict[str, np.ndarray],
                                targets: Dict[str, np.ndarray]):
        """Plot risk stratification analysis"""
        print("Creating risk stratification plot...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, task_name in enumerate(sorted(COMPLICATIONS.keys())):
            ax = axes[idx]
            
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Create risk bins
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bin_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            
            pred_binned = pd.cut(pred, bins=bins, labels=bin_labels)
            
            # Calculate actual event rate in each bin
            df = pd.DataFrame({
                'predicted_risk': pred_binned,
                'actual_outcome': target
            })
            
            stratification = df.groupby('predicted_risk')['actual_outcome'].agg(['mean', 'count'])
            stratification['se'] = np.sqrt(stratification['mean'] * (1 - stratification['mean']) / stratification['count'])
            
            # Plot
            x_pos = np.arange(len(stratification))
            ax.bar(x_pos, stratification['mean'], 
                  yerr=stratification['se'], 
                  alpha=0.7, capsize=5, color='steelblue', edgecolor='black')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(stratification.index, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Actual Event Rate', fontsize=10)
            ax.set_title(COMPLICATIONS[task_name]['name'], fontsize=11, pad=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1])
            
            # Add count annotations
            for i, (idx_val, row) in enumerate(stratification.iterrows()):
                ax.text(i, row['mean'] + row['se'] + 0.05, 
                       f"n={int(row['count'])}", 
                       ha='center', fontsize=8)
        
        plt.suptitle('Risk Stratification: Predicted vs Actual Event Rates', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = self.save_dir / 'risk_stratification.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_feature_contribution_timeline(self,
                                         time_series: np.ndarray,
                                         feature_names: List[str],
                                         shap_values: np.ndarray = None):
        """Plot feature contributions over time"""
        print("Creating feature contribution timeline...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Feature values over time
        ax = axes[0]
        
        seq_len = time_series.shape[0]
        time_points = np.arange(seq_len)
        
        # Select top 5 most variable features
        variances = np.var(time_series, axis=0)
        top_features = np.argsort(variances)[-5:][::-1]
        
        for feat_idx in top_features:
            feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature {feat_idx}'
            ax.plot(time_points, time_series[:, feat_idx], 
                   label=feature_name, linewidth=2, alpha=0.8)
        
        # Mark surgery time (middle of sequence)
        ax.axvline(x=seq_len//2, color='red', linestyle='--', 
                  linewidth=2, alpha=0.5, label='Surgery Time')
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Feature Value (Normalized)', fontsize=11)
        ax.set_title('Top 5 Variable Features Over Time', fontsize=13, pad=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: SHAP values over time (if available)
        ax = axes[1]
        
        if shap_values is not None:
            # Plot SHAP values for same features
            for feat_idx in top_features:
                if feat_idx < shap_values.shape[1]:
                    feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature {feat_idx}'
                    ax.plot(time_points, shap_values[:, feat_idx], 
                           label=feature_name, linewidth=2, alpha=0.8)
            
            ax.axvline(x=seq_len//2, color='red', linestyle='--', 
                      linewidth=2, alpha=0.5, label='Surgery Time')
            
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('SHAP Value', fontsize=11)
            ax.set_title('Feature Contributions (SHAP Values) Over Time', fontsize=13, pad=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'SHAP values not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'feature_contribution_timeline.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    def plot_uncertainty_analysis(self,
                                 predictions: Dict[str, np.ndarray],
                                 uncertainties: Dict[str, np.ndarray],
                                 targets: Dict[str, np.ndarray]):
        """Plot uncertainty analysis"""
        print("Creating uncertainty analysis...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, task_name in enumerate(sorted(COMPLICATIONS.keys())):
            ax = axes[idx]
            
            pred = predictions[task_name]
            unc = uncertainties[task_name]
            target = targets[task_name]
            
            # Scatter plot: prediction vs uncertainty, colored by correctness
            correct = (np.round(pred) == target)
            
            ax.scatter(pred[correct], unc[correct], 
                      c='green', alpha=0.5, s=20, label='Correct')
            ax.scatter(pred[~correct], unc[~correct], 
                      c='red', alpha=0.5, s=20, label='Incorrect')
            
            ax.set_xlabel('Predicted Probability', fontsize=10)
            ax.set_ylabel('Uncertainty (Std Dev)', fontsize=10)
            ax.set_title(COMPLICATIONS[task_name]['name'], fontsize=11, pad=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Prediction Uncertainty Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = self.save_dir / 'uncertainty_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")