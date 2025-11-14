# explainability/attention_viz.py
"""
Attention visualization for transformer-based models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from config import EXPLAINABILITY_CONFIG, FIGURES_DIR

class AttentionVisualizer:
    """
    Visualize attention weights from transformer models
    
    Features:
    - Attention heatmaps
    - Multi-head attention visualization
    - Cross-modal attention analysis
    - Temporal attention patterns
    """
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.config = EXPLAINABILITY_CONFIG['attention']
        self.figures_dir = Path(FIGURES_DIR) / 'attention'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_attention(self,
                          batch: Dict[str, torch.Tensor],
                          sample_idx: int = 0) -> Dict:
        """
        Extract and visualize attention weights
        
        Args:
            batch: Input batch
            sample_idx: Index of sample to visualize
            
        Returns:
            Dictionary with attention weights and plots
        """
        self.model.eval()
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass with attention extraction
        with torch.no_grad():
            outputs = self.model(
                time_series=batch['time_series_full'],
                phase_markers=batch['phase_markers'],
                ts_attention_mask=batch['mask_full'],
                text_embedding=batch['text_combined'],
                static_features=batch['static'],
                return_attention=True
            )
        
        # Extract attention weights
        # This depends on model architecture
        attention_weights = self._extract_attention_weights(outputs)
        
        # Visualize
        if self.config.get('save_heatmaps', True):
            self._plot_attention_heatmaps(attention_weights, sample_idx)
            self._plot_temporal_attention(attention_weights, sample_idx)
        
        return {
            'attention_weights': attention_weights,
            'sample_idx': sample_idx
        }
    
    def _extract_attention_weights(self, outputs: Dict) -> Dict:
        """Extract attention weights from model outputs"""
        # This is model-specific
        # For transformers, attention weights are typically in outputs
        
        attention = {
            'time_series': None,
            'cross_modal': None
        }
        
        # Extract if available
        if 'attentions' in outputs:
            attention['time_series'] = outputs['attentions']
        
        return attention
    
    def _plot_attention_heatmaps(self, 
                                attention_weights: Dict,
                                sample_idx: int):
        """Plot attention heatmaps"""
        print("Creating attention heatmaps...")
        
        if attention_weights['time_series'] is not None:
            # Plot time series self-attention
            for layer_idx in self.config['visualize_layers']:
                if layer_idx < len(attention_weights['time_series']):
                    
                    # Get attention for this layer
                    layer_attention = attention_weights['time_series'][layer_idx][sample_idx]
                    
                    # Average across heads if specified
                    if self.config['head_reduction'] == 'mean':
                        layer_attention = layer_attention.mean(dim=0)
                    elif self.config['head_reduction'] == 'max':
                        layer_attention = layer_attention.max(dim=0)[0]
                    
                    # Convert to numpy
                    attn_matrix = layer_attention.cpu().numpy()
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    sns.heatmap(
                        attn_matrix,
                        cmap='viridis',
                        ax=ax,
                        cbar_kws={'label': 'Attention Weight'}
                    )
                    
                    ax.set_title(f'Self-Attention Heatmap - Layer {layer_idx} (Sample {sample_idx})', 
                               fontsize=14)
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
                    
                    plt.tight_layout()
                    
                    save_path = self.figures_dir / f'attention_layer{layer_idx}_sample{sample_idx}.png'
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Saved: {save_path}")
    
    def _plot_temporal_attention(self,
                                attention_weights: Dict,
                                sample_idx: int):
        """Plot temporal attention patterns"""
        print("Creating temporal attention plots...")
        
        if attention_weights['time_series'] is not None:
            # Average attention across all layers and heads
            all_attention = torch.stack([
                attn[sample_idx].mean(dim=0) 
                for attn in attention_weights['time_series']
            ])
            
            # Average across layers
            mean_attention = all_attention.mean(dim=0).cpu().numpy()
            
            # Plot attention over time
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot each query position
            for i in range(0, mean_attention.shape[0], 5):  # Plot every 5th position
                ax.plot(mean_attention[i], alpha=0.3, linewidth=0.5)
            
            # Plot mean
            ax.plot(mean_attention.mean(axis=0), 
                   color='red', linewidth=2, label='Mean Attention')
            
            ax.set_xlabel('Key Position (Time Steps)', fontsize=12)
            ax.set_ylabel('Attention Weight', fontsize=12)
            ax.set_title(f'Temporal Attention Pattern (Sample {sample_idx})', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = self.figures_dir / f'temporal_attention_sample{sample_idx}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {save_path}")
    
    def visualize_cross_modal_attention(self,
                                      batch: Dict[str, torch.Tensor],
                                      sample_idx: int = 0):
        """Visualize cross-modal attention between text and time series"""
        print("Visualizing cross-modal attention...")
        
        # This requires access to intermediate cross-attention layers
        # Implementation depends on model architecture
        
        # Placeholder for cross-modal attention visualization
        pass