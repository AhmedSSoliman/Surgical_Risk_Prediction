# models/model.py
"""
Complete Multimodal Surgical Risk Prediction Model
Combines time series, text, static features with cross-attention fusion
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional  # Add all type hints
import math

from config import MODEL_CONFIG, COMPLICATIONS
from .simple_text_encoder import SimpleTextEncoder

# Optionally try to import VibeTunedBiomedicalEncoder
try:
    from .vibe_tuning import VibeTunedBiomedicalEncoder
    VIBE_AVAILABLE = True
except Exception:
    VIBE_AVAILABLE = False
    print("WARNING: VibeTunedBiomedicalEncoder not available, using SimpleTextEncoder")



class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding"""
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesEncoder(nn.Module):
    """
    Transformer-based time series encoder
    
    Handles temporal sequences of labs and vitals with:
    - Multi-head self-attention
    - Positional encoding
    - Phase awareness (preop vs intraop)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.hidden_size = config['hidden_size']
        
        # Input projection
        self.input_projection = nn.Linear(
            config.get('input_size', 1),
            self.hidden_size
        )
        
        # Positional encoding
        if config.get('use_positional_encoding', True):
            self.pos_encoder = PositionalEncoding(self.hidden_size)
        else:
            self.pos_encoder = None
        
        # Phase embedding
        self.phase_embedding = nn.Embedding(2, self.hidden_size)  # 0=preop, 1=intraop
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config['num_heads'],
            dim_feedforward=self.hidden_size * 4,
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers']
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, 
                x: torch.Tensor,
                phase_markers: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Time series [batch_size, seq_len, num_features]
            phase_markers: Phase markers [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with encoded representations
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.pos_encoder:
            x = self.pos_encoder(x)
        
        # Add phase embedding
        if phase_markers is not None:
            phase_emb = self.phase_embedding(phase_markers.long())
            x = x + phase_emb
        
        # Create mask for transformer
        if attention_mask is not None:
            # Transformer expects True for positions to attend to
            src_key_padding_mask = (attention_mask == 0)
            # Safety check: if all positions are masked (all zeros), create dummy output
            if attention_mask.sum() == 0:
                # Return zeros with correct shape in dictionary format
                batch_size = x.size(0)
                seq_len = x.size(1)
                zero_sequence = torch.zeros(batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype)
                zero_pooled = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
                return {
                    'sequence_output': zero_sequence,
                    'pooled_output': zero_pooled,
                    'attention_mask': attention_mask
                }
        else:
            src_key_padding_mask = None
        
        # Apply transformer
        encoded = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Layer norm
        encoded = self.layer_norm(encoded)
        
        # Pooling: mean over valid positions
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            sum_encoded = (encoded * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_encoded / count
        else:
            pooled = encoded.mean(dim=1)
        
        return {
            'sequence_output': encoded,
            'pooled_output': pooled,
            'attention_mask': attention_mask
        }


class CrossModalAttention(nn.Module):
    """
    Cross-attention fusion between modalities
    
    Allows text to attend to time series and vice versa
    """
    
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                query: torch.Tensor,
                key_value: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Cross attention
        
        Args:
            query: Query tensor [batch_size, query_len, hidden_size]
            key_value: Key/Value tensor [batch_size, kv_len, hidden_size]
            attention_mask: Mask [batch_size, kv_len]
        """
        # Cross attention
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        attended, _ = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + norm
        output = self.layer_norm(query + self.dropout(attended))
        
        return output


class MultiTaskPredictionHead(nn.Module):
    """
    Multi-task prediction head with uncertainty estimation
    
    Predicts all 9 complications with:
    - Shared layers
    - Task-specific layers
    - Monte Carlo dropout for uncertainty
    """
    
    def __init__(self, 
                 input_size: int,
                 config: Dict):
        super().__init__()
        
        self.config = config
        self.num_tasks = len(COMPLICATIONS)
        
        # Shared layers
        shared_layers = []
        prev_size = input_size
        
        for hidden_size in config['shared_layers']:
            linear = nn.Linear(prev_size, hidden_size)
            # Better initialization for stability
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(linear.bias)
            
            shared_layers.extend([
                linear,
                nn.LayerNorm(hidden_size) if config.get('use_batch_norm', True) else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(config['dropout'])
            ])
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for comp_name in sorted(COMPLICATIONS.keys()):
            task_layers = []
            task_prev_size = prev_size
            
            for hidden_size in config['task_specific_layers']:
                linear = nn.Linear(task_prev_size, hidden_size)
                nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(linear.bias)
                
                task_layers.extend([
                    linear,
                    nn.ReLU(),
                    nn.Dropout(config['dropout'])
                ])
                task_prev_size = hidden_size
            
            # Final prediction layer with very small initialization
            final_layer = nn.Linear(task_prev_size, 1)
            nn.init.xavier_uniform_(final_layer.weight, gain=0.001)  # Very small gain
            nn.init.constant_(final_layer.bias, 0.0)  # Neutral bias
            task_layers.append(final_layer)
            
            self.task_heads[comp_name] = nn.Sequential(*task_layers)
        
        # Uncertainty estimation
        self.uncertainty_estimation = config.get('uncertainty_estimation', True)
        self.mc_dropout = config.get('monte_carlo_dropout', True)
        self.mc_samples = config.get('mc_samples', 10)
    
    def forward(self, x: torch.Tensor, compute_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, input_size]
            compute_uncertainty: Whether to compute uncertainty via MC dropout
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        # Validate and clean input
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Shared representation
        shared_features = self.shared(x)
        
        # Validate shared features
        if torch.isnan(shared_features).any() or torch.isinf(shared_features).any():
            shared_features = torch.nan_to_num(shared_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Task-specific predictions
        predictions = {}
        
        for comp_name, head in self.task_heads.items():
            logit = head(shared_features).squeeze(-1)
            
            # Clamp logits to reasonable range to prevent numerical issues
            logit = torch.clamp(logit, min=-20.0, max=20.0)
            
            # Handle potential NaN or inf values in logits
            if torch.isnan(logit).any() or torch.isinf(logit).any():
                logit = torch.nan_to_num(logit, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Apply sigmoid to get probabilities
            pred = torch.sigmoid(logit)
            
            # Ensure predictions are strictly in valid range [0, 1]
            pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
            
            # Final validation
            if torch.isnan(pred).any() or (pred < 0).any() or (pred > 1).any():
                print(f"Warning: Invalid predictions for {comp_name}, forcing to valid range")
                pred = torch.clamp(torch.nan_to_num(pred, nan=0.5), min=1e-7, max=1-1e-7)
            
            predictions[comp_name] = pred
        
        # Uncertainty estimation via MC Dropout
        if compute_uncertainty and self.uncertainty_estimation and self.mc_dropout:
            uncertainties = self._compute_uncertainty(shared_features)
        else:
            uncertainties = {comp: torch.zeros_like(pred) for comp, pred in predictions.items()}
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'shared_features': shared_features
        }
    
    def _compute_uncertainty(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute uncertainty via Monte Carlo Dropout"""
        self.train()  # Enable dropout
        
        uncertainties = {}
        
        with torch.no_grad():
            for comp_name, head in self.task_heads.items():
                # Multiple forward passes with dropout
                samples = []
                for _ in range(self.mc_samples):
                    logit = head(shared_features).squeeze(-1)
                    pred = torch.sigmoid(logit)
                    samples.append(pred)
                
                # Stack samples
                samples = torch.stack(samples, dim=0)  # [mc_samples, batch_size]
                
                # Uncertainty = standard deviation
                uncertainty = samples.std(dim=0)
                uncertainties[comp_name] = uncertainty
        
        self.eval()  # Back to eval mode
        
        return uncertainties


class MultimodalSurgicalRiskModel(nn.Module):
    """
    Complete Multimodal Surgical Risk Prediction Model
    
    Architecture:
    1. Time Series Encoder (Transformer)
    2. Text Encoder (Vibe-Tuned Biomedical BERT)
    3. Static Feature Encoder
    4. Cross-Modal Attention Fusion
    5. Multi-Task Prediction Heads (9 complications)
    
    Features:
    - Parameter-efficient fine-tuning (Vibe-Tuning)
    - Cross-attention between modalities
    - Uncertainty estimation
    - Multi-task learning with shared representations
    """
    
    def __init__(self, 
                 ts_input_size: int,
                 static_input_size: int,
                 text_input_size: int = None,
                 config: Dict = None):
        super().__init__()
        
        self.config = config or MODEL_CONFIG
        
        # Time series encoder
        ts_config = self.config['time_series'].copy()
        ts_config['input_size'] = ts_input_size
        self.time_series_encoder = TimeSeriesEncoder(ts_config)
        
        # Text encoder (use simple encoder if Vibe-Tuning causes issues)
        use_simple_encoder = self.config.get('use_simple_text_encoder', True)
        if use_simple_encoder or not VIBE_AVAILABLE:
            print("Using SimpleTextEncoder (stable, MPS-compatible)")
            self.text_encoder = SimpleTextEncoder(self.config.get('vibe_tuning', {}))
        else:
            print("Using VibeTunedBiomedicalEncoder (may cause stability issues on MPS)")
            self.text_encoder = VibeTunedBiomedicalEncoder(self.config['vibe_tuning'])
        
        # Determine text input dimension
        if text_input_size is None:
            text_input_size = self.config.get('vibe_tuning', {}).get('output_dim', 256)
        self.text_input_size = text_input_size
        
        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Cross-modal fusion
        fusion_hidden = self.config['fusion']['hidden_sizes'][0]
        
        # Text attends to time series
        self.text_to_ts_attention = CrossModalAttention(
            fusion_hidden,
            self.config['fusion']['num_attention_heads'],
            self.config['fusion']['attention_dropout']
        )
        
        # Time series attends to text
        self.ts_to_text_attention = CrossModalAttention(
            fusion_hidden,
            self.config['fusion']['num_attention_heads'],
            self.config['fusion']['attention_dropout']
        )
        
        # Projection layers to common dimension
        self.ts_projection = nn.Linear(
            self.config['time_series']['hidden_size'],
            fusion_hidden
        )
        
        self.text_projection = nn.Linear(
            self.text_input_size,
            fusion_hidden
        )
        
        self.static_projection = nn.Linear(256, fusion_hidden)
        
        # Fusion layers
        fusion_layers = []
        prev_size = fusion_hidden * 3  # concat ts, text, static
        
        for hidden_size in self.config['fusion']['hidden_sizes']:
            fusion_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size) if self.config['fusion'].get('use_layer_norm', True) else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.config['fusion']['attention_dropout'])
            ])
            prev_size = hidden_size
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # Multi-task prediction heads
        self.prediction_heads = MultiTaskPredictionHead(
            input_size=prev_size,
            config=self.config['prediction_heads']
        )
    
    def forward(self,
                time_series: torch.Tensor,
                phase_markers: torch.Tensor,
                ts_attention_mask: torch.Tensor,
                text_embedding: torch.Tensor,
                static_features: torch.Tensor,
                compute_uncertainty: bool = False,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            time_series: [batch_size, seq_len, ts_features]
            phase_markers: [batch_size, seq_len]
            ts_attention_mask: [batch_size, seq_len]
            text_embedding: [batch_size, text_dim]
            static_features: [batch_size, static_dim]
            compute_uncertainty: Whether to compute prediction uncertainty
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions, uncertainties, and optionally attention weights
        """
        # Encode time series
        ts_encoded = self.time_series_encoder(
            time_series,
            phase_markers,
            ts_attention_mask
        )
        ts_pooled = ts_encoded['pooled_output']  # [batch_size, hidden_size]
        ts_sequence = ts_encoded['sequence_output']  # [batch_size, seq_len, hidden_size]
        
        # Encode static features
        static_encoded = self.static_encoder(static_features)  # [batch_size, 256]
        
        # Project to common dimension
        ts_proj = self.ts_projection(ts_pooled)  # [batch_size, fusion_hidden]
        text_proj = self.text_projection(text_embedding)  # [batch_size, fusion_hidden]
        static_proj = self.static_projection(static_encoded)  # [batch_size, fusion_hidden]
        
        # Cross-modal attention
        # Add sequence dimension for attention
        text_seq = text_proj.unsqueeze(1)  # [batch_size, 1, fusion_hidden]
        ts_seq_proj = self.ts_projection(ts_sequence)  # [batch_size, seq_len, fusion_hidden]
        
        # Text attends to time series
        text_attended = self.text_to_ts_attention(
            query=text_seq,
            key_value=ts_seq_proj,
            attention_mask=ts_attention_mask
        ).squeeze(1)  # [batch_size, fusion_hidden]
        
        # Time series attends to text
        ts_attended = self.ts_to_text_attention(
            query=ts_seq_proj,
            key_value=text_seq,
            attention_mask=None
        )  # [batch_size, seq_len, fusion_hidden]
        
        # Pool time series attended
        if ts_attention_mask is not None:
            mask_expanded = ts_attention_mask.unsqueeze(-1).expand_as(ts_attended)
            ts_attended_pooled = (ts_attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            ts_attended_pooled = ts_attended.mean(dim=1)
        
        # Concatenate all modalities
        fused = torch.cat([
            ts_attended_pooled,
            text_attended,
            static_proj
        ], dim=-1)  # [batch_size, fusion_hidden * 3]
        
        # Apply fusion network
        fused_features = self.fusion_network(fused)  # [batch_size, final_hidden]
        
        # Multi-task predictions
        outputs = self.prediction_heads(fused_features, compute_uncertainty)
        
        # Add intermediate representations
        outputs['ts_encoded'] = ts_pooled
        outputs['text_encoded'] = text_proj
        outputs['static_encoded'] = static_proj
        outputs['fused_features'] = fused_features
        
        return outputs
    
    def predict(self, batch: Dict[str, torch.Tensor], compute_uncertainty: bool = True) -> Dict:
        """
        Convenience method for prediction
        
        Args:
            batch: Batch dictionary from dataloader
            compute_uncertainty: Whether to estimate uncertainty
            
        Returns:
            Predictions with uncertainties
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                time_series=batch['time_series_full'],
                phase_markers=batch['phase_markers'],
                ts_attention_mask=batch['mask_full'],
                text_embedding=batch['text_combined'],
                static_features=batch['static'],
                compute_uncertainty=compute_uncertainty
            )
        
        return outputs
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }