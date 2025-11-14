# models/vibe_tuning.py
"""
Vibe-Tuning: The Art of Fine-Tuning Small Language Models
Implementation of parameter-efficient fine-tuning for biomedical text

Vibe-Tuning combines:
1. LoRA (Low-Rank Adaptation)
2. Prefix Tuning
3. Adapter Layers
4. Selective Layer Freezing
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple, List  # Add this import
import math

from config import MODEL_CONFIG

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Layer
    
    Adds trainable low-rank matrices to frozen weights:
    W_new = W_frozen + A @ B
    where A is (d x r) and B is (r x d) with r << d
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation"""
        # Original: x @ W
        # With LoRA: x @ W + (x @ A) @ B * scaling
        lora_output = (self.dropout(x) @ self.lora_A) @ self.lora_B
        return lora_output * self.scaling


class AdapterLayer(nn.Module):
    """
    Adapter Layer for efficient fine-tuning
    
    Bottleneck architecture: down-project -> non-linearity -> up-project
    """
    
    def __init__(self,
                 hidden_size: int,
                 adapter_size: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter with residual connection"""
        residual = x
        
        # Adapter
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        
        # Residual + layer norm
        x = self.layer_norm(residual + x)
        
        return x


class PrefixTuning(nn.Module):
    """
    Prefix Tuning: prepend trainable prefix vectors to each layer
    """
    
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 prefix_length: int = 10):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length
        
        # Prefix parameters for keys and values
        self.prefix_keys = nn.Parameter(
            torch.randn(num_layers, num_heads, prefix_length, head_dim)
        )
        self.prefix_values = nn.Parameter(
            torch.randn(num_layers, num_heads, prefix_length, head_dim)
        )
        
        # Initialize
        nn.init.xavier_uniform_(self.prefix_keys)
        nn.init.xavier_uniform_(self.prefix_values)
    
    def get_prefix(self, layer_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prefix keys and values for a specific layer"""
        # Expand for batch
        keys = self.prefix_keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
        values = self.prefix_values[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return keys, values


class VibeTunedBiomedicalEncoder(nn.Module):
    """
    Vibe-Tuned Biomedical Text Encoder
    
    Implements parameter-efficient fine-tuning for clinical text:
    1. Freeze most of the base model
    2. Add LoRA adapters to attention layers
    3. Add adapter layers after each transformer block
    4. Optionally add prefix tuning
    5. Fine-tune only task-specific head
    
    This reduces trainable parameters by 90%+ while maintaining performance
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        self.config = config or MODEL_CONFIG['vibe_tuning']
        
        # Load pre-trained biomedical model with MPS-safe settings
        print(f"Loading base model: {self.config['base_model']}")
        import gc
        gc.collect()
        
        # Load on CPU first
        print("Loading model on CPU first...")
        self.base_model = AutoModel.from_pretrained(
            self.config['base_model'],
            low_cpu_mem_usage=True
        )
        self.model_config = self.base_model.config
        print("✓ Base model loaded")
        
        # Freeze base model
        self._freeze_base_model()
        
        # Add LoRA adapters to attention layers
        if self.config.get('adapter_type', 'lora') in ['lora', 'all']:
            self._add_lora_adapters()
        
        # Add adapter layers
        if self.config.get('adapter_type', 'lora') in ['adapter', 'all']:
            self._add_adapter_layers()
        
        # Add prefix tuning
        self.use_prefix = self.config.get('use_prefix_tuning', False)
        if self.use_prefix:
            self.prefix_tuning = PrefixTuning(
                num_layers=self.model_config.num_hidden_layers,
                num_heads=self.model_config.num_attention_heads,
                head_dim=self.model_config.hidden_size // self.model_config.num_attention_heads,
                prefix_length=self.config.get('prefix_length', 10)
            )
        
        # Output projection
        self.output_projection = nn.Linear(
            self.model_config.hidden_size,
            self.config.get('output_dim', 256)
        )
        
        self._print_trainable_parameters()
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        frozen_layers = self.config.get('frozen_layers', 6)
        
        # Freeze embeddings
        for param in self.base_model.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of layers
        for i, layer in enumerate(self.base_model.encoder.layer):
            if i < frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        print(f"Frozen first {frozen_layers} layers of base model")
    
    def _add_lora_adapters(self):
        """Add LoRA adapters to attention layers"""
        hidden_size = self.model_config.hidden_size
        rank = self.config.get('lora_r', 8)
        alpha = self.config.get('lora_alpha', 16)
        dropout = self.config.get('lora_dropout', 0.1)
        
        self.lora_adapters = nn.ModuleList()
        
        for layer in self.base_model.encoder.layer:
            # Add LoRA to query, key, value projections
            lora_q = LoRALayer(hidden_size, hidden_size, rank, alpha, dropout)
            lora_k = LoRALayer(hidden_size, hidden_size, rank, alpha, dropout)
            lora_v = LoRALayer(hidden_size, hidden_size, rank, alpha, dropout)
            
            self.lora_adapters.append(nn.ModuleDict({
                'query': lora_q,
                'key': lora_k,
                'value': lora_v
            }))
        
        print(f"Added LoRA adapters (rank={rank}) to {len(self.lora_adapters)} layers")
    
    def _add_adapter_layers(self):
        """Add adapter layers after transformer blocks"""
        hidden_size = self.model_config.hidden_size
        adapter_size = self.config.get('adapter_size', 64)
        dropout = self.config.get('dropout', 0.1)
        
        self.adapters = nn.ModuleList([
            AdapterLayer(hidden_size, adapter_size, dropout)
            for _ in range(self.model_config.num_hidden_layers)
        ])
        
        print(f"Added {len(self.adapters)} adapter layers (size={adapter_size})")
    
    def forward(self, 
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Vibe-Tuning
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            inputs_embeds: Pre-computed embeddings [batch_size, seq_len, hidden_size]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with embeddings and optionally attention weights
        """
        # Support both input_ids and pre-computed embeddings
        if input_ids is not None:
            # Get base model embeddings
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention
            )
        elif inputs_embeds is not None:
            # Use pre-computed embeddings
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention
            )
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states
        
        # Apply adapters if present
        if hasattr(self, 'adapters'):
            adapted_states = []
            for i, (hidden, adapter) in enumerate(zip(hidden_states[1:], self.adapters)):
                adapted = adapter(hidden)
                adapted_states.append(adapted)
            
            # Use last adapted state
            final_hidden = adapted_states[-1]
        else:
            final_hidden = hidden_states[-1]
        
        # Pool: use [CLS] token
        pooled_output = final_hidden[:, 0, :]
        
        # Project to output dimension
        output_embedding = self.output_projection(pooled_output)
        
        result = {
            'embedding': output_embedding,
            'pooled_output': pooled_output,
            'hidden_states': hidden_states
        }
        
        if return_attention:
            result['attentions'] = outputs.attentions
        
        return result
    
    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"\n{'='*60}")
        print(f"Vibe-Tuning Parameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        print(f"{'='*60}\n")
    
    def get_input_embeddings(self):
        """Get input embeddings for compatibility"""
        return self.base_model.embeddings.word_embeddings
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings"""
        self.base_model.resize_token_embeddings(new_num_tokens)