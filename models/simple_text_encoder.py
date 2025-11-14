# models/simple_text_encoder.py
"""
Simple Text Encoder - Fallback when transformer models cause issues
Uses pre-computed embeddings from preprocessing
"""

import torch
import torch.nn as nn


class SimpleTextEncoder(nn.Module):
    """
    Simple text encoder that processes pre-computed embeddings
    
    This is a lightweight alternative to VibeTunedBiomedicalEncoder
    for when transformer models cause stability issues (segfaults with MPS).
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        input_dim = 768  # Standard BERT embedding size
        output_dim = config.get('output_dim', 256)
        
        # Simple projection network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"✓ Simple text encoder initialized ({input_dim} -> {output_dim})")
        self._print_trainable_parameters()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,}")
    
    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass
        
        Args:
            embeddings: Pre-computed text embeddings [batch_size, embedding_dim]
            attention_mask: Ignored (for API compatibility)
            
        Returns:
            Encoded embeddings [batch_size, output_dim]
        """
        return self.encoder(embeddings)
