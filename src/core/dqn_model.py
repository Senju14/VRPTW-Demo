"""Attention-based DQN model for VRPTW."""

import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model configuration
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dqn_model.safetensors")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape
        
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        
        return self.out_proj(out)


class GraphEncoder(nn.Module):
    """Transformer encoder for graph nodes."""
    
    def __init__(self, input_dim: int = 5, embed_dim: int = 128, 
                 num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadAttention(embed_dim, num_heads),
                'norm1': nn.LayerNorm(embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer['norm1'](x + layer['attn'](x, mask))
            x = layer['norm2'](x + layer['ff'](x))
        
        return x


class AttentionDecoder(nn.Module):
    """Bahdanau attention decoder for action selection."""
    
    def __init__(self, state_dim: int = 3, embed_dim: int = 128):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)
    
    def forward(self, state: torch.Tensor, nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.state_proj(state).unsqueeze(1)
        scores = self.v(torch.tanh(self.W_q(query) + self.W_k(nodes))).squeeze(-1)
        return scores.masked_fill(mask, float('-inf'))


class AttentionDQN(nn.Module):
    """Attention-based Deep Q-Network for VRPTW."""
    
    def __init__(self, input_dim: int = 5, state_dim: int = 3,
                 embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.encoder = GraphEncoder(input_dim, embed_dim, num_heads, num_layers)
        self.decoder = AttentionDecoder(state_dim, embed_dim)
    
    def forward(self, features: torch.Tensor, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        node_embeddings = self.encoder(features, mask)
        return self.decoder(state, node_embeddings, mask)


# Global model cache
_model_cache = None


def load_dqn_model() -> AttentionDQN:
    """Load pretrained DQN model."""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    from safetensors.torch import load_file
    
    model = AttentionDQN(
        input_dim=5, state_dim=3,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    model.load_state_dict(load_file(MODEL_PATH))
    model.eval()
    
    _model_cache = model
    return model


def get_device() -> torch.device:
    """Get compute device."""
    return DEVICE
