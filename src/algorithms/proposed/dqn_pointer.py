"""Attention-based Pointer Network for VRPTW (Stage 1)."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
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
            # Mask should be (B, S) -> unsqueeze to (B, 1, 1, S) for broadcast
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int = 5, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3):
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


class PointerDecoder(nn.Module):
    """Bahdanau-style Pointer Decoder."""
    def __init__(self, state_dim: int = 3, embed_dim: int = 128):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)
    
    def forward(self, state: torch.Tensor, nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # state: (B, state_dim)
        # nodes: (B, N, embed_dim)
        # mask: (B, N)
        query = self.state_proj(state).unsqueeze(1) # (B, 1, embed_dim)
        
        # Attention scores (B, N, 1) -> (B, N)
        scores = self.v(torch.tanh(self.W_q(query) + self.W_k(nodes))).squeeze(-1)
        
        # Pointer mechanism: mask invalid actions
        return scores.masked_fill(mask, float('-inf'))


class VRPTWPointerNetwork(nn.Module):
    def __init__(self, input_dim: int = 5, state_dim: int = 3, embed_dim: int = 128, 
                 num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.encoder = GraphEncoder(input_dim, embed_dim, num_heads, num_layers)
        self.decoder = PointerDecoder(state_dim, embed_dim)
        
    def forward(self, features: torch.Tensor, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: (B, N, input_dim)
        # state: (B, state_dim)
        # mask: (B, N)
        embeddings = self.encoder(features, mask)
        return self.decoder(state, embeddings, mask)
