import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)




class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalAttention(nn.Module):
    def init(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb, config.bias)
        self.c_proj = nn.Linear(config.n_emb, config.n_emb, config.bias)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C= x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        attn = q @ k.transpose(-1,-2) * (1/np.sqrt(k.size(-1)))  # (B, nh, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = attn 
