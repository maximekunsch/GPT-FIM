
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
from torch.nn.attention import and_masks, or_masks

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class TunedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, config.bias)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Fixed: changed 'flex' to match the actual check
        self.flex = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flex:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x, suffix_prefix_length: list=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)    # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)    # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)    # (B, nh, T, hs)
        
        if self.flex:
            # Define score_mod function for softcapping
            def score_mod(score, b, h, q_idx, kv_idx):
                score = score / self.softcap
                score = torch.tanh(score)
                score = score * self.softcap
                return score
            
            # Define sliding window causal mask
            def sliding_window_causal(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                window_mask = q_idx - kv_idx <= self.sliding_window
                return causal_mask & window_mask
            
            # If suffix_prefix_length is provided, create suffix_prefix mask
            if suffix_prefix_length is not None:
                def suffix_prefix_mask(b, h, q_idx, kv_idx):
                    return kv_idx < suffix_prefix_length[b]
                
                # Combine prefix and sliding window causal masks
                combined_mask = or_masks(suffix_prefix_mask, sliding_window_causal)
            else:
                combined_mask = sliding_window_causal
            
            # Create block mask
            block_mask = create_block_mask(combined_mask, B=B, H=None, Q_LEN=T, KV_LEN=T)
            
            # Apply flex attention
            y = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        else:
            # Manual attention computation for non-flex path
            # Save original scores before masking for prefix unmask later
            score_orig = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
            score = score_orig.clone()
            
            # Start with causal + sliding window mask
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            
            # Apply sliding window: only attend within window
            sliding_mask = torch.zeros(T, T, device=x.device, dtype=torch.bool)
            for i in range(T):
                for j in range(T):
                    if i >= j and (i - j) <= self.sliding_window:
                        sliding_mask[i, j] = True
            
            # Combine: causal AND sliding window
            combined_mask = causal_mask & sliding_mask
            
            # If suffix_prefix_length is provided, allow attention to prefix tokens
            if suffix_prefix_length is not None:
                # For each batch element, allow attention to its prefix
                for b in range(B):
                    # All positions can attend to tokens before suffix_prefix_length[b]
                    combined_mask_b = combined_mask.clone()
                    combined_mask_b[:, :suffix_prefix_length[b]] = True
                    score[b, :, :, :] = score_orig[b, :, :, :].masked_fill(~combined_mask_b, float('-inf'))
            else:
                score = score.masked_fill(~combined_mask.view(1, 1, T, T), float('-inf'))
            
            # Apply softcapping
            score = score / self.softcap
            score = torch.tanh(score)
            score = score * self.softcap
            
            # Softmax and attention
            att = F.softmax(score, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        
        return y


class GLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 2 fully connected layer
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # 4 is tunable
        self.c_fc_2  = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # 4 because must be the same size than the first layer
        
        # Activation function
        self.silu    = nn.SiLU()
        
        # Projection layer
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        # One side
        x_1 = self.c_fc(x)
        x_1 = self.silu(x_1)
        
        # Other side
        x_2 = self.c_fc_2(x)
        
        # "Re-assemble" (AVENGERSSSS)
        x = x_1 * x_2
        
        # Project
        x = self.c_proj(x)
        
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TunedSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GLU(config)

    def forward(self, x, suffix_prefix_length: list=None):
        x = x + self.attn(self.ln_1(x), suffix_prefix_length=suffix_prefix_length)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 4096
    sliding_window: int = 1024
    vocab_size: int = 100608 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency and multiplied by 2
    n_layer: int = 32         
    n_head: int = 32
    n_embd: int = 2560
    softcap: int = 20
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
