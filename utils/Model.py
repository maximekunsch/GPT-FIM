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
        assert config.n_embd % config.n_head == 0
        
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


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, suffix_prefix_length=None, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, suffix_prefix_length=suffix_prefix_length)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        # Crop position embeddings
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        
        # Update all attention modules
        for block in self.transformer.h:
            # Crop mask buffer if it exists (non-flex path)
            if hasattr(block.attn, 'mask'):
                block.attn.mask = block.attn.mask[:,:,:block_size,:block_size]
            
            # Update block_size in attention
            block.attn.block_size = block_size
            
            # Optionally adjust sliding window proportionally
            # if block.attn.sliding_window > block_size:
            #     block.attn.sliding_window = block_size

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 283.8e12
        mfu = flops_achieved / flops_promised
        return mfu