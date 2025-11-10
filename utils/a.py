import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class LayerNorm(nn.MModule):
    def __init__(self, ndim, bias):
        super().__init()
        self.weight = nn.Parameters(torch.ones(ndim))
        self.bias = nn.Parameters(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps = 1e-5)

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        
        self.n_embd = config.n_embd
        self.n_head = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim = -1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        
        attn = (q @ k.transpose(-1, -2)) * (1 / np.sqrt(k.size(-1))) #(B, nh, T, T)
        attn = F.softmax(attn, dim = -1) #(B, nh, T, T)
        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        
        return y

class GLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias = config.bias)
        self.glu = nn.SiLU()
        
        self.c_fc2 = nn.Linear(config.n_embd, 4 * config.n_embd, bias = config.bias)
        
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias = config.bias)
    
    def forward(self, x):
        x1 = self.c_fc1(x)
        x1 = self.glu(x1)
        
        x2 = self.c_fc2(x)
        
        x = x1 * x2
        
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias = config.bias)
        self.attn = CausalAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias = config.bias)
        self.mlp = GLU(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init()
        self.config = config
        self.transformer = nn.ModuleDict(dict)(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.0),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias = config.bias)
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self,idx, targets = None):
        B, T = idx.size() # batch size, sequence length
        device = idx.device
        
        pos = torch.arange(0, T, dtype = torch.long, device = device).unsqueeze(0)
        
        tok_emb = self.transformer.wte(idx) #(B, T, C)
        pos_emb = self.transformer.wpe(pos) #(1, T, C)
        x = tok_emb + pos_emb #(B, T, C)
        for block in range(self.transformer.h):
            x = block(x) #(B, T, C)
        x = self.transformer.ln_f(x) #(B, T, C)
        
        if targets is not None:
            logits = self.lm_head(x) #(B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:,[-1], :]) #(B, 1, vocab_size)
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond) # (trials, T, vocab_size)
            pred_probs = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(pred_probs, min(top_k, pred_probs.size(-1)))
                pred_probs[pred_probs < v[:, [-1]]] = -float('Inf')
            pred_probs = F.softmax(pred_probs / temperature, dim = -1)
            next_id = torch.multinomial(pred_probs, num_samples = 1)
            idx = torch.cat((idx, next_id), dim = 1)
        return idx