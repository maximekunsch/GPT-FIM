from Model import GPTConfig, GPT
import torch
import tiktoken
import random
import numpy as np

from logging_config import logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster


class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('python_alpaca_train.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        self.current_pos = 0
        
        logger.info(f"loaded :{len(self.tokens)} tokens")
        logger.info(f"1 epoch :{len(self.tokens) // (B*T)} batches")
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_pos += B * T
        
        if self.current_pos + B*T +1 > len(self.tokens):
            self.current_pos = 0
        return x, y

class DataLoaderFIM:
    def __init__(self, B, T, fim_rate=0.5):
        self.B = B
        self.T = T
        self.fim_rate = fim_rate  # Probability of using FIM
        
        with open('python_alpaca_train.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        
        # Add FIM tokens if not in vocabulary
        self.fim_prefix_id = enc.encode('FIM_PREFIX')[0]
        self.fim_middle_id = enc.encode('FIM_MIDDLE')[0]
        self.fim_suffix_id = enc.encode('FIM_SUFFIX')[0]
        
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_pos = 0
        
        logger.info(f"loaded :{len(self.tokens)}")
        logger.info(f"1 epoch :{len(self.tokens) // (B*T)} batches")
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos + B*T + 1]
        
        # Decide whether to use FIM for this batch
        if random.random() < self.fim_rate:
            # Apply FIM transformation to the whole buffer
            x, y = self._create_fim_single(buf, B, T)
        else:
            # Standard autoregressive training
            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
        
        self.current_pos += B * T
        if self.current_pos + B*T + 1 > len(self.tokens):
            self.current_pos = 0
        
        return x, y
    
    def _create_fim_single(self, buf, B, T):
        # Do ONE FIM transformation on the whole sequence
        # Then reshape to (B, T)
        total_len = len(buf) - 1
        
        split1 = random.randint(1, total_len - 2)
        split2 = random.randint(split1 + 1, total_len - 1)
        
        prefix = buf[:split1]
        middle = buf[split1:split2]
        suffix = buf[split2:-1]
        
        # SPM format
        x_seq = torch.cat([
            torch.tensor([self.fim_suffix_id]),
            suffix,
            torch.tensor([self.fim_prefix_id]),
            prefix,
            torch.tensor([self.fim_middle_id]),
            middle
        ])
        
        # Standard next-token prediction targets
        y_seq = x_seq[1:]
        x_seq = x_seq[:-1]
        
        # Truncate or pad to B*T
        if len(x_seq) > B*T:
            x_seq = x_seq[:B*T]
            y_seq = y_seq[:B*T]
        else:
            pad_len = B*T - len(x_seq)
            x_seq = torch.cat([x_seq, torch.zeros(pad_len, dtype=torch.long)])
            y_seq = torch.cat([y_seq, torch.full((pad_len,), -100)])
        
        return x_seq.view(B, T), y_seq.view(B, T)

texte = DataLoaderFIM(B=4, T=32)

config = GPTConfig(
    block_size=128,
    sliding_window=32,
    vocab_size=50304,  # ← Changed! (50257 rounded up to nearest multiple of 64)
    n_layer=26,
    n_head=32,
    n_embd=1024,
    softcap=20,
    dropout=0,
    bias=False
)
model = GPT(config)
model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr= 2e-4)

for i in range(5000):
    x, y = texte.next_batch()
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optim.step()
    logger.info(f"Step {i} loss: {loss.item()}")

model.eval()
max_iter = 45
trials = 5
enc = tiktoken.get_encoding('gpt2')
x = " Complete this function that sums a and b def sum(a, b):"
tokens = enc.encode(x)
tokens = torch.tensor(tokens, dtype= torch.long)
tokens = tokens.unsqueeze(0).repeat(trials, 1)

x = tokens.to(device)

# Generate - note the correct argument order
y = model.generate(x, max_iter, temperature=1.0, top_k=50)  # ← Fixed order

# Decode each generated sequence
for i in range(trials):
    # Get just the generated tokens (skip the prompt)
    generated_tokens = y[i].tolist()  # ← Convert to list
    output = enc.decode(generated_tokens)
    logger.info(f"Trial {i+1}:\n{output}\n")