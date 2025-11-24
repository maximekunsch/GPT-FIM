from Model import GPTConfig, GPT
import torch
import tiktoken
import random
import numpy as np

from logging_config import logger

import wandb

# Initialize a new run
wandb.init(
    project="gpt-training",  # your project name
    config={
        "batch_size": 1,
        "block_size": 256,
        "learning_rate": 8e-5,
        "n_layer": 9,
        "n_head": 16,
        "n_embd": 2048,
        "dropout": 0
    }
)

config = wandb.config



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
        
        enc = tiktoken.get_encoding('cl100k_base')
        vocab_size = enc.max_token_value + 1
        print(vocab_size)  
        
        # Add FIM tokens if not in vocabulary
        allowed = {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endoftext|>"}
        
        self.fim_prefix_id = enc.encode("<|fim_prefix|>", allowed_special=allowed)[0]
        self.fim_middle_id = enc.encode("<|fim_middle|>", allowed_special=allowed)[0]
        self.fim_suffix_id = enc.encode("<|fim_suffix|>", allowed_special=allowed)[0]
        self.end_id = enc.encode("<|endoftext|>", allowed_special=allowed)[0]
        
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
        
        split1 = random.randint(1, total_len - 2) # Where Prefix ends
        split2 = random.randint(split1 + 1, total_len - 1) #Where Suffix starts 
        
        prefix = buf[:split1]
        middle = buf[split1:split2]
        suffix = buf[split2:]
        
        # SPM format
        x_seq = torch.cat([
            torch.tensor([self.fim_suffix_id]),
            suffix,
            torch.tensor([self.fim_prefix_id]),
            prefix,
            torch.tensor([self.fim_middle_id]),
            middle,
            torch.tensor([self.end_id]),
        ])
        
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

texte = DataLoaderFIM(B=1, T=256)

config = GPTConfig(
    block_size=256,
    sliding_window=32,
    vocab_size=100277,
    n_layer=9,
    n_head=16,
    n_embd=2048,
    softcap=20,
    dropout=0,
    bias=False
)
model = GPT(config)
model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr= 8e-5)

for i in range(500):
    x, y = texte.next_batch()
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optim.step()
    logger.info(f"Step {i} loss: {loss.item()}")
    wandb.log({"step": i, "loss": loss.item()})

torch.save(model.state_dict(), "gpt_model.pt")
# wandb.save("gpt_model.pt")

model.eval()
max_iter = 45
trials = 5
enc = tiktoken.get_encoding('gpt2')
allowed = {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endoftext|>"}
x = "<|fim_suffix|> return c <|fim_prefix|> def sum(a, b): <|fim_middle|> c ="
tokens = enc.encode(x, allowed_special=allowed)
tokens = torch.tensor(tokens, dtype= torch.long)
tokens = tokens.unsqueeze(0).repeat(trials, 1) #(trials, len(tokens))

x = tokens.to(device)

# Generate
y = model.generate(x, max_iter, temperature=0.3, top_k=5)

# Decode each generated sequence
for i in range(trials):
    generated_tokens = y[i].tolist() 
    output = enc.decode(generated_tokens)
    logger.info(f"Trial {i+1}:\n{output}\n")
