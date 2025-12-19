from Model import GPTConfig, GPT
import torch
import tiktoken
import random
import numpy as np
from datasets import load_dataset
import math

from logging_config import logger

import wandb

# Initialize a new run
wandb.init(
    project="gpt-training", 
    config={
        "batch_size": 1,
        "block_size": 1024,
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

# If classic L2R
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

# Our case
class DataLoaderFIM:
    def __init__(self, B, T, fim_rate=0.5):
        self.B = B
        self.T = T
        self.fim_rate = fim_rate
        
        # Streaming dataset only python
        self.ds = load_dataset(
            "bigcode/the-stack",
            data_dir="data/python",
            split="train",
            streaming=True
        )
        
        self.ds_iter = iter(self.ds)
        
        # Tokenizer
        self.enc = tiktoken.get_encoding('cl100k_base')
        
        # Sanity check
        vocab_size = self.enc.n_vocab
        logger.info(f"Vocab size: {vocab_size}")
        
        # Couldn't find a startoftext token
        allowed = {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endoftext|>", '<|endofprompt|>'}
        
        self.fim_prefix_id = self.enc.encode("<|fim_prefix|>", allowed_special=allowed)[0]
        self.fim_middle_id = self.enc.encode("<|fim_middle|>", allowed_special=allowed)[0]
        self.fim_suffix_id = self.enc.encode("<|fim_suffix|>", allowed_special=allowed)[0]
        self.end_of_text_id = self.enc.encode("<|endoftext|>", allowed_special=allowed)[0]
        
        # rolling token buffer
        self.buffer = torch.empty(0, dtype=torch.long)
        
        logger.info("Streaming dataset loader ready.")
    
    def _fill_buffer(self, min_needed):
        """Ensure buffer has at least min_needed tokens"""
        while len(self.buffer) < min_needed:
            try:
                sample = next(self.ds_iter)
            except StopIteration:
                self.ds_iter = iter(self.ds)
                sample = next(self.ds_iter)
            
            code = sample["content"]
            tokens =  [self.end_of_text_id] + self.enc.encode(code)
            self.buffer = torch.cat([self.buffer, torch.tensor(tokens, dtype=torch.long)])
    
    def next_batch(self):
        B, T = self.B, self.T
        needed = B * T + 1
        self._fill_buffer(needed)
        
        # This is the tokens I will be using for this batch
        buf = self.buffer[:needed]
        
        # This are the remaining tokens for next batch
        self.buffer = self.buffer[needed:]
        
        if random.random() < self.fim_rate:
            # FIM
            x, y = self._create_fim_single(buf, B, T)
        else:
            # L2R classic
            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
        
        return x, y
    
    def _create_fim_single(self, buf, B, T):
        total_len = len(buf)
        
        # ensure at least 4 tokens for middle, 1 for prefix, 3 for suffix
        min_middle = 4
        min_suffix = 3 # True is 0 , see later
        min_prefix = 1
        
        # Determine safe range for split1 (prefix/middle boundary) so where prefix ends
        split1 = random.randint(min_prefix, total_len - min_middle - min_suffix)
        
        # Determine safe range for split2 (middle/suffix boundary) so where suffix starts
        split2 = random.randint(split1 + min_middle, total_len - min_suffix)
        
        prefix = buf[:split1]       # non-empty
        middle = buf[split1:split2] # at least min_middle tokens
        #suffix = buf[split2:]       # at least min_suffix tokens
        
        suffix = buf[split2:-3]       # Because we are about to add 3 tokens, if we want to fit in model we have to crop suffix
        
        
        x_seq = torch.cat([
            torch.tensor([self.fim_suffix_id]),
            suffix,
            torch.tensor([self.fim_prefix_id]),
            prefix,
            torch.tensor([self.fim_middle_id]),
            middle
        ])
        
        y_seq = x_seq[1:]
        x_seq = x_seq[:-1]
        
        # Ignore CE loss for index -100, look class GPT.forward, computes loss for middle only, should I average the loss with reduction = 'mean' ? to test
        mask = torch.full_like(y_seq, -100)
        mask[-len(middle):] = y_seq[-len(middle):]
        y_seq = mask
        
        return x_seq.view(B, T), y_seq.view(B, T)

# Old one, if we are using a .txt file
class DataLoaderFIMold:
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
            x, y, mask= self._create_fim_single(buf, B, T)
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
        split2 = random.randint(split1 + 1, total_len - 1) # Where Suffix starts 
        
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
            middle
        ])
        
        
        y_seq = x_seq[1:]
        x_seq = x_seq[:-1]
        
        mask = torch.full_like(y_seq, -100)
        mask[-(len(middle) + 2):] = y_seq[-(len(middle) + 2):]
        
        # Truncate or pad to B*T
        if len(x_seq) > B*T:
            x_seq = x_seq[:B*T]
            y_seq = y_seq[:B*T]
            mask = mask[:B*T]
        else:
            pad_len = B*T - len(x_seq)
            x_seq = torch.cat([x_seq, torch.zeros(pad_len, dtype=torch.long)])
            y_seq = torch.cat([y_seq, torch.full((pad_len,), -100)])
            mask = torch.cat([mask, torch.full((pad_len,), -100)])
        
        return x_seq.view(B, T), y_seq.view(B, T), mask.view(B, T)



def eval_generate(model):
    model.eval()
    
    max_iter = 45
    trials = 5
    
    enc = tiktoken.get_encoding('cl100k_base')
    allowed = {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endoftext|>"}
    x = "<|fim_suffix|> return c <|fim_prefix|> def sum(a: int, b: int): <|fim_middle|>"
    x_2 = "def sum(a: int, b: int): # returns the sum of a and b"
    
    tokens = enc.encode(x, allowed_special=allowed)
    tokens = torch.tensor(tokens, dtype= torch.long)
    tokens = tokens.unsqueeze(0).repeat(trials, 1) #(trials, len(tokens))
    
    tokens_2 = enc.encode(x_2, allowed_special=allowed)
    tokens_2 = torch.tensor(tokens_2, dtype= torch.long)
    tokens_2 = tokens_2.unsqueeze(0).repeat(trials, 1) #(trials, len(tokens))
    
    x = tokens.to(device)
    x_2 = tokens_2.to(device)
    
    # Generate
    with torch.no_grad():
        y = model.generate(x, max_iter, temperature=0.3, top_k=5)
        y_2 = model.generate(x_2, max_iter, temperature=0.3, top_k=5)
    
    # Decode each generated sequence
    for i in range(trials):
        generated_tokens = y[i].tolist() 
        output = enc.decode(generated_tokens)
        generated_tokens_2 = y_2[i].tolist() 
        output_2 = enc.decode(generated_tokens_2)
        logger.info(f"Trial {i+1}:\n{output}\n")
        logger.info(f"Trial {i+1}:\n{output_2}\n")
    
    model.train()


def eval_dataset(model, x, y_true):
    # WIP, not operational
    
    
    model.eval()
    
    max_iter = 45
    
    fim_middle_id = 100259
    
    # Find indices where x equals fim_middle_id
    indices = torch.where(x == fim_middle_id)[0]
    
    # Take the first occurrence
    idx = indices[0].item()
    
    # Slice up to and including that token
    x_slice = x[:idx + 1]
    
    # Generate
    with torch.no_grad():
        y = model.generate(x_slice, max_iter, temperature=0.3, top_k=5)
    
    enc = tiktoken.get_encoding('cl100k_base')
    
    # Decode ground truth and prediction
    x_text      = enc.decode(x_slice[0].tolist())
    gt_text     = enc.decode(y_true[0][y_true[0] != -100].tolist())
    pred_text   = enc.decode(y[0].tolist())
    
    logger.info("===== INPUT PROMPT =====")
    logger.info(x_text)
    
    logger.info("===== GROUND TRUTH =====")
    logger.info(gt_text)
    
    logger.info("===== MODEL PREDICTION =====")
    logger.info(pred_text)
    
    # optional: log to wandb
    wandb.log({
        "eval/input": x_text,
        "eval/ground_truth": gt_text,
        "eval/prediction": pred_text
    })
    
    model.train()

B = 1
T = 512
texte = DataLoaderFIM(B=B, T=T)

#eval = DataLoaderFIMeval(B=1, T=1024)
#x_eval, y_eval = eval.next_batch()
#x_eval, y_eval = x_eval.to(device), y_eval.to(device)

config = GPTConfig(
    block_size=512,
    sliding_window=512, # Currenly no sliding window, 
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
# optim = torch.optim.AdamW(model.parameters(), lr= 8e-5)
optim = model.configure_optimizers(weight_decay=0.1, learning_rate=8e-5, betas=(0.9, 0.95), device_type=device)

total_tokens_per_step = 524288 # 2**19
micro_tokens = B * T
total_steps = 19000
warmup_steps = 300

grad_acc_steps = total_tokens_per_step // micro_tokens


def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

model.train()
for i in range(total_steps):
    optim.zero_grad()
    for micro_step in range(grad_acc_steps):
        logger.info(f"Accumulation {micro_step}")
        x, y = texte.next_batch()
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, y)
        
        # Backward
        loss.backward()
    
    # Ensure FIM stability
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer
    optim.step()
    
    # Scheduler
    scheduler.step()
    
    # WANDB
    current_lr = optim.param_groups[0]["lr"]
    logger.info(f"Step {i} loss: {loss.item()}")
    wandb.log({"step": i, "loss": loss.item(), "lr": current_lr, 'norm': norm})
    if i % 500 == 0:
        eval_generate(model)
        # eval_dataset(model, x, y) not ready yet


torch.save(model.state_dict(), "gpt_model.pt")
# wandb.save("gpt_model.pt")

logger.success('FINISHEDDDDDD !')