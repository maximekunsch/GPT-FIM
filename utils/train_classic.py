import argparse
import sys
import os
import torch
import tiktoken
import random
import numpy as np
from datasets import load_dataset, interleave_datasets
import math
import time

# Ensure imports work whether running from root or utils/ directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.logging_config import logger
    from utils.Model import GPTConfig, GPT
except ImportError:
    from logging_config import logger
    from Model import GPTConfig, GPT

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='GPT Classic Training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--block_size', type=int, default=512, help='Block size')
    parser.add_argument('--compile', action='store_true', default=True, help='Use PyTorch 2.0 compile')
    parser.add_argument('--wandb_project', type=str, default='gpt-training-classic', help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token (or set HF_TOKEN env var)')
    return parser.parse_args()


def run_training(args):
    # Initialize wandb if not disabled
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "learning_rate": 8e-5,
                "n_layer": 9,
                "n_head": 16,
                "n_embd": 2048,
                "dropout": 0
            }
        )
        config = wandb.config
    else:
        config = argparse.Namespace(
            batch_size=args.batch_size,
            block_size=args.block_size,
            learning_rate=8e-5,
            n_layer=9,
            n_head=16,
            n_embd=2048,
            dropout=0
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    # Get HF token from args or environment
    hf_token = args.hf_token or os.getenv('HF_TOKEN')

    # FineWeb multilingual dataloader (EN + FR) with cl100k_base tokenizer
    class DataLoader:
        def __init__(self, B, T):
            self.B = B
            self.T = T

            # Load FineWeb datasets
            fw_en = load_dataset("HuggingFaceFW/fineweb", name="sample-10B", split="train", streaming=True, token=hf_token)
            time.sleep(60)  # Wait for HF quota reset between datasets

            fw_fr = load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn", split="train", streaming=True, token=hf_token)

            self.ds = interleave_datasets([fw_en, fw_fr], probabilities=[0.5, 0.5], seed=42)

            self.enc = tiktoken.get_encoding('cl100k_base')

            # Rolling buffer
            self.buffer = torch.empty(0, dtype=torch.long)
            self.ds_iter = iter(self.ds)

            logger.info("FineWeb EN/FR streaming loader ready with cl100k_base tokenizer.")

        def _fill_buffer(self, min_needed):
            while len(self.buffer) < min_needed:
                try:
                    sample = next(self.ds_iter)
                    time.sleep(0.5)  # HF rate limit
                    text = sample["text"]
                    tokens = self.enc.encode(text)
                    self.buffer = torch.cat([self.buffer, torch.tensor(tokens, dtype=torch.long)])
                except StopIteration:
                    self.ds_iter = iter(self.ds)
                    sample = next(self.ds_iter)
                    time.sleep(0.5)  # HF rate limit
                    text = sample["text"]
                    tokens = self.enc.encode(text)
                    self.buffer = torch.cat([self.buffer, torch.tensor(tokens, dtype=torch.long)])

        def next_batch(self):
            B, T = self.B, self.T
            needed = B * T + 1
            self._fill_buffer(needed)

            buf = self.buffer[:needed]
            self.buffer = self.buffer[needed:]

            x = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
            return x, y

    B = args.batch_size
    T = args.block_size
    texte = DataLoader(B=B, T=T)

    gpt_config = GPTConfig(
        block_size=T,
        sliding_window=T,
        vocab_size=100277,
        n_layer=9,
        n_head=16,
        n_embd=2048,
        softcap=20,
        dropout=0,
        bias=False
    )

    model = GPT(gpt_config)
    model.to(device)

    if args.compile:
        model = torch.compile(model)

    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=8e-5, betas=(0.9, 0.95), device_type=device_type)

    total_tokens_per_step = 524288
    micro_tokens = B * T
    total_steps = 31000
    warmup_steps = 1000

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
        if not args.no_wandb:
            wandb.log({"step": i, "loss": loss.item(), "lr": current_lr, 'norm': norm})

        # Checkpoint every 5K steps + final
        if i % 5000 == 0 or i == total_steps - 1:
            torch.save(model.state_dict(), f"gpt_model_classic_{i}.pt")

    torch.save(model.state_dict(), "gpt_model_classic.pt")

    logger.success('FINISHEDDDDDD !')


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
