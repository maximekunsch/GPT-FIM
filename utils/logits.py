import torch
import torch.nn.functional as F
from Model import GPTConfig, GPT
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Configure model ---
config = GPTConfig(
    block_size=128,
    sliding_window=32,
    vocab_size=1024,
    n_layer=4,
    n_head=8,
    n_embd=256,
    softcap=20,
    dropout=0.1,
    bias=False
)

model = GPT(config).to(device)
model.eval()  # Disable dropout etc.

# --- Dummy input ---
batch_size, seq_len = 2, 64
idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

# --- Run both modes ---
with torch.no_grad():
    model.flex = True
    start_flex = time.time()
    logits_flex, _ = model(idx)
    final_flex = time.time() - start_flex

    model.flex = False
    start_manual = time.time()
    logits_manual, _ = model(idx)
    final_manual = time.time() - start_manual

# --- Compare numerically ---
max_diff = (logits_flex - logits_manual).abs().max()
mean_diff = (logits_flex - logits_manual).abs().mean()

print(f"Max abs diff: {max_diff.item():.6f}")
print(f"Mean abs diff: {mean_diff.item():.6f}")
print(f"Time diff: {final_manual - final_flex:.6f}")

# Optional sanity check: are they numerically close?
print("Close:", torch.allclose(logits_flex, logits_manual, atol=1e-4, rtol=1e-4))
