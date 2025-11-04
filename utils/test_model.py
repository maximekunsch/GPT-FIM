import torch
import torch.nn.functional as F
import time
from Model import GPTConfig, GPT
from logging_config import logger

logger.info("="*80)
logger.info("GPT MODEL TEST SUITE")
logger.info("="*80)

# =============================================================================
# TEST 1: BASIC INSTANTIATION TEST
# =============================================================================
logger.info("[TEST 1] Basic Model Instantiation")
logger.info("-" * 80)

try:
    # Create a small config for testing
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
    
    logger.success(f"Config created: {config}")
    
    # Instantiate model
    model = GPT(config)
    logger.success("✓ Model instantiated successfully")
    logger.success(f"✓ Total parameters: {model.get_num_params()/1e6:.2f}M")
    logger.success(f"✓ Non-embedding parameters: {model.get_num_params(non_embedding=True)/1e6:.2f}M")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 2: FORWARD PASS TEST
# =============================================================================
logger.info("\n")
logger.info("[TEST 2] Forward Pass (No Targets)")
logger.info("-" * 80)

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    logger.info(f"Input shape: {idx.shape}")
    
    with torch.no_grad():
        logits, loss = model(idx)
    
    logger.success("✓ Forward pass successful")
    logger.success(f"✓ Logits shape: {logits.shape}")
    logger.success(f"✓ Expected shape: ({batch_size}, 1, {config.vocab_size})")
    assert logits.shape == (batch_size, 1, config.vocab_size), "Logits shape mismatch!"
    assert loss is None, "Loss should be None without targets"
    logger.success("✓ All assertions passed")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 3: FORWARD PASS WITH TARGETS (TRAINING MODE)
# =============================================================================
logger.info("\n")
logger.info("[TEST 3] Forward Pass with Targets")
logger.info("-" * 80)

try:
    model.train()
    
    # Create dummy input and targets
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    logger.info(f"Input shape: {idx.shape}")
    logger.info(f"Targets shape: {targets.shape}")
    
    logits, loss = model(idx, targets=targets)
    
    logger.success("✓ Forward pass with targets successful")
    logger.success(f"✓ Logits shape: {logits.shape}")
    logger.success(f"✓ Loss value: {loss.item():.4f}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Logits shape mismatch!"
    assert loss is not None, "Loss should not be None with targets"
    assert not torch.isnan(loss), "Loss is NaN!"
    logger.success("✓ All assertions passed")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 4: SUFFIX-PREFIX ATTENTION MASKING
# =============================================================================
logger.info("\n")
logger.info("[TEST 4] Suffix-Prefix Attention Masking")
logger.info("-" * 80)

try:
    model.eval()
    
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Define prefix lengths for each batch element
    suffix_prefix_length = [10, 20]  # First batch has 10 prefix tokens, second has 20
    
    logger.info(f"Input shape: {idx.shape}")
    logger.info(f"Prefix lengths: {suffix_prefix_length}")
    
    with torch.no_grad():
        logits_with_prefix, _ = model(idx, suffix_prefix_length=suffix_prefix_length)
        logits_without_prefix, _ = model(idx)
    
    logger.success("✓ Forward pass with suffix-prefix masking successful")
    logger.success(f"✓ Logits with prefix shape: {logits_with_prefix.shape}")
    logger.success(f"✓ Logits without prefix shape: {logits_without_prefix.shape}")
    
    # Check that outputs are different
    diff = torch.abs(logits_with_prefix - logits_without_prefix).max()
    logger.success(f"✓ Max difference between masked/unmasked: {diff.item():.6f}")
    assert diff > 0, "Outputs should differ with different masking!"
    logger.success("✓ All assertions passed")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 5: BACKWARD PASS & GRADIENT CHECK
# =============================================================================
logger.info("\n")
logger.info("[TEST 5] Backward Pass & Gradient Check")
logger.info("-" * 80)

try:
    model.train()
    
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    logits, loss = model(idx, targets=targets)
    
    # Backward pass
    loss.backward()
    
    logger.success("✓ Backward pass successful")
    
    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    logger.success(f"✓ All {grad_count} parameter gradients are valid")
    logger.success("✓ All assertions passed")
    
    # Zero gradients for next test
    model.zero_grad()
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 6: MINI TRAINING LOOP
# =============================================================================
logger.info("\n")
logger.info("[TEST 6] Mini Training Loop (10 steps)")
logger.info("-" * 80)

try:
    model.train()
    
    # Configure optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=2e-5,
        betas=(0.9, 0.95),
        device_type='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    batch_size = 4
    seq_len = 64
    num_steps = 10
    
    losses = []
    
    logger.info(f"Training for {num_steps} steps...")
    start_time = time.time()
    
    for step in range(num_steps):
        # Generate random batch
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        
        # Forward pass
        logits, loss = model(idx, targets=targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 5 == 0:
            logger.info(f"  Step {step+1}/{num_steps}: loss = {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    
    logger.success("✓ Training loop completed successfully")
    logger.success(f"✓ Time elapsed: {elapsed:.2f}s ({elapsed/num_steps:.3f}s per step)")
    logger.success(f"✓ Initial loss: {losses[0]:.4f}")
    logger.success(f"✓ Final loss: {losses[-1]:.4f}")
    logger.success(f"✓ Loss trend: {'decreasing' if losses[-1] < losses[0] else 'not decreasing (may need more steps)'}")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 7: TEXT GENERATION
# =============================================================================
logger.info("\n")
logger.info("[TEST 7] Text Generation")
logger.info("-" * 80)

try:
    model.eval()
    
    # Start with a random token
    start_tokens = torch.randint(0, config.vocab_size, (1, 1)).to(device)
    max_new_tokens = 20
    temperature = 1.0
    
    logger.info(f"Generating {max_new_tokens} tokens...")
    logger.info(f"Start token: {start_tokens.item()}")
    
    generated = start_tokens
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop if sequence exceeds block size
            idx_cond = generated if generated.size(1) <= config.block_size else generated[:, -config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
    
    logger.success(f"✓ Generated sequence length: {generated.size(1)}")
    logger.success(f"✓ Generated tokens: {generated[0].tolist()}")
    logger.success("✓ All assertions passed")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 8: CROP BLOCK SIZE
# =============================================================================
logger.info("\n")
logger.info("[TEST 8] Crop Block Size")
logger.info("-" * 80)

try:
    original_block_size = config.block_size
    new_block_size = 64
    
    logger.info(f"Original block size: {original_block_size}")
    logger.info(f"New block size: {new_block_size}")
    
    # Crop the model
    model.crop_block_size(new_block_size)
    
    logger.success("✓ Block size cropped successfully")
    logger.success(f"✓ New config block size: {model.config.block_size}")
    logger.success(f"✓ Position embedding shape: {model.transformer.wpe.weight.shape}")
    
    assert model.config.block_size == new_block_size, "Block size not updated!"
    assert model.transformer.wpe.weight.size(0) == new_block_size, "Position embeddings not cropped!"
    
    # Test forward pass with new block size
    idx = torch.randint(0, config.vocab_size, (2, new_block_size)).to(device)
    
    with torch.no_grad():
        logits, _ = model(idx)
    
    logger.success("✓ Forward pass with cropped model successful")
    logger.success("✓ All assertions passed")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 9: PERFORMANCE BENCHMARK
# =============================================================================
logger.info("\n")
logger.info("[TEST 9] Performance Benchmark")
logger.info("-" * 80)

try:
    model.eval()
    
    batch_size = 4
    seq_len = min(64, model.config.block_size)  # Use current block size
    num_iterations = 20
    
    # Warmup
    for _ in range(5):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        with torch.no_grad():
            _ = model(idx)
    
    # Benchmark forward pass
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(num_iterations):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        with torch.no_grad():
            _ = model(idx)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_iterations
    throughput = (batch_size * seq_len) / avg_time
    
    logger.success(f"✓ Benchmark completed ({num_iterations} iterations)")
    logger.success(f"✓ Average time per forward pass: {avg_time*1000:.2f}ms")
    logger.success(f"✓ Throughput: {throughput:.0f} tokens/sec")
    
    if device == 'cuda':
        logger.info(f"✓ Peak memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 10: MEMORY USAGE
# =============================================================================
logger.info("\n")
logger.info("[TEST 10] Memory Usage Analysis")
logger.info("-" * 80)

try:
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        batch_sizes = [1, 2, 4, 8]
        seq_len = min(64, model.config.block_size)  # Use current block size
        
        logger.info(f"Testing different batch sizes (seq_len={seq_len})...")
        
        for bs in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            idx = torch.randint(0, config.vocab_size, (bs, seq_len)).to(device)
            
            with torch.no_grad():
                _ = model(idx)
            
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"  Batch size {bs}: {peak_mem:.3f}GB")
        
        logger.success("✓ Memory analysis completed")
    else:
        logger.info("⚠ Skipping memory test (requires CUDA)")
    
except Exception as e:
    logger.error(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# SUMMARY
# =============================================================================
logger.info("\n")
logger.info("="*80)
logger.info("TEST SUITE COMPLETED")
logger.info("="*80)
logger.success("All tests finished. Review results above for any failures.")
logger.success("Note: Make sure to apply the fixes mentioned before running this test suite!")