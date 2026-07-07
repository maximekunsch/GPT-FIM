"""
Model Serving Pipeline

Option 1: FastAPI server (works with current custom GPT model)
Option 2: vLLM server (requires vllm package and HF model format)

Usage:
    # FastAPI server (default)
    python utils/serve.py
    
    # vLLM server
    python utils/serve.py --backend vllm --model_path gpt_model.pt
    
    # With custom host/port
    python utils/serve.py --host 0.0.0.0 --port 8000
"""

import argparse
import sys
import os
import json
import logging
from typing import Optional, List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Serve GPT model')
    parser.add_argument('--backend', type=str, default='fastapi', choices=['fastapi', 'vllm'], 
                        help='Serving backend: fastapi or vllm')
    parser.add_argument('--model_path', type=str, default='gpt_model.pt', 
                        help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='localhost', 
                        help='Host to serve on')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Port to serve on')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use: cuda or cpu')
    parser.add_argument('--dtype', type=str, default='bfloat16', 
                        help='Data type: bfloat16, float16, float32')
    parser.add_argument('--max_length', type=int, default=512, 
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.7, 
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, 
                        help='Top-k sampling')
    return parser.parse_args()


# ============================================================================
# Backend: FastAPI (works with custom GPT model)
# ============================================================================

def serve_fastapi(args):
    """Serve model using FastAPI - works with the custom GPT implementation"""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        logger.error("FastAPI or uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)
    
    # Import model components
    try:
        from utils.Model import GPT, GPTConfig
        from utils.logging_config import logger as file_logger
        import torch
        import tiktoken
    except ImportError as e:
        logger.error(f"Failed to import model components: {e}")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Create model config (must match how model was trained)
    config = GPTConfig(
        block_size=args.max_length,
        sliding_window=args.max_length,
        vocab_size=100277,  # cl100k_base tokenizer
        n_layer=9,
        n_head=16,
        n_embd=2048,
        softcap=20,
        dropout=0,
        bias=False
    )
    
    model = GPT(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Tokenizer
    enc = tiktoken.get_encoding('cl100k_base')
    allowed = {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endoftext|>", "<|endofprompt|>"}
    fim_prefix_id = enc.encode("<|fim_prefix|>", allowed_special=allowed)[0]
    fim_middle_id = enc.encode("<|fim_middle|>", allowed_special=allowed)[0]
    fim_suffix_id = enc.encode("<|fim_suffix|>", allowed_special=allowed)[0]
    
    app = FastAPI(title="GPT-FIM Model Server")
    
    @app.post("/generate")
    async def generate(request: Request):
        """
        Generate text from a prompt.
        
        Request body:
        {
            "prompt": "Your prompt here",
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_k": 50
        }
        """
        try:
            body = await request.json()
            prompt = body.get("prompt", "")
            max_new_tokens = body.get("max_new_tokens", args.max_length)
            temperature = body.get("temperature", args.temperature)
            top_k = body.get("top_k", args.top_k)
            
            # Tokenize
            tokens = enc.encode(prompt, allowed_special=allowed)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            
            # Generate
            with torch.no_grad():
                output = model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            
            # Decode
            output_text = enc.decode(output[0].tolist())
            
            return JSONResponse({
                "generated_text": output_text,
                "input_tokens": len(tokens[0]),
                "output_tokens": len(output[0])
            })
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": "gpt-fim", "device": device}
    
    @app.get("/info")
    async def info():
        return {
            "model": "gpt-fim",
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "device": device
        }
    
    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


# ============================================================================
# Backend: vLLM (requires vllm and HF model format)
# ============================================================================

def serve_vllm(args):
    """
    Serve model using vLLM - requires model in HuggingFace format.
    
    Note: Your current model is custom. You have two options:
    1. Convert your model to HF format first
    2. Use the FastAPI backend instead
    
    For vLLM to work, you need to:
    - pip install vllm
    - Convert your model to HF Transformers format
    - Save with HuggingFace's save_pretrained()
    """
    try:
        from vllm import LLM, SamplingParams
        import torch
    except ImportError:
        logger.error(
            "vLLM not installed. Install with: pip install vllm\n"
            "Note: vLLM requires CUDA and Linux.\n"
            "Falling back to FastAPI backend."
        )
        serve_fastapi(args)
        return
    
    logger.info("Loading model with vLLM...")
    
    # vLLM expects a HuggingFace model directory
    # For your custom model, you'd need to convert it first
    # This is a placeholder - you'd need to implement model conversion
    try:
        # Try to load as HF model
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=1,  # Single GPU
            dtype=args.dtype,
        )
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_length,
        )
        
        # Create a simple HTTP server with vLLM
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import uvicorn
        
        app = FastAPI(title="GPT-FIM vLLM Server")
        
        @app.post("/generate")
        async def generate(request: Request):
            body = await request.json()
            prompts = [body.get("prompt", "")]
            outputs = llm.generate(prompts, sampling_params)
            
            return JSONResponse({
                "generated_text": [o.outputs[0].text for o in outputs],
                "model": llm.llm_engine.model_config.model
            })
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "backend": "vllm"}
        
        logger.info(f"Starting vLLM server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"vLLM loading failed: {e}")
        logger.error("Model may not be in HuggingFace format. Use --backend fastapi or convert your model.")
        sys.exit(1)


# ============================================================================
# Helper: Convert custom model to HuggingFace format
# ============================================================================

def convert_to_hf(args):
    """
    Convert your custom GPT model to HuggingFace Transformers format.
    
    This creates a directory with:
    - config.json
    - pytorch_model.bin
    - tokenizer.json / tokenizer_config.json
    """
    import torch
    from pathlib import Path
    
    try:
        from utils.Model import GPT, GPTConfig
        import tiktoken
    except ImportError as e:
        logger.error(f"Failed to import: {e}")
        return
    
    logger.info("Converting model to HuggingFace format...")
    
    # Load your model
    device = 'cpu'  # Convert on CPU
    config = GPTConfig(
        block_size=512,
        sliding_window=512,
        vocab_size=100277,
        n_layer=9,
        n_head=16,
        n_embd=2048,
        softcap=20,
        dropout=0,
        bias=False
    )
    
    model = GPT(config)
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Create output directory
    hf_dir = Path(args.model_path).parent / (Path(args.model_path).stem + "_hf")
    hf_dir.mkdir(exist_ok=True)
    
    # Save config
    hf_config = {
        "model_type": "gpt2",
        "vocab_size": config.vocab_size,
        "n_positions": config.block_size,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "activation_function": "gelu_new",
        "resid_pdrop": config.dropout,
        "embd_pdrop": config.dropout,
        "attn_pdrop": config.dropout,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "summary_activation": None,
        "summary_proj_to_labels": True,
    }
    
    with open(hf_dir / "config.json", 'w') as f:
        json.dump(hf_config, f, indent=2)
    
    # Save model weights (need to map to HF names)
    hf_state_dict = {}
    
    # Map your layer names to HF GPT2 names
    for i in range(config.n_layer):
        # Attention layers
        hf_state_dict[f'transformer.h.{i}.attn.c_attn.weight'] = state_dict[f'transformer.h.{i}.attn.c_attn.weight']
        hf_state_dict[f'transformer.h.{i}.attn.c_attn.bias'] = state_dict.get(f'transformer.h.{i}.attn.c_attn.bias', torch.zeros_like(state_dict[f'transformer.h.{i}.attn.c_attn.weight'][:config.n_embd]))
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.weight'] = state_dict[f'transformer.h.{i}.attn.c_proj.weight']
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.bias'] = state_dict.get(f'transformer.h.{i}.attn.c_proj.bias', torch.zeros(config.n_embd))
        
        # Layer norm
        hf_state_dict[f'transformer.h.{i}.ln_1.weight'] = state_dict[f'transformer.h.{i}.ln_1.weight']
        hf_state_dict[f'transformer.h.{i}.ln_1.bias'] = state_dict.get(f'transformer.h.{i}.ln_1.bias', torch.zeros(config.n_embd))
        
        # MLP
        hf_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = state_dict[f'transformer.h.{i}.mlp.c_fc.weight']
        hf_state_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = state_dict.get(f'transformer.h.{i}.mlp.c_fc.bias', torch.zeros(4 * config.n_embd))
        hf_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = state_dict[f'transformer.h.{i}.mlp.c_proj.weight']
        hf_state_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = state_dict.get(f'transformer.h.{i}.mlp.c_proj.bias', torch.zeros(config.n_embd))
        
        # Layer norm 2
        hf_state_dict[f'transformer.h.{i}.ln_2.weight'] = state_dict[f'transformer.h.{i}.ln_2.weight']
        hf_state_dict[f'transformer.h.{i}.ln_2.bias'] = state_dict.get(f'transformer.h.{i}.ln_2.bias', torch.zeros(config.n_embd))
    
    # Embeddings and final layer norm
    hf_state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight']
    hf_state_dict['transformer.wpe.weight'] = state_dict['transformer.wpe.weight']
    hf_state_dict['transformer.ln_f.weight'] = state_dict['transformer.ln_f.weight']
    hf_state_dict['transformer.ln_f.bias'] = state_dict.get('transformer.ln_f.bias', torch.zeros(config.n_embd))
    hf_state_dict['lm_head.weight'] = state_dict['lm_head.weight']
    
    torch.save(hf_state_dict, hf_dir / "pytorch_model.bin")
    
    # Save tokenizer
    # vLLM will use its own tokenizer, but we can save for reference
    enc = tiktoken.get_encoding('cl100k_base')
    # Note: tiktoken doesn't save to files easily; you'd need to use HF's tokenizer
    # For now, just log that tokenizer needs manual setup
    logger.warning("Tokenizer not saved. For vLLM, use: tokenizer = 'tiktoken' or provide a tokenizer.json")
    
    logger.info(f"Model saved to {hf_dir}")
    logger.info("To use with vLLM: python utils/serve.py --backend vllm --model_path <hf_dir>")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    if args.backend == 'vllm':
        serve_vllm(args)
    elif args.backend == 'fastapi':
        serve_fastapi(args)
    else:
        logger.error(f"Unknown backend: {args.backend}")
        sys.exit(1)


if __name__ == "__main__":
    main()
