"""
Convert custom GPT model to HuggingFace Transformers format for vLLM.

Usage:
    python utils/convert_to_hf.py --model_path gpt_model.pt --output_dir gpt_model_hf

This creates a HuggingFace-compatible model directory that can be loaded with:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt_model_hf")

For vLLM:
    from vllm import LLM
    llm = LLM(model="gpt_model_hf", dtype="bfloat16")
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert custom GPT to HuggingFace format')
    parser.add_argument('--model_path', type=str, default='gpt_model.pt',
                        help='Path to custom model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <model_path>_hf)')
    parser.add_argument('--n_layer', type=int, default=9,
                        help='Number of layers')
    parser.add_argument('--n_head', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=2048,
                        help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=512,
                        help='Block size / context length')
    parser.add_argument('--vocab_size', type=int, default=100277,
                        help='Vocabulary size (cl100k_base = 100277)')
    parser.add_argument('--bias', action='store_true',
                        help='Use bias in layers')
    return parser.parse_args()


def create_hf_config(args: argparse.Namespace) -> Dict:
    """Create HuggingFace GPT-2 style config."""
    return {
        "model_type": "gpt2",
        "vocab_size": args.vocab_size,
        "n_positions": args.block_size,
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "summary_activation": None,
        "summary_proj_to_labels": True,
        "use_cache": True,
        "bos_token_id": 100259,  # <|fim_prefix|> or custom BOS
        "eos_token_id": 100260,  # <|fim_middle|> or custom EOS
    }


def map_state_dict(state_dict: Dict, args: argparse.Namespace) -> Dict:
    """Map custom model state dict to HuggingFace GPT-2 names."""
    hf_state_dict = {}
    
    for i in range(args.n_layer):
        prefix = f'transformer.h.{i}.'
        custom_prefix = f'transformer.h.{i}.'
        
        # Attention projection (QKV combined -> split)
        # Custom: c_attn.weight shape [3*n_embd, n_embd]
        # HF: q_attn.weight, k_attn.weight, v_attn.weight each [n_embd, n_embd]
        c_attn_weight = state_dict[f'{custom_prefix}attn.c_attn.weight']
        n_embd = args.n_embd
        q_w, k_w, v_w = c_attn_weight[:n_embd, :], c_attn_weight[n_embd:2*n_embd, :], c_attn_weight[2*n_embd:3*n_embd, :]
        
        hf_state_dict[f'{prefix}attn.q_attn.weight'] = q_w.T if q_w.shape[0] != n_embd else q_w
        hf_state_dict[f'{prefix}attn.k_attn.weight'] = k_w.T if k_w.shape[0] != n_embd else k_w
        hf_state_dict[f'{prefix}attn.v_attn.weight'] = v_w.T if v_w.shape[0] != n_embd else v_w
        
        # Attention output projection
        hf_state_dict[f'{prefix}attn.c_proj.weight'] = state_dict[f'{custom_prefix}attn.c_proj.weight']
        
        # Layer norm 1
        hf_state_dict[f'{prefix}ln_1.weight'] = state_dict[f'{custom_prefix}ln_1.weight']
        if args.bias:
            hf_state_dict[f'{prefix}ln_1.bias'] = state_dict.get(f'{custom_prefix}ln_1.bias', torch.zeros(args.n_embd))
        
        # MLP (GLU)
        # Your model uses GLU with 4*n_embd intermediate size
        # HF GPT-2 uses c_fc and c_proj
        # We'll map your GLU layers to approximate HF structure
        glu_fc_weight = state_dict[f'{custom_prefix}mlp.c_fc.weight']
        glu_fc_2_weight = state_dict[f'{custom_prefix}mlp.c_fc_2.weight']
        
        # For now, simple mapping - this may need adjustment
        hf_state_dict[f'{prefix}mlp.c_fc.weight'] = glu_fc_weight
        hf_state_dict[f'{prefix}mlp.c_fc.bias'] = state_dict.get(f'{custom_prefix}mlp.c_fc.bias', torch.zeros(4 * args.n_embd))
        hf_state_dict[f'{prefix}mlp.c_proj.weight'] = state_dict[f'{custom_prefix}mlp.c_proj.weight']
        hf_state_dict[f'{prefix}mlp.c_proj.bias'] = state_dict.get(f'{custom_prefix}mlp.c_proj.bias', torch.zeros(args.n_embd))
        
        # Layer norm 2
        hf_state_dict[f'{prefix}ln_2.weight'] = state_dict[f'{custom_prefix}ln_2.weight']
        if args.bias:
            hf_state_dict[f'{prefix}ln_2.bias'] = state_dict.get(f'{custom_prefix}ln_2.bias', torch.zeros(args.n_embd))
    
    # Token and position embeddings
    hf_state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight']
    hf_state_dict['transformer.wpe.weight'] = state_dict['transformer.wpe.weight']
    
    # Final layer norm
    hf_state_dict['transformer.ln_f.weight'] = state_dict['transformer.ln_f.weight']
    if args.bias:
        hf_state_dict['transformer.ln_f.bias'] = state_dict.get('transformer.ln_f.bias', torch.zeros(args.n_embd))
    
    # LM head (weight-tied with wte)
    hf_state_dict['lm_head.weight'] = state_dict['lm_head.weight']
    
    return hf_state_dict


def main():
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        model_stem = Path(args.model_path).stem
        args.output_dir = str(Path(args.model_path).parent / f"{model_stem}_hf")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    
    print("Mapping state dict to HuggingFace format...")
    hf_state_dict = map_state_dict(state_dict, args)
    
    print(f"Saving to {output_dir}...")
    
    # Save config
    config = create_hf_config(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save model
    torch.save(hf_state_dict, output_dir / "pytorch_model.bin")
    
    # Create tokenizer config (for cl100k_base)
    tokenizer_config = {
        "tokenizer_class": "ByteLevelBPETokenizer",
        "model_max_length": args.block_size,
    }
    with open(output_dir / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Note about tokenizer
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print(f"HuggingFace model saved to: {output_dir}")
    print("\nFor vLLM, you need the tokenizer. Options:")
    print("  1. Use tiktoken: tokenizer='cl100k_base' (recommended)")
    print("  2. Or copy tiktoken's vocab to this directory")
    print("\nTo load with vLLM:")
    print(f"  from vllm import LLM")
    print(f"  llm = LLM(model='{output_dir}', tokenizer='cl100k_base', dtype='bfloat16')")
    print("="*60)


if __name__ == "__main__":
    main()
