#!/usr/bin/env python3
"""Estimate parameters for model configurations."""

import json
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def count_parameters(config: dict) -> dict:
    """Calculate parameter counts for a model configuration."""
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_hidden_layers = config["num_hidden_layers"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)
    intermediate_size = config["intermediate_size"]
    tie_word_embeddings = config.get("tie_word_embeddings", True)
    
    # Embedding parameters
    embedding_params = vocab_size * hidden_size
    
    # Output projection (if not tied)
    if tie_word_embeddings:
        output_params = 0
    else:
        output_params = vocab_size * hidden_size
    
    # Per-layer parameters
    # Attention: Q, K, V projections + output projection
    # GQA: Q has num_attention_heads * head_dim, K and V have num_key_value_heads * head_dim
    q_proj = hidden_size * (num_attention_heads * head_dim)
    k_proj = hidden_size * (num_key_value_heads * head_dim)
    v_proj = hidden_size * (num_key_value_heads * head_dim)
    o_proj = (num_attention_heads * head_dim) * hidden_size
    
    attention_params = q_proj + k_proj + v_proj + o_proj
    
    # MLP: SwiGLU has 3 matrices (gate, up, down)
    mlp_gate = hidden_size * intermediate_size
    mlp_up = hidden_size * intermediate_size
    mlp_down = intermediate_size * hidden_size
    
    mlp_params = mlp_gate + mlp_up + mlp_down
    
    # RMSNorm: 2 per layer (attention norm and mlp norm)
    norm_params = 2 * hidden_size
    
    # Total per layer
    per_layer_params = attention_params + mlp_params + norm_params
    
    # Total transformer parameters
    transformer_params = num_hidden_layers * per_layer_params
    
    # Total parameters
    total_params = embedding_params + output_params + transformer_params
    
    return {
        "embedding_params": embedding_params,
        "output_params": output_params,
        "attention_params_per_layer": attention_params,
        "mlp_params_per_layer": mlp_params,
        "norm_params_per_layer": norm_params,
        "per_layer_params": per_layer_params,
        "transformer_params": transformer_params,
        "total_params": total_params,
        "total_params_millions": total_params / 1e6,
    }


def estimate_memory(params: int, dtype: str = "float32") -> dict:
    """Estimate memory requirements for model parameters."""
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "q4": 0.5,
    }.get(dtype, 4)
    
    param_memory_bytes = params * bytes_per_param
    param_memory_mb = param_memory_bytes / (1024 ** 2)
    param_memory_gb = param_memory_bytes / (1024 ** 3)
    
    # Add overhead for gradients and optimizer states (for training)
    gradient_memory = param_memory_bytes  # Same size as parameters
    adam_state_memory = 2 * param_memory_bytes  # Adam has 2 state variables
    
    training_memory_bytes = param_memory_bytes + gradient_memory + adam_state_memory
    training_memory_gb = training_memory_bytes / (1024 ** 3)
    
    return {
        "dtype": dtype,
        "param_memory_mb": param_memory_mb,
        "param_memory_gb": param_memory_gb,
        "training_memory_gb": training_memory_gb,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate model parameters")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.json",
        help="Path to model config JSON",
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    params = count_parameters(config)
    memory_fp32 = estimate_memory(params["total_params"], "float32")
    memory_fp16 = estimate_memory(params["total_params"], "float16")
    memory_q4 = estimate_memory(params["total_params"], "q4")
    
    print(f"Parameter estimation for: {config_path}")
    print("=" * 60)
    print(f"Vocab size: {config['vocab_size']}")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Num layers: {config['num_hidden_layers']}")
    print(f"Num attention heads: {config['num_attention_heads']}")
    print(f"Num KV heads: {config.get('num_key_value_heads', config['num_attention_heads'])}")
    print(f"Intermediate size: {config['intermediate_size']}")
    print(f"Tie word embeddings: {config.get('tie_word_embeddings', True)}")
    print("=" * 60)
    print(f"Embedding parameters: {params['embedding_params']:,}")
    print(f"Output parameters: {params['output_params']:,}")
    print(f"Attention params per layer: {params['attention_params_per_layer']:,}")
    print(f"MLP params per layer: {params['mlp_params_per_layer']:,}")
    print(f"Norm params per layer: {params['norm_params_per_layer']:,}")
    print(f"Total per layer: {params['per_layer_params']:,}")
    print(f"Transformer parameters: {params['transformer_params']:,}")
    print(f"TOTAL PARAMETERS: {params['total_params']:,} ({params['total_params_millions']:.2f}M)")
    print("=" * 60)
    print(f"Memory estimate (FP32): {memory_fp32['param_memory_gb']:.2f} GB")
    print(f"Memory estimate (FP16/BF16): {memory_fp16['param_memory_gb']:.2f} GB")
    print(f"Memory estimate (Q4): {memory_q4['param_memory_gb']:.2f} GB")
    print(f"Training memory estimate (FP32 with Adam): {memory_fp32['training_memory_gb']:.2f} GB")
    print("=" * 60)
    
    # Check against budget
    budget = 550_000_000
    if params["total_params"] > budget:
        print(f"WARNING: Parameters exceed budget of {budget:,}")
        print(f"Excess: {params['total_params'] - budget:,} parameters")
        return 1
    else:
        print(f"OK: Parameters within budget of {budget:,}")
        print(f"Remaining budget: {budget - params['total_params']:,} parameters")
        return 0


if __name__ == "__main__":
    exit(main())
