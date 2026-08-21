"""
Parameter Count Utilities

Functions for counting model parameters and estimating memory usage.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .config import ModelConfig


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in a PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = 0
    trainable = 0
    non_trainable = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total += num_params
        
        if param.requires_grad:
            trainable += num_params
        else:
            non_trainable += num_params
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def estimate_parameters_from_config(config: ModelConfig) -> Dict[str, int]:
    """
    Estimate parameter count from model configuration without creating model.
    
    Args:
        config (ModelConfig): Model configuration
        
    Returns:
        Dictionary with detailed parameter breakdown
    """
    # Embedding parameters
    embedding_params = config.vocab_size * config.hidden_size
    
    # Per-layer parameters
    # Attention: Q + K + V + O projections
    attn_q_params = config.hidden_size * (config.num_attention_heads * config.head_dim)
    attn_k_params = config.hidden_size * (config.num_key_value_heads * config.head_dim)
    attn_v_params = config.hidden_size * (config.num_key_value_heads * config.head_dim)
    attn_o_params = config.hidden_size * config.hidden_size
    attn_total = attn_q_params + attn_k_params + attn_v_params + attn_o_params
    
    # MLP (SwiGLU): gate + value + output projections
    mlp_gate_params = config.hidden_size * config.intermediate_size
    mlp_value_params = config.hidden_size * config.intermediate_size
    mlp_output_params = config.intermediate_size * config.hidden_size
    mlp_total = mlp_gate_params + mlp_value_params + mlp_output_params
    
    # Normalization (2x RMSNorm per layer)
    norm_params = 2 * config.hidden_size
    
    # Per layer total
    per_layer_params = attn_total + mlp_total + norm_params
    
    # Total transformer layers
    transformer_params = config.num_hidden_layers * per_layer_params
    
    # Total model parameters
    # Note: Output projection is tied with embedding, so not counted separately
    total_params = embedding_params + transformer_params
    
    # If embeddings are not tied, add output projection
    if not config.tie_word_embeddings:
        total_params += config.vocab_size * config.hidden_size
    
    return {
        'embedding': embedding_params,
        'attention_per_layer': attn_total,
        'mlp_per_layer': mlp_total,
        'norm_per_layer': norm_params,
        'per_layer_total': per_layer_params,
        'transformer_total': transformer_params,
        'output_projection': config.vocab_size * config.hidden_size if not config.tie_word_embeddings else 0,
        'total': total_params,
        'within_budget': total_params <= config.parameter_budget_max
    }


def estimate_memory(
    num_params: int,
    dtype: str = 'float32'
) -> Dict[str, float]:
    """
    Estimate memory usage for model parameters.

    Args:
        num_params (int): Number of parameters
        dtype (str): Data type ('float32', 'float16', 'bfloat16', 'q4')

    Returns:
        Dictionary with memory estimates in GB
    """
    # Bytes per element based on dtype
    dtype_bytes = {'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1, 'q4': 0.5}.get(dtype, 4)

    # Parameter memory
    param_memory_bytes = num_params * dtype_bytes
    param_memory_gb = param_memory_bytes / (1024 ** 3)

    # Training overhead (gradients + optimizer states)
    gradient_memory = param_memory_bytes
    adam_state_memory = 2 * param_memory_bytes
    training_memory_bytes = param_memory_bytes + gradient_memory + adam_state_memory
    training_memory_gb = training_memory_bytes / (1024 ** 3)

    return {
        'param_memory_gb': round(param_memory_gb, 2),
        'training_memory_gb': round(training_memory_gb, 2),
    }



def estimate_memory_usage(
    config: ModelConfig,
    batch_size: int = 1,
    seq_len: int = 1024,
    dtype: str = 'float32'
) -> Dict[str, float]:
    """
    Estimate memory usage for model inference.
    
    Args:
        config (ModelConfig): Model configuration
        batch_size (int): Batch size
        seq_len (int): Sequence length
        dtype (str): Data type ('float32', 'float16', 'bfloat16')
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Bytes per element based on dtype
    dtype_bytes = {'float32': 4, 'float16': 2, 'bfloat16': 2}.get(dtype, 4)
    
    # Get parameter count
    param_counts = estimate_parameters_from_config(config)
    total_params = param_counts['total']
    
    # Model weights memory
    weights_memory_mb = (total_params * dtype_bytes) / (1024 ** 2)
    
    # KV Cache memory
    # For GQA: 2 * num_kv_heads * head_dim * hidden_size * seq_len * batch_size
    # But we need to account for the repeat_interleave in attention
    kv_cache_per_token = (
        2 *  # key + value
        config.num_key_value_heads *
        config.head_dim *
        config.hidden_size // config.num_attention_heads *  # effective dim per head
        config.num_attention_heads  # after repeat_interleave
    )
    kv_cache_memory_mb = (
        kv_cache_per_token * seq_len * batch_size * dtype_bytes
    ) / (1024 ** 2)
    
    # Activation memory (rough estimate)
    # Each layer stores: hidden_states, attention outputs, mlp outputs
    activation_per_layer = (
        3 *  # multiple intermediates
        batch_size * seq_len * config.hidden_size * dtype_bytes
    )
    activation_memory_mb = (
        activation_per_layer * config.num_hidden_layers
    ) / (1024 ** 2)
    
    # Input/output buffers
    io_memory_mb = (
        2 * batch_size * seq_len * config.hidden_size * dtype_bytes
    ) / (1024 ** 2)
    
    # Total
    total_memory_mb = weights_memory_mb + kv_cache_memory_mb + activation_memory_mb + io_memory_mb
    
    return {
        'weights_mb': round(weights_memory_mb, 2),
        'kv_cache_mb': round(kv_cache_memory_mb, 2),
        'activations_mb': round(activation_memory_mb, 2),
        'io_buffers_mb': round(io_memory_mb, 2),
        'total_mb': round(total_memory_mb, 2),
        'total_gb': round(total_memory_mb / 1024, 2)
    }


def check_budget_compliance(config: ModelConfig) -> Tuple[bool, str]:
    """
    Check if configuration is within parameter budget.
    
    Args:
        config (ModelConfig): Model configuration
        
    Returns:
        Tuple of (is_within_budget, message)
    """
    estimates = estimate_parameters_from_config(config)
    
    if estimates['within_budget']:
        remaining = config.parameter_budget_max - estimates['total']
        percentage = (estimates['total'] / config.parameter_budget_max) * 100
        return True, f"Within budget: {estimates['total']:,} / {config.parameter_budget_max:,} ({percentage:.1f}%), {remaining:,} remaining"
    else:
        excess = estimates['total'] - config.parameter_budget_max
        percentage = (estimates['total'] / config.parameter_budget_max) * 100
        return False, f"Exceeds budget by {excess:,}: {estimates['total']:,} / {config.parameter_budget_max:,} ({percentage:.1f}%)"


def adjust_config_for_budget(config: ModelConfig) -> ModelConfig:
    """
    Adjust configuration to fit within parameter budget.
    
    Primarily adjusts intermediate_size to reduce parameters.
    
    Args:
        config (ModelConfig): Original configuration
        
    Returns:
        Adjusted configuration
    """
    import copy
    new_config = copy.deepcopy(config)
    
    estimates = estimate_parameters_from_config(new_config)
    
    if estimates['within_budget']:
        return new_config
    
    # Binary search for appropriate intermediate_size
    min_intermediate = 64
    max_intermediate = new_config.intermediate_size
    
    while min_intermediate < max_intermediate:
        mid = (min_intermediate + max_intermediate) // 2
        new_config.intermediate_size = mid
        
        estimates = estimate_parameters_from_config(new_config)
        
        if estimates['within_budget']:
            min_intermediate = mid + 1
        else:
            max_intermediate = mid
    
    # Set to largest valid value
    new_config.intermediate_size = min_intermediate - 1 if min_intermediate > 1 else 64
    
    # Verify final config
    final_estimates = estimate_parameters_from_config(new_config)
    
    if not final_estimates['within_budget']:
        # If still over budget, reduce hidden_size
        while not final_estimates['within_budget'] and new_config.hidden_size > 256:
            new_config.hidden_size -= 64
            new_config.intermediate_size = new_config.hidden_size * 3  # Maintain ratio
            final_estimates = estimate_parameters_from_config(new_config)
    
    return new_config


def print_parameter_summary(config: ModelConfig):
    """Print formatted parameter summary."""
    estimates = estimate_parameters_from_config(config)
    budget_ok, budget_msg = check_budget_compliance(config)
    
    print("=" * 60)
    print("Nusa Tiny LLM - Parameter Summary")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention Heads: {config.num_attention_heads}Q / {config.num_key_value_heads}KV")
    print(f"  Intermediate Size: {config.intermediate_size}")
    print(f"  Vocabulary Size: {config.vocab_size}")
    print(f"  Context Length: {config.max_position_embeddings}")
    print(f"  Tied Embeddings: {config.tie_word_embeddings}")
    print()
    print("Parameter Breakdown:")
    print(f"  Embedding:     {estimates['embedding']:>12,}")
    print(f"  Attention/Layer: {estimates['attention_per_layer']:>12,}")
    print(f"  MLP/Layer:     {estimates['mlp_per_layer']:>12,}")
    print(f"  Norm/Layer:    {estimates['norm_per_layer']:>12,}")
    print(f"  Per Layer:     {estimates['per_layer_total']:>12,}")
    print(f"  Transformer:   {estimates['transformer_total']:>12,}")
    if estimates['output_projection'] > 0:
        print(f"  Output Proj:   {estimates['output_projection']:>12,}")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:         {estimates['total']:>12,}")
    print()
    print(f"Budget Status: {budget_msg}")
    print("=" * 60)
