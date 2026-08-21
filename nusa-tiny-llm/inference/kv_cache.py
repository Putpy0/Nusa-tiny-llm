"""
KV Cache for Efficient Inference

Implements key-value caching to avoid recomputing attention keys/values
for previous tokens during generation.
"""

import torch
from typing import Optional, Tuple, List, Dict


class KVCache:
    """
    Key-Value cache for transformer inference.
    
    Stores key and value tensors for each layer to avoid
    recomputation during autoregressive generation.
    """
    
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        
        # Initialize cache tensors
        # Shape: (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.key_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        self.value_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Track current sequence length for each batch item
        self.seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_start: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.
        
        Args:
            layer_idx: Layer index
            key_states: New key states (batch, heads, seq_len, head_dim)
            value_states: New value states (batch, heads, seq_len, head_dim)
            seq_start: Starting position in sequence (for incremental decoding)
            
        Returns:
            Tuple of (cached_keys, cached_values) for the full sequence
        """
        if seq_start is None:
            # Use tracked sequence lengths
            seq_start = self.seq_lengths[0].item() if self.batch_size == 1 else 0
        
        current_len = key_states.shape[-2]  # Sequence length of new states
        
        # Store in cache
        self.key_cache[layer_idx, :, :, seq_start:seq_start + current_len] = key_states.cpu()
        self.value_cache[layer_idx, :, :, seq_start:seq_start + current_len] = value_states.cpu()
        
        # Update sequence lengths
        self.seq_lengths += current_len
        
        # Return full cached sequence up to current position
        total_len = seq_start + current_len
        cached_keys = self.key_cache[layer_idx, :, :, :total_len].to(key_states.device)
        cached_values = self.value_cache[layer_idx, :, :, :total_len].to(value_states.device)
        
        return cached_keys, cached_values
    
    def get_cached_state(
        self,
        layer_idx: int,
        up_to: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached keys and values for a layer.
        
        Args:
            layer_idx: Layer index
            up_to: Maximum sequence length to retrieve
            
        Returns:
            Tuple of (keys, values)
        """
        if up_to is None:
            up_to = self.seq_lengths.max().item()
        
        keys = self.key_cache[layer_idx, :, :, :up_to]
        values = self.value_cache[layer_idx, :, :, :up_to]
        
        return keys, values
    
    def reset(self):
        """Reset the cache."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.seq_lengths.zero_()
    
    def clear_layer(self, layer_idx: int):
        """Clear cache for a specific layer."""
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()
    
    @property
    def current_length(self) -> int:
        """Get current sequence length."""
        return self.seq_lengths.max().item()
    
    def to_dict(self) -> Dict[str, any]:
        """Export cache state as dictionary."""
        return {
            "key_cache": self.key_cache.clone(),
            "value_cache": self.value_cache.clone(),
            "seq_lengths": self.seq_lengths.clone()
        }
    
    @classmethod
    def from_dict(cls, state: Dict[str, any]) -> 'KVCache':
        """Create KVCache from dictionary state."""
        key_cache = state["key_cache"]
        value_cache = state["value_cache"]
        seq_lengths = state["seq_lengths"]
        
        num_layers, batch_size, num_kv_heads, max_seq_len, head_dim = key_cache.shape
        
        cache = cls(
            num_layers=num_layers,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=key_cache.dtype,
            device=key_cache.device.type if hasattr(key_cache.device, 'type') else "cpu"
        )
        
        cache.key_cache = key_cache
        cache.value_cache = value_cache
        cache.seq_lengths = seq_lengths
        
        return cache


def create_kv_cache_from_config(
    config: dict,
    batch_size: int = 1,
    device: str = "cpu"
) -> KVCache:
    """
    Create KVCache from model configuration.
    
    Args:
        config: Model configuration dictionary
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Initialized KVCache
    """
    return KVCache(
        num_layers=config.get("num_hidden_layers", 24),
        batch_size=batch_size,
        num_kv_heads=config.get("num_key_value_heads", 4),
        head_dim=config.get("head_dim", 64),
        max_seq_len=config.get("max_position_embeddings", 1024),
        dtype=torch.float32,
        device=device
    )
