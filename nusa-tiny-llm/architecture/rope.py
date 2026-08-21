"""
Rotary Position Embedding (RoPE)

Implementation of Rotary Position Embeddings for transformer models.
Provides relative position encoding through rotation matrices.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    
    Applies rotation to query and key vectors based on their position
    in the sequence. This encodes relative position information without
    adding learnable parameters.
    
    Args:
        dim (int): Dimension of each attention head
        max_position_embeddings (int): Maximum sequence length
        theta (float): Base for frequency calculation
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 1024,
        theta: float = 10000.0
    ):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        
        # Generate inverse frequencies
        # freqs[i] = theta^(-2i/dim) for i = 0, 1, ..., dim/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for sin/cos values
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._seq_len_cached: int = 0
    
    def _get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached or compute new cos/sin values.
        
        Args:
            seq_len (int): Sequence length
            device (torch.device): Device to use
            dtype (torch.dtype): Data type
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len > self._seq_len_cached:
            # Compute new positions
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # Outer product: t × inv_freq
            freqs = torch.outer(t, self.inv_freq)
            
            # Interleave for full dimension
            # [freq, freq, freq, ...] -> [freq, freq, freq, freq, ...]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Compute cos and sin
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
            
            # Update cache
            self._cos_cached = cos
            self._sin_cached = sin
            self._seq_len_cached = seq_len
        
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings.
        
        Args:
            x (torch.Tensor): Query or Key tensor of shape (batch, heads, seq_len, dim)
            position_ids (torch.Tensor, optional): Explicit position IDs
            
        Returns:
            Tuple of (cos, sin) for applying rotation
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Get cos/sin values
        cos, sin = self._get_cos_sin(seq_len, x.device, x.dtype)
        
        # Handle explicit position IDs
        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            # Shape: (1, 1, seq_len, dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dims by 90 degrees.
    
    Args:
        x (torch.Tensor): Input tensor with even last dimension
        
    Returns:
        Rotated tensor
    """
    # Split into two halves
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    
    # Rotate: [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Formula:
        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin
    
    Args:
        q (torch.Tensor): Query tensor (batch, heads, seq_len, dim)
        k (torch.Tensor): Key tensor (batch, heads, seq_len, dim)
        cos (torch.Tensor): Cosine values
        sin (torch.Tensor): Sine values
        
    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Expand cos/sin for element-wise multiplication
    # cos shape: (1, 1, seq_len, dim)
    
    # Apply rotation
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    
    return q_rotated, k_rotated


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embedding to a single tensor (for inference).
    
    Args:
        x (torch.Tensor): Query or Key tensor
        cos (torch.Tensor): Cosine values
        sin (torch.Tensor): Sine values
        
    Returns:
        Rotated tensor
    """
    return x * cos + rotate_half(x) * sin
