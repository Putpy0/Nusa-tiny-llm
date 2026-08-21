"""
Grouped-Query Attention (GQA)

Implementation of Grouped-Query Attention for efficient inference.
Multiple query heads share the same key-value heads to reduce memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rope import apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention implementation.
    
    In GQA, multiple query heads share the same key and value heads.
    This reduces the KV cache size during inference while maintaining
    most of the quality of multi-head attention.
    
    Args:
        hidden_size (int): Size of hidden dimension
        num_heads (int): Number of query heads
        num_kv_heads (int): Number of key-value heads (must divide num_heads)
        head_dim (int): Dimension of each head
        max_position_embeddings (int): Maximum sequence length
        rope_theta (float): RoPE theta parameter
        dropout (float): Attention dropout
        use_bias (bool): Whether to use bias in projections
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        use_bias: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_bias = use_bias
        
        # Check divisibility
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )
        
        self.num_groups = num_heads // num_kv_heads
        
        # Query projection (all heads)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=use_bias)
        
        # Key and Value projections (KV heads only)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=use_bias)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)
        
        # Rotary embeddings
        from .rope import RotaryEmbedding
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            theta=rope_theta
        )
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def _reshape_for_attention(
        self,
        x: torch.Tensor,
        num_heads: int,
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, hidden)
            num_heads (int): Number of heads
            batch_size (int): Batch size
            seq_len (int): Sequence length
            
        Returns:
            Reshaped tensor (batch, num_heads, seq_len, head_dim)
        """
        return x.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for grouped-query attention.
        
        Args:
            hidden_states (torch.Tensor): Input tensor (batch, seq_len, hidden)
            attention_mask (torch.Tensor, optional): Causal mask
            position_ids (torch.Tensor, optional): Position IDs for RoPE
            past_key_value (tuple, optional): Cached KV states
            use_cache (bool): Whether to return KV cache
            
        Returns:
            Tuple of (output, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = self._reshape_for_attention(
            query_states, self.num_heads, batch_size, seq_len
        )
        key_states = self._reshape_for_attention(
            key_states, self.num_kv_heads, batch_size, seq_len
        )
        value_states = self._reshape_for_attention(
            value_states, self.num_kv_heads, batch_size, seq_len
        )
        
        # Apply RoPE
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        
        # Handle KV cache for inference
        if past_key_value is not None:
            # Concatenate with cached keys/values
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_kv = None
        if use_cache:
            past_kv = (key_states, value_states)
        
        # Expand KV heads for each query group
        # Shape: (batch, num_heads, seq_len, head_dim)
        if self.num_kv_heads < self.num_heads:
            # Repeat KV heads for each query group
            key_states = key_states.repeat_interleave(self.num_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_groups, dim=1)
        
        # Compute attention scores
        # Scale factor
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale
        
        # Apply causal mask
        if attention_mask is not None:
            # attention_mask shape: (1, 1, seq_len, seq_len) or similar
            # -inf for positions that should be masked
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)
        
        # Apply dropout
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to (batch, seq_len, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, past_kv


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a causal attention mask.
    
    Args:
        seq_len (int): Sequence length
        device (torch.device): Device
        dtype (torch.dtype): Data type
        
    Returns:
        Causal mask tensor
    """
    # Create upper triangular mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype),
        diagonal=1
    )
    
    # Convert to -inf mask
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    # Reshape for broadcasting: (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(0)
