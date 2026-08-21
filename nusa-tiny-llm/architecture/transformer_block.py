"""
Transformer Block

Single transformer layer combining attention, MLP, and normalization.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .rmsnorm import RMSNorm
from .attention import GroupedQueryAttention
from .mlp import SwiGLU


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    
    Architecture:
        Input → RMSNorm1 → Attention → Residual → RMSNorm2 → MLP → Residual → Output
    
    Uses pre-normalization architecture for better training stability.
    
    Args:
        hidden_size (int): Size of hidden dimension
        num_attention_heads (int): Number of query heads
        num_key_value_heads (int): Number of KV heads
        head_dim (int): Dimension per head
        intermediate_size (int): MLP intermediate dimension
        max_position_embeddings (int): Maximum sequence length
        rope_theta (float): RoPE theta parameter
        dropout (float): Dropout probability
        use_bias (bool): Whether to use bias in layers
        layer_norm_epsilon (float): Normalization epsilon
    """
    
    def __init__(
        self,
        hidden_size: int = 1280,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        use_bias: bool = False,
        layer_norm_epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Pre-attention normalization
        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=layer_norm_epsilon
        )
        
        # Self-attention
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            dropout=dropout,
            use_bias=use_bias
        )
        
        # Pre-MLP normalization
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=layer_norm_epsilon
        )
        
        # MLP (SwiGLU)
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_bias=use_bias,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for transformer block.
        
        Args:
            hidden_states (torch.Tensor): Input (batch, seq_len, hidden)
            attention_mask (torch.Tensor, optional): Causal mask
            position_ids (torch.Tensor, optional): Position IDs
            past_key_value (tuple, optional): Cached KV states
            use_cache (bool): Whether to return KV cache
            
        Returns:
            Tuple of (output, past_key_value)
        """
        # Residual connection 1: Attention
        residual = hidden_states
        
        # Pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states, present_kv = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # Apply dropout
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Residual connection 2: MLP
        residual = hidden_states
        
        # Pre-norm
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        # Apply dropout
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states, present_kv


def create_attention_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create causal attention mask for a batch.
    
    Args:
        batch_size (int): Batch size
        seq_len (int): Sequence length
        device (torch.device): Device
        dtype (torch.dtype): Data type
        
    Returns:
        Attention mask tensor
    """
    from .attention import create_causal_mask
    
    mask = create_causal_mask(seq_len, device, dtype)
    
    # Expand to batch size
    # Shape: (batch, 1, 1, seq_len)
    # Mask shape is (seq_len, seq_len), need to expand properly
        # Shape: (batch, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
