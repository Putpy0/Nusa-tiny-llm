"""
SwiGLU - Swish Gated Linear Unit

Implementation of SwiGLU activation for transformer feed-forward networks.
Provides better performance than ReLU/GeLU in language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU).
    
    Combines Swish activation with gating mechanism:
        SwiGLU(x) = Swish(xW₁ + b₁) ⊗ (xW₂ + b₂)
    
    where ⊗ is element-wise multiplication.
    
    This implementation uses two parallel projections followed by
    gated activation and output projection.
    
    Args:
        hidden_size (int): Input/output dimension
        intermediate_size (int): Hidden dimension in MLP
        use_bias (bool): Whether to use bias in linear layers
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_bias = use_bias
        
        # Gate projection (W₁)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        
        # Value projection (W₂)
        self.value_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        
        # Output projection (W₃)
        self.output_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU.
        
        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor of same shape as input
        """
        # Project through gate and value paths
        gate = self.gate_proj(x)
        value = self.value_proj(x)
        
        # Apply Swish activation to gate
        # Swish(x) = x * sigmoid(x)
        gate_activated = F.silu(gate)
        
        # Element-wise multiplication (gating)
        hidden = gate_activated * value
        
        # Apply dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        
        # Output projection
        output = self.output_proj(hidden)
        
        return output


def swiglu_functional(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    gate_bias: torch.Tensor = None,
    value_bias: torch.Tensor = None,
    output_bias: torch.Tensor = None,
    dropout_p: float = 0.0,
    training: bool = False
) -> torch.Tensor:
    """
    Functional SwiGLU implementation.
    
    Args:
        x (torch.Tensor): Input tensor
        gate_weight (torch.Tensor): Gate projection weight
        value_weight (torch.Tensor): Value projection weight
        output_weight (torch.Tensor): Output projection weight
        gate_bias (torch.Tensor, optional): Gate bias
        value_bias (torch.Tensor, optional): Value bias
        output_bias (torch.Tensor, optional): Output bias
        dropout_p (float): Dropout probability
        training (bool): Training mode flag
        
    Returns:
        Output tensor
    """
    # Linear projections
    gate = F.linear(x, gate_weight, gate_bias)
    value = F.linear(x, value_weight, value_bias)
    
    # Swish activation on gate
    gate_activated = F.silu(gate)
    
    # Gating
    hidden = gate_activated * value
    
    # Dropout
    if training and dropout_p > 0:
        hidden = F.dropout(hidden, p=dropout_p)
    
    # Output projection
    output = F.linear(hidden, output_weight, output_bias)
    
    return output


class SwiGLUBlock(nn.Module):
    """
    Alternative SwiGLU implementation with single intermediate representation.
    
    More memory efficient but slightly different computation pattern.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Combined projection for gate and value
        self.gate_value_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=use_bias
        )
        self.output_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Project to combined gate+value
        combined = self.gate_value_proj(x)
        
        # Split into gate and value
        gate, value = combined.chunk(2, dim=-1)
        
        # Apply Swish and multiply
        hidden = F.silu(gate) * value
        
        # Dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        
        # Output projection
        return self.output_proj(hidden)
