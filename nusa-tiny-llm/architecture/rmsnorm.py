"""
RMSNorm - Root Mean Square Layer Normalization

Implementation of RMSNorm as used in modern transformer architectures.
More efficient than LayerNorm with comparable performance.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Unlike LayerNorm which normalizes using mean and variance,
    RMSNorm only uses the root mean square.
    
    Formula:
        y = x / sqrt(mean(x²) + ε) * γ
    
    where γ is a learnable scale parameter.
    
    Args:
        hidden_size (int): Size of the hidden dimension
        eps (float): Epsilon value for numerical stability
        elementwise_affine (bool): Whether to learn scale parameter
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Learnable scale parameter (gamma)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Normalized tensor of same shape
        """
        # Calculate RMS (root mean square)
        # x can be (batch, seq_len, hidden) or (batch, hidden)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        
        # Normalize
        x = x * torch.rsqrt(variance + self.eps)
        
        # Apply scale parameter if present
        if self.weight is not None:
            x = x * self.weight
        
        return x
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f'hidden_size={self.hidden_size}, eps={self.eps}, ' \
               f'elementwise_affine={self.elementwise_affine}'


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Functional RMSNorm implementation.
    
    Args:
        x (torch.Tensor): Input tensor
        weight (torch.Tensor): Scale parameter
        eps (float): Epsilon for stability
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    
    if weight is not None:
        x = x * weight
    
    return x
