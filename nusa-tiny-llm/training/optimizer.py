"""
Optimizer Factory for Nusa Tiny LLM

Creates and configures optimizers for training.
"""

import torch
from torch.optim import Optimizer, AdamW, SGD
from typing import Dict, Any, Optional


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    momentum: float = 0.9,
    **kwargs
) -> Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        betas: Beta parameters for Adam-style optimizers
        eps: Epsilon for numerical stability
        momentum: Momentum for SGD
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer instance
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't apply weight decay to biases and LayerNorm parameters
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=0.0,  # Adam doesn't use weight decay properly
            **kwargs
        )
    elif optimizer_type == "sgd":
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_optimizer_with_config(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> Optimizer:
    """
    Create optimizer from configuration dictionary.
    
    Args:
        model: The model to optimize
        config: Configuration dictionary
        
    Returns:
        Configured optimizer instance
    """
    return create_optimizer(
        model=model,
        optimizer_type=config.get("optimizer_type", "adamw"),
        learning_rate=config.get("learning_rate", 3e-4),
        weight_decay=config.get("weight_decay", 0.1),
        betas=tuple(config.get("betas", [0.9, 0.95])),
        eps=config.get("eps", 1e-8),
        momentum=config.get("momentum", 0.9)
    )


def get_optimizer_state(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get optimizer state for checkpointing.
    
    Args:
        optimizer: Optimizer instance
        
    Returns:
        State dictionary
    """
    return optimizer.state_dict()


def load_optimizer_state(
    optimizer: Optimizer,
    state_dict: Dict[str, Any]
):
    """
    Load optimizer state from checkpoint.
    
    Args:
        optimizer: Optimizer instance
        state_dict: State dictionary from checkpoint
    """
    optimizer.load_state_dict(state_dict)
