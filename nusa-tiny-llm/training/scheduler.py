"""
Learning Rate Scheduler for Nusa Tiny LLM

Implements learning rate schedules with warmup and decay.
"""

import math
from typing import Optional, Dict, Any
import torch


class WarmupCosineSchedule:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_step: int = -1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = last_step + 1
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Apply initial learning rate
        self.step()
    
    def get_lr_multiplier(self) -> float:
        """
        Calculate the learning rate multiplier for current step.
        
        Returns:
            Multiplier between 0 and 1
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            # Cosine schedule
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Scale to [min_lr_ratio, 1.0]
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
    
    def step(self):
        """Update learning rate for current step."""
        multiplier = self.get_lr_multiplier()
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * multiplier
        
        self.current_step += 1
    
    def get_last_lr(self) -> list:
        """Get the last computed learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict.get("current_step", 0)
        # base_lrs will be recomputed from optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1
):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR ratio after decay
        
    Returns:
        Scheduler instance
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "cosine":
        return WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
    elif scheduler_type == "linear":
        return WarmupLinearSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
    elif scheduler_type == "constant":
        return ConstantSchedule(
            optimizer=optimizer
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmupLinearSchedule:
    """Learning rate scheduler with linear warmup and linear decay."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_step: int = -1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = last_step + 1
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()
    
    def get_lr_multiplier(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (1.0 - progress)
    
    def step(self):
        multiplier = self.get_lr_multiplier()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * multiplier
        self.current_step += 1
    
    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]


class ConstantSchedule:
    """Constant learning rate (no scheduling)."""
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        pass
    
    def get_last_lr(self) -> list:
        return self.base_lrs
