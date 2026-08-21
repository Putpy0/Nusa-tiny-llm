"""
Checkpoint Manager for Nusa Tiny LLM

Handles saving and loading model checkpoints during training.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


class CheckpointManager:
    """
    Manages checkpoint saving and loading for training.
    
    Features:
    - Save model weights
    - Save optimizer state
    - Save scheduler state
    - Save training metadata
    - Keep only best N checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 3,
        save_every_steps: int = 1000
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.save_every_steps = save_every_steps
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved checkpoints
        self.saved_checkpoints: List[str] = []
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        loss: float,
        config: Dict[str, Any],
        is_best: bool = False
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            step: Current training step
            loss: Current loss value
            config: Model/training configuration
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step}"
        if is_best:
            checkpoint_name = "checkpoint_best"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Create checkpoint dictionary
        checkpoint = {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
            "config": config
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track saved checkpoints
        if checkpoint_name not in self.saved_checkpoints:
            self.saved_checkpoints.append(checkpoint_name)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save metadata
        self._save_metadata(step, loss)
        
        print(f"Saved checkpoint: {checkpoint_path} (step {step}, loss {loss:.4f})")
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self.saved_checkpoints) <= self.keep_last_n:
            return
        
        # Sort by step number
        def get_step(name):
            if name == "checkpoint_best":
                return float('inf')
            try:
                return int(name.split("_")[-2])
            except:
                return 0
        
        sorted_checkpoints = sorted(self.saved_checkpoints, key=get_step)
        
        # Remove oldest checkpoints
        to_remove = sorted_checkpoints[:-self.keep_last_n]
        for checkpoint_name in to_remove:
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            self.saved_checkpoints.remove(checkpoint_name)
    
    def _save_metadata(self, step: int, loss: float):
        """Save training metadata."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        
        metadata = {
            "last_step": step,
            "last_loss": loss,
            "total_checkpoints": len(self.saved_checkpoints)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Path to checkpoint (uses latest if None)
            
        Returns:
            Metadata from checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None:
            print("No checkpoint found")
            return {}
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        metadata = {
            "step": checkpoint.get("step", 0),
            "loss": checkpoint.get("loss", 0.0),
            "config": checkpoint.get("config", {})
        }
        
        print(f"Loaded checkpoint: {checkpoint_path} (step {metadata['step']})")
        return metadata
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*"))
        
        if not checkpoints:
            return None
        
        # Sort by step number
        def get_step(path):
            try:
                return int(path.name.split("_")[-2])
            except:
                return 0
        
        latest = max(checkpoints, key=get_step)
        return str(latest)
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return [str(p) for p in self.checkpoint_dir.glob("checkpoint_*")]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "checkpoint_best"
        if best_path.exists():
            return str(best_path)
        return None
