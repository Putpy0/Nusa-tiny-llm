"""
Training Loop for Nusa Tiny LLM

Main trainer class that orchestrates the training process.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


class Trainer:
    """
    Main trainer class for Nusa Tiny LLM.
    
    Handles:
    - Training loop
    - Gradient accumulation
    - Loss computation
    - Logging and metrics
    - Checkpointing
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
        checkpoint_manager: Any = None,
        device: str = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.checkpoint_manager = checkpoint_manager
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training parameters
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_steps = config.get("max_steps", 100000)
        self.logging_every_steps = config.get("logging_every_steps", 10)
        self.checkpoint_every_steps = config.get("checkpoint_every_steps", 1000)
        self.gradient_clipping = config.get("gradient_clipping", 1.0)
        
        # Loss function
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history: List[Dict[str, float]] = []
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss value
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        
        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten tokens
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on validation set.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / max(1, num_batches)
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            Training summary
        """
        # Resume from checkpoint if specified
        if resume_from and self.checkpoint_manager:
            metadata = self.checkpoint_manager.load_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                resume_from
            )
            self.global_step = metadata.get("step", 0)
            print(f"Resumed training from step {self.global_step}")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        epoch = 0
        accumulated_loss = 0.0
        
        while self.global_step < self.max_steps:
            epoch += 1
            
            for batch in self.train_dataloader:
                if self.global_step >= self.max_steps:
                    break
                
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                # Gradient accumulation
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clipping
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Logging
                    if (self.global_step + 1) % self.logging_every_steps == 0:
                        avg_loss = accumulated_loss / self.logging_every_steps
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        log_entry = {
                            "step": self.global_step + 1,
                            "loss": avg_loss,
                            "learning_rate": current_lr
                        }
                        self.training_history.append(log_entry)
                        
                        print(f"Step {self.global_step + 1}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                        
                        accumulated_loss = 0.0
                    
                    # Validation
                    if self.val_dataloader and (self.global_step + 1) % self.checkpoint_every_steps == 0:
                        val_loss = self.evaluate()
                        print(f"Validation loss at step {self.global_step + 1}: {val_loss:.4f}")
                        
                        # Save checkpoint
                        if self.checkpoint_manager:
                            is_best = val_loss < self.best_loss
                            if is_best:
                                self.best_loss = val_loss
                            
                            self.checkpoint_manager.save_checkpoint(
                                self.model,
                                self.optimizer,
                                self.scheduler,
                                self.global_step + 1,
                                val_loss,
                                self.config,
                                is_best=is_best
                            )
                
                self.global_step += 1
        
        # Final evaluation and checkpoint
        if self.val_dataloader:
            final_val_loss = self.evaluate()
            print(f"Final validation loss: {final_val_loss:.4f}")
        
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.global_step,
                self.training_history[-1]["loss"] if self.training_history else 0.0,
                self.config
            )
        
        return {
            "final_step": self.global_step,
            "best_loss": self.best_loss,
            "training_history": self.training_history
        }
    
    def save_model(self, output_path: str):
        """Save model weights without optimizer/scheduler state."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), output_dir / "model.pt")
        
        # Save config
        if self.config:
            with open(output_dir / "config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
        
        print(f"Model saved to {output_dir}")
