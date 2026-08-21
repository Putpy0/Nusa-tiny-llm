"""
Curriculum Learning Scheduler for Nusa Tiny LLM

Implements curriculum learning by gradually increasing sequence length
and difficulty during training.
"""

from typing import List, Dict, Any, Optional
import math


class CurriculumScheduler:
    """
    Manages curriculum learning progression.
    
    Strategies:
    - Length-based: Start with shorter sequences, increase over time
    - Difficulty-based: Start with easier samples, progress to harder
    - Combined: Both length and difficulty progression
    """
    
    def __init__(
        self,
        strategy: str = "length",
        total_steps: int = 100000,
        initial_length: int = 64,
        final_length: int = 1024,
        warmup_ratio: float = 0.1
    ):
        self.strategy = strategy
        self.total_steps = total_steps
        self.initial_length = initial_length
        self.final_length = final_length
        self.warmup_steps = int(total_steps * warmup_ratio)
        
        # Current state
        self.current_step = 0
    
    def get_current_length(self, step: Optional[int] = None) -> int:
        """
        Get the current maximum sequence length for training.
        
        Args:
            step: Current training step (uses internal counter if None)
            
        Returns:
            Maximum sequence length for this step
        """
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            # During warmup, use initial length
            return self.initial_length
        
        # Progress from initial to final length
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        # Linear progression
        length = self.initial_length + (self.final_length - self.initial_length) * progress
        
        return int(length)
    
    def get_difficulty_weight(self, step: Optional[int] = None) -> float:
        """
        Get difficulty weight for sample selection.
        
        Higher values mean harder samples should be prioritized.
        
        Args:
            step: Current training step
            
        Returns:
            Difficulty weight between 0 and 1
        """
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            return 0.0
        
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return min(1.0, max(0.0, progress))
    
    def filter_by_curriculum(
        self,
        samples: List[Dict[str, Any]],
        step: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter samples based on current curriculum stage.
        
        Args:
            samples: List of data samples
            step: Current training step
            
        Returns:
            Filtered list of samples appropriate for current stage
        """
        if step is None:
            step = self.current_step
        
        current_length = self.get_current_length(step)
        difficulty_weight = self.get_difficulty_weight(step)
        
        filtered = []
        for sample in samples:
            prompt_len = len(sample.get('prompt', ''))
            response_len = len(sample.get('response', ''))
            total_len = prompt_len + response_len
            
            # Filter by length
            if total_len > current_length * 4:  # Rough character to token ratio
                continue
            
            # Optionally filter by quality/difficulty
            quality = sample.get('quality', 0.5)
            
            if self.strategy == "difficulty":
                # Early stages: prefer high-quality easy samples
                # Later stages: include more diverse samples
                if quality < difficulty_weight * 0.5:
                    continue
            
            filtered.append(sample)
        
        return filtered
    
    def step(self):
        """Increment the current step counter."""
        self.current_step += 1
    
    def set_step(self, step: int):
        """Set the current step explicitly."""
        self.current_step = step
    
    def get_status(self) -> Dict[str, Any]:
        """Get current curriculum status."""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.current_step / self.total_steps,
            "current_length": self.get_current_length(),
            "difficulty_weight": self.get_difficulty_weight(),
            "strategy": self.strategy
        }


def create_curriculum_scheduler(
    config: Dict[str, Any]
) -> CurriculumScheduler:
    """
    Create a curriculum scheduler from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CurriculumScheduler instance
    """
    return CurriculumScheduler(
        strategy=config.get("curriculum_strategy", "length"),
        total_steps=config.get("max_steps", 100000),
        initial_length=config.get("initial_sequence_length", 64),
        final_length=config.get("max_position_embeddings", 1024),
        warmup_ratio=config.get("warmup_ratio", 0.1)
    )
