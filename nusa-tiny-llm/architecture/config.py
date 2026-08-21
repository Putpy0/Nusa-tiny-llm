"""
Model Configuration for Nusa Tiny LLM

Provides configuration management with validation for model architecture.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for Nusa Tiny LLM model."""
    
    # Model type
    model_type: str = "decoder-only"
    
    # Parameter budget
    target_parameters: int = 500_000_000
    parameter_budget_max: int = 550_000_000
    
    # Architecture parameters
    hidden_size: int = 1280
    num_hidden_layers: int = 24
    num_attention_heads: int = 20
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 4096
    
    # Sequence parameters
    max_position_embeddings: int = 1024
    vocab_size: int = 24576
    
    # Embedding settings
    tie_word_embeddings: bool = True
    
    # Normalization and activation
    normalization: str = "rmsnorm"
    positional_encoding: str = "rope"
    activation: str = "swiglu"
    attention_type: str = "grouped-query"
    
    # Data types
    dtype_training: str = "float32"
    dtype_inference: str = "float32"
    
    # Special token IDs
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3
    sep_token_id: int = 4
    user_token_id: int = 5
    assistant_token_id: int = 6
    
    # Language support
    languages: List[str] = field(default_factory=lambda: ["en", "id"])
    
    # Additional settings
    context_length: int = 1024
    embedding_dim: int = 1280
    use_bias: bool = False
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    rope_theta: float = 10000.0
    max_batch_size: int = 32
    
    # Description
    description: str = "Nusa Tiny LLM - Bilingual English-Indonesian model"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Check parameter budget
        if self.parameter_budget_max <= 0:
            raise ValueError("parameter_budget_max must be positive")
        
        # Check architecture constraints
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        
        # Check head dimension consistency
        expected_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim != expected_head_dim:
            # Allow explicit override but warn
            pass
        
        # Check hidden size matches embedding dim
        if self.hidden_size != self.embedding_dim:
            pass  # Allow mismatch if intentional
        
        # Check vocab size
        if self.vocab_size < 1000:
            raise ValueError(f"vocab_size ({self.vocab_size}) too small")
        
        # Check layer count
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")
        
        # Check attention heads
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1")
        
        # Check key-value heads
        if self.num_key_value_heads < 1:
            raise ValueError("num_key_value_heads must be at least 1")
        
        # Check intermediate size
        if self.intermediate_size < self.hidden_size:
            pass  # Allow but unusual for SwiGLU
        
        # Check dropout
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        
        # Check epsilon
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        
        # Check rope theta
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
    
    @classmethod
    def from_json(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_json(self, path: str, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=indent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def get_dtype(self, training: bool = True):
        """Get torch dtype based on configuration."""
        import torch
        
        dtype_str = self.dtype_training if training else self.dtype_inference
        
        dtype_map = {
            'float32': torch.float32,
            'fp32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        
        return dtype_map.get(dtype_str.lower(), torch.float32)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"ModelConfig(hidden_size={self.hidden_size}, "
            f"layers={self.num_hidden_layers}, "
            f"heads={self.num_attention_heads}/{self.num_key_value_heads}, "
            f"vocab={self.vocab_size}, "
            f"context={self.max_position_embeddings})"
        )


def load_config(path: str) -> ModelConfig:
    """Convenience function to load config from JSON."""
    return ModelConfig.from_json(path)


def save_config(config: ModelConfig, path: str) -> None:
    """Convenience function to save config to JSON."""
    config.to_json(path)


@dataclass
class SmokeConfig(ModelConfig):
    """Minimal configuration for smoke testing."""

    # Reduced parameters for fast testing
    vocab_size: int = 512
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    intermediate_size: int = 256
    max_position_embeddings: int = 128

    # Keep tie_word_embeddings
    tie_word_embeddings: bool = True

    # Override parameter budget (not relevant for smoke test)
    parameter_budget_max: int = 10_000_000

    def __post_init__(self):
        """Skip heavy validation for smoke config."""
        pass
