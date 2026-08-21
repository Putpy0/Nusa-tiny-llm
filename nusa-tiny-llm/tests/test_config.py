"""
Test Configuration Validation

Tests for config loading and validation.
"""

import json
import pytest
from pathlib import Path


def test_model_config_exists():
    """Test that model config file exists."""
    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    assert config_path.exists(), "model_config.json should exist"


def test_smoke_config_exists():
    """Test that smoke config file exists."""
    config_path = Path(__file__).parent.parent / "configs" / "smoke_config.json"
    assert config_path.exists(), "smoke_config.json should exist"


def test_tokenizer_config_exists():
    """Test that tokenizer config file exists."""
    config_path = Path(__file__).parent.parent / "configs" / "tokenizer_config.json"
    assert config_path.exists(), "tokenizer_config.json should exist"


def test_training_config_exists():
    """Test that training config file exists."""
    config_path = Path(__file__).parent.parent / "configs" / "training_config.json"
    assert config_path.exists(), "training_config.json should exist"


def test_inference_config_exists():
    """Test that inference config file exists."""
    config_path = Path(__file__).parent.parent / "configs" / "inference_config.json"
    assert config_path.exists(), "inference_config.json should exist"


def test_model_config_valid_json():
    """Test that model config is valid JSON."""
    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    assert "hidden_size" in config
    assert "num_hidden_layers" in config
    assert "vocab_size" in config


def test_smoke_config_small():
    """Test that smoke config has small values for testing."""
    config_path = Path(__file__).parent.parent / "configs" / "smoke_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Smoke config should be much smaller than full config
    assert config["hidden_size"] < 256
    assert config["num_hidden_layers"] < 4
    assert config["vocab_size"] < 1024


def test_config_parameters_within_budget():
    """Test that model config parameters are within budget."""
    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Rough parameter estimate
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]
    intermediate_size = config["intermediate_size"]
    
    # Embedding params
    embedding_params = vocab_size * hidden_size
    
    # Per layer params (attention + MLP)
    # Attention: 4 * hidden^2 (Q, K, V, O projections)
    # MLP: 2 * hidden * intermediate (SwiGLU)
    per_layer_params = 4 * hidden_size * hidden_size + 2 * hidden_size * intermediate_size
    
    total_params = embedding_params + num_layers * per_layer_params
    
    # Should be under 550M budget
    assert total_params < 550000000, f"Total params {total_params} exceeds budget"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
