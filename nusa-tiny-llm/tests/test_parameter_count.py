"""
Test Parameter Count Estimation

Tests for parameter counting functionality.
"""

import pytest
import json
from pathlib import Path

# Import parameter counting module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from architecture.parameter_count import count_parameters, estimate_memory


def test_count_parameters_function_exists():
    """Test that count_parameters function exists."""
    assert callable(count_parameters)


def test_estimate_memory_function_exists():
    """Test that estimate_memory function exists."""
    assert callable(estimate_memory)


def test_smoke_config_parameter_count():
    """Test parameter count for smoke config."""
    config_path = Path(__file__).parent.parent / "configs" / "smoke_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    result = count_parameters(config)
    
    # Smoke config should have very few parameters
    assert result["total_params"] < 10000000  # Less than 10M
    assert result["total_params"] > 0


def test_full_config_within_budget():
    """Test that full model config is within parameter budget."""
    config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    result = count_parameters(config)
    
    # Should be under 550M budget
    assert result["total_params"] < 550000000
    
    # Should be close to 0.5B target (within 20%)
    assert result["total_params"] > 400000000


def test_parameter_breakdown():
    """Test that parameter breakdown is provided."""
    config_path = Path(__file__).parent.parent / "configs" / "smoke_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    result = count_parameters(config)
    
    # Should have breakdown
    assert "embedding_params" in result
    assert "layer_params" in result or "per_layer_params" in result
    assert "total_params" in result


def test_memory_estimation():
    """Test memory estimation function."""
    config_path = Path(__file__).parent.parent / "configs" / "smoke_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    param_count = count_parameters(config)["total_params"]
    
    # FP32 memory
    fp32_mem = estimate_memory(param_count, dtype="fp32")
    assert fp32_mem > 0
    
    # FP16 memory should be half of FP32
    fp16_mem = estimate_memory(param_count, dtype="fp16")
    assert fp16_mem < fp32_mem
    
    # Q4 memory should be much smaller
    q4_mem = estimate_memory(param_count, dtype="q4")
    assert q4_mem < fp16_mem


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
