"""
Test Model Forward Pass

Tests for model architecture forward pass.
"""

import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from architecture.config import ModelConfig, SmokeConfig
from architecture.model import NusaTinyLLM


def test_model_creation():
    """Test that model can be created."""
    config = SmokeConfig()
    model = NusaTinyLLM(config)
    
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward_pass():
    """Test forward pass with dummy input."""
    config = SmokeConfig()
    model = NusaTinyLLM(config)
    model.eval()
    
    # Create dummy input (batch_size=1, seq_len=16)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    
    # Check output shape
    assert outputs.logits.shape == (1, 16, config.vocab_size)


def test_model_with_attention_mask():
    """Test forward pass with attention mask."""
    config = SmokeConfig()
    model = NusaTinyLLM(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    assert outputs.logits.shape == (1, 16, config.vocab_size)


def test_model_parameter_count():
    """Test that model has expected number of parameters."""
    config = SmokeConfig()
    model = NusaTinyLLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Smoke model should have relatively few parameters
    assert total_params < 10000000  # Less than 10M
    assert total_params > 100000   # More than 100K


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    config = SmokeConfig()
    model = NusaTinyLLM(config)
    model.train()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    # Backward pass
    loss.backward()
    
    # Check that some parameters have gradients
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have non-zero gradients"


def test_model_generation():
    """Test token generation."""
    config = SmokeConfig()
    model = NusaTinyLLM(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
    
    assert next_token.shape == (1,)
    assert next_token.item() >= 0
    assert next_token.item() < config.vocab_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
