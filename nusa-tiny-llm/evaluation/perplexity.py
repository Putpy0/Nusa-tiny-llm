"""
Perplexity Computation for Nusa Tiny LLM

Calculates perplexity metric on test datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import math
import torch
from torch.nn import CrossEntropyLoss


def compute_perplexity(
    model: torch.nn.Module,
    data_path: str,
    tokenizer: Any = None,
    max_length: int = 512,
    batch_size: int = 4,
    device: str = None
) -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        data_path: Path to JSONL data file
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        device: Device to run on
        
    Returns:
        Perplexity score (lower is better)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    
    # Load data
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
    
    if len(samples) == 0:
        return float('inf')
    
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    
    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        
        # Tokenize batch
        input_ids_list = []
        labels_list = []
        
        for sample in batch_samples:
            text = f"{sample.get('prompt', '')} {sample.get('response', '')}"
            
            if tokenizer is not None and hasattr(tokenizer, 'encode'):
                tokens = tokenizer.encode(text)
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
            else:
                # Fallback: character-level
                tokens = [ord(c) % 256 for c in text[:max_length]]
            
            # Truncate
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            input_ids_list.append(tokens)
            labels_list.append(tokens)
        
        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        
        input_ids = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
        labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        
        for j, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            labels[j, :len(lbls)] = torch.tensor(lbls)
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Count valid tokens
            valid_tokens = (shift_labels != -100).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    # Compute average loss and perplexity
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def compute_perplexity_from_texts(
    model: torch.nn.Module,
    texts: List[str],
    tokenizer: Any = None,
    max_length: int = 512,
    device: str = None
) -> float:
    """
    Compute perplexity on a list of raw texts.
    
    Args:
        model: Model to evaluate
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        device: Device to run on
        
    Returns:
        Perplexity score
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    
    loss_fn = CrossEntropyLoss(reduction='none')
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        if tokenizer is not None and hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
        else:
            tokens = [ord(c) % 256 for c in text[:max_length]]
        
        if len(tokens) < 2:
            continue
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_tokens = input_ids[..., 1:].contiguous()
            
            # Compute per-token loss
            losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_tokens.view(-1)
            )
            
            total_loss += losses.sum().item()
            total_tokens += len(shift_tokens[0])
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def quick_perplexity_test(
    model: torch.nn.Module,
    tokenizer: Any = None,
    device: str = None
) -> Dict[str, float]:
    """
    Quick perplexity test with dummy data.
    
    Useful for smoke testing.
    """
    dummy_texts = [
        "Hello, how are you?",
        "Halo, apa kabar?",
        "The weather is nice today.",
        "Cuaca hari ini cerah.",
        "I like to learn new things.",
        "Saya suka belajar hal baru."
    ]
    
    perplexity = compute_perplexity_from_texts(
        model=model,
        texts=dummy_texts,
        tokenizer=tokenizer,
        max_length=64,
        device=device
    )
    
    return {
        "perplexity": perplexity,
        "num_texts": len(dummy_texts),
        "status": "ok" if perplexity < 1000 else "high"
    }
