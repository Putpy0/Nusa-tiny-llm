"""
Training Dataset for Nusa Tiny LLM

Handles loading and tokenizing JSONL data for training.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """
    Dataset for causal language modeling training.
    
    Loads JSONL files and converts them to token sequences.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 1024,
        cache_dir: str = ".cache"
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        
        # Load or cache tokenized data
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from JSONL file."""
        samples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        return samples
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using the provided tokenizer."""
        if self.tokenizer is None:
            # Fallback: simple character-level tokenization
            return [ord(c) % 256 for c in text[:self.max_length]]
        
        encoded = self.tokenizer.encode(text)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        
        # Truncate if necessary
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        
        return encoded
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Combine prompt and response for causal LM
        text = f"{sample['prompt']} {sample['response']}"
        
        # Tokenize
        tokens = self._tokenize_text(text)
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Create attention mask (all ones for valid tokens)
        attention_mask = torch.ones_like(input_ids)
        
        # Labels are the same as input for causal LM
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.
    
    Pads sequences to the same length within a batch.
    """
    # Find max length in batch
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Pad sequences
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for item in batch:
        length = len(item["input_ids"])
        padding_length = max_length - length
        
        # Pad input_ids
        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])
        input_ids_list.append(padded_input_ids)
        
        # Pad attention mask
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(padding_length, dtype=torch.long)
        ])
        attention_mask_list.append(padded_attention_mask)
        
        # Pad labels (use -100 for padding to ignore in loss)
        padded_labels = torch.cat([
            item["labels"],
            torch.full((padding_length,), -100, dtype=torch.long)
        ])
        labels_list.append(padded_labels)
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list)
    }


def load_dataset(
    data_path: str,
    tokenizer=None,
    max_length: int = 1024,
    split: str = "train",
    val_ratio: float = 0.1
) -> Tuple[TrainingDataset, Optional[TrainingDataset]]:
    """
    Load dataset with optional train/val split.
    
    Args:
        data_path: Path to JSONL file
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        split: 'train', 'val', or 'both'
        val_ratio: Validation split ratio
        
    Returns:
        Tuple of (train_dataset, val_dataset) or single dataset
    """
    full_dataset = TrainingDataset(data_path, tokenizer, max_length)
    
    if split == "train":
        return full_dataset, None
    elif split == "val":
        # Return last portion as validation
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        return torch.utils.data.random_split(full_dataset, [train_size, val_size])
    else:
        # Return both
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        return train_dataset, val_dataset
