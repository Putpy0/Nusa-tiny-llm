"""
Nusa Tokenizer - Main tokenizer interface for Nusa Tiny LLM.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from .bpe import BPETrainer, BPETokenizer


class NusaTokenizer:
    """
    Main tokenizer class for Nusa Tiny LLM.
    
    Provides a unified interface for encoding and decoding text
    with support for special tokens and batch processing.
    
    Args:
        vocab_path (str, optional): Path to vocabulary file
        vocab_size (int): Vocabulary size
        special_tokens (dict): Special token mappings
    """
    
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vocab_size: int = 24576,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        self.vocab_size = vocab_size
        
        # Default special tokens
        self.special_tokens = special_tokens or {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<sep>': 4,
            '<user>': 5,
            '<assistant>': 6
        }
        
        # Reverse mapping
        self.special_ids_to_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Initialize BPE tokenizer
        self.bpe_tokenizer: Optional[BPETokenizer] = None
        
        if vocab_path:
            self.load(vocab_path)
        else:
            # Create empty trainer for later training
            self.trainer = BPETrainer(vocab_size=vocab_size)
    
    def load(self, vocab_path: str):
        """Load tokenizer from file."""
        path = Path(vocab_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load trainer
        self.trainer = BPETrainer.load(vocab_path)
        self.bpe_tokenizer = BPETokenizer(self.trainer)
        
        # Update special tokens if present
        if 'special_tokens' in data:
            self.special_tokens = data['special_tokens']
            self.special_ids_to_tokens = {v: k for k, v in self.special_tokens.items()}
    
    def save(self, vocab_path: str):
        """Save tokenizer to file."""
        if self.trainer:
            self.trainer.save(vocab_path)
    
    def train(self, texts: List[str], show_progress: bool = True):
        """
        Train tokenizer on text corpus.
        
        Args:
            texts: List of training texts
            show_progress: Show training progress
        """
        self.trainer = BPETrainer(
            vocab_size=self.vocab_size,
            show_progress=show_progress
        )
        self.trainer.train(texts, self.special_tokens)
        self.bpe_tokenizer = BPETokenizer(self.trainer)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            truncation: Truncate to max_length
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        if not self.bpe_tokenizer:
            raise ValueError("Tokenizer not trained or loaded")
        
        # Get token IDs
        token_ids = self.bpe_tokenizer.encode(text)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.special_tokens['<bos>']] + token_ids + [self.special_tokens['<eos>']]
        
        # Truncation
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not self.bpe_tokenizer:
            raise ValueError("Tokenizer not trained or loaded")
        
        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [
                tid for tid in token_ids 
                if tid not in self.special_tokens.values()
            ]
        
        return self.bpe_tokenizer.decode(token_ids)
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        max_length: Optional[int] = None
    ) -> Union[List[List[int]], Dict[str, List[List[int]]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts
            add_special_tokens: Add BOS/EOS tokens
            padding: Pad to max_length
            max_length: Maximum sequence length
            
        Returns:
            List of token ID lists or BatchEncoding dict
        """
        encoded = []
        max_len = 0
        
        for text in texts:
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            encoded.append(ids)
            max_len = max(max_len, len(ids))
        
        # Apply padding
        if padding and max_length:
            pad_id = self.special_tokens['<pad>']
            for i in range(len(encoded)):
                padding_length = max_length - len(encoded[i])
                if padding_length > 0:
                    encoded[i].extend([pad_id] * padding_length)
        
        return encoded
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        if self.trainer:
            return self.trainer.vocab.copy()
        return {}
    
    def vocab_size(self) -> int:
        """Get actual vocabulary size."""
        if self.trainer:
            return len(self.trainer.vocab)
        return self.vocab_size
    
    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size()
    
    @classmethod
    def from_pretrained(cls, vocab_path: str) -> 'NusaTokenizer':
        """Load pretrained tokenizer."""
        return cls(vocab_path=vocab_path)
