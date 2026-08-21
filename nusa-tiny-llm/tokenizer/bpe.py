"""
Byte Pair Encoding (BPE) Implementation

Custom BPE tokenizer trained from scratch without external dependencies.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re


class BPETrainer:
    """
    Train Byte Pair Encoding tokenizer.
    
    Implements byte-level BPE for robust tokenization of any text.
    """
    
    def __init__(
        self,
        vocab_size: int = 24576,
        min_frequency: int = 2,
        show_progress: bool = True
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.show_progress = show_progress
        
        # Vocabulary and merges
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # Special tokens
        self.special_tokens: Dict[str, int] = {}
    
    def _get_stats(
        self,
        corpus: List[List[str]]
    ) -> Dict[Tuple[str, str], int]:
        """Get pair frequencies in corpus."""
        pairs = defaultdict(int)
        for words in corpus:
            for word in words:
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs
    
    def _merge_corpus(
        self,
        corpus: List[List[str]],
        pair: Tuple[str, str]
    ) -> List[List[str]]:
        """Merge a pair in all words of corpus."""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        new_corpus = []
        for words in corpus:
            new_words = []
            text = ' '.join(words)
            text = text.replace(bigram, replacement)
            new_words = text.split()
            new_corpus.append(new_words)
        
        return new_corpus
    
    def train(
        self,
        texts: List[str],
        special_tokens: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Train BPE on text corpus.
        
        Args:
            texts: List of text strings
            special_tokens: Dictionary of special tokens
            
        Returns:
            Final vocabulary
        """
        # Set special tokens
        if special_tokens:
            self.special_tokens = special_tokens
            self.vocab = special_tokens.copy()
        else:
            self.special_tokens = {}
            self.vocab = {}
        
        # Add byte fallback tokens (256 bytes)
        for i in range(256):
            byte_token = f"<byte_{i:02x}>"
            self.vocab[byte_token] = len(self.vocab)
        
        # Preprocess texts into character-level tokens
        corpus = []
        for text in texts:
            # Convert to bytes then to our representation
            words = text.split()
            word_symbols = []
            for word in words:
                # Each character as separate symbol initially
                chars = list(word)
                word_symbols.append(' '.join(chars))
            corpus.append(word_symbols)
        
        # Count initial characters
        char_counts = defaultdict(int)
        for words in corpus:
            for word in words:
                for char in word.split():
                    char_counts[char] += 1
        
        # Add frequent characters to vocab
        for char, count in sorted(char_counts.items(), key=lambda x: -x[1]):
            if char not in self.vocab and count >= self.min_frequency:
                if len(self.vocab) < self.vocab_size:
                    self.vocab[char] = len(self.vocab)
        
        # BPE merge loop
        while len(self.vocab) < self.vocab_size:
            # Get pair statistics
            pairs = self._get_stats(corpus)
            
            if not pairs:
                break
            
            # Find best pair
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < self.min_frequency:
                break
            
            # Merge the pair
            corpus = self._merge_corpus(corpus, best_pair)
            self.merges.append(best_pair)
            
            # Add merged token to vocab
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
            
            if self.show_progress and len(self.vocab) % 1000 == 0:
                print(f"Vocab size: {len(self.vocab)}")
        
        return self.vocab
    
    def save(self, path: str):
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETrainer':
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trainer = cls(
            vocab_size=data.get('vocab_size', 24576),
            min_frequency=data.get('min_frequency', 2)
        )
        trainer.vocab = data['vocab']
        trainer.merges = data['merges']
        trainer.special_tokens = data.get('special_tokens', {})
        
        return trainer


class BPETokenizer:
    """
    BPE Tokenizer for encoding and decoding text.
    """
    
    def __init__(self, trainer: BPETrainer):
        self.trainer = trainer
        self.vocab = trainer.vocab
        
        # Create reverse vocab (ID to token)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Build merge rules for fast encoding
        self.merge_ranks = {
            merge: i for i, merge in enumerate(trainer.merges)
        }
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Split into words
        words = text.split()
        tokens = []
        
        for word in words:
            # Convert word to character sequence
            word_tokens = list(word)
            
            # Apply merges greedily
            while len(word_tokens) > 1:
                # Find best merge
                best_score = float('inf')
                best_idx = -1
                
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    rank = self.merge_ranks.get(pair, float('inf'))
                    
                    if rank < best_score:
                        best_score = rank
                        best_idx = i
                
                if best_idx == -1:
                    break
                
                # Perform merge
                merged = word_tokens[best_idx] + word_tokens[best_idx + 1]
                word_tokens[best_idx] = merged
                del word_tokens[best_idx + 1]
            
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Fallback to byte representation
                    for char in token:
                        byte_val = ord(char)
                        byte_token = f"<byte_{byte_val:02x}>"
                        if byte_token in self.vocab:
                            tokens.append(self.vocab[byte_token])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '')
            
            # Handle byte tokens
            if token.startswith('<byte_') and token.endswith('>'):
                hex_val = token[6:-1]
                try:
                    char = chr(int(hex_val, 16))
                    tokens.append(char)
                except:
                    pass
            else:
                tokens.append(token)
        
        # Join and clean up
        text = ''.join(tokens)
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
