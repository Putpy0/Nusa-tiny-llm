"""
Deduplication Module for Nusa Tiny LLM Data Pipeline

Removes duplicate samples from the dataset using hash-based deduplication.
"""

import hashlib
import json
from typing import List, Dict, Any, Set


class Deduplicator:
    """
    Removes duplicate samples from datasets.
    
    Uses multiple strategies:
    - Exact match deduplication
    - Normalized text deduplication
    - Fuzzy matching (optional)
    """
    
    def __init__(self, method: str = "exact"):
        """
        Initialize deduplicator.
        
        Args:
            method: Deduplication method ('exact', 'normalized')
        """
        self.method = method
        self.seen_hashes: Set[str] = set()
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        - Lowercase
        - Remove extra whitespace
        - Remove punctuation
        """
        import re
        
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _get_dedup_key(self, sample: Dict[str, Any]) -> str:
        """
        Get deduplication key from a sample.
        
        Args:
            sample: Data sample dictionary
            
        Returns:
            Hash string for deduplication
        """
        if self.method == "normalized":
            prompt = self._normalize_text(sample.get('prompt', ''))
            response = self._normalize_text(sample.get('response', ''))
        else:
            prompt = sample.get('prompt', '')
            response = sample.get('response', '')
        
        # Combine prompt and response for deduplication
        combined = f"{prompt}|||{response}"
        return self._compute_hash(combined)
    
    def deduplicate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Check if sample is duplicate and track it.
        
        Args:
            sample: Data sample
            
        Returns:
            True if sample is new (not duplicate), False if duplicate
        """
        key = self._get_dedup_key(sample)
        
        if key in self.seen_hashes:
            return False
        
        self.seen_hashes.add(key)
        return True
    
    def deduplicate_dataset(
        self,
        samples: List[Dict[str, Any]],
        reset: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicates from a dataset.
        
        Args:
            samples: List of data samples
            reset: Whether to reset seen hashes before processing
            
        Returns:
            Deduplicated list of samples
        """
        if reset:
            self.seen_hashes = set()
        
        deduplicated = []
        duplicate_count = 0
        
        for sample in samples:
            if self.deduplicate_sample(sample):
                deduplicated.append(sample)
            else:
                duplicate_count += 1
        
        print(f"Deduplication: removed {duplicate_count} duplicates, kept {len(deduplicated)} samples")
        return deduplicated
    
    def get_statistics(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            "unique_hashes": len(self.seen_hashes),
            "method": self.method
        }


def deduplicate_jsonl_file(input_path: str, output_path: str, method: str = "exact") -> int:
    """
    Deduplicate a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        method: Deduplication method
        
    Returns:
        Number of samples written
    """
    deduplicator = Deduplicator(method=method)
    
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
    
    deduplicated = deduplicator.deduplicate_dataset(samples)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in deduplicated:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return len(deduplicated)
