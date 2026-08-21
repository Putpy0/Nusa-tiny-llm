"""
Content Filters for Nusa Tiny LLM Data Pipeline

Filters out low-quality, harmful, or invalid content from synthetic data.
"""

import re
from typing import Dict, Any, List, Optional


class ContentFilter:
    """
    Filters content based on safety and quality criteria.
    """
    
    # Patterns that indicate potentially harmful content
    HARMFUL_PATTERNS = [
        r'\b(hate|racist|sexist|discriminatory)\b',
        r'\b(violence|kill|murder|attack)\b',
        r'\b(explicit|porn|nude|sexual)\b',
        r'\b(drug|cocaine|heroin|meth)\b',
    ]
    
    # Indonesian harmful patterns
    HARMFUL_PATTERNS_ID = [
        r'\b(benci|rasis|diskriminasi)\b',
        r'\b(kekerasan|bunuh|serang)\b',
        r'\b(eksplisit|porno|telanjang)\b',
        r'\b(narkoba|sabu|ganja)\b',
    ]
    
    def __init__(self, language: Optional[str] = None):
        self.language = language
        self.patterns_en = [re.compile(p, re.IGNORECASE) for p in self.HARMFUL_PATTERNS]
        self.patterns_id = [re.compile(p, re.IGNORECASE) for p in self.HARMFUL_PATTERNS_ID]
    
    def is_harmful(self, text: str) -> bool:
        """Check if text contains harmful content."""
        # Check English patterns
        for pattern in self.patterns_en:
            if pattern.search(text):
                return True
        
        # Check Indonesian patterns
        for pattern in self.patterns_id:
            if pattern.search(text):
                return True
        
        return False
    
    def has_valid_length(self, text: str, min_len: int = 10, max_len: int = 2000) -> bool:
        """Check if text has valid length."""
        return min_len <= len(text) <= max_len
    
    def has_valid_characters(self, text: str) -> bool:
        """Check if text has valid character composition."""
        # Too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars > len(text) * 0.3:
            return False
        
        # Too many newlines
        newline_count = text.count('\n')
        if newline_count > 20:
            return False
        
        return True
    
    def filter_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter a single data sample.
        
        Returns None if sample should be filtered out.
        """
        prompt = sample.get('prompt', '')
        response = sample.get('response', '')
        
        # Check for harmful content
        if self.is_harmful(prompt) or self.is_harmful(response):
            return None
        
        # Check length validity
        if not self.has_valid_length(prompt) or not self.has_valid_length(response):
            return None
        
        # Check character validity
        if not self.has_valid_characters(prompt) or not self.has_valid_characters(response):
            return None
        
        return sample


class QualityFilter:
    """
    Filters content based on quality metrics.
    """
    
    def __init__(self, min_quality_score: float = 0.5):
        self.min_quality_score = min_quality_score
    
    def check_repetition(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text has excessive repetition."""
        words = text.split()
        if len(words) < 10:
            return True
        
        # Check for word repetition
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        max_repetition = max(word_counts.values()) / len(words)
        return max_repetition < threshold
    
    def check_language_consistency(self, text: str, expected_lang: str) -> bool:
        """Basic check for language consistency."""
        # Simple heuristic: count common words in each language
        en_common = {'the', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        id_common = {'yang', 'dan', 'atau', 'adalah', 'dengan', 'untuk', 'dari', 'pada'}
        
        words = set(text.lower().split())
        
        en_count = len(words & en_common)
        id_count = len(words & id_common)
        
        if expected_lang == 'en':
            return en_count >= id_count
        elif expected_lang == 'id':
            return id_count >= en_count
        
        return True
    
    def filter_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter a sample based on quality metrics."""
        response = sample.get('response', '')
        lang = sample.get('lang', 'en')
        quality = sample.get('quality', 0)
        
        # Check quality score
        if quality < self.min_quality_score:
            return None
        
        # Check repetition
        if not self.check_repetition(response):
            return None
        
        # Check language consistency
        if not self.check_language_consistency(response, lang):
            return None
        
        return sample


def filter_dataset(
    samples: List[Dict[str, Any]],
    content_filter: Optional[ContentFilter] = None,
    quality_filter: Optional[QualityFilter] = None
) -> List[Dict[str, Any]]:
    """
    Apply filters to a dataset.
    
    Args:
        samples: List of data samples
        content_filter: Optional content filter
        quality_filter: Optional quality filter
        
    Returns:
        Filtered list of samples
    """
    filtered = []
    
    for sample in samples:
        # Apply content filter
        if content_filter is not None:
            sample = content_filter.filter_sample(sample)
            if sample is None:
                continue
        
        # Apply quality filter
        if quality_filter is not None:
            sample = quality_filter.filter_sample(sample)
            if sample is None:
                continue
        
        filtered.append(sample)
    
    return filtered
