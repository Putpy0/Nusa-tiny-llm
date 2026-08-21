"""
Data Package

Synthetic data generation pipeline for Nusa Tiny LLM.
"""

from .synthetic_generator import SyntheticDataGenerator
from .teacher_adapter import TeacherAdapter, MockTeacherAdapter
from .filters import ContentFilter
from .dedup import Deduplicator

__all__ = [
    'SyntheticDataGenerator',
    'TeacherAdapter',
    'MockTeacherAdapter',
    'ContentFilter',
    'Deduplicator'
]
