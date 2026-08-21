"""
Evaluation Module for Nusa Tiny LLM

Handles model evaluation, testing, and perplexity computation.
"""

from .eval import Evaluator, load_bilingual_tests
from .perplexity import compute_perplexity

__all__ = [
    "Evaluator",
    "load_bilingual_tests",
    "compute_perplexity",
]
