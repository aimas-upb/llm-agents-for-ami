"""Intent matching modules for signifier retrieval.

This package provides pluggable intent matching algorithms
with version management.
"""

from src.matching.base import IntentMatcher, MatchResult
from src.matching.registry import IntentMatcherRegistry
from src.matching.string_matcher import StringContainsMatcher
from src.matching.embedding_matcher import EmbeddingMatcher

__all__ = [
    "IntentMatcher",
    "MatchResult",
    "IntentMatcherRegistry",
    "StringContainsMatcher",
    "EmbeddingMatcher",
]
