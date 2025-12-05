"""Base interface for Intent Matcher implementations.

This module defines the abstract interface that all Intent Matcher
versions must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of intent matching for a single signifier.

    Args:
        signifier_id: Signifier identifier
        similarity: Similarity score (0.0 to 1.0)
        metadata: Optional metadata about the match
    """

    signifier_id: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = {
            "signifier_id": self.signifier_id,
            "similarity": self.similarity,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class IntentMatcher(ABC):
    """Abstract base class for Intent Matcher implementations.

    All Intent Matcher versions must implement this interface to ensure
    they can be used interchangeably in the retrieval pipeline.
    """

    def __init__(self, version: str):
        """Initialize the intent matcher.

        Args:
            version: Version identifier (e.g., 'v0', 'v1')
        """
        self.version = version
        logger.info(f"Initialized Intent Matcher {self.__class__.__name__} ({version})")

    @abstractmethod
    def match(
        self,
        intent_query: str,
        signifiers: List[Dict[str, Any]],
        k: int = 10,
        **kwargs,
    ) -> List[MatchResult]:
        """Match intent query against signifiers.

        Args:
            intent_query: Natural language intent query
            signifiers: List of signifier dictionaries
            k: Number of top results to return
            **kwargs: Additional algorithm-specific parameters

        Returns:
            List of MatchResult objects sorted by similarity (descending)

        Raises:
            ValueError: If input is invalid
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about this matcher version.

        Returns:
            Dictionary with version, name, description, parameters
        """
        pass

    def get_version(self) -> str:
        """Get the version identifier.

        Returns:
            Version string
        """
        return self.version
