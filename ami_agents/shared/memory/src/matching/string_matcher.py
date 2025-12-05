"""String Contains Matcher (IM v0).

This module implements a simple string-based intent matching algorithm
using token containment.
"""

import logging
import re
from typing import Any, Dict, List

from src.matching.base import IntentMatcher, MatchResult

logger = logging.getLogger(__name__)


class StringContainsMatcher(IntentMatcher):
    """Intent Matcher v0 - String Contains.

    Simple string matching algorithm that searches for query tokens
    in the intent natural language text and structured JSON.
    """

    def __init__(self):
        """Initialize the String Contains Matcher."""
        super().__init__(version="v0")

    def match(
        self,
        intent_query: str,
        signifiers: List[Dict[str, Any]],
        k: int = 10,
        case_sensitive: bool = False,
        **kwargs,
    ) -> List[MatchResult]:
        """Match intent query using string containment.

        Args:
            intent_query: Natural language intent query
            signifiers: List of signifier dictionaries
            k: Number of top results to return
            case_sensitive: Whether matching should be case-sensitive
            **kwargs: Additional parameters (ignored)

        Returns:
            List of MatchResult objects sorted by similarity

        Raises:
            ValueError: If inputs are invalid
        """
        if not intent_query:
            raise ValueError("intent_query cannot be empty")

        if not signifiers:
            logger.warning("No signifiers provided for matching")
            return []

        query_tokens = self._tokenize(intent_query, case_sensitive)

        results = []
        for signifier in signifiers:
            similarity = self._compute_similarity(
                query_tokens, signifier, case_sensitive
            )

            if similarity > 0:
                results.append(
                    MatchResult(
                        signifier_id=signifier.get("signifier_id", "unknown"),
                        similarity=similarity,
                        metadata={
                            "matcher_version": self.version,
                            "matched_tokens": self._get_matched_tokens(
                                query_tokens, signifier, case_sensitive
                            ),
                        },
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)

        logger.info(
            f"String matching found {len(results)} matches, returning top {k}"
        )
        return results[:k]

    def _tokenize(self, text: str, case_sensitive: bool) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize
            case_sensitive: Whether to preserve case

        Returns:
            List of tokens
        """
        text = text if case_sensitive else text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 2]

    def _compute_similarity(
        self,
        query_tokens: List[str],
        signifier: Dict[str, Any],
        case_sensitive: bool,
    ) -> float:
        """Compute similarity score for a signifier.

        Args:
            query_tokens: Tokenized query
            signifier: Signifier dictionary
            case_sensitive: Whether matching is case-sensitive

        Returns:
            Similarity score between 0 and 1
        """
        intent = signifier.get("intent", {})
        nl_text = intent.get("nl_text", "")
        structured = intent.get("structured", {})

        nl_tokens = self._tokenize(nl_text, case_sensitive)

        structured_text = str(structured) if structured else ""
        structured_tokens = self._tokenize(structured_text, case_sensitive)

        all_signifier_tokens = set(nl_tokens + structured_tokens)

        if not all_signifier_tokens:
            return 0.0

        matches = sum(1 for token in query_tokens if token in all_signifier_tokens)

        similarity = matches / len(query_tokens) if query_tokens else 0.0

        return similarity

    def _get_matched_tokens(
        self,
        query_tokens: List[str],
        signifier: Dict[str, Any],
        case_sensitive: bool,
    ) -> List[str]:
        """Get list of matched tokens.

        Args:
            query_tokens: Tokenized query
            signifier: Signifier dictionary
            case_sensitive: Whether matching is case-sensitive

        Returns:
            List of tokens that matched
        """
        intent = signifier.get("intent", {})
        nl_text = intent.get("nl_text", "")
        structured = intent.get("structured", {})

        nl_tokens = self._tokenize(nl_text, case_sensitive)
        structured_text = str(structured) if structured else ""
        structured_tokens = self._tokenize(structured_text, case_sensitive)

        all_signifier_tokens = set(nl_tokens + structured_tokens)

        matched = [token for token in query_tokens if token in all_signifier_tokens]

        return matched

    def get_info(self) -> Dict[str, Any]:
        """Get information about this matcher.

        Returns:
            Matcher information dictionary
        """
        return {
            "version": self.version,
            "name": "String Contains Matcher",
            "description": "Simple token-based string matching algorithm",
            "parameters": {
                "case_sensitive": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether matching should be case-sensitive",
                }
            },
            "latency_budget_ms": 30,
        }
