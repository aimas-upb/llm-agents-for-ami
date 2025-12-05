"""Intent Matcher Registry for version management.

This module provides a registry for managing different Intent Matcher
versions and selecting them at runtime.
"""

import logging
from typing import Dict, List, Optional

from src.matching.base import IntentMatcher, MatchResult
from src.matching.embedding_matcher import EmbeddingMatcher
from src.matching.string_matcher import StringContainsMatcher

logger = logging.getLogger(__name__)


class IntentMatcherRegistry:
    """Registry for managing Intent Matcher versions.

    This registry allows registering multiple matcher implementations
    and selecting them by version at runtime.
    """

    def __init__(self, default_version: str = "v0"):
        """Initialize the Intent Matcher Registry.

        Args:
            default_version: Default matcher version to use
        """
        self.default_version = default_version
        self._matchers: Dict[str, IntentMatcher] = {}
        self._initialize_default_matchers()
        logger.info(
            f"Initialized Intent Matcher Registry "
            f"(default={default_version})"
        )

    def _initialize_default_matchers(self) -> None:
        """Initialize and register default matchers."""
        self.register(StringContainsMatcher())

        try:
            self.register(EmbeddingMatcher())
            logger.info("Registered EmbeddingMatcher (v1)")
        except ImportError as e:
            logger.warning(
                f"Could not register EmbeddingMatcher: {e}. "
                "Install sentence-transformers to enable embedding matching."
            )

    def register(self, matcher: IntentMatcher) -> None:
        """Register a new matcher version.

        Args:
            matcher: Intent matcher instance

        Raises:
            ValueError: If version is already registered
        """
        version = matcher.get_version()

        if version in self._matchers:
            logger.warning(f"Overwriting existing matcher version: {version}")

        self._matchers[version] = matcher
        logger.info(
            f"Registered matcher {matcher.__class__.__name__} as {version}"
        )

    def get_matcher(self, version: Optional[str] = None) -> IntentMatcher:
        """Get a matcher by version.

        Args:
            version: Matcher version (uses default if None)

        Returns:
            Intent matcher instance

        Raises:
            ValueError: If version is not registered
        """
        version_to_use = version or self.default_version

        if version_to_use not in self._matchers:
            available = list(self._matchers.keys())
            raise ValueError(
                f"Matcher version '{version_to_use}' not registered. "
                f"Available versions: {available}"
            )

        return self._matchers[version_to_use]

    def list_versions(self) -> List[str]:
        """List all registered matcher versions.

        Returns:
            List of version identifiers
        """
        return list(self._matchers.keys())

    def get_all_info(self) -> Dict[str, Dict]:
        """Get information about all registered matchers.

        Returns:
            Dictionary mapping versions to matcher info
        """
        return {
            version: matcher.get_info()
            for version, matcher in self._matchers.items()
        }

    def match(
        self,
        intent_query: str,
        signifiers: List[Dict],
        k: int = 10,
        version: Optional[str] = None,
        **kwargs,
    ) -> List[MatchResult]:
        """Match intent using specified or default matcher.

        Args:
            intent_query: Natural language intent query
            signifiers: List of signifier dictionaries
            k: Number of top results to return
            version: Matcher version to use (optional)
            **kwargs: Additional matcher-specific parameters

        Returns:
            List of MatchResult objects

        Raises:
            ValueError: If version is invalid or matching fails
        """
        matcher = self.get_matcher(version)

        logger.info(
            f"Matching with {matcher.__class__.__name__} ({matcher.version})"
        )

        results = matcher.match(intent_query, signifiers, k=k, **kwargs)

        logger.info(
            f"Matched {len(results)} signifiers using {matcher.version}"
        )

        return results

    def set_default_version(self, version: str) -> None:
        """Set the default matcher version.

        Args:
            version: Version to set as default

        Raises:
            ValueError: If version is not registered
        """
        if version not in self._matchers:
            available = list(self._matchers.keys())
            raise ValueError(
                f"Cannot set default to unregistered version '{version}'. "
                f"Available versions: {available}"
            )

        self.default_version = version
        logger.info(f"Set default matcher version to: {version}")

    def get_default_version(self) -> str:
        """Get the default matcher version.

        Returns:
            Default version identifier
        """
        return self.default_version
