"""Environment Variable Matcher (IM v3).

This module implements environment-variable-based matching for implicit intents.
Matches on exact equality of affected environment variables rather than semantic similarity.
"""

import logging
from typing import Any, Dict, List, Optional

from src.matching.base import IntentMatcher, MatchResult

logger = logging.getLogger(__name__)


class EnvironmentVariableMatcher(IntentMatcher):
    """Intent Matcher v3 - Environment Variable Matching.

    Matches implicit intents based on exact equality of affected environment variables.
    Used when an LLM has inferred which environment variables (temperature, luminosity, etc.)
    the user wants changed, and in which direction (increase, decrease).

    This matcher is appropriate for implicit intents where the user did not specify
    a specific device or action, but the system inferred their desire from environment context.
    """

    def __init__(self):
        """Initialize the Environment Variable Matcher."""
        super().__init__(version="v3")

    def match(
        self,
        intent_query: str,
        signifiers: List[Dict[str, Any]],
        k: int = 10,
        affected_env_vars: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[MatchResult]:
        """Match intent query using environment variable equality.

        Args:
            intent_query: Natural language intent query (used for logging only)
            signifiers: List of signifier dictionaries
            k: Number of top results to return (ignored for v3, returns all exact matches)
            affected_env_vars: List of {variable, direction} dicts inferred from the intent.
                Expected format:
                [
                    {"variable": "luminosity", "direction": "increase"},
                    {"variable": "temperature", "direction": "increase"}
                ]
            **kwargs: Additional parameters (ignored)

        Returns:
            List of MatchResult objects with similarity=1.0 for exact matches

        Raises:
            ValueError: If affected_env_vars is not provided or malformed
        """
        if not affected_env_vars:
            raise ValueError(
                "v3 (Environment Variable) matcher requires affected_env_vars parameter"
            )

        if not isinstance(affected_env_vars, list):
            raise ValueError("affected_env_vars must be a list")

        if not signifiers:
            logger.warning("No signifiers provided for matching")
            return []

        # Convert query env vars to comparable format (set of tuples)
        try:
            query_vars = {(v["variable"], v["direction"]) for v in affected_env_vars}
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Invalid affected_env_vars format: each item must have 'variable' and 'direction' keys. Error: {e}"
            )

        logger.debug(
            f"v3 matcher: looking for exact match on variables={query_vars}"
        )

        results = []
        for signifier in signifiers:
            # Extract signifier's affected_env_vars
            signifier_vars_raw = signifier.get("affected_env_vars", [])
            if not isinstance(signifier_vars_raw, list):
                logger.debug(
                    f"Skipping signifier {signifier.get('signifier_id')}: "
                    f"affected_env_vars is not a list"
                )
                continue

            # Convert signifier env vars to same format
            try:
                signifier_vars = {
                    (v.get("variable"), v.get("direction")) for v in signifier_vars_raw
                }
            except (AttributeError, TypeError) as e:
                logger.debug(
                    f"Skipping signifier {signifier.get('signifier_id')}: "
                    f"could not parse affected_env_vars: {e}"
                )
                continue

            # v3 match: exact match on affected_env_vars (both must have identical elements)
            if query_vars == signifier_vars:
                logger.debug(
                    f"v3 matcher: exact env-var match found for signifier_id={signifier.get('signifier_id')}"
                )
                results.append(
                    MatchResult(
                        signifier_id=signifier.get("signifier_id", "unknown"),
                        similarity=1.0,  # Exact match
                        metadata={
                            "matcher_version": self.version,
                            "match_type": "exact_env_var_match",
                            "matched_variables": list(query_vars),
                        },
                    )
                )

        logger.info(
            f"v3 matcher: found {len(results)} exact env-var matches for intent '{intent_query}'"
        )
        return results

    def get_version(self) -> str:
        """Get the matcher version identifier.

        Returns:
            Version string ('v3')
        """
        return self.version

    def get_info(self) -> Dict[str, Any]:
        """Get information about this matcher.

        Returns:
            Dictionary with matcher metadata
        """
        return {
            "version": self.version,
            "name": "Environment Variable Matcher",
            "description": "Matches implicit intents based on exact equality of affected environment variables",
            "input_format": {
                "affected_env_vars": "List[Dict[str, str]] with 'variable' and 'direction' keys"
            },
            "similarity_range": [1.0],  # Only exact matches (1.0) or no match
        }
