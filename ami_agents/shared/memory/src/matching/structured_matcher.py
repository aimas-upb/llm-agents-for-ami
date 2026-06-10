"""Structured Intent Matcher (IM v2).

This module implements a structured intent matching algorithm that matches
on intent structure (action, artifact, parameter) rather than natural language text.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.matching.base import IntentMatcher, MatchResult

logger = logging.getLogger(__name__)


class StructuredIntentMatcher(IntentMatcher):
    """Intent Matcher v2 - Structured Intent Matching.

    Matches intents based on structured components:
    - action: The action type (set, check, modify)
    - artifact: The target artifact ID (with partial matching support)
    - parameter: The parameter being manipulated (optional, for bonus scoring)
    """

    def __init__(self):
        """Initialize the Structured Intent Matcher."""
        super().__init__(version="v2")

    def match(
        self,
        intent_query: str,
        signifiers: List[Dict[str, Any]],
        k: int = 10,
        min_similarity: float = 0.0,
        query_structured_intent: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[MatchResult]:
        """Match intent query using structured intent comparison.

        Args:
            intent_query: Natural language intent query (used as fallback)
            signifiers: List of signifier dictionaries
            k: Number of top results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            query_structured_intent: Structured intent dict with keys:
                - action: str (e.g., "set", "check", "modify")
                - artifact: str (e.g., "lights_308")
                - parameter: str (optional, e.g., "on_off")
                - value: Any (optional)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of MatchResult objects sorted by similarity

        Raises:
            ValueError: If inputs are invalid
        """
        if not intent_query and not query_structured_intent:
            raise ValueError("Either intent_query or query_structured_intent must be provided")

        if not signifiers:
            logger.warning("No signifiers provided for matching")
            return []

        # If no structured intent provided, cannot do structured matching
        if not query_structured_intent or not isinstance(query_structured_intent, dict):
            logger.warning(
                "No structured_intent provided for v2 matcher, falling back to zero matches"
            )
            return []

        query_action = query_structured_intent.get("action", "").lower()
        query_artifact = query_structured_intent.get("artifact", "").lower()
        query_parameter = query_structured_intent.get("parameter", "").lower()
        query_value = query_structured_intent.get("value")

        if not query_artifact:
            logger.warning("No artifact in query structured_intent, cannot match")
            return []

        results = []
        for signifier in signifiers:
            similarity = self._compute_similarity(
                query_action,
                query_artifact,
                query_parameter,
                query_value,
                signifier,
            )

            # Filter by minimum similarity threshold
            if similarity >= min_similarity:
                results.append(
                    MatchResult(
                        signifier_id=signifier.get("signifier_id", "unknown"),
                        similarity=similarity,
                        metadata={
                            "matcher_version": self.version,
                            "matched_components": self._get_matched_components(
                                query_action, query_artifact, query_parameter, query_value, signifier
                            ),
                        },
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)

        logger.info(
            f"Structured matching found {len(results)} matches (min_similarity={min_similarity}), returning top {k}"
        )
        return results[:k]

    def _compute_similarity(
        self,
        query_action: str,
        query_artifact: str,
        query_parameter: str,
        query_value: Any,
        signifier: Dict[str, Any],
    ) -> float:
        """Compute similarity score based on structured intent components.

        Args:
            query_action: Action type from query (e.g., "set")
            query_artifact: Artifact ID from query (e.g., "lights_308")
            query_parameter: Parameter from query (e.g., "on_off")
            signifier: Signifier dictionary

        Returns:
            Similarity score between 0 and 1
        """
        # Extract structured_intent from signifier
        intent = signifier.get("intent", {})
        if not isinstance(intent, dict):
            return 0.0

        structured = intent.get("structured", {})
        if not isinstance(structured, dict):
            return 0.0

        # Try to get structured_intent (new format)
        sig_structured_intent = structured.get("structured_intent")
        if sig_structured_intent and isinstance(sig_structured_intent, dict):
            sig_action = sig_structured_intent.get("action", "").lower()
            sig_artifact = sig_structured_intent.get("artifact", "").lower()
            sig_parameter = sig_structured_intent.get("parameter", "").lower()
            sig_value = sig_structured_intent.get("value")
        else:
            # Fallback: try to extract from action_name (backward compatibility)
            sig_action = ""  # Cannot infer action from action_name reliably
            sig_artifact = ""  # Cannot infer artifact without structured_intent
            sig_parameter = ""
            sig_value = None

            # If no structured_intent, cannot do structured matching
            logger.debug(
                f"Signifier {signifier.get('signifier_id')} has no structured_intent, skipping"
            )
            return 0.0

        # Component-based scoring with weights
        action_weight = 0.25
        artifact_weight = 0.5
        parameter_weight = 0.1
        value_weight = 0.15

        # 1. Action similarity
        action_score = 1.0 if sig_action == query_action else 0.0

        # 2. Artifact similarity (with partial matching)
        artifact_score = self._compute_artifact_similarity(query_artifact, sig_artifact)

        # 3. Parameter similarity (bonus if both have same parameter)
        parameter_score = 0.0
        if query_parameter and sig_parameter:
            parameter_score = 1.0 if sig_parameter == query_parameter else 0.0

        # 4. Value similarity. For structured "set" intents, the target value is
        # semantically significant and opposite boolean values should not match.
        value_score = 0.0
        has_comparable_value = query_value is not None and sig_value is not None
        if has_comparable_value:
            value_score = 1.0 if self._normalize_value(query_value) == self._normalize_value(sig_value) else 0.0
            if query_action == "set" and value_score == 0.0:
                return 0.0

        # Weighted combination
        total_score = (
            action_weight * action_score
            + artifact_weight * artifact_score
            + parameter_weight * parameter_score
            + value_weight * value_score
        )

        # If artifact doesn't match at all, return 0 (critical component)
        if artifact_score == 0.0:
            return 0.0

        return round(total_score, 4)

    def _compute_artifact_similarity(self, query_artifact: str, sig_artifact: str) -> float:
        """Compute artifact similarity with partial matching.

        Rules:
        - Exact match: 1.0
        - Same prefix (e.g., "lights_308" vs "lights_309"): 0.7
        - Different types (e.g., "lights_308" vs "blinds_308"): 0.0

        Args:
            query_artifact: Artifact ID from query
            sig_artifact: Artifact ID from signifier

        Returns:
            Similarity score between 0 and 1
        """
        if not query_artifact or not sig_artifact:
            return 0.0

        query_artifact = self._normalize_artifact_id(query_artifact)
        sig_artifact = self._normalize_artifact_id(sig_artifact)

        # Exact match
        if query_artifact == sig_artifact:
            return 1.0

        if query_artifact in sig_artifact or sig_artifact in query_artifact:
            return 0.8

        # Extract artifact type (prefix before underscore/number)
        query_type = self._extract_artifact_type(query_artifact)
        sig_type = self._extract_artifact_type(sig_artifact)

        # Different types -> no match
        if query_type != sig_type:
            return 0.0

        # Same type, different ID -> partial match
        # This allows "lights_308" to partially match "lights_309"
        return 0.7

    def _normalize_artifact_id(self, artifact_id: str) -> str:
        """Normalize artifact labels from intent extraction and stored signifiers."""
        artifact = re.sub(r"[^a-z0-9]+", "_", str(artifact_id or "").lower()).strip("_")
        artifact = re.sub(r"(_cover)+$", "", artifact).strip("_")
        return artifact

    def _extract_artifact_type(self, artifact_id: str) -> str:
        """Extract artifact type from artifact ID.

        Examples:
        - "lights_308" -> "lights"
        - "blinds_308" -> "blinds"
        - "temperature_sensor_1" -> "temperature_sensor"

        Args:
            artifact_id: Full artifact identifier

        Returns:
            Artifact type (prefix before last underscore followed by digits)
        """
        # Remove trailing digits and underscore
        # Pattern: remove _<digits> at the end
        match = re.match(r"^(.+?)_\d+$", artifact_id)
        if match:
            return match.group(1)

        # No numeric suffix, return as-is
        return artifact_id

    def _normalize_value(self, value: Any) -> Any:
        """Normalize structured intent values for comparison."""
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return lowered
        return value

    def _get_matched_components(
        self,
        query_action: str,
        query_artifact: str,
        query_parameter: str,
        query_value: Any,
        signifier: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get information about which components matched.

        Args:
            query_action: Action from query
            query_artifact: Artifact from query
            query_parameter: Parameter from query
            signifier: Signifier dictionary

        Returns:
            Dictionary with matched component details
        """
        intent = signifier.get("intent", {})
        structured = intent.get("structured", {}) if isinstance(intent, dict) else {}
        sig_structured_intent = structured.get("structured_intent", {}) if isinstance(structured, dict) else {}

        if not isinstance(sig_structured_intent, dict):
            return {}

        sig_action = sig_structured_intent.get("action", "").lower()
        sig_artifact = sig_structured_intent.get("artifact", "").lower()
        sig_parameter = sig_structured_intent.get("parameter", "").lower()
        sig_value = sig_structured_intent.get("value")

        return {
            "action_match": sig_action == query_action,
            "artifact_match": sig_artifact == query_artifact,
            "artifact_type_match": self._extract_artifact_type(sig_artifact) == self._extract_artifact_type(query_artifact),
            "parameter_match": sig_parameter == query_parameter if query_parameter and sig_parameter else None,
            "value_match": self._normalize_value(sig_value) == self._normalize_value(query_value)
            if query_value is not None and sig_value is not None
            else None,
        }

    def get_info(self) -> Dict[str, Any]:
        """Get information about this matcher.

        Returns:
            Matcher information dictionary
        """
        return {
            "version": self.version,
            "name": "Structured Intent Matcher",
            "description": "Matches intents based on structured components (action, artifact, parameter)",
            "parameters": {
                "min_similarity": {
                    "type": "float",
                    "default": 0.0,
                    "description": "Minimum similarity threshold (0.0 to 1.0)",
                },
                "query_structured_intent": {
                    "type": "dict",
                    "required": True,
                    "description": "Structured intent with action, artifact, parameter keys",
                },
            },
            "latency_budget_ms": 10,
        }
