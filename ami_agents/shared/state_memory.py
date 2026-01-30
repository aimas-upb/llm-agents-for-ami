"""
Simple state memory cache for UserAssistant.

Maps property_uri -> last_known_value so that state queries can be served
from cache when the environment hasn't changed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("StateMemory")


class StateMemoryCache:
    """
    In-memory cache for environment property values.

    Used by UserAssistant to:
    - Cache state query results (property_uri -> value)
    - Serve repeated state lookups without re-querying EnvExplorer
    - Update cached values after plan execution changes state
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def store(self, property_uri: str, value: Any) -> None:
        """Store a property value."""
        self._cache[property_uri] = value

    def get(self, property_uri: str, default: Any = None) -> Any:
        """Get a cached property value."""
        return self._cache.get(property_uri, default)

    def has(self, property_uri: str) -> bool:
        """Check if a property is cached."""
        return property_uri in self._cache

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def all(self) -> dict[str, Any]:
        """Return a copy of all cached values."""
        return dict(self._cache)

    def store_bulk(self, state_data: dict) -> int:
        """
        Store multiple property values from a state response.

        Accepts either:
        - Flat dict: {property_uri: value, ...}
        - Nested dict: {artifact_id: {property_uri: value, ...}, ...}

        Returns the number of properties stored.
        """
        count = 0
        for key, val in state_data.items():
            if isinstance(val, dict):
                # Nested: artifact_id -> {property_uri: value}
                for prop_uri, prop_val in val.items():
                    self._cache[prop_uri] = prop_val
                    count += 1
            else:
                self._cache[key] = val
                count += 1
        return count
