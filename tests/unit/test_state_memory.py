"""
Unit tests for UserAssistant state memory cache.

Tests the StateMemoryCache from ami_agents.shared.state_memory.
"""

import pytest

from ami_agents.shared.state_memory import StateMemoryCache


class TestStateMemoryCache:
    """Tests for the state memory cache."""

    def test_cache_stores_value(self):
        cache = StateMemoryCache()
        cache.store("http://localhost/props/state", "on")
        assert cache.get("http://localhost/props/state") == "on"

    def test_cache_returns_cached(self):
        cache = StateMemoryCache()
        cache.store("http://localhost/props/brightness", 75)
        # Multiple gets return the same value
        assert cache.get("http://localhost/props/brightness") == 75
        assert cache.get("http://localhost/props/brightness") == 75

    def test_cache_updates_on_new_store(self):
        cache = StateMemoryCache()
        cache.store("http://localhost/props/state", "off")
        assert cache.get("http://localhost/props/state") == "off"

        cache.store("http://localhost/props/state", "on")
        assert cache.get("http://localhost/props/state") == "on"

    def test_cache_returns_default_for_unknown(self):
        cache = StateMemoryCache()
        assert cache.get("http://localhost/props/unknown") is None
        assert cache.get("http://localhost/props/unknown", "default") == "default"

    def test_cache_has(self):
        cache = StateMemoryCache()
        assert cache.has("http://localhost/props/state") is False
        cache.store("http://localhost/props/state", "on")
        assert cache.has("http://localhost/props/state") is True

    def test_cache_clear(self):
        cache = StateMemoryCache()
        cache.store("http://localhost/props/a", 1)
        cache.store("http://localhost/props/b", 2)
        assert len(cache.all()) == 2

        cache.clear()
        assert len(cache.all()) == 0

    def test_cache_all(self):
        cache = StateMemoryCache()
        cache.store("http://localhost/props/a", 1)
        cache.store("http://localhost/props/b", 2)
        all_vals = cache.all()
        assert all_vals == {
            "http://localhost/props/a": 1,
            "http://localhost/props/b": 2,
        }

    def test_store_bulk_flat(self):
        cache = StateMemoryCache()
        count = cache.store_bulk({
            "http://localhost/props/state": "on",
            "http://localhost/props/brightness": 75,
        })
        assert count == 2
        assert cache.get("http://localhost/props/state") == "on"
        assert cache.get("http://localhost/props/brightness") == 75

    def test_store_bulk_nested(self):
        cache = StateMemoryCache()
        count = cache.store_bulk({
            "light308": {
                "http://localhost/props/state": "on",
                "http://localhost/props/brightness": 80,
            },
            "blinds308": {
                "http://localhost/props/position": 50,
            },
        })
        assert count == 3
        assert cache.get("http://localhost/props/state") == "on"
        assert cache.get("http://localhost/props/brightness") == 80
        assert cache.get("http://localhost/props/position") == 50

    def test_store_bulk_empty(self):
        cache = StateMemoryCache()
        count = cache.store_bulk({})
        assert count == 0
        assert len(cache.all()) == 0
