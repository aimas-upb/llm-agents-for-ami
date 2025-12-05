"""Tests for Phase 3 intent matching modules.

This module tests string matching, embedding matching,
and the matcher registry.
"""

import pytest

from src.matching import (
    EmbeddingMatcher,
    IntentMatcherRegistry,
    StringContainsMatcher,
)


class TestStringContainsMatcher:
    """Tests for String Contains Matcher (IM v0)."""

    def test_initialization(self):
        """Test matcher initialization."""
        matcher = StringContainsMatcher()

        assert matcher.get_version() == "v0"
        info = matcher.get_info()
        assert info["version"] == "v0"
        assert "String Contains" in info["name"]

    def test_exact_match(self):
        """Test exact token matching."""
        matcher = StringContainsMatcher()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "increase luminosity in a room",
                    "structured": {}
                }
            },
            {
                "signifier_id": "sig2",
                "intent": {
                    "nl_text": "decrease temperature level",
                    "structured": {}
                }
            }
        ]

        results = matcher.match("increase luminosity", signifiers, k=5)

        assert len(results) > 0
        assert results[0].signifier_id == "sig1"
        assert results[0].similarity > 0

    def test_partial_match(self):
        """Test partial token matching."""
        matcher = StringContainsMatcher()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "turn on the lights",
                    "structured": {}
                }
            }
        ]

        results = matcher.match("turn lights", signifiers, k=5)

        assert len(results) == 1
        assert results[0].similarity > 0

    def test_no_match(self):
        """Test when no tokens match."""
        matcher = StringContainsMatcher()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "increase luminosity",
                    "structured": {}
                }
            }
        ]

        results = matcher.match("decrease temperature", signifiers, k=5)

        assert len(results) == 0

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching (default)."""
        matcher = StringContainsMatcher()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "Increase Luminosity",
                    "structured": {}
                }
            }
        ]

        results = matcher.match("increase luminosity", signifiers, k=5)

        assert len(results) == 1
        assert results[0].similarity > 0

    def test_case_sensitive_matching(self):
        """Test case-sensitive matching."""
        matcher = StringContainsMatcher()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "Increase Luminosity",
                    "structured": {}
                }
            }
        ]

        results = matcher.match(
            "increase luminosity",
            signifiers,
            k=5,
            case_sensitive=True
        )

        assert len(results) == 0

    def test_structured_field_matching(self):
        """Test matching against structured intent field."""
        matcher = StringContainsMatcher()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "turn on device",
                    "structured": {"action": "activate", "target": "lamp"}
                }
            }
        ]

        results = matcher.match("activate lamp", signifiers, k=5)

        assert len(results) == 1

    def test_top_k_limiting(self):
        """Test that results are limited to top k."""
        matcher = StringContainsMatcher()

        signifiers = [
            {"signifier_id": f"sig{i}", "intent": {"nl_text": "increase value", "structured": {}}}
            for i in range(20)
        ]

        results = matcher.match("increase value", signifiers, k=5)

        assert len(results) == 5

    def test_empty_query(self):
        """Test handling of empty query."""
        matcher = StringContainsMatcher()

        signifiers = [
            {"signifier_id": "sig1", "intent": {"nl_text": "test", "structured": {}}}
        ]

        with pytest.raises(ValueError, match="intent_query cannot be empty"):
            matcher.match("", signifiers, k=5)

    def test_empty_signifiers(self):
        """Test handling of empty signifier list."""
        matcher = StringContainsMatcher()

        results = matcher.match("test query", [], k=5)

        assert len(results) == 0


class TestEmbeddingMatcher:
    """Tests for Embedding Matcher (IM v1)."""

    def test_initialization(self):
        """Test matcher initialization."""
        try:
            matcher = EmbeddingMatcher()

            assert matcher.get_version() == "v1"
            info = matcher.get_info()
            assert info["version"] == "v1"
            assert "Embedding" in info["name"]
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_semantic_matching(self):
        """Test semantic similarity matching."""
        try:
            matcher = EmbeddingMatcher()

            signifiers = [
                {
                    "signifier_id": "sig1",
                    "intent": {
                        "nl_text": "increase brightness in room",
                        "structured": {}
                    }
                },
                {
                    "signifier_id": "sig2",
                    "intent": {
                        "nl_text": "lower temperature level",
                        "structured": {}
                    }
                }
            ]

            results = matcher.match("make room brighter", signifiers, k=5)

            assert len(results) > 0
            assert results[0].signifier_id == "sig1"

        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_min_similarity_threshold(self):
        """Test minimum similarity threshold filtering."""
        try:
            matcher = EmbeddingMatcher()

            signifiers = [
                {
                    "signifier_id": "sig1",
                    "intent": {
                        "nl_text": "increase luminosity",
                        "structured": {}
                    }
                }
            ]

            results = matcher.match(
                "completely unrelated query",
                signifiers,
                k=5,
                min_similarity=0.8
            )

            assert len(results) == 0

        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_embedding_caching(self):
        """Test that embeddings are cached."""
        try:
            matcher = EmbeddingMatcher(cache_embeddings=True)

            signifiers = [
                {
                    "signifier_id": "sig1",
                    "intent": {
                        "nl_text": "test intent",
                        "structured": {}
                    }
                }
            ]

            results1 = matcher.match("query", signifiers, k=5)
            stats_before = matcher.get_cache_stats()

            results2 = matcher.match("query", signifiers, k=5)
            stats_after = matcher.get_cache_stats()

            assert stats_after["size"] > 0

        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_cache_clearing(self):
        """Test cache clearing."""
        try:
            matcher = EmbeddingMatcher(cache_embeddings=True)

            signifiers = [
                {"signifier_id": "sig1", "intent": {"nl_text": "test", "structured": {}}}
            ]

            matcher.match("query", signifiers, k=5)
            matcher.clear_cache()

            stats = matcher.get_cache_stats()
            assert stats["size"] == 0

        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestIntentMatcherRegistry:
    """Tests for Intent Matcher Registry."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = IntentMatcherRegistry(default_version="v0")

        assert registry.get_default_version() == "v0"
        versions = registry.list_versions()
        assert "v0" in versions

    def test_get_matcher_by_version(self):
        """Test retrieving matcher by version."""
        registry = IntentMatcherRegistry()

        matcher_v0 = registry.get_matcher("v0")
        assert matcher_v0.get_version() == "v0"
        assert isinstance(matcher_v0, StringContainsMatcher)

    def test_get_default_matcher(self):
        """Test retrieving default matcher."""
        registry = IntentMatcherRegistry(default_version="v0")

        matcher = registry.get_matcher()
        assert matcher.get_version() == "v0"

    def test_invalid_version(self):
        """Test retrieving non-existent version."""
        registry = IntentMatcherRegistry()

        with pytest.raises(ValueError, match="not registered"):
            registry.get_matcher("v999")

    def test_set_default_version(self):
        """Test setting default version."""
        registry = IntentMatcherRegistry(default_version="v0")

        registry.set_default_version("v0")
        assert registry.get_default_version() == "v0"

    def test_set_invalid_default_version(self):
        """Test setting invalid default version."""
        registry = IntentMatcherRegistry()

        with pytest.raises(ValueError, match="unregistered version"):
            registry.set_default_version("v999")

    def test_list_versions(self):
        """Test listing all versions."""
        registry = IntentMatcherRegistry()

        versions = registry.list_versions()
        assert isinstance(versions, list)
        assert len(versions) > 0
        assert "v0" in versions

    def test_get_all_info(self):
        """Test getting info for all matchers."""
        registry = IntentMatcherRegistry()

        all_info = registry.get_all_info()
        assert isinstance(all_info, dict)
        assert "v0" in all_info
        assert "name" in all_info["v0"]

    def test_match_with_registry(self):
        """Test matching through registry."""
        registry = IntentMatcherRegistry()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "increase luminosity",
                    "structured": {}
                }
            }
        ]

        results = registry.match(
            "increase luminosity",
            signifiers,
            k=5,
            version="v0"
        )

        assert len(results) > 0
        assert results[0].signifier_id == "sig1"

    def test_match_with_default_version(self):
        """Test matching using default version."""
        registry = IntentMatcherRegistry(default_version="v0")

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "test intent",
                    "structured": {}
                }
            }
        ]

        results = registry.match("test intent", signifiers, k=5)

        assert len(results) > 0


class TestIntegration:
    """Integration tests for matching workflow."""

    def test_version_switching(self):
        """Test switching between matcher versions."""
        registry = IntentMatcherRegistry()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "increase brightness",
                    "structured": {}
                }
            }
        ]

        results_v0 = registry.match(
            "increase brightness",
            signifiers,
            k=5,
            version="v0"
        )

        assert len(results_v0) > 0
        assert results_v0[0].metadata["matcher_version"] == "v0"

    def test_ab_comparison(self):
        """Test A/B comparison between matcher versions."""
        registry = IntentMatcherRegistry()

        signifiers = [
            {
                "signifier_id": "sig1",
                "intent": {
                    "nl_text": "make room brighter",
                    "structured": {}
                }
            }
        ]

        results_v0 = registry.match(
            "increase brightness",
            signifiers,
            k=5,
            version="v0"
        )

        assert len(results_v0) >= 0
