"""
Unit tests for intent_type validation and correction logic.

Tests that the system correctly detects when LLM infers artifact IDs
not present in user's original message and overrides intent_type to 'implicit'.
"""

import pytest
from ami_agents.agents.user_assistant.models import validate_intent_type


class TestIntentTypeValidation:
    """Tests for intent_type consistency validation."""

    def test_implicit_vague_light_no_artifact_id(self):
        """User says 'turn on the light' (vague) - should stay IMPLICIT."""
        result = validate_intent_type(
            ["turn on light"], "implicit", "turn on the light"
        )
        assert result == "implicit"

    def test_explicit_with_artifact_id_matches(self):
        """User says 'turn on lights_308' (explicit) - should stay EXPLICIT."""
        result = validate_intent_type(
            ["turn on lights_308"], "explicit", "turn on lights_308"
        )
        assert result == "explicit"

    def test_llm_inferred_artifact_id_override_to_implicit(self):
        """LLM inferred 'lights_308' when user said 'the light' - override to IMPLICIT."""
        result = validate_intent_type(
            ["turn on lights_308"], "explicit", "turn on the light"
        )
        assert result == "implicit"

    def test_vague_dark_statement_override(self):
        """User says 'it's dark' - LLM inferred lights_308 - override to IMPLICIT."""
        result = validate_intent_type(
            ["turn on lights_308"], "explicit", "it's kind of dark in here"
        )
        assert result == "implicit"

    def test_artifact_id_without_underscore_variation(self):
        """User says 'lights308' (no underscore) - should match 'lights_308'."""
        result = validate_intent_type(
            ["turn on lights_308"], "explicit", "turn on lights308"
        )
        assert result == "explicit"

    def test_artifact_id_underscore_mismatch_plural(self):
        """User says 'light308' but intent has 'lights_308' - different word, override to implicit."""
        result = validate_intent_type(
            ["turn on lights_308"], "explicit", "turn on light308"
        )
        assert result == "implicit"

    def test_multiple_intents_one_inferred(self):
        """Multiple intents, one has inferred artifact ID - override to IMPLICIT."""
        result = validate_intent_type(
            ["turn off lights_308", "set blinds_308 position to 50"],
            "explicit",
            "turn off the light and open blinds_308 to 50%",
        )
        assert result == "implicit"

    def test_no_user_message_trust_llm(self):
        """No user message available - trust LLM classification."""
        result = validate_intent_type(
            ["turn on lights_308"], "explicit", ""
        )
        assert result == "explicit"

    def test_implicit_already_correct_no_override(self):
        """LLM correctly classified as IMPLICIT - no override needed."""
        result = validate_intent_type(
            ["turn on light"], "implicit", "turn on a light"
        )
        assert result == "implicit"

    def test_case_insensitive_matching(self):
        """Artifact ID matching should be case-insensitive."""
        result = validate_intent_type(
            ["turn on LIGHTS_308"], "explicit", "turn on lights_308"
        )
        assert result == "explicit"


class TestIntentTypeValidationIntegration:
    """Integration tests for end-to-end validation."""

    def test_validation_logged_correctly(self, caplog):
        """Test that validation warnings are logged."""
        import logging
        caplog.set_level(logging.WARNING)

        result = validate_intent_type(
            ["turn on lights_308"], "explicit", "it's dark in here"
        )

        assert result == "implicit"
        assert "INTENT_TYPE MISMATCH DETECTED" in caplog.text
        assert "lights_308" in caplog.text
        assert "Overriding intent_type" in caplog.text
