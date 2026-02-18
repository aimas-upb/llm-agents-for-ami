"""
Unit tests for intent_type validation and correction logic.

Tests that the system correctly detects when LLM infers artifact IDs
not present in user's original message and overrides intent_type to 'implicit'.
"""

import pytest
from ami_agents.agents.user_assistant.tools import RequestInteractionPlanTool


class TestIntentTypeValidation:
    """Tests for intent_type consistency validation."""

    def setup_method(self):
        """Create tool instance for testing."""
        self.tool = RequestInteractionPlanTool()

    def test_implicit_vague_light_no_artifact_id(self):
        """User says 'turn on the light' (vague) - should stay IMPLICIT."""
        intents = ["turn on light"]  # NO artifact ID
        intent_type = "implicit"
        user_message = "turn on the light"

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "implicit"  # Correct classification

    def test_explicit_with_artifact_id_matches(self):
        """User says 'turn on lights_308' (explicit) - should stay EXPLICIT."""
        intents = ["turn on lights_308"]  # WITH artifact ID
        intent_type = "explicit"
        user_message = "turn on lights_308"

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "explicit"  # Correct classification

    def test_llm_inferred_artifact_id_override_to_implicit(self):
        """LLM inferred 'lights_308' when user said 'the light' - override to IMPLICIT."""
        intents = ["turn on lights_308"]  # LLM inferred artifact ID!
        intent_type = "explicit"  # LLM wrongly classified as explicit
        user_message = "turn on the light"  # User didn't say "lights_308"

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "implicit"  # Overridden to implicit!

    def test_vague_dark_statement_override(self):
        """User says 'it's dark' - LLM inferred lights_308 - override to IMPLICIT."""
        intents = ["turn on lights_308"]  # LLM inferred artifact ID
        intent_type = "explicit"  # LLM wrongly classified
        user_message = "it's kind of dark in here"  # Vague statement

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "implicit"  # Overridden!

    def test_artifact_id_without_underscore_variation(self):
        """User says 'light308' (no underscore) - should match 'lights_308'."""
        intents = ["turn on lights_308"]  # With underscore
        intent_type = "explicit"
        user_message = "turn on light308"  # Without underscore

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "explicit"  # Matches! (variation handled)

    def test_multiple_intents_one_inferred(self):
        """Multiple intents, one has inferred artifact ID - override to IMPLICIT."""
        intents = [
            "turn off lights_308",  # Inferred!
            "set blinds_308 position to 50"  # User said this
        ]
        intent_type = "explicit"
        user_message = "turn off the light and open blinds_308 to 50%"

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "implicit"  # Override because lights_308 was inferred

    def test_no_user_message_trust_llm(self):
        """No user message available - trust LLM classification."""
        intents = ["turn on lights_308"]
        intent_type = "explicit"
        user_message = ""  # No message

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "explicit"  # Trust LLM when no user message

    def test_implicit_already_correct_no_override(self):
        """LLM correctly classified as IMPLICIT - no override needed."""
        intents = ["turn on light"]  # No artifact ID
        intent_type = "implicit"  # Already correct
        user_message = "turn on a light"

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "implicit"  # No change needed

    def test_case_insensitive_matching(self):
        """Artifact ID matching should be case-insensitive."""
        intents = ["turn on LIGHTS_308"]  # Uppercase
        intent_type = "explicit"
        user_message = "turn on lights_308"  # Lowercase

        result = self.tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "explicit"  # Matches (case-insensitive)


class TestIntentTypeValidationIntegration:
    """Integration tests for end-to-end validation."""

    @pytest.mark.asyncio
    async def test_validation_logged_correctly(self, caplog):
        """Test that validation warnings are logged."""
        import logging
        caplog.set_level(logging.WARNING)

        tool = RequestInteractionPlanTool()
        intents = ["turn on lights_308"]
        intent_type = "explicit"
        user_message = "it's dark in here"  # Vague, no artifact ID

        result = tool._validate_and_correct_intent_type(
            intents, intent_type, user_message
        )

        assert result == "implicit"
        # Check that warning was logged
        assert "INTENT_TYPE MISMATCH DETECTED" in caplog.text
        assert "lights_308" in caplog.text
        assert "Overriding intent_type" in caplog.text
