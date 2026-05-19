"""
Unit tests for UserAssistant data models (Intent, ConversationState, ConversationPhase).
"""

import pytest

from ami_agents.agents.user_assistant.models import (
    Intent,
    ConversationPhase,
    ConversationState,
)


class TestIntent:
    """Tests for Intent dataclass (slim version with intent_text only)."""

    def test_construct_intent(self):
        i = Intent(intent_text="turn on the light")
        assert i.intent_text == "turn on the light"

    def test_intent_text_empty(self):
        i = Intent(intent_text="")
        assert i.intent_text == ""

    def test_to_query_string(self):
        i = Intent(intent_text="turn on the light")
        assert i.to_query_string() == "turn on the light"

    def test_to_query_string_with_special_chars(self):
        i = Intent(intent_text="set the living room light to 75%")
        assert i.to_query_string() == "set the living room light to 75%"


class TestConversationPhase:
    """Tests for ConversationPhase enum."""

    def test_all_phases_exist(self):
        phases = [p.value for p in ConversationPhase]
        assert "idle" in phases
        assert "extracting_intents" in phases
        assert "awaiting_plan" in phases
        assert "summarizing_plan" in phases
        assert "awaiting_confirmation" in phases
        assert "executing" in phases

    def test_default_is_idle(self):
        conv = ConversationState()
        assert conv.phase == ConversationPhase.IDLE


class TestConversationState:
    """Tests for ConversationState dataclass."""

    def test_default_state(self):
        conv = ConversationState()
        assert conv.phase == ConversationPhase.IDLE
        assert conv.user_message == ""
        assert conv.intents == []
        assert conv.workspace_id is None
        assert conv.plan_json is None
        assert conv.plan_hash is None
        assert conv.plan_summary is None
        assert conv.is_query is False

    def test_clear_plan(self):
        from ami_agents.shared.models.intents import ImplicitGoalIntent
        conv = ConversationState(
            phase=ConversationPhase.AWAITING_CONFIRMATION,
            intents=[ImplicitGoalIntent(text_intent="turn on the light", reason="User wants to turn on light")],
            workspace_id="lab308",
            plan_json='{"tree": {}}',
            plan_hash="abc123",
            plan_summary="Turn on the light.",
        )
        conv.clear_plan()
        assert conv.plan_json is None
        assert conv.plan_hash is None
        assert conv.plan_summary is None
        assert conv.intents == []
        assert conv.workspace_id is None

    def test_clear_plan_preserves_phase_and_message(self):
        conv = ConversationState(
            phase=ConversationPhase.EXECUTING,
            user_message="turn on the light",
        )
        conv.clear_plan()
        # Phase and user_message are NOT cleared by clear_plan
        assert conv.phase == ConversationPhase.EXECUTING
        assert conv.user_message == "turn on the light"
