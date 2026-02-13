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
    """Tests for Intent dataclass and canonical string conversion."""

    def test_turn_on(self):
        i = Intent(action="turn_on", artifact="light308")
        assert i.to_canonical_string() == "turn on light308"

    def test_turn_off(self):
        i = Intent(action="turn_off", artifact="blinds308")
        assert i.to_canonical_string() == "turn off blinds308"

    def test_set_with_parameter(self):
        i = Intent(action="set", artifact="light308", parameter="brightness", value=75)
        assert i.to_canonical_string() == "set light308 brightness to 75"

    def test_set_with_string_value(self):
        i = Intent(action="set", artifact="light308", parameter="color", value="warm_white")
        assert i.to_canonical_string() == "set light308 color to warm_white"

    def test_check_status(self):
        i = Intent(action="check_status", artifact="light308")
        assert i.to_canonical_string() == "check status of light308"

    def test_fallback_for_unknown_action(self):
        i = Intent(action="toggle", artifact="light308")
        result = i.to_canonical_string()
        assert "toggle" in result
        assert "light308" in result

    def test_to_dict(self):
        i = Intent(action="set", artifact="light308", parameter="brightness", value=100)
        d = i.to_dict()
        assert d == {
            "action": "set",
            "artifact": "light308",
            "parameter": "brightness",
            "value": 100,
        }

    def test_to_dict_minimal(self):
        i = Intent(action="turn_on", artifact="light308")
        d = i.to_dict()
        assert d == {"action": "turn_on", "artifact": "light308"}
        assert "parameter" not in d
        assert "value" not in d

    def test_from_dict(self):
        d = {"action": "set", "artifact": "light308", "parameter": "brightness", "value": 75}
        i = Intent.from_dict(d)
        assert i.action == "set"
        assert i.artifact == "light308"
        assert i.parameter == "brightness"
        assert i.value == 75

    def test_from_dict_minimal(self):
        d = {"action": "turn_on", "artifact": "light308"}
        i = Intent.from_dict(d)
        assert i.action == "turn_on"
        assert i.artifact == "light308"
        assert i.parameter is None
        assert i.value is None

    def test_from_dict_empty(self):
        i = Intent.from_dict({})
        assert i.action == ""
        assert i.artifact == ""

    def test_roundtrip(self):
        original = Intent(action="set", artifact="light308", parameter="brightness", value=50)
        d = original.to_dict()
        restored = Intent.from_dict(d)
        assert restored.action == original.action
        assert restored.artifact == original.artifact
        assert restored.parameter == original.parameter
        assert restored.value == original.value
        assert restored.to_canonical_string() == original.to_canonical_string()


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
        conv = ConversationState(
            phase=ConversationPhase.AWAITING_CONFIRMATION,
            intents=[Intent(action="turn_on", artifact="light308")],
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
