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

    # set action (subsumes old turn_on / turn_off)

    def test_set_boolean_on(self):
        i = Intent(action="set", artifact="light308", parameter="on_off", value=True)
        assert i.to_canonical_string() == "set light308 on_off to True"

    def test_set_boolean_off(self):
        i = Intent(action="set", artifact="blinds308", parameter="open_close", value=False)
        assert i.to_canonical_string() == "set blinds308 open_close to False"

    def test_set_with_parameter(self):
        i = Intent(action="set", artifact="light308", parameter="brightness", value=75)
        assert i.to_canonical_string() == "set light308 brightness to 75"

    def test_set_with_string_value(self):
        i = Intent(action="set", artifact="light308", parameter="color", value="warm_white")
        assert i.to_canonical_string() == "set light308 color to warm_white"

    # check action

    def test_check(self):
        i = Intent(action="check", artifact="light308")
        assert i.to_canonical_string() == "check light308"

    def test_check_with_parameter(self):
        i = Intent(action="check", artifact="light308", parameter="brightness")
        assert i.to_canonical_string() == "check light308 brightness"

    # modify action

    def test_modify_with_value(self):
        i = Intent(action="modify", artifact="light308", parameter="brightness", value=10)
        assert i.to_canonical_string() == "modify light308 brightness by 10"

    def test_modify_with_negative_value(self):
        i = Intent(action="modify", artifact="light308", parameter="brightness", value=-20)
        assert i.to_canonical_string() == "modify light308 brightness by -20"

    def test_modify_without_value(self):
        """Vague modify (e.g. 'dim the light') → value is None."""
        i = Intent(action="modify", artifact="light308", parameter="brightness")
        assert i.to_canonical_string() == "modify light308 brightness"

    # fallback

    def test_fallback_for_unknown_action(self):
        i = Intent(action="toggle", artifact="light308")
        result = i.to_canonical_string()
        assert "toggle" in result
        assert "light308" in result

    # intent_text field

    def test_intent_text_field(self):
        i = Intent(action="set", artifact="light308", parameter="on_off", value=True,
                   intent_text="turn on the light")
        assert i.intent_text == "turn on the light"

    def test_intent_text_default_none(self):
        i = Intent(action="set", artifact="light308", parameter="on_off", value=True)
        assert i.intent_text is None

    # serialization

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
        i = Intent(action="check", artifact="light308")
        d = i.to_dict()
        assert d == {"action": "check", "artifact": "light308"}
        assert "parameter" not in d
        assert "value" not in d
        assert "intent_text" not in d

    def test_to_dict_with_intent_text(self):
        i = Intent(action="set", artifact="light308", parameter="on_off", value=True,
                   intent_text="turn on the light")
        d = i.to_dict()
        assert d["intent_text"] == "turn on the light"

    def test_from_dict(self):
        d = {"action": "set", "artifact": "light308", "parameter": "brightness", "value": 75}
        i = Intent.from_dict(d)
        assert i.action == "set"
        assert i.artifact == "light308"
        assert i.parameter == "brightness"
        assert i.value == 75
        assert i.intent_text is None

    def test_from_dict_with_intent_text(self):
        d = {"action": "set", "artifact": "light308", "parameter": "on_off", "value": True,
             "intent_text": "turn on the light"}
        i = Intent.from_dict(d)
        assert i.intent_text == "turn on the light"

    def test_from_dict_minimal(self):
        d = {"action": "check", "artifact": "light308"}
        i = Intent.from_dict(d)
        assert i.action == "check"
        assert i.artifact == "light308"
        assert i.parameter is None
        assert i.value is None

    def test_from_dict_empty(self):
        i = Intent.from_dict({})
        assert i.action == ""
        assert i.artifact == ""

    def test_roundtrip(self):
        original = Intent(action="set", artifact="light308", parameter="brightness", value=50,
                          intent_text="set the brightness to 50")
        d = original.to_dict()
        restored = Intent.from_dict(d)
        assert restored.action == original.action
        assert restored.artifact == original.artifact
        assert restored.parameter == original.parameter
        assert restored.value == original.value
        assert restored.intent_text == original.intent_text
        assert restored.to_canonical_string() == original.to_canonical_string()

    def test_roundtrip_modify(self):
        original = Intent(action="modify", artifact="light308", parameter="brightness", value=10,
                          intent_text="increase the brightness by 10")
        d = original.to_dict()
        restored = Intent.from_dict(d)
        assert restored.to_canonical_string() == "modify light308 brightness by 10"
        assert restored.intent_text == "increase the brightness by 10"


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
            intents=[Intent(action="set", artifact="light308", parameter="on_off", value=True)],
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
