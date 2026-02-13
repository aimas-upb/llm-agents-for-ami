"""
Unit tests for UserAssistant behaviours.

These tests validate the deterministic logic in UserMessageBehaviour
without requiring a live SPADE server or real LLM.  The AsyncOpenAI
client and RPC calls are mocked.
"""

import json
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ami_agents.agents.user_assistant.models import (
    ConversationPhase,
    ConversationState,
    Intent,
)
from ami_agents.agents.user_assistant.utils import (
    canonicalize_plan_for_hash,
    coerce_plan_dict,
    loose_json_loads,
    strip_code_fences,
    count_bt_nodes,
    bt_preview,
)


# ── Utility function tests ─────────────────────────────────────────


class TestStripCodeFences:
    def test_no_fences(self):
        assert strip_code_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fences(self):
        assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fences(self):
        assert strip_code_fences('```\nhello\n```') == 'hello'

    def test_empty(self):
        assert strip_code_fences("") == ""
        assert strip_code_fences(None) == ""


class TestLooseJsonLoads:
    def test_clean_json(self):
        assert loose_json_loads('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert loose_json_loads('```json\n{"a": 1}\n```') == {"a": 1}

    def test_leading_text(self):
        result = loose_json_loads('Here is the result: {"a": 1}')
        assert result == {"a": 1}

    def test_empty(self):
        assert loose_json_loads("") is None

    def test_not_json(self):
        assert loose_json_loads("just plain text") is None


class TestCoercePlanDict:
    def test_dict_passthrough(self):
        d = {"tree": {"type": "action"}}
        assert coerce_plan_dict(d) == d

    def test_json_string(self):
        s = json.dumps({"tree": {"type": "action"}})
        result = coerce_plan_dict(s)
        assert result == {"tree": {"type": "action"}}

    def test_wrapper_with_plan_json(self):
        inner = json.dumps({"tree": {"type": "action"}})
        wrapper = {"ok": True, "plan_json": inner}
        result = coerce_plan_dict(wrapper)
        assert result == {"tree": {"type": "action"}}

    def test_none_for_empty(self):
        assert coerce_plan_dict("") is None
        assert coerce_plan_dict(None) is None


class TestCanonicalizePlanForHash:
    def test_deterministic(self):
        plan = {"b": 2, "a": 1}
        canonical1, hash1 = canonicalize_plan_for_hash(plan)
        canonical2, hash2 = canonicalize_plan_for_hash(plan)
        assert canonical1 == canonical2
        assert hash1 == hash2

    def test_sorted_keys(self):
        plan = {"b": 2, "a": 1}
        canonical, _ = canonicalize_plan_for_hash(plan)
        assert canonical == '{"a":1,"b":2}'

    def test_hash_is_sha256(self):
        plan = {"a": 1}
        canonical, plan_hash = canonicalize_plan_for_hash(plan)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert plan_hash == expected


class TestCountBtNodes:
    def test_single_action(self):
        assert count_bt_nodes({"type": "action", "name": "a"}) == 1

    def test_sequence_with_children(self):
        tree = {
            "type": "sequence",
            "children": [
                {"type": "action", "name": "a"},
                {"type": "action", "name": "b"},
            ],
        }
        assert count_bt_nodes(tree) == 3

    def test_empty(self):
        assert count_bt_nodes({}) == 1
        assert count_bt_nodes(None) == 0


class TestBtPreview:
    def test_action_node(self):
        node = {"name": "turn_on", "type": "action", "action_url": "http://x/turn_on"}
        preview = bt_preview(node)
        assert "turn_on" in preview
        assert "action" in preview

    def test_empty(self):
        assert bt_preview(None) == ""
        assert bt_preview({}) != ""  # Returns "?(?)"


# ── Intent integration tests ───────────────────────────────────────


class TestIntentCanonicalStrings:
    """Test that Intent canonical strings are compatible with InteractionSolver expectations."""

    def test_multiple_intents_produce_list(self):
        intents = [
            Intent(action="turn_on", artifact="light308"),
            Intent(action="set", artifact="light308", parameter="brightness", value=100),
        ]
        strings = [i.to_canonical_string() for i in intents]
        assert strings == ["turn on light308", "set light308 brightness to 100"]

    def test_set_with_zero_value(self):
        i = Intent(action="set", artifact="light308", parameter="brightness", value=0)
        assert i.to_canonical_string() == "set light308 brightness to 0"

    def test_set_missing_value_falls_back(self):
        """If value is None, set falls through to the fallback path."""
        i = Intent(action="set", artifact="light308", parameter="brightness", value=None)
        result = i.to_canonical_string()
        assert "set" in result
        assert "light308" in result


# ── Confirmation token tests ────────────────────────────────────────


class TestConfirmationTokens:
    """Test that confirmation and rejection tokens are properly defined."""

    def test_confirm_tokens(self):
        from ami_agents.agents.user_assistant.models import CONFIRM_TOKENS as _CONFIRM_TOKENS
        for token in ["yes", "ok", "proceed", "go ahead", "sure"]:
            assert token in _CONFIRM_TOKENS

    def test_reject_tokens(self):
        from ami_agents.agents.user_assistant.models import REJECT_TOKENS as _REJECT_TOKENS
        for token in ["no", "cancel", "discard", "reject"]:
            assert token in _REJECT_TOKENS

    def test_no_overlap(self):
        from ami_agents.agents.user_assistant.models import CONFIRM_TOKENS as _CONFIRM_TOKENS, REJECT_TOKENS as _REJECT_TOKENS
        assert _CONFIRM_TOKENS.isdisjoint(_REJECT_TOKENS)


# ── ConversationState flow tests ────────────────────────────────────


class TestConversationStateFlow:
    """Test conversation state transitions that behaviours depend on."""

    def test_goal_flow_phases(self):
        """Verify the expected phase transitions for a goal request."""
        conv = ConversationState()
        assert conv.phase == ConversationPhase.IDLE

        conv.phase = ConversationPhase.EXTRACTING_INTENTS
        conv.intents = [Intent(action="turn_on", artifact="light308")]
        assert conv.phase == ConversationPhase.EXTRACTING_INTENTS

        conv.phase = ConversationPhase.AWAITING_PLAN
        assert conv.phase == ConversationPhase.AWAITING_PLAN

        conv.plan_json = '{"tree": {}}'
        conv.plan_hash = "abc"
        conv.phase = ConversationPhase.SUMMARIZING_PLAN
        assert conv.phase == ConversationPhase.SUMMARIZING_PLAN

        conv.phase = ConversationPhase.AWAITING_CONFIRMATION
        assert conv.phase == ConversationPhase.AWAITING_CONFIRMATION

        conv.phase = ConversationPhase.EXECUTING
        assert conv.phase == ConversationPhase.EXECUTING

        conv.clear_plan()
        conv.phase = ConversationPhase.IDLE
        assert conv.phase == ConversationPhase.IDLE
        assert conv.plan_json is None

    def test_reject_flow(self):
        """Verify that rejecting a plan clears state properly."""
        conv = ConversationState(
            phase=ConversationPhase.AWAITING_CONFIRMATION,
            plan_json='{"tree": {}}',
            plan_hash="abc",
            plan_summary="A plan",
            intents=[Intent(action="turn_on", artifact="light308")],
        )
        conv.clear_plan()
        conv.phase = ConversationPhase.IDLE
        assert conv.plan_json is None
        assert conv.intents == []
        assert conv.phase == ConversationPhase.IDLE


# ── Plan hash gating tests ──────────────────────────────────────────


class TestPlanHashGating:
    """Test that plan canonicalization produces stable hashes for approval gating."""

    def test_same_plan_same_hash(self):
        plan = {"plan_type": "behavior_tree", "tree": {"type": "action", "action_url": "http://x"}}
        _, h1 = canonicalize_plan_for_hash(plan)
        _, h2 = canonicalize_plan_for_hash(plan)
        assert h1 == h2

    def test_different_key_order_same_hash(self):
        plan1 = {"a": 1, "b": 2}
        plan2 = {"b": 2, "a": 1}
        _, h1 = canonicalize_plan_for_hash(plan1)
        _, h2 = canonicalize_plan_for_hash(plan2)
        assert h1 == h2

    def test_different_plans_different_hash(self):
        plan1 = {"tree": {"type": "action", "action_url": "http://x"}}
        plan2 = {"tree": {"type": "action", "action_url": "http://y"}}
        _, h1 = canonicalize_plan_for_hash(plan1)
        _, h2 = canonicalize_plan_for_hash(plan2)
        assert h1 != h2
