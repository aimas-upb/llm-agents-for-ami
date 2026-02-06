"""
Unit tests for BT <-> Signifier conversion (signifier_bridge).
"""

import pytest

from ami_agents.bt_planning.signifier_bridge import (
    extract_signifiers_from_bt,
    build_bt_from_signifiers,
)


class TestExtractSignifiers:
    """Tests for extract_signifiers_from_bt()."""

    def test_extract_single_action(self, sample_action_spec):
        sigs = extract_signifiers_from_bt(
            tree_spec=sample_action_spec,
            intents=["turn on the light"],
        )
        assert len(sigs) == 1
        assert sigs[0]["affordance_uri"].endswith("/turn_on")
        assert sigs[0]["intent"] == "turn on the light"
        assert sigs[0]["source"] == "bt_execution"

    def test_extract_multiple_actions(self, sample_sequence_spec):
        sigs = extract_signifiers_from_bt(
            tree_spec=sample_sequence_spec,
            intents=["increase light level"],
        )
        assert len(sigs) == 3
        urls = [s["affordance_uri"] for s in sigs]
        assert any("turn_on" in u for u in urls)
        assert any("set_brightness" in u for u in urls)
        assert any("set_position" in u for u in urls)

    def test_extract_nested_actions(self, sample_nested_spec):
        sigs = extract_signifiers_from_bt(
            tree_spec=sample_nested_spec,
            intents=["turn on and brighten light"],
        )
        # selector > [condition, action] + action = 2 action nodes
        assert len(sigs) == 2

    def test_extract_skips_conditions(self, sample_selector_spec):
        sigs = extract_signifiers_from_bt(
            tree_spec=sample_selector_spec,
            intents=["turn on the light"],
        )
        # selector > [condition, action] -> only 1 action extracted
        assert len(sigs) == 1
        assert sigs[0]["node_name"] == "TurnOnLight"

    def test_extract_maps_intent(self):
        spec = {
            "name": "TurnOnLight",
            "type": "action",
            "action_url": "http://localhost/light/turn_on",
        }
        sigs = extract_signifiers_from_bt(
            tree_spec=spec,
            intents=["light turn on request"],
        )
        assert sigs[0]["intent"] == "light turn on request"

    def test_extract_with_parameters(self):
        spec = {
            "name": "SetBrightness",
            "type": "action",
            "action_url": "http://localhost/light/set_brightness",
            "parameters": {"brightness": 75},
        }
        sigs = extract_signifiers_from_bt(
            tree_spec=spec,
            intents=["set brightness"],
        )
        assert sigs[0]["payload_hint"] == {"brightness": 75}

    def test_extract_from_empty_tree(self):
        sigs = extract_signifiers_from_bt(tree_spec={}, intents=["test"])
        assert sigs == []

    def test_extract_workspace_id(self, sample_action_spec):
        sigs = extract_signifiers_from_bt(
            tree_spec=sample_action_spec,
            intents=["test"],
            workspace_id="http://localhost:8080/workspaces/lab308",
        )
        assert sigs[0]["workspace_id"] == "http://localhost:8080/workspaces/lab308"


class TestBuildBTFromSignifiers:
    """Tests for build_bt_from_signifiers()."""

    def test_build_single_intent(self, sample_signifier_matches):
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=["increase light level"],
        )
        assert bt is not None
        assert bt["type"] == "action"
        assert "set_brightness" in bt["action_url"]
        assert bt["parameters"] == {"brightness": 100}

    def test_build_multiple_intents(self):
        matches = {
            "turn on light": {
                "matches": [{"signifier_id": "s1", "affordance_uri": "http://localhost/turn_on"}],
                "final_matches": ["s1"],
            },
            "open blinds": {
                "matches": [{"signifier_id": "s2", "affordance_uri": "http://localhost/set_position", "payload_hint": {"position": 100}}],
                "final_matches": ["s2"],
            },
        }
        bt = build_bt_from_signifiers(
            signifier_matches=matches,
            intents=["turn on light", "open blinds"],
        )
        assert bt is not None
        assert bt["type"] == "parallel"
        assert len(bt["children"]) == 2

    def test_build_returns_none_if_no_matches(self):
        bt = build_bt_from_signifiers(
            signifier_matches={},
            intents=["test"],
        )
        assert bt is None

    def test_build_returns_none_if_partial_match(self, sample_signifier_matches):
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=["increase light level", "unmatched intent"],
        )
        assert bt is None

    def test_build_returns_none_if_empty_intents(self):
        bt = build_bt_from_signifiers(
            signifier_matches={"x": {"matches": [], "final_matches": []}},
            intents=[],
        )
        assert bt is None

    def test_build_returns_none_if_match_has_no_affordance_uri(self):
        matches = {
            "test": {
                "matches": [{"signifier_id": "s1", "affordance_uri": ""}],
                "final_matches": ["s1"],
            },
        }
        bt = build_bt_from_signifiers(
            signifier_matches=matches,
            intents=["test"],
        )
        assert bt is None
