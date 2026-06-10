"""
Unit tests for BT <-> Signifier conversion (signifier_bridge).
"""

import pytest

from ami_agents.agents.user_assistant.models import Intent
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

    def test_extract_tdsosa_metadata_and_filters_context(self):
        action_url = "http://localhost:8080/workspaces/lab308/artifacts/blinds/ha/cover/close_cover"
        state_snapshot = {
            "artifacts": {
                "http://localhost:8080/workspaces/lab308/artifacts/glare_sensor": {
                    "name": "glare sensor",
                    "workspace_id": "lab308",
                    "state": {"state": 900},
                },
                "http://localhost:8080/workspaces/lab308/artifacts/co2_sensor": {
                    "name": "co2 sensor",
                    "workspace_id": "lab308",
                    "state": {"state": 600},
                },
            },
            "td_sosa_action_effects": {
                action_url: [
                    {
                        "property_uri": "http://localhost:8080/workspaces/lab308/environment/glare",
                        "direction": "decrease",
                        "settling_time_seconds": 15,
                        "readable_property_urls": [
                            "http://localhost:8080/workspaces/lab308/artifacts/glare_sensor/properties/state"
                        ],
                    }
                ]
            },
        }
        sigs = extract_signifiers_from_bt(
            tree_spec={"name": "CloseBlinds", "type": "action", "action_url": action_url},
            intents=["reduce glare"],
            workspace_id="lab308",
            state_snapshot=state_snapshot,
            td_sosa_supported=True,
        )

        assert len(sigs) == 1
        assert sigs[0]["td_sosa"]["affected_observable_property_uris"] == [
            "http://localhost:8080/workspaces/lab308/environment/glare"
        ]
        assert sigs[0]["td_sosa"]["effect_directions"] == ["decrease"]
        assert sigs[0]["td_sosa"]["settling_time_seconds"] == 15
        assert len(sigs[0]["structured_conditions"]) == 1
        assert sigs[0]["structured_conditions"][0]["artifact"].endswith("/glare_sensor")


class TestBuildBTFromSignifiers:
    """Tests for build_bt_from_signifiers()."""

    def _intent(self, intent_text, action="unknown", artifact="unknown", parameter=None, value=None):
        """Helper to create an Intent with intent_text matching the signifier_matches key."""
        return Intent(action=action, artifact=artifact, parameter=parameter, value=value, intent_text=intent_text)

    def test_build_single_intent(self, sample_signifier_matches):
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=[self._intent("increase light level")],
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
            intents=[self._intent("turn on light"), self._intent("open blinds")],
        )
        assert bt is not None
        assert bt["type"] == "parallel"
        assert len(bt["children"]) == 2

    def test_build_returns_none_if_no_matches(self):
        bt = build_bt_from_signifiers(
            signifier_matches={},
            intents=[self._intent("test")],
        )
        assert bt is None

    def test_build_returns_none_if_partial_match(self, sample_signifier_matches):
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=[self._intent("increase light level"), self._intent("unmatched intent")],
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
            intents=[self._intent("test")],
        )
        assert bt is None

    # Multi-action signifier reuse tests

    def test_build_multi_action_single_intent(self, sample_multi_action_signifier_matches):
        """1 intent with 2 final_matches → sequence of 2 actions."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_multi_action_signifier_matches,
            intents=[self._intent("increase light level")],
        )
        assert bt is not None
        assert bt["type"] == "sequence"
        assert len(bt["children"]) == 2
        assert bt["children"][0]["type"] == "action"
        assert bt["children"][1]["type"] == "action"

    def test_build_multi_action_preserves_order(self, sample_multi_action_signifier_matches):
        """Actions appear in same order as final_matches."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_multi_action_signifier_matches,
            intents=[self._intent("increase light level")],
        )
        assert bt is not None
        urls = [c["action_url"] for c in bt["children"]]
        assert "turn_on" in urls[0]
        assert "set_brightness" in urls[1]

    def test_build_multi_action_with_payloads(self, sample_multi_action_signifier_matches):
        """Each action retains its own payload_hint."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_multi_action_signifier_matches,
            intents=[self._intent("increase light level")],
        )
        assert bt is not None
        # First action has empty payload (no parameters key)
        assert "parameters" not in bt["children"][0] or bt["children"][0].get("parameters") == {}
        # Second action has brightness payload
        assert bt["children"][1]["parameters"] == {"brightness": 100}

    def test_build_multi_intent_one_multi_action(self, sample_multi_action_signifier_matches):
        """2 intents, one with 2 actions → parallel of [sequence, action]."""
        matches = dict(sample_multi_action_signifier_matches)
        matches["open blinds"] = {
            "matches": [{"signifier_id": "s3", "affordance_uri": "http://localhost/set_position", "payload_hint": {"position": 100}}],
            "final_matches": ["s3"],
        }
        bt = build_bt_from_signifiers(
            signifier_matches=matches,
            intents=[self._intent("increase light level"), self._intent("open blinds")],
        )
        assert bt is not None
        assert bt["type"] == "parallel"
        assert len(bt["children"]) == 2
        # First child: sequence (multi-action intent)
        assert bt["children"][0]["type"] == "sequence"
        assert len(bt["children"][0]["children"]) == 2
        # Second child: single action
        assert bt["children"][1]["type"] == "action"

    def test_build_multi_intent_both_multi_action(self):
        """2 intents, both with 2+ actions → parallel of [sequence, sequence]."""
        matches = {
            "increase light": {
                "matches": [
                    {"signifier_id": "s1", "affordance_uri": "http://localhost/turn_on"},
                    {"signifier_id": "s2", "affordance_uri": "http://localhost/set_brightness", "payload_hint": {"brightness": 100}},
                ],
                "final_matches": ["s1", "s2"],
            },
            "adjust blinds": {
                "matches": [
                    {"signifier_id": "s3", "affordance_uri": "http://localhost/open_blinds"},
                    {"signifier_id": "s4", "affordance_uri": "http://localhost/set_position", "payload_hint": {"position": 80}},
                ],
                "final_matches": ["s3", "s4"],
            },
        }
        bt = build_bt_from_signifiers(
            signifier_matches=matches,
            intents=[self._intent("increase light"), self._intent("adjust blinds")],
        )
        assert bt is not None
        assert bt["type"] == "parallel"
        assert len(bt["children"]) == 2
        assert bt["children"][0]["type"] == "sequence"
        assert bt["children"][1]["type"] == "sequence"

    def test_build_multi_action_single_intent_single_match(self, sample_signifier_matches):
        """1 intent with 1 final_match → still returns single action node."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=[self._intent("increase light level")],
        )
        assert bt is not None
        assert bt["type"] == "action"
        assert "set_brightness" in bt["action_url"]

    # Intent-aware payload tests

    def test_build_set_intent_overrides_payload(self, sample_signifier_matches):
        """set intent with explicit value overrides the signifier's payload_hint."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=[Intent(action="set", artifact="light308", parameter="brightness", value=50,
                            intent_text="increase light level")],
        )
        assert bt is not None
        # Should use intent's value (50) not signifier's payload_hint (100)
        assert bt["parameters"] == {"brightness": 50}

    def test_build_returns_none_for_modify_intent(self, sample_signifier_matches):
        """modify intents cannot be fast-pathed (need read-compute-set)."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=[Intent(action="modify", artifact="light308", parameter="brightness", value=10,
                            intent_text="increase light level")],
        )
        assert bt is None

    def test_build_returns_none_for_modify_without_value(self, sample_signifier_matches):
        """modify intents with null value also bail from fast path."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=[Intent(action="modify", artifact="light308", parameter="brightness",
                            intent_text="increase light level")],
        )
        assert bt is None

    def test_build_check_intent_has_no_parameters(self):
        """check intents produce action nodes without parameters."""
        matches = {
            "check light": {
                "matches": [{"signifier_id": "s1", "affordance_uri": "http://localhost/light/status", "payload_hint": {"brightness": 75}}],
                "final_matches": ["s1"],
            },
        }
        bt = build_bt_from_signifiers(
            signifier_matches=matches,
            intents=[Intent(action="check", artifact="light308", intent_text="check light")],
        )
        assert bt is not None
        assert "parameters" not in bt

    def test_build_set_on_off_skips_parameters(self):
        """set on_off intent produces action node WITHOUT parameters.

        The on/off semantic is encoded in the affordance_uri (turn_on vs turn_off),
        so no API parameter should be sent.
        """
        matches = {
            "turn on the light": {
                "matches": [{"signifier_id": "s1", "affordance_uri": "http://localhost/light/turn_on", "payload_hint": {"on_off": True}}],
                "final_matches": ["s1"],
            },
        }
        bt = build_bt_from_signifiers(
            signifier_matches=matches,
            intents=[Intent(action="set", artifact="light308", parameter="on_off", value=True,
                            intent_text="turn on the light")],
        )
        assert bt is not None
        assert "parameters" not in bt

    def test_build_set_on_off_false_skips_parameters(self):
        """set on_off=False (turn off) also skips parameters."""
        matches = {
            "turn off the light": {
                "matches": [{"signifier_id": "s1", "affordance_uri": "http://localhost/light/turn_off"}],
                "final_matches": ["s1"],
            },
        }
        bt = build_bt_from_signifiers(
            signifier_matches=matches,
            intents=[Intent(action="set", artifact="light308", parameter="on_off", value=False,
                            intent_text="turn off the light")],
        )
        assert bt is not None
        assert "parameters" not in bt

    def test_build_unknown_action_uses_payload_hint(self, sample_signifier_matches):
        """Intent with unknown action falls back to signifier payload_hint."""
        bt = build_bt_from_signifiers(
            signifier_matches=sample_signifier_matches,
            intents=[self._intent("increase light level")],
        )
        assert bt is not None
        assert bt["parameters"] == {"brightness": 100}
