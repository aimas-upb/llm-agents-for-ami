"""
The BT planner names affordances by device, td:title and class -- no ids, no URLs.

    catalog      the device list the model is shown: actions and properties
                 apart, no URLs; an action and a property may share a name
    resolution   a leaf's (artifact, affordance, affordance_class) to its URL:
                 action nodes among actions, conditions / waits among
                 properties; anything else an error the retry loop reads
    end to end   generate_bt with a fake model: prompt, retry, recovered tool call
    parameters   an action's parameters checked against its input
    gates        the equality-gate anti-pattern (unchanged)
    hints        memory and observable-property hints named the same way
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ami_agents.bt_planning.planning.affordance_refs import AffordanceCatalog
from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.prompts import (
    format_capability_context,
    format_observable_property_hints,
    format_signifier_hints,
)

BASE = "http://localhost:8080/workspaces/lab308/artifacts"

LIGHT_ACTION = {
    "artifact_id": f"{BASE}/light308#artifact",
    "artifact_name": "light308",
    "affordance_type": "action",
    "action_name": "setBrightness",
    "semantic_types": ["saref:SetLevelCommand"],
    "method": "POST",
    "target": f"{BASE}/light308/setBrightness",
    "input_schema": {
        "type": "object",
        "properties": {"brightness": {"type": "integer"}},
        "required": ["brightness"],
    },
}

LIGHT_PROPERTY = {
    "artifact_id": f"{BASE}/light308#artifact",
    "artifact_name": "light308",
    "affordance_type": "property",
    "name": "brightness",
    "semantic_types": ["homeont:DimmableLightBrightness"],
    "method": "GET",
    "target": f"{BASE}/light308/properties/brightness",
}

# A washer's mode: an action and a property with the same name, in the shape
# `specific_capability_context` builds.
MODE_ACTION = {
    "name": "currentMode", "type": "action_affordance", "artifact_name": "washer_1",
    "artifact_id": f"{BASE}/washer_1#artifact",
    "semantic_types": ["homeont:SetModeCommand"],
    "target": f"{BASE}/washer_1/actions/currentMode",
    "form": {"href": f"{BASE}/washer_1/actions/currentMode", "method": "POST"},
    "description": 'acts upon "currentMode" (homeont:LaundryWasherCurrentMode)',
    "input_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                     "required": ["value"]},
}
MODE_PROPERTY = {
    "name": "currentMode", "type": "property_affordance", "artifact_name": "washer_1",
    "artifact_id": f"{BASE}/washer_1#artifact",
    "semantic_types": ["homeont:LaundryWasherCurrentMode"],
    "target": f"{BASE}/washer_1/properties/currentMode",
    "form": {"href": f"{BASE}/washer_1/properties/currentMode", "method": "GET"},
    "output_schema": {"type": "integer"}, "description": 'reads {"type": "integer"}',
}

AFFORDANCES = [LIGHT_ACTION, LIGHT_PROPERTY]
WASHER = [MODE_ACTION, MODE_PROPERTY]


def _leaf(node_type, device, name, cls, **fields):
    """A leaf naming its affordance as the model writes it."""
    return {"name": "N", "type": node_type, "artifact": device, "affordance": name,
            "affordance_class": cls, **fields}


SET_BRIGHTNESS = _leaf("action", "light308", "setBrightness", "saref:SetLevelCommand",
                       parameters={"brightness": 75})


@pytest.fixture
def planner():
    return AsyncBTPlanner(max_attempts=1)


# --------------------------------------------------------------------------
# The device list
# --------------------------------------------------------------------------

class TestCatalog:
    def test_devices_list_actions_and_properties_apart_without_urls(self):
        text = format_capability_context(AffordanceCatalog(WASHER + AFFORDANCES))
        assert "washer_1\n  actions:\n  - currentMode | homeont:SetModeCommand" in text
        assert "  properties:\n  - currentMode | homeont:LaundryWasherCurrentMode" in text
        assert "http://" not in text and "affordance_id" not in text
        assert "light308" in text

    def test_a_property_reading_is_shown_once(self):
        """Its output schema is shown; the description repeating it is not."""
        text = AffordanceCatalog([MODE_PROPERTY]).render()
        assert text.count('{"type": "integer"}') == 1

    def test_entries_are_found_by_url(self):
        catalog = AffordanceCatalog(WASHER)
        assert catalog.entry_for_url(MODE_PROPERTY["target"] + "/").kind == "property"
        assert catalog.entry_for_url("http://elsewhere") is None

    def test_junk_is_ignored(self):
        assert AffordanceCatalog([None, "junk", {"type": "action"}]).devices == {}

    def test_no_affordances(self):
        assert format_capability_context(AffordanceCatalog([])) == "No affordances available."

    def test_state_section(self):
        state = {"artifacts": {f"{BASE}/light308#artifact": {"brightness": 10}}}
        text = format_capability_context(AffordanceCatalog(AFFORDANCES), state)
        assert "### Current State" in text and "light308: brightness=10" in text

    def test_state_uris_are_shortened_and_aggregates_flattened(self):
        state = {"artifacts": {f"{BASE}/light308": {
            "metadata": {"x": 1},
            "state": {f"{BASE}/light308/properties/brightness": 10,
                      f"{BASE}/light308/properties/on": True}}}}
        text = format_capability_context(AffordanceCatalog(AFFORDANCES), state)
        assert "brightness: 10" in text and "on: True" in text and "metadata" not in text


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

class TestResolution:
    def test_an_action_resolves_among_actions(self, planner):
        spec = _leaf("action", "washer_1", "currentMode", "homeont:SetModeCommand",
                     parameters={"value": 2})
        assert planner._resolve_affordance_refs(spec, AffordanceCatalog(WASHER)) == []
        assert spec["action_url"] == MODE_ACTION["target"]

    @pytest.mark.parametrize("node_type", ["condition", "wait_condition"])
    def test_a_reading_resolves_among_properties(self, planner, node_type):
        """The episode's failure: a wait on the mode reads the property, never
        the action of the same name."""
        spec = _leaf(node_type, "washer_1", "currentMode", "homeont:LaundryWasherCurrentMode",
                     expected_value=2)
        assert planner._resolve_affordance_refs(spec, AffordanceCatalog(WASHER)) == []
        assert spec["property_url"] == MODE_PROPERTY["target"]

    def test_a_reading_given_the_actions_class_is_an_error(self, planner):
        spec = _leaf("wait_condition", "washer_1", "currentMode", "homeont:SetModeCommand",
                     expected_value=2)
        (error,) = planner._resolve_affordance_refs(spec, AffordanceCatalog(WASHER))
        assert "property 'currentMode' is not of class" in error
        assert "homeont:LaundryWasherCurrentMode" in error and "property_url" not in spec

    def test_an_action_naming_a_property_is_an_error_listing_the_actions(self, planner):
        spec = _leaf("action", "light308", "brightness", "homeont:DimmableLightBrightness")
        (error,) = planner._resolve_affordance_refs(spec, AffordanceCatalog(AFFORDANCES))
        assert "has no action 'brightness'" in error
        assert "setBrightness (saref:SetLevelCommand)" in error

    def test_an_unknown_device_is_an_error_listing_the_devices(self, planner):
        spec = _leaf("action", "light999", "setBrightness", "saref:SetLevelCommand")
        (error,) = planner._resolve_affordance_refs(spec, AffordanceCatalog(AFFORDANCES))
        assert "no device 'light999'" in error and "light308" in error

    def test_a_missing_field_is_an_error(self, planner):
        spec = {"name": "N", "type": "action", "artifact": "light308"}
        (error,) = planner._resolve_affordance_refs(spec, AffordanceCatalog(AFFORDANCES))
        assert "'affordance'" in error and "'affordance_class'" in error

    def test_an_entry_listed_without_a_class_takes_any(self, planner):
        bare = {k: v for k, v in LIGHT_ACTION.items() if k != "semantic_types"}
        spec = _leaf("action", "light308", "setBrightness", "whatever")
        assert planner._resolve_affordance_refs(spec, AffordanceCatalog([bare])) == []

    def test_nested_children_resolve_and_validate(self, planner):
        spec = {"name": "Root", "type": "selector", "children": [
            _leaf("condition", "light308", "brightness", "homeont:DimmableLightBrightness",
                  expected_value=75, operator=">="),
            dict(SET_BRIGHTNESS)]}
        assert planner._resolve_affordance_refs(spec, AffordanceCatalog(AFFORDANCES)) == []
        assert spec["children"][0]["property_url"] == LIGHT_PROPERTY["target"]
        assert spec["children"][1]["action_url"] == LIGHT_ACTION["target"]
        assert planner._validate_tree(spec) == []


def _tool_call_response(tree: dict, explanation: str = "ok", impossible: bool = False):
    arguments = json.dumps({"tree": tree, "explanation": explanation, "impossible": impossible})
    tool_call = SimpleNamespace(id="call-1", function=SimpleNamespace(arguments=arguments))
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _content_response(content: str):
    """A response with no structured tool call, only text content."""
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestParseToolCallContent:
    def test_fenced_tool_call_wrapper(self):
        content = (
            "Here is the plan:\n```json\n"
            '{"name": "generate_behavior_tree", "arguments": '
            '{"tree": {"name": "T", "type": "action", "artifact": "a", "affordance": "b"}, '
            '"explanation": "x", "impossible": false}}\n```'
        )
        args = AsyncBTPlanner._parse_tool_call_content(content)
        assert args["tree"]["affordance"] == "b"

    def test_bare_arguments_without_wrapper(self):
        content = '```json\n{"tree": {"name": "T", "type": "action"}, "impossible": false}\n```'
        args = AsyncBTPlanner._parse_tool_call_content(content)
        assert args["tree"]["name"] == "T"

    def test_arguments_as_json_string(self):
        content = json.dumps({"name": "generate_behavior_tree", "arguments": json.dumps({"tree": {"name": "T"}})})
        args = AsyncBTPlanner._parse_tool_call_content(content)
        assert args["tree"]["name"] == "T"

    def test_unfenced_json_with_surrounding_prose(self):
        content = 'Sure! {"tree": {"name": "T", "type": "action"}, "impossible": false} Hope this helps.'
        args = AsyncBTPlanner._parse_tool_call_content(content)
        assert args["tree"]["name"] == "T"

    def test_prose_returns_none(self):
        assert AsyncBTPlanner._parse_tool_call_content("I cannot generate a plan right now.") is None
        assert AsyncBTPlanner._parse_tool_call_content(None) is None
        assert AsyncBTPlanner._parse_tool_call_content("") is None



class TestGenerateBTEndToEnd:
    @pytest.mark.asyncio
    async def test_generates_a_resolved_tree_from_names(self):
        planner = AsyncBTPlanner(max_attempts=1)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_tool_call_response(dict(SET_BRIGHTNESS)))

        result = await planner.generate_bt(
            intents=["set the brightness to 75"], affordances=AFFORDANCES,
            state={"artifacts": {"light308": {"brightness": 10}}},
            client=client, model="gpt-4o-mini")

        assert result["impossible"] is False
        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]
        # The prompt names devices and affordances; it carries no URL.
        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "setBrightness | saref:SetLevelCommand" in system_prompt
        assert LIGHT_ACTION["target"] not in system_prompt
        assert "affordance_id" not in system_prompt

    @pytest.mark.asyncio
    async def test_a_wrong_name_is_retried_with_what_the_device_has(self):
        planner = AsyncBTPlanner(max_attempts=2)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _tool_call_response(_leaf("action", "light308", "doesNotExist",
                                      "saref:SetLevelCommand")),
            _tool_call_response(dict(SET_BRIGHTNESS)),
        ])

        result = await planner.generate_bt(intents=["set the brightness"],
                                           affordances=AFFORDANCES, client=client,
                                           model="gpt-4o-mini")

        assert client.chat.completions.create.await_count == 2
        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]
        retry_messages = client.chat.completions.create.call_args.kwargs["messages"]
        feedback = [m for m in retry_messages if m.get("role") == "tool"]
        assert feedback and "has no action 'doesNotExist'" in feedback[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_call_recovered_from_text_content(self):
        # Ollama models often emit the tool call as fenced JSON text.
        planner = AsyncBTPlanner(max_attempts=1)
        client = MagicMock()
        content = "```json\n" + json.dumps({
            "name": "generate_behavior_tree",
            "arguments": {"tree": dict(SET_BRIGHTNESS), "explanation": "ok",
                          "impossible": False}}) + "\n```"
        client.chat.completions.create = AsyncMock(return_value=_content_response(content))

        result = await planner.generate_bt(intents=["set the brightness"],
                                           affordances=AFFORDANCES, client=client,
                                           model="qwen2.5-coder:3b")
        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]

    @pytest.mark.asyncio
    async def test_prose_response_retries_then_succeeds(self):
        planner = AsyncBTPlanner(max_attempts=2)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _content_response("I think you should turn up the light."),
            _tool_call_response(dict(SET_BRIGHTNESS)),
        ])

        result = await planner.generate_bt(intents=["set the brightness"],
                                           affordances=AFFORDANCES, client=client,
                                           model="qwen2.5-coder:3b")

        assert client.chat.completions.create.await_count == 2
        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]
        # Without a tool_call_id the retry feedback must be a user message.
        retry_messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert [m for m in retry_messages
                if m.get("role") == "user" and "invalid" in str(m.get("content"))]


SET_TEMPERATURE = {
    "artifact_id": "http://h/w/artifacts/ac1#artifact",
    "artifact_name": "ac1",
    "affordance_type": "action",
    "action_name": "ClimateSetTemperature",
    "semantic_types": ["saref:SetLevelCommand"],
    "target": "http://h/w/artifacts/ac1/ha/climate/set_temperature",
    "input_schema": {
        "type": "object",
        "properties": {
            "temperature": {"type": "number"},
            "target_temp_high": {"type": "number"},
            "target_temp_low": {"type": "number"},
            "hvac_mode": {"type": "string"},
        },
    },
}


class TestActionParameterValidation:
    def _resolve(self, planner, params, affordance=SET_TEMPERATURE):
        spec = _leaf("action", affordance["artifact_name"], affordance["action_name"],
                     "saref:SetLevelCommand", parameters=params)
        return planner._resolve_affordance_refs(spec, AffordanceCatalog([affordance])), spec

    def test_temperature_only_is_valid(self, planner):
        errors, _ = self._resolve(planner, {"temperature": 24.0, "hvac_mode": "cool"})
        assert errors == []

    def test_range_pair_is_valid(self, planner):
        errors, _ = self._resolve(planner, {"target_temp_high": 26.0, "target_temp_low": 22.0})
        assert errors == []

    def test_mixed_styles_rejected(self, planner):
        errors, _ = self._resolve(
            planner, {"temperature": 24.0, "target_temp_high": 30.0, "target_temp_low": 20.0})
        assert len(errors) == 1 and "never both styles" in errors[0]

    def test_partial_range_rejected(self, planner):
        errors, _ = self._resolve(planner, {"target_temp_high": 26.0})
        assert len(errors) == 1 and "both" in errors[0]

    def test_hvac_mode_alone_rejected(self, planner):
        # The exact HA 400 from ablation 005/006.
        errors, _ = self._resolve(planner, {"hvac_mode": "cool"})
        assert len(errors) == 1 and "'hvac_mode' alone is not valid" in errors[0]

    def test_unknown_parameter_rejected(self, planner):
        errors, _ = self._resolve(planner, {"temperature": 24.0, "brightness": 50})
        assert len(errors) == 1
        assert "unknown parameter" in errors[0] and "brightness" in errors[0]

    def test_schema_known_param_passes(self, planner):
        errors, _ = self._resolve(planner, {"brightness": 75}, affordance=LIGHT_ACTION)
        assert errors == []

    def test_no_schema_skips_unknown_check(self, planner):
        aff = {k: v for k, v in LIGHT_ACTION.items() if k != "input_schema"}
        errors, _ = self._resolve(planner, {"anything": 1}, affordance=aff)
        assert errors == []


# A SimuHome light's on/off and an AC's mode, as their TDs give the inputs.
ONOFF_ACTION = {
    "name": "onOff", "type": "action_affordance", "artifact_name": "kitchen_light_1",
    "artifact_id": f"{BASE}/kitchen_light_1#artifact",
    "semantic_types": ["homeont:SetOnOffCommand"],
    "target": f"{BASE}/kitchen_light_1/actions/onOff",
    "input_schema": {"type": "object", "properties": {"value": {"type": "boolean"}},
                     "required": ["value"]},
}
MODE_ENUM_ACTION = {
    "name": "hvacMode", "type": "action_affordance", "artifact_name": "ac_1",
    "artifact_id": f"{BASE}/ac_1#artifact", "semantic_types": ["homeont:SetModeCommand"],
    "target": f"{BASE}/ac_1/actions/hvacMode",
    "input_schema": {"type": "object", "properties": {"value": {
        "type": "integer", "enum": [0, 3, 4]}}, "required": ["value"]},
}
LEVEL_ACTION = {
    "name": "brightness", "type": "action_affordance", "artifact_name": "tv_1",
    "artifact_id": f"{BASE}/tv_1#artifact", "semantic_types": ["saref:SetLevelCommand"],
    "target": f"{BASE}/tv_1/actions/brightness",
    "input_schema": {"type": "object", "properties": {"value": {
        "type": "integer", "minimum": 1, "maximum": 254}}, "required": ["value"]},
}


class TestParameterValues:
    """Each parameter's value against its own schema, so a malformed one goes
    back to the model in the retry instead of reaching the device."""

    def _errors(self, planner, affordance, value):
        spec = _leaf("action", affordance["artifact_name"], affordance["name"],
                     affordance["semantic_types"][0], parameters={"value": value})
        return planner._resolve_affordance_refs(spec, AffordanceCatalog([affordance]))

    def test_the_schema_copied_as_a_boolean_is_an_error(self, planner):
        """qt2 seed 45's slip: {"type": "boolean", "value": true} for true."""
        (error,) = self._errors(planner, ONOFF_ACTION, {"type": "boolean", "value": True})
        assert "parameter 'value' must be a boolean" in error

    def test_a_boolean_passes(self, planner):
        assert self._errors(planner, ONOFF_ACTION, True) == []

    @pytest.mark.parametrize("value, ok", [(3, True), (3.0, True), (2, False),
                                           ("3", False), (True, False)])
    def test_an_enum_takes_only_its_values_as_integers(self, planner, value, ok):
        """3.0 is the integer 3; a string or a bool is not an integer."""
        assert (self._errors(planner, MODE_ENUM_ACTION, value) == []) is ok

    @pytest.mark.parametrize("value, ok", [(1, True), (254, True), (0, False), (300, False)])
    def test_a_number_within_its_bounds(self, planner, value, ok):
        assert (self._errors(planner, LEVEL_ACTION, value) == []) is ok


class TestEqualityGateDetection:
    def _seq(self, condition, action=None):
        action = action or {"name": "Act", "type": "action", "action_url": "http://x/a"}
        return {"name": "S", "type": "sequence", "children": [condition, action]}

    def test_numeric_equality_gate_before_action_is_flagged(self):
        planner = AsyncBTPlanner()
        cond = {"name": "C", "type": "condition", "property_url": "http://x/p",
                "operator": "==", "expected_value": 71}
        errors = planner._detect_equality_gates(self._seq(cond))
        assert len(errors) == 1
        assert "selector" in errors[0]

    def test_default_operator_counts_as_equality(self):
        planner = AsyncBTPlanner()
        cond = {"name": "C", "type": "condition", "property_url": "http://x/p",
                "expected_value": 36.124}
        errors = planner._detect_equality_gates(self._seq(cond))
        assert len(errors) == 1

    def test_range_operator_is_fine(self):
        planner = AsyncBTPlanner()
        cond = {"name": "C", "type": "condition", "property_url": "http://x/p",
                "operator": "<=", "expected_value": 50}
        assert planner._detect_equality_gates(self._seq(cond)) == []

    def test_discrete_and_boolean_values_are_fine(self):
        planner = AsyncBTPlanner()
        for expected in ("on", True):
            cond = {"name": "C", "type": "condition", "property_url": "http://x/p",
                    "operator": "==", "expected_value": expected}
            assert planner._detect_equality_gates(self._seq(cond)) == []

    def test_selector_idempotent_pattern_is_fine(self):
        planner = AsyncBTPlanner()
        tree = {"name": "S", "type": "selector", "children": [
            {"name": "C", "type": "condition", "property_url": "http://x/p",
             "operator": "==", "expected_value": 71},
            {"name": "Act", "type": "action", "action_url": "http://x/a"},
        ]}
        assert planner._detect_equality_gates(tree) == []

    def test_post_action_verification_is_fine(self):
        planner = AsyncBTPlanner()
        tree = {"name": "S", "type": "sequence", "children": [
            {"name": "Act", "type": "action", "action_url": "http://x/a"},
            {"name": "C", "type": "condition", "property_url": "http://x/p",
             "operator": "==", "expected_value": 75},
        ]}
        assert planner._detect_equality_gates(tree) == []

    def test_nested_sequences_are_checked(self):
        planner = AsyncBTPlanner()
        inner = self._seq({"name": "C", "type": "condition", "property_url": "http://x/p",
                           "operator": "==", "expected_value": 10})
        tree = {"name": "Root", "type": "selector", "children": [inner]}
        errors = planner._detect_equality_gates(tree)
        assert len(errors) == 1
        assert errors[0].startswith("tree.children[0].children[0]")



class TestHintFormatting:
    def test_signifier_hints_name_what_the_past_plan_used(self):
        hints = format_signifier_hints(
            {"intent_text": "it is too dark", "signifier_id": "sig-1",
             "used": {"actions": [{"url": LIGHT_ACTION["target"], "artifact": "light308",
                                   "action": "setBrightness",
                                   "parameters": {"brightness": 100}}],
                      "properties": [{"url": LIGHT_PROPERTY["target"],
                                      "artifact": "light308", "property": "brightness"}]}},
            AffordanceCatalog(AFFORDANCES))
        assert ("- light308 / action setBrightness (saref:SetLevelCommand) "
                "with parameters {'brightness': 100}") in hints
        assert "- read light308 / property brightness (homeont:DimmableLightBrightness)" in hints
        assert '"it is too dark"' in hints and "http://" not in hints

    def test_a_hint_outside_the_device_list_says_so(self):
        hints = format_signifier_hints(
            {"used": {"actions": [{"url": "http://elsewhere/unknown"}]}}, AffordanceCatalog([]))
        assert "- unknown (not in the device list)" in hints

    def test_nothing_used_is_no_hint(self):
        assert format_signifier_hints({"used": {"actions": [], "properties": []}}) == ""
        assert format_signifier_hints(None) == ""

    def test_observable_hints_name_actions_and_readable_properties(self):
        hints = format_observable_property_hints({"results": [{
            "property_uri": "glare of lab308",
            "target_max": 60,
            "readable_property_urls": [LIGHT_PROPERTY["target"]],
            "actions": [{"artifact_title": "light308", "action_name": "setBrightness",
                         "direction": "decreases", "action_target": LIGHT_ACTION["target"],
                         "settling_time_seconds": 10}]}]}, AffordanceCatalog(AFFORDANCES))
        assert "- light308 / property brightness (homeont:DimmableLightBrightness)" in hints
        assert "- light308 / action setBrightness (saref:SetLevelCommand) -> decreases" in hints
        assert "settling_time=10s" in hints and "http://" not in hints
