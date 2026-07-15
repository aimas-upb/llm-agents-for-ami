"""
Unit tests for compact affordance ids in BT planning prompts and their
resolution back to target URLs in the planner.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.prompts import (
    build_affordance_index,
    build_url_to_ref,
    format_capability_context,
    format_observable_property_hints,
    format_signifier_hints,
)


LIGHT_ACTION = {
    "artifact_id": "http://localhost:8080/workspaces/lab308/artifacts/light308#artifact",
    "artifact_name": "light308",
    "affordance_id": "http://localhost:8080/workspaces/lab308/artifacts/light308/setBrightness",
    "affordance_type": "action",
    "action_name": "setBrightness",
    "method": "POST",
    "target": "http://localhost:8080/workspaces/lab308/artifacts/light308/setBrightness",
    "input_schema": {
        "type": "object",
        "properties": {"brightness": {"type": "integer"}},
        "required": ["brightness"],
    },
}

LIGHT_PROPERTY = {
    "artifact_id": "http://localhost:8080/workspaces/lab308/artifacts/light308#artifact",
    "artifact_name": "light308",
    "affordance_id": "http://localhost:8080/workspaces/lab308/artifacts/light308/properties/brightness",
    "affordance_type": "property",
    "name": "brightness",
    "method": "GET",
    "target": "http://localhost:8080/workspaces/lab308/artifacts/light308/properties/brightness",
}

AFFORDANCES = [LIGHT_ACTION, LIGHT_PROPERTY]


@pytest.fixture
def planner():
    return AsyncBTPlanner(max_attempts=1)


class TestBuildAffordanceIndex:
    def test_assigns_artifact_slash_name_refs(self):
        index = build_affordance_index(AFFORDANCES)
        assert index["light308/setBrightness"] is LIGHT_ACTION
        assert index["light308/brightness"] is LIGHT_PROPERTY

    def test_collisions_get_numeric_suffix(self):
        dup = dict(LIGHT_ACTION)
        index = build_affordance_index([LIGHT_ACTION, dup])
        assert set(index) == {"light308/setBrightness", "light308/setBrightness-2"}

    def test_url_to_ref_round_trip(self):
        index = build_affordance_index(AFFORDANCES)
        url_to_ref = build_url_to_ref(index)
        assert url_to_ref[LIGHT_ACTION["target"]] == "light308/setBrightness"
        assert url_to_ref[LIGHT_PROPERTY["target"]] == "light308/brightness"

    def test_ignores_non_dict_entries(self):
        index = build_affordance_index([None, "junk", LIGHT_ACTION])
        assert list(index) == ["light308/setBrightness"]


class TestFormatCapabilityContext:
    def test_contains_refs_but_not_urls_or_schemas(self):
        context = format_capability_context(AFFORDANCES)
        assert "light308/setBrightness" in context
        assert "light308/brightness" in context
        assert "brightness:integer*" in context
        assert "http://localhost:8080" not in context
        assert '"properties"' not in context

    def test_state_section_is_preserved(self):
        state = {"artifacts": {"light308": {"brightness": 50}}}
        context = format_capability_context(AFFORDANCES, state)
        assert "### Current State" in context
        assert "light308: brightness=50" in context

    def test_no_affordances(self):
        assert format_capability_context([]) == "No affordances available."

    def test_state_uris_are_shortened_and_aggregates_flattened(self):
        artifact_uri = "http://localhost:8080/workspaces/lab308/artifacts/light308#artifact"
        base = artifact_uri.split("#")[0]
        state = {
            "artifacts": {
                artifact_uri: {
                    "name": "Light 308",
                    "workspace_id": "http://localhost:8080/workspaces/lab308#workspace",
                    f"{base}/properties/metadata": {"vendor": "x"},
                    # Some adapters nest the actual values in one aggregate dict
                    f"{base}/properties/state": {
                        f"{base}/properties/brightness": 50,
                        f"{base}/properties/on_off": True,
                        f"{base}/properties/metadata": {"vendor": "x"},
                    },
                }
            }
        }
        context = format_capability_context(AFFORDANCES, state)
        state_sec = context.split("### Current State")[1]
        assert "- light308:" in state_sec
        assert "- brightness: 50" in state_sec
        assert "- on_off: True" in state_sec
        assert "http://localhost:8080" not in state_sec
        assert "metadata" not in state_sec
        assert "Light 308" not in state_sec

    def test_single_prop_artifact_renders_one_line(self):
        state = {"artifacts": {"cover1": {"state": "open", "persistent": False}}}
        context = format_capability_context(AFFORDANCES, state)
        assert "- cover1: state=open" in context
        assert "persistent" not in context

    def test_sensor_only_property_affordances_not_listed_but_resolvable(self):
        sensor_prop = {
            "artifact_id": "http://localhost:8080/workspaces/lab308/artifacts/sensor42#artifact",
            "artifact_name": "sensor42",
            "affordance_id": "http://localhost:8080/workspaces/lab308/artifacts/sensor42/properties/glare",
            "affordance_type": "property",
            "name": "glare",
            "method": "GET",
            "target": "http://localhost:8080/workspaces/lab308/artifacts/sensor42/properties/glare",
        }
        affordances = AFFORDANCES + [sensor_prop]
        index = build_affordance_index(affordances)
        context = format_capability_context(affordances, index=index)

        # Not listed (sensor42 has no actions), light308 property still listed
        assert "sensor42/glare" not in context
        assert "light308/brightness" in context

        # Still resolvable through the full index (e.g. when a hint names it)
        planner = AsyncBTPlanner(max_attempts=1)
        spec = {
            "name": "CheckGlare",
            "type": "condition",
            "affordance_id": "sensor42/glare",
            "expected_value": 40,
        }
        assert planner._resolve_affordance_refs(spec, index) == []
        assert spec["property_url"] == sensor_prop["target"]


class TestResolveAffordanceRefs:
    def test_resolves_action_ref_to_action_url(self, planner):
        index = build_affordance_index(AFFORDANCES)
        spec = {
            "name": "SetBrightness",
            "type": "action",
            "affordance_id": "light308/setBrightness",
            "parameters": {"brightness": 75},
        }
        errors = planner._resolve_affordance_refs(spec, index)
        assert errors == []
        assert spec["action_url"] == LIGHT_ACTION["target"]

    def test_resolves_condition_ref_to_property_url(self, planner):
        index = build_affordance_index(AFFORDANCES)
        spec = {
            "name": "CheckBrightness",
            "type": "condition",
            "affordance_id": "light308/brightness",
            "expected_value": 75,
        }
        errors = planner._resolve_affordance_refs(spec, index)
        assert errors == []
        assert spec["property_url"] == LIGHT_PROPERTY["target"]

    def test_resolves_nested_children(self, planner):
        index = build_affordance_index(AFFORDANCES)
        spec = {
            "name": "Root",
            "type": "selector",
            "children": [
                {
                    "name": "AlreadyBright",
                    "type": "condition",
                    "affordance_id": "light308/brightness",
                    "expected_value": 75,
                },
                {
                    "name": "SetBrightness",
                    "type": "action",
                    "affordance_id": "light308/setBrightness",
                },
            ],
        }
        errors = planner._resolve_affordance_refs(spec, index)
        assert errors == []
        assert spec["children"][0]["property_url"] == LIGHT_PROPERTY["target"]
        assert spec["children"][1]["action_url"] == LIGHT_ACTION["target"]

    def test_unknown_ref_is_error(self, planner):
        index = build_affordance_index(AFFORDANCES)
        spec = {"name": "Bad", "type": "action", "affordance_id": "light308/hallucinated"}
        errors = planner._resolve_affordance_refs(spec, index)
        assert len(errors) == 1
        assert "unknown affordance_id" in errors[0]
        assert "action_url" not in spec

    def test_unknown_ref_suggests_close_matches(self, planner):
        index = build_affordance_index(AFFORDANCES)
        # Typical hallucination: near-miss mash-up of a real id
        spec = {"name": "Bad", "type": "action", "affordance_id": "light309/setBrightness"}
        errors = planner._resolve_affordance_refs(spec, index)
        assert len(errors) == 1
        assert "did you mean" in errors[0]
        assert "light308/setBrightness" in errors[0]

    def test_action_node_rejects_property_ref(self, planner):
        index = build_affordance_index(AFFORDANCES)
        spec = {"name": "Bad", "type": "action", "affordance_id": "light308/brightness"}
        errors = planner._resolve_affordance_refs(spec, index)
        assert len(errors) == 1
        assert "property affordance" in errors[0]

    def test_condition_accepts_action_ref(self, planner):
        # No-TD-SOSA path: readable GET targets are exposed as action affordances.
        index = build_affordance_index(AFFORDANCES)
        spec = {
            "name": "Check",
            "type": "condition",
            "affordance_id": "light308/setBrightness",
            "expected_value": 75,
        }
        errors = planner._resolve_affordance_refs(spec, index)
        assert errors == []
        assert spec["property_url"] == LIGHT_ACTION["target"]

    def test_condition_accepts_literal_url_passthrough(self, planner):
        # TD-SOSA hints and state keys provide readable property URLs verbatim.
        index = build_affordance_index(AFFORDANCES)
        url = "http://localhost:8080/workspaces/lab308/artifacts/sensor1/properties/glare"
        spec = {
            "name": "CheckGlare",
            "type": "wait_condition",
            "affordance_id": url,
            "expected_value": 50,
            "operator": "<=",
        }
        errors = planner._resolve_affordance_refs(spec, index)
        assert errors == []
        assert spec["property_url"] == url

    def test_legacy_tree_with_urls_still_valid(self, planner):
        # Signifier-reuse trees carry action_url/property_url directly.
        index = build_affordance_index(AFFORDANCES)
        spec = {
            "name": "Legacy",
            "type": "action",
            "action_url": LIGHT_ACTION["target"],
        }
        errors = planner._resolve_affordance_refs(spec, index)
        assert errors == []
        assert spec["action_url"] == LIGHT_ACTION["target"]

    def test_missing_ref_and_url_is_error(self, planner):
        errors = planner._resolve_affordance_refs(
            {"name": "Bad", "type": "action"}, build_affordance_index(AFFORDANCES)
        )
        assert len(errors) == 1
        assert "require 'affordance_id'" in errors[0]

    def test_resolved_tree_passes_validation(self, planner):
        index = build_affordance_index(AFFORDANCES)
        spec = {
            "name": "SetBrightness",
            "type": "action",
            "affordance_id": "light308/setBrightness",
        }
        assert planner._resolve_affordance_refs(spec, index) == []
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
            '{"tree": {"name": "T", "type": "action", "affordance_id": "a/b"}, '
            '"explanation": "x", "impossible": false}}\n```'
        )
        args = AsyncBTPlanner._parse_tool_call_content(content)
        assert args["tree"]["affordance_id"] == "a/b"

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
    async def test_generates_resolved_tree_from_refs(self):
        planner = AsyncBTPlanner(max_attempts=1)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_tool_call_response(
                {
                    "name": "SetBrightness",
                    "type": "action",
                    "affordance_id": "light308/setBrightness",
                    "parameters": {"brightness": 75},
                }
            )
        )

        result = await planner.generate_bt(
            intents=["set the brightness to 75"],
            affordances=AFFORDANCES,
            state={"artifacts": {"light308": {"brightness": 10}}},
            client=client,
            model="gpt-4o-mini",
        )

        assert result["impossible"] is False
        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]

        # Prompt must be compact: refs shown, affordance URLs hidden.
        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "light308/setBrightness" in system_prompt
        assert LIGHT_ACTION["target"] not in system_prompt

    @pytest.mark.asyncio
    async def test_hallucinated_ref_triggers_retry_then_success(self):
        planner = AsyncBTPlanner(max_attempts=2)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[
                _tool_call_response(
                    {"name": "Bad", "type": "action", "affordance_id": "light999/doesNotExist"}
                ),
                _tool_call_response(
                    {"name": "Good", "type": "action", "affordance_id": "light308/setBrightness"}
                ),
            ]
        )

        result = await planner.generate_bt(
            intents=["set the brightness"],
            affordances=AFFORDANCES,
            client=client,
            model="gpt-4o-mini",
        )

        assert client.chat.completions.create.await_count == 2
        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]

        # Retry feedback must mention the unknown id.
        retry_messages = client.chat.completions.create.call_args.kwargs["messages"]
        feedback = [m for m in retry_messages if m.get("role") == "tool"]
        assert feedback and "unknown affordance_id" in feedback[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_call_recovered_from_text_content(self):
        # Ollama models often emit the tool call as fenced JSON text.
        planner = AsyncBTPlanner(max_attempts=1)
        client = MagicMock()
        content = (
            "```json\n"
            + json.dumps({
                "name": "generate_behavior_tree",
                "arguments": {
                    "tree": {
                        "name": "SetBrightness",
                        "type": "action",
                        "affordance_id": "light308/setBrightness",
                    },
                    "explanation": "ok",
                    "impossible": False,
                },
            })
            + "\n```"
        )
        client.chat.completions.create = AsyncMock(return_value=_content_response(content))

        result = await planner.generate_bt(
            intents=["set the brightness"],
            affordances=AFFORDANCES,
            client=client,
            model="qwen2.5-coder:3b",
        )

        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]

    @pytest.mark.asyncio
    async def test_prose_response_retries_then_succeeds(self):
        planner = AsyncBTPlanner(max_attempts=2)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[
                _content_response("I think you should turn up the light."),
                _tool_call_response(
                    {"name": "Good", "type": "action", "affordance_id": "light308/setBrightness"}
                ),
            ]
        )

        result = await planner.generate_bt(
            intents=["set the brightness"],
            affordances=AFFORDANCES,
            client=client,
            model="qwen2.5-coder:3b",
        )

        assert client.chat.completions.create.await_count == 2
        assert result["tree"]["action_url"] == LIGHT_ACTION["target"]

        # Without a tool_call_id the retry feedback must be a user message.
        retry_messages = client.chat.completions.create.call_args.kwargs["messages"]
        feedback = [m for m in retry_messages if m.get("role") == "user" and "invalid" in str(m.get("content"))]
        assert feedback


SEED1_AC_ACTION = {
    "artifact_id": "http://h/w/artifacts/qt2_feasible_seed_1_utility_room_environment#artifact",
    "artifact_name": "qt2_feasible_seed_1_utility_room_environment",
    "affordance_type": "action",
    "action_name": "getHumidityInPercent",
    "target": "http://h/w/artifacts/qt2_feasible_seed_1_utility_room_environment/getHumidityInPercent",
}

SET_TEMPERATURE = {
    "artifact_id": "http://h/w/artifacts/ac1#artifact",
    "artifact_name": "ac1",
    "affordance_type": "action",
    "action_name": "ClimateSetTemperature",
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


class TestSuffixMatchResolution:
    def test_unique_token_boundary_suffix_resolves(self, planner):
        # The exact failure from ablation 006: prefix dropped from long ids.
        index = build_affordance_index([SEED1_AC_ACTION])
        spec = {
            "name": "N", "type": "action",
            "affordance_id": "utility_room_environment/getHumidityInPercent",
        }
        errors = planner._resolve_affordance_refs(spec, index)
        assert errors == []
        assert spec["action_url"] == SEED1_AC_ACTION["target"]
        assert spec["affordance_id"] == "qt2_feasible_seed_1_utility_room_environment/getHumidityInPercent"

    def test_case_insensitive_exact_match(self, planner):
        index = build_affordance_index([LIGHT_ACTION])
        spec = {"name": "N", "type": "action", "affordance_id": "Light308/setBrightness"}
        assert planner._resolve_affordance_refs(spec, index) == []
        assert spec["action_url"] == LIGHT_ACTION["target"]

    def test_mid_token_suffix_does_not_match(self, planner):
        # 'room_environment/...' must not match 'bathroom_environment/...'
        aff = dict(SEED1_AC_ACTION)
        aff["artifact_name"] = "qt2_bathroom_environment"
        index = build_affordance_index([aff])
        spec = {"name": "N", "type": "action", "affordance_id": "room_environment/getHumidityInPercent"}
        errors = planner._resolve_affordance_refs(spec, index)
        assert len(errors) == 1 and "unknown affordance_id" in errors[0]

    def test_ambiguous_suffix_is_error_listing_candidates(self, planner):
        aff2 = dict(SEED1_AC_ACTION)
        aff2["artifact_name"] = "qt2_feasible_seed_2_utility_room_environment"
        index = build_affordance_index([SEED1_AC_ACTION, aff2])
        spec = {"name": "N", "type": "action", "affordance_id": "utility_room_environment/getHumidityInPercent"}
        errors = planner._resolve_affordance_refs(spec, index)
        assert len(errors) == 1
        assert "ambiguous" in errors[0]
        assert "seed_1" in errors[0] and "seed_2" in errors[0]

    def test_suffix_match_still_enforces_type_check(self, planner):
        index = build_affordance_index(AFFORDANCES)
        spec = {"name": "N", "type": "action", "affordance_id": "light308/brightness"}
        # exact match already covers this; sanity: suffix path re-enters the
        # normal resolution incl. the action-vs-property check
        errors = planner._resolve_affordance_refs(spec, index)
        assert len(errors) == 1 and "property affordance" in errors[0]


class TestActionParameterValidation:
    def _resolve(self, planner, params):
        index = build_affordance_index([SET_TEMPERATURE])
        spec = {
            "name": "N", "type": "action",
            "affordance_id": "ac1/ClimateSetTemperature",
            "parameters": params,
        }
        return planner._resolve_affordance_refs(spec, index), spec

    def test_temperature_only_is_valid(self, planner):
        errors, _ = self._resolve(planner, {"temperature": 24.0, "hvac_mode": "cool"})
        assert errors == []

    def test_range_pair_is_valid(self, planner):
        errors, _ = self._resolve(planner, {"target_temp_high": 26.0, "target_temp_low": 22.0})
        assert errors == []

    def test_mixed_styles_rejected(self, planner):
        errors, _ = self._resolve(
            planner, {"temperature": 24.0, "target_temp_high": 30.0, "target_temp_low": 20.0}
        )
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
        index = build_affordance_index([LIGHT_ACTION])
        spec = {
            "name": "N", "type": "action",
            "affordance_id": "light308/setBrightness",
            "parameters": {"brightness": 75},
        }
        assert planner._resolve_affordance_refs(spec, index) == []

    def test_no_schema_skips_unknown_check(self, planner):
        aff = {k: v for k, v in LIGHT_ACTION.items() if k != "input_schema"}
        index = build_affordance_index([aff])
        spec = {
            "name": "N", "type": "action",
            "affordance_id": "light308/setBrightness",
            "parameters": {"anything": 1},
        }
        assert planner._resolve_affordance_refs(spec, index) == []


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
    def test_signifier_hints_show_refs(self):
        index = build_affordance_index(AFFORDANCES)
        url_to_ref = build_url_to_ref(index)
        hints = format_signifier_hints(
            {
                "turn on the light": {
                    "final_matches": ["sig-1"],
                    "matches": [
                        {
                            "signifier_id": "sig-1",
                            "affordance_uri": LIGHT_ACTION["target"],
                            "intent_similarity": 0.97,
                            "payload": {"brightness": 100},
                        }
                    ],
                }
            },
            url_to_ref=url_to_ref,
        )
        assert "Recommended affordance_id: light308/setBrightness" in hints

    def test_signifier_hints_fall_back_to_url(self):
        hints = format_signifier_hints(
            {
                "x": {
                    "final_matches": ["sig-1"],
                    "matches": [
                        {
                            "signifier_id": "sig-1",
                            "affordance_uri": "http://elsewhere/unknown",
                            "intent_similarity": 0.9,
                        }
                    ],
                }
            },
            url_to_ref={},
        )
        assert "Recommended affordance_id: http://elsewhere/unknown" in hints

    def test_observable_hints_map_urls_to_refs(self):
        index = build_affordance_index(AFFORDANCES)
        url_to_ref = build_url_to_ref(index)
        hints = format_observable_property_hints(
            {
                "results": [
                    {
                        "property_uri": "http://localhost:8080/workspaces/lab308/environment/glare",
                        "target_max": 60,
                        "readable_property_urls": [LIGHT_PROPERTY["target"]],
                        "actions": [
                            {
                                "artifact_title": "light308",
                                "action_name": "setBrightness",
                                "direction": "decreases",
                                "action_target": LIGHT_ACTION["target"],
                                "settling_time_seconds": 10,
                            }
                        ],
                    }
                ]
            },
            url_to_ref=url_to_ref,
        )
        assert "light308/brightness" in hints
        assert "affordance_id: light308/setBrightness" in hints
        assert "settling_time=10s" in hints
