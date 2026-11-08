"""Prompt variants: the registry, the goal-structuring options, and the harness side.

No LLM, no simulator, no EnvExplorer: the options are plain functions over a small
structured goal context, the behaviour runs with its RPC and LLM client patched (as
`test_goal_structuring.py` does), and the harness side reads variant dicts and files.
What each group pins:

* the registry (`shared/utils/prompt_variants.py`): defaults, the errors for a
  misspelt step / axis / option, normalising a variant file's choices, writing them
  into the agents' config, and that a second step can be declared the same way;
* the structuring options: the default call is the old one byte for byte; each option
  does what its docstring says (the JSON context, the focused prompt, the strict schema
  and the class-name repair);
* `GoalStructuringBehaviour` with variants configured: the schema reaches the API call,
  the repaired reply is what is structured, a context without structure is refused;
* the harness: `prompt_variants:` in a variant file, its record, its cache key, the
  arm files shipped in `tests/homebench/variants/`.
"""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from ami_agents.agents.user_assistant.behaviours.goal_structuring import (
    GoalStructuringBehaviour,
)
from ami_agents.agents.user_assistant.prompts.goal_structuring_prompts import (
    CONTEXT_GUIDE,
    CONTEXT_GUIDE_JSON,
    DEVICE_CONVENTIONS,
    FOCUSED_EXAMPLES,
    PROMPTS,
    render_goal_prompt,
)
from ami_agents.agents.user_assistant.utils import goal_structuring_variants as gsv
from ami_agents.agents.user_assistant.utils.goal_context import GoalContext, vocabulary
from ami_agents.shared.utils import prompt_variants as pv
from tests.simuhome.bench import models
from tests.simuhome.bench.run import apply_prompt_variants

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARM_FILES = sorted((PROJECT_ROOT / "tests" / "homebench" / "variants").glob(
    "homebench.ecai.M3.pv-*.yaml"))
MODULE = "ami_agents.agents.user_assistant.behaviours.goal_structuring"


# -- a small home, as EnvExplorer's structured view ----------------------------

def _home_data():
    """The structured goal context of a two-room home (shape of
    `actuation_context.build_actuation_context`)."""
    light = {
        "title": "master_bedroom_light", "hmas_types": ["hmas:Artifact"],
        "saref_types": ["saref:Actuator"], "homeont_types": ["homeont:DimmableLight"],
        "label": "Dimmable Light",
        "actions": [
            {"title": "brightness", "command_types": ["saref:SetAbsoluteLevelCommand"],
             "input": {"form": "direct value", "type": "js:IntegerSchema",
                       "minimum": 0, "maximum": 100, "unit": "unit:PERCENT"}},
            {"title": "onOff", "command_types": ["homeont:SetOnOffCommand"],
             "input": {"form": "direct value", "type": "js:BooleanSchema"}}]}
    washer = {
        "title": "laundry_washer", "hmas_types": [], "saref_types": [],
        "homeont_types": ["homeont:Washer"],
        "actions": [
            {"title": "state", "command_types": ["homeont:SetOperationalStateCommand"],
             "input": {"form": "parameterized call", "parameters": [
                 {"parameter_name": "operation",
                  "parameter_schema": {"form": "direct value", "type": "js:StringSchema"}}]}}]}
    return {"workspaces": [{
        "title": "home6", "artifacts": [], "workspaces": [
            {"title": "Master Bedroom", "space_class": "homeont:MasterBedroom",
             "environment": {"label": "env", "variables": [
                 {"title": "illuminance", "class": "homeont:Illuminance",
                  "quantity_kind": "quantitykind:Illuminance", "unit": "unit:LUX",
                  "observed_by": []}]},
             "artifacts": [light]},
            {"title": "Laundry", "space_class": "homeont:LaundryRoom",
             "artifacts": [washer]}]}]}


def _context(with_data=True):
    return GoalContext(text="HOME TEXT", data=_home_data() if with_data else None)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_no_selection_is_every_default(self):
        selection = pv.resolve("structuring")
        assert selection.chosen == {"context": "text", "prompt": "full",
                                    "constraint": "none", "repair": "none"}
        assert selection.non_default() == {} and selection.label() == "default"

    def test_a_config_without_the_key_is_the_default(self):
        assert pv.selection_from_config(None, "structuring").label() == "default"
        assert pv.selection_from_config({}, "structuring").label() == "default"
        assert pv.selection_from_config({"prompt_variants": {}}, "structuring"
                                        ).label() == "default"

    def test_an_axis_left_out_keeps_its_default(self):
        selection = pv.selection_from_config(
            {"prompt_variants": {"structuring": {"constraint": "schema"}}}, "structuring")
        assert selection.chosen["constraint"] == "schema"
        assert selection.chosen["context"] == "text"
        assert selection.label() == "constraint=schema"
        assert selection["constraint"] is gsv.schema_constraint

    @pytest.mark.parametrize("request_, message", [
        ({"contexte": "json"}, "no axis 'contexte'"),
        ({"context": "jsonn"}, "no option 'jsonn'"),
    ])
    def test_a_typo_is_an_error_listing_what_exists(self, request_, message):
        with pytest.raises(ValueError, match=message) as error:
            pv.resolve("structuring", request_)
        # The message names the valid choices, so the fix is in the error.
        assert ("context" in str(error.value)) and ("text" in str(error.value)
                                                      or "constraint" in str(error.value))

    def test_an_unknown_step_is_an_error(self):
        with pytest.raises(ValueError, match="unknown step 'structuring2'"):
            pv.resolve("structuring2")

    def test_normalize_keeps_only_what_differs_from_the_defaults(self):
        assert pv.normalize({"structuring": {"context": "text", "repair": "class_names"}}
                            ) == {"structuring": {"repair": "class_names"}}
        # A step that names only defaults is the same as one not named.
        assert pv.normalize({"structuring": {"context": "text"}}) == {}
        assert pv.normalize(None) == {}

    def test_normalize_validates(self):
        with pytest.raises(ValueError):
            pv.normalize({"structuring": {"repair": "nope"}})

    def test_apply_to_config_writes_under_the_steps_agent(self):
        config = {"user_assistant": {"prompt_variants": {"structuring": {
            "context": "json", "prompt": "focused"}}}}
        pv.apply_to_config(config, {"structuring": {"constraint": "schema"}})
        # Replaced whole: the variant file decides every axis of a step it names.
        assert config["user_assistant"]["prompt_variants"]["structuring"] == {
            "constraint": "schema"}
        assert pv.selection_from_config(config["user_assistant"], "structuring"
                                        ).label() == "constraint=schema"

    def test_apply_to_config_creates_the_sections(self):
        config = {}
        pv.apply_to_config(config, {"structuring": {"repair": "class_names"}})
        assert config == {"user_assistant": {"prompt_variants": {
            "structuring": {"repair": "class_names"}}}}

    def test_catalog_lists_every_axis_with_its_default_and_options(self):
        axes = pv.catalog()["structuring"]
        assert axes["context"]["default"] == "text"
        assert axes["context"]["options"] == ["text", "json"]
        assert axes["prompt"]["options"] == ["full", "focused"]
        assert axes["constraint"]["options"] == ["none", "schema"]
        assert axes["repair"]["options"] == ["none", "class_names"]

    def test_describe_marks_what_is_not_the_default(self):
        lines = pv.describe(pv.resolve("structuring", {"context": "json"}))
        assert "context=json *" in lines and "prompt=full" in lines

    def test_another_step_is_declared_the_same_way(self, monkeypatch):
        """The mechanism is not the structuring step's: a step declared elsewhere
        resolves, validates and applies the same, under its own agent."""
        monkeypatch.setattr(pv, "_STEPS", dict(pv._STEPS))
        monkeypatch.setitem(pv.STEP_MODULES, "planning", "unused: declared below")
        step = pv.declare_step("planning", agent="interaction_solver")
        tone = step.axis("tone", "plain", "tone of the system prompt")

        @tone.option("plain")
        def plain() -> str:
            return "plain"

        @tone.option("terse")
        def terse() -> str:
            return "terse"

        selection = pv.resolve("planning", {"tone": "terse"})
        assert selection["tone"]() == "terse" and selection.label() == "tone=terse"
        with pytest.raises(ValueError, match="no option 'loud'"):
            pv.resolve("planning", {"tone": "loud"})
        config = {}
        pv.apply_to_config(config, pv.normalize({"planning": {"tone": "terse"}}))
        assert config == {"interaction_solver": {"prompt_variants": {
            "planning": {"tone": "terse"}}}}

    def test_an_option_name_cannot_be_registered_twice(self):
        with pytest.raises(ValueError, match="already has an option"):
            gsv.CONTEXT.option("text")(lambda context: None)

    def test_an_axis_whose_default_is_not_registered_is_caught(self, monkeypatch):
        monkeypatch.setattr(pv, "_STEPS", dict(pv._STEPS))
        monkeypatch.setitem(pv.STEP_MODULES, "broken", "unused")
        step = pv.declare_step("broken", agent="user_assistant")
        step.axis("x", "missing", "no option registered")
        with pytest.raises(ValueError, match="default 'missing' is not a registered"):
            pv.resolve("broken")


# ---------------------------------------------------------------------------
# the structuring options
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_names_and_classes_of_the_context(self):
        vocab = vocabulary(_home_data())
        assert vocab.room_classes == ["homeont:MasterBedroom", "homeont:LaundryRoom"]
        assert vocab.artifact_names == ["master_bedroom_light", "laundry_washer"]
        assert vocab.artifact_classes == ["homeont:DimmableLight", "homeont:Washer"]
        assert vocab.affordance_names == ["brightness", "onOff", "state"]
        assert vocab.affordance_classes == [
            "saref:SetAbsoluteLevelCommand", "homeont:SetOnOffCommand",
            "homeont:SetOperationalStateCommand"]
        assert vocab.parameter_names == ["operation"]
        assert vocab.variable_classes == ["homeont:Illuminance"]
        assert vocab.units == ["unit:LUX"]

    def test_a_context_without_structure_is_refused_by_name(self):
        with pytest.raises(ValueError, match="repair=class_names needs the structured"):
            _context(with_data=False).require_data("repair=class_names")


class TestContextOption:
    def test_text_is_what_structuring_always_sent(self):
        rendered = gsv.text_context(_context())
        assert (rendered.guide, rendered.body) == (CONTEXT_GUIDE, "HOME TEXT")

    def test_json_is_the_structured_view_without_nulls(self):
        data = _home_data()
        data["workspaces"][0]["workspaces"][0]["artifacts"][0]["manufacturer"] = None
        rendered = gsv.json_context(GoalContext(text="x", data=data))
        body = json.loads(rendered.body)
        light = body["workspaces"][0]["workspaces"][0]["artifacts"][0]
        assert light["title"] == "master_bedroom_light" and "manufacturer" not in light
        # An empty list is a fact (nothing reads the variable), so it stays.
        env = body["workspaces"][0]["workspaces"][0]["environment"]["variables"][0]
        assert env["observed_by"] == []
        assert rendered.guide == CONTEXT_GUIDE_JSON

    def test_json_needs_the_structure(self):
        with pytest.raises(ValueError, match="context=json needs the structured"):
            gsv.json_context(_context(with_data=False))


class TestPromptOption:
    @pytest.mark.parametrize("structure", ["simple", "dependency"])
    def test_default_is_the_old_prompt_byte_for_byte(self, structure):
        rendered = gsv.text_context(_context())
        assert gsv.full_prompt(structure, rendered, ["explicit", "achievement"]) == \
            render_goal_prompt(structure, "HOME TEXT")

    def test_the_default_prompts_are_unchanged_since_before_variants(self):
        """Pinned by hash: the control arm (A) must be the prompt of the runs made
        before variants existed, so the comparison with them stays valid."""
        assert hashlib.sha1(PROMPTS["simple"].encode()).hexdigest() == \
            "39555101d903d1654bafa15a715f93f28af15526"
        assert hashlib.sha1(PROMPTS["dependency"].encode()).hexdigest() == \
            "2aa791ded47cf21a180749bb452633e2140b7c6d"

    def test_full_with_the_json_guide_swaps_the_guide_only(self):
        rendered = gsv.json_context(_context())
        prompt = gsv.full_prompt("simple", rendered, ["explicit", "achievement"])
        assert CONTEXT_GUIDE_JSON in prompt and CONTEXT_GUIDE not in prompt
        assert prompt.count(rendered.body) == 1 and "{goal_context}" not in prompt

    def test_focused_drops_what_an_explicit_goal_does_not_need(self):
        rendered = gsv.text_context(_context())
        focused = gsv.focused_prompt("simple", rendered, ["explicit", "achievement"])
        full = gsv.full_prompt("simple", rendered, ["explicit", "achievement"])
        assert "### ambiguous goals" in full and "### ambiguous goals" not in focused
        assert "For an ambiguous goal" not in focused
        assert FOCUSED_EXAMPLES in focused and "HOME TEXT" in focused
        # The rules of explicit and incomplete goals are all still there.
        assert "### explicit and incomplete goals" in focused
        assert "{" + "conventions}" not in focused and "{context_guide}" not in focused

    def test_focused_keeps_the_cycle_convention_only_for_a_home_with_a_cycle_device(self):
        plain = gsv.focused_prompt("simple", gsv.text_context(_context()),
                                   ["explicit", "achievement"])
        assert DEVICE_CONVENTIONS not in plain
        with_cycle = gsv.focused_prompt(
            "simple", gsv.RenderedContext(CONTEXT_GUIDE, "command class: "
                                          "homeont:SetOperationalStateCommand"),
            ["explicit", "achievement"])
        assert DEVICE_CONVENTIONS in with_cycle

    @pytest.mark.parametrize("structure, qualifiers", [
        ("simple", ["ambiguous", "achievement"]),
        ("dependency", ["explicit", "logical_dependency", "achievement"]),
    ])
    def test_focused_is_the_full_prompt_for_the_goals_it_leaves_out(self, structure, qualifiers):
        rendered = gsv.text_context(_context())
        assert gsv.focused_prompt(structure, rendered, qualifiers) == \
            gsv.full_prompt(structure, rendered, qualifiers)

    def test_the_prompt_report_can_still_take_a_focused_prompt_apart(self):
        """A focused prompt is a registered template, so the prompt report shows its
        parts and the home context as a field, not one anonymous message."""
        from tests.homebench import prompt_parts
        rendered = gsv.text_context(_context())
        part = prompt_parts.decompose_message(
            gsv.focused_prompt("simple", rendered, ["explicit", "achievement"]))
        assert part.kind == "template" and part.name == "GOAL_STRUCTURING_FOCUSED_PROMPT"


class TestSchemaConstraint:
    def _schema(self, qualifiers=("explicit", "achievement")):
        constraint = gsv.schema_constraint("simple", list(qualifiers), _context())
        assert constraint["type"] == "json_schema"
        assert constraint["json_schema"]["strict"] is True
        return constraint["json_schema"]["schema"]

    def _reply(self, **changes):
        reply = {"goal_specificity": "explicit", "goal_kind": "achievement",
                 "intent_text": "Set the brightness of the light to 10",
                 "location_class": "homeont:MasterBedroom",
                 "artifact_class": "homeont:DimmableLight",
                 "artifact_name": "master_bedroom_light", "device_reference": "description",
                 "affordance_class": "saref:SetAbsoluteLevelCommand", "parent_class": None,
                 "affordance_name": "brightness", "parameter_name": "NA",
                 "goal_effect": "set", "target_value_text": "10",
                 "target_value_determined": "10", "target_value_is_percentage": False,
                 "quantifier": None, "unmatched_device": None, "unmatched_action": None}
        reply.update(changes)
        return reply

    def test_strict_shape_every_property_required_and_nothing_extra(self):
        schema = self._schema()
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])

    def test_it_follows_the_prompts_output_shape(self):
        """The schema has the fields of SIMPLE_OUTPUT_GOAL, in its order: the prompt
        and the schema describe one reply, and a field added to one shows here."""
        import re

        from ami_agents.agents.user_assistant.prompts.goal_structuring_prompts import (
            SIMPLE_OUTPUT_GOAL,
        )
        shown = re.findall(r'"([a-z_]+)":', SIMPLE_OUTPUT_GOAL)
        assert shown == list(self._schema()["properties"])

    def test_slots_that_name_something_take_the_homes_names(self):
        props = self._schema()["properties"]

        def values(slot):
            return props[slot]["anyOf"][0]["enum"]
        assert values("location_class") == ["homeont:MasterBedroom", "homeont:LaundryRoom"]
        assert values("artifact_name") == ["master_bedroom_light", "laundry_washer"]
        assert values("affordance_name") == ["brightness", "onOff", "state"]
        # "NA" and null both stay possible for a parameter.
        assert values("parameter_name") == ["operation", "NA"]
        assert {"type": "null"} in props["parameter_name"]["anyOf"]

    def test_a_correct_reply_validates_and_the_bare_class_name_does_not(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = self._schema()
        jsonschema.validate(self._reply(), schema)
        for slot, bare in (("artifact_class", "DimmableLight"),
                           ("location_class", "Master bedroom"),
                           ("affordance_class", "SetAbsoluteLevelCommand"),
                           ("artifact_name", "bedroom_light")):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(self._reply(**{slot: bare}), schema)

    def test_an_ambiguous_goal_gets_the_environment_shape(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = self._schema(("ambiguous", "achievement"))
        assert "implied_environment_vars" in schema["properties"]
        assert "artifact_name" not in schema["properties"]
        jsonschema.validate({
            "goal_specificity": "ambiguous", "goal_kind": "achievement",
            "intent_text": "too dark in here",
            "implied_environment_vars": [{
                "property_name": "illuminance", "property_class": "homeont:Illuminance",
                "property_quantity": "quantitykind:Illuminance",
                "property_measurement_unit": "unit:LUX",
                "property_sensed_space": {"space_class": "homeont:MasterBedroom",
                                          "space_name": "Master Bedroom"},
                "property_change_direction": "increase"}]}, schema)

    def test_the_dependency_structure_gets_no_schema(self):
        assert gsv.schema_constraint(
            "dependency", ["explicit", "logical_dependency"], _context()) is None

    def test_a_slot_the_context_has_no_values_for_stays_a_free_string(self):
        empty = GoalContext(text="x", data={"workspaces": []})
        schema = gsv.schema_constraint("simple", ["explicit", "achievement"], empty)[
            "json_schema"]["schema"]
        assert schema["properties"]["artifact_name"]["anyOf"][0] == {"type": "string"}

    def test_it_needs_the_structure(self):
        with pytest.raises(ValueError, match="constraint=schema needs the structured"):
            gsv.schema_constraint("simple", ["explicit"], _context(with_data=False))

    def test_none_is_no_response_format(self):
        assert gsv.no_constraint("simple", ["explicit"], _context()) is None


class TestClassNameRepair:
    CANDIDATES = ["homeont:Foyer", "homeont:MasterBedroom", "saref:Light", "homeont:Light"]

    def test_a_missing_prefix_is_restored(self):
        assert gsv.canonical_class("Foyer", self.CANDIDATES) == "homeont:Foyer"

    def test_case_and_spacing_are_notation_too(self):
        assert gsv.canonical_class("Master bedroom", self.CANDIDATES) == \
            "homeont:MasterBedroom"
        assert gsv.canonical_class("master_bedroom", self.CANDIDATES) == \
            "homeont:MasterBedroom"

    def test_a_class_the_home_has_is_kept(self):
        assert gsv.canonical_class("saref:Light", self.CANDIDATES) == "saref:Light"

    def test_a_name_that_fits_two_classes_is_a_guess_and_is_left_alone(self):
        assert gsv.canonical_class("Light", self.CANDIDATES) == "Light"

    def test_an_unknown_class_is_left_alone(self):
        assert gsv.canonical_class("Curtain", self.CANDIDATES) == "Curtain"
        assert gsv.canonical_class("Foyer", []) == "Foyer"

    def test_a_wrong_prefix_on_a_known_name_is_corrected(self):
        assert gsv.canonical_class("saref:Foyer", self.CANDIDATES) == "homeont:Foyer"

    def test_the_reply_is_repaired_slot_by_slot_and_the_input_kept(self):
        parsed = {"location_class": "Master bedroom", "artifact_class": "DimmableLight",
                  "affordance_class": "SetAbsoluteLevelCommand",
                  "artifact_name": "master_bedroom_light", "parent_class": None}
        original = copy.deepcopy(parsed)
        repaired, log = gsv.class_names_repair(parsed, _context())
        assert repaired["location_class"] == "homeont:MasterBedroom"
        assert repaired["artifact_class"] == "homeont:DimmableLight"
        assert repaired["affordance_class"] == "saref:SetAbsoluteLevelCommand"
        # Names are not classes: untouched. The input is not mutated.
        assert repaired["artifact_name"] == "master_bedroom_light"
        assert parsed == original
        assert len(log) == 3 and "location_class: 'Master bedroom'" in log[0]

    def test_a_class_in_the_wrong_slot_is_not_moved(self):
        """Slot confusion is the model's error: the repair is notation only."""
        repaired, log = gsv.class_names_repair(
            {"location_class": "Washer"}, _context())
        assert repaired == {"location_class": "Washer"} and log == []

    def test_the_dependency_shape_is_repaired_inside_goals_and_predicates(self):
        parsed = {"goals": {"G1": {"location_class": "LaundryRoom",
                                   "implied_environment_vars": [{
                                       "property_class": "Illuminance",
                                       "property_sensed_space": {
                                           "space_class": "MasterBedroom"}}]}},
                  "predicates": {"P1": {"artifact_class": "Washer"}}}
        repaired, log = gsv.class_names_repair(parsed, _context())
        assert repaired["goals"]["G1"]["location_class"] == "homeont:LaundryRoom"
        var = repaired["goals"]["G1"]["implied_environment_vars"][0]
        assert var["property_class"] == "homeont:Illuminance"
        assert var["property_sensed_space"]["space_class"] == "homeont:MasterBedroom"
        assert repaired["predicates"]["P1"]["artifact_class"] == "homeont:Washer"
        assert len(log) == 4

    def test_none_changes_nothing(self):
        parsed = {"location_class": "Foyer"}
        assert gsv.no_repair(parsed, _context()) == (parsed, [])

    def test_it_needs_the_structure(self):
        with pytest.raises(ValueError, match="repair=class_names needs the structured"):
            gsv.class_names_repair({}, _context(with_data=False))


# ---------------------------------------------------------------------------
# the behaviour, with variants configured
# ---------------------------------------------------------------------------

def _llm_returning(payload):
    completion = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(payload)))])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return SimpleNamespace(client=client, model="test-model")


async def _run(variants, payload, context, qualifiers=("explicit", "achievement")):
    behaviour = GoalStructuringBehaviour("set the brightness of the light to 10 in the "
                                         "master bedroom", list(qualifiers))
    behaviour.agent = MagicMock()
    behaviour.agent.config = {"prompt_variants": {"structuring": variants}}
    llm = _llm_returning(payload)
    with patch(f"{MODULE}.pipeline") as pl, \
         patch(f"{MODULE}.build_behaviour_llm_client", return_value=llm), \
         patch(f"{MODULE}.build_llm_call_kwargs", return_value={"temperature": 0}):
        pl.fetch_goal_context_data = AsyncMock(return_value=context)
        pl.resolve_candidates = AsyncMock(return_value={})
        await behaviour.run()
    return behaviour, llm.client.chat.completions.create.await_args.kwargs


BARE = {"goal_specificity": "explicit", "goal_kind": "achievement",
        "location_class": "Master bedroom", "artifact_class": "DimmableLight",
        "artifact_name": "master_bedroom_light",
        "affordance_class": "SetAbsoluteLevelCommand", "affordance_name": "brightness",
        "parameter_name": "NA", "goal_effect": "set", "target_value_determined": "10"}


class TestBehaviourWithVariants:
    @pytest.mark.asyncio
    async def test_default_sends_the_old_call(self):
        behaviour, call = await _run({}, BARE, _context())
        assert call["messages"][0]["content"] == render_goal_prompt("simple", "HOME TEXT")
        assert "response_format" not in call and call["temperature"] == 0
        # Nothing repaired: the reply's notation reaches the structure as written.
        assert behaviour.repairs == []
        assert behaviour.result.goals["G1"].artifact_class == "DimmableLight"

    @pytest.mark.asyncio
    async def test_schema_reaches_the_api_call(self):
        _, call = await _run({"constraint": "schema"}, BARE, _context())
        fmt = call["response_format"]
        assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
        assert "homeont:DimmableLight" in json.dumps(fmt)

    @pytest.mark.asyncio
    async def test_json_context_and_focused_prompt_reach_the_system_message(self):
        _, call = await _run({"context": "json", "prompt": "focused"}, BARE, _context())
        system = call["messages"][0]["content"]
        assert CONTEXT_GUIDE_JSON in system and FOCUSED_EXAMPLES in system
        assert '"title": "master_bedroom_light"' in system
        assert "HOME TEXT" not in system

    @pytest.mark.asyncio
    async def test_repair_is_what_gets_structured_and_is_recorded(self):
        behaviour, _ = await _run({"repair": "class_names"}, BARE, _context())
        goal = behaviour.result.goals["G1"]
        assert goal.location_class == "homeont:MasterBedroom"
        assert goal.artifact_class == "homeont:DimmableLight"
        assert goal.affordance_class == "saref:SetAbsoluteLevelCommand"
        assert len(behaviour.repairs) == 3

    @pytest.mark.asyncio
    async def test_a_variant_that_needs_structure_fails_the_goal_when_there_is_none(self):
        behaviour = GoalStructuringBehaviour("turn on the light", ["explicit", "achievement"])
        behaviour.agent = MagicMock()
        behaviour.agent.config = {"prompt_variants": {"structuring": {"repair": "class_names"}}}
        with patch(f"{MODULE}.pipeline") as pl, \
             patch(f"{MODULE}.build_behaviour_llm_client",
                   return_value=_llm_returning(BARE)), \
             patch(f"{MODULE}.build_llm_call_kwargs", return_value={}):
            pl.fetch_goal_context_data = AsyncMock(
                return_value=GoalContext(text="HOME TEXT"))
            await behaviour.run()
        assert behaviour.result is None
        assert "needs the structured goal context" in behaviour.error

    def test_a_misspelt_variant_stops_the_agent_at_startup(self):
        """Not a run on the default, and not an error per goal: the agent refuses to
        be built (its config is checked in `__init__`)."""
        from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
        config = {"llm": {}, "default_llm": {"default_model": "m"},
                  "prompt_variants": {"structuring": {"constraint": "schemaa"}}}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}),              pytest.raises(ValueError, match="no option 'schemaa'"):
            UserAssistantAgent("ua@localhost", "pw", config, {})

    @pytest.mark.asyncio
    async def test_a_misspelt_variant_fails_the_goal_with_the_reason(self):
        behaviour, _ = await _run_failing({"constraint": "schemaa"})
        assert behaviour.result is None and "no option 'schemaa'" in behaviour.error


async def _run_failing(variants):
    """`_run` for a goal whose LLM call never happens: the behaviour and its error."""
    behaviour = GoalStructuringBehaviour("turn on the light", ["explicit", "achievement"])
    behaviour.agent = MagicMock()
    behaviour.agent.config = {"prompt_variants": {"structuring": variants}}
    with patch(f"{MODULE}.pipeline") as pl,          patch(f"{MODULE}.build_behaviour_llm_client", return_value=_llm_returning(BARE)),          patch(f"{MODULE}.build_llm_call_kwargs", return_value={}):
        pl.fetch_goal_context_data = AsyncMock(return_value=_context())
        await behaviour.run()
    return behaviour, None


# ---------------------------------------------------------------------------
# the harness side
# ---------------------------------------------------------------------------

def _variant(prompts=None):
    raw = {"models": {"structuring": {"model": "m", "temperature": 0}}}
    if prompts is not None:
        raw["prompt_variants"] = prompts
    return models.variant_from_dict(raw, "t")


class TestVariantFile:
    def test_a_variant_without_prompt_variants_is_what_it_was(self):
        variant = _variant()
        assert variant.prompts == {} and "prompt_variants" not in variant.as_dict()

    def test_prompt_variants_are_read_and_recorded(self):
        variant = _variant({"structuring": {"constraint": "schema", "context": "text"}})
        assert variant.prompts == {"structuring": {"constraint": "schema"}}
        assert variant.as_dict()["prompt_variants"] == variant.prompts

    def test_a_typo_in_the_file_is_an_error(self):
        with pytest.raises(ValueError, match="no option"):
            _variant({"structuring": {"constraint": "strict"}})

    def test_the_step_cache_key_follows_the_prompt_variants(self):
        base = _variant().key("segmentation", "structuring")
        schema = _variant({"structuring": {"constraint": "schema"}}
                          ).key("segmentation", "structuring")
        repair = _variant({"structuring": {"repair": "class_names"}}
                          ).key("segmentation", "structuring")
        assert len({base, schema, repair}) == 3
        # Naming only defaults is the same variant: it must reuse the same cache.
        assert _variant({"structuring": {"context": "text"}}
                        ).key("segmentation", "structuring") == base
        # A step the key does not cover does not change it.
        assert _variant({"structuring": {"constraint": "schema"}}).key("planning.explicit") \
            == _variant().key("planning.explicit")

    def test_the_key_of_a_variant_without_them_is_the_key_it_always_had(self):
        """Pinned: a step cache written before prompt variants existed still matches."""
        variant = _variant()
        steps = ("segmentation", "structuring")
        legacy = hashlib.sha1(json.dumps(
            {s: models.asdict(variant.specs[s]) for s in steps},
            sort_keys=True).encode()).hexdigest()[:12]
        assert variant.key(*steps) == legacy

    def test_run_writes_them_into_the_agents_config(self):
        config = {}
        apply_prompt_variants(config, _variant({"structuring": {"context": "json"}}))
        assert pv.selection_from_config(config["user_assistant"], "structuring"
                                        ).label() == "context=json"

    def test_a_variant_without_them_leaves_the_config_to_agents_yaml(self):
        config = {"user_assistant": {"prompt_variants": {"structuring": {"repair": "none"}}}}
        apply_prompt_variants(config, _variant())
        assert config == {"user_assistant": {"prompt_variants": {
            "structuring": {"repair": "none"}}}}


class TestArmFiles:
    EXPECTED = {
        "homebench.ecai.M3.pv-schema": {"constraint": "schema"},
        "homebench.ecai.M3.pv-json": {"context": "json"},
        "homebench.ecai.M3.pv-focused": {"prompt": "focused"},
        "homebench.ecai.M3.pv-repair": {"repair": "class_names"},
    }

    def test_every_arm_changes_exactly_one_axis_of_the_control(self):
        assert {p.stem for p in ARM_FILES} == set(self.EXPECTED)
        for path in ARM_FILES:
            variant = models.load_variant(str(path))
            assert variant.prompts == {"structuring": self.EXPECTED[path.stem]}, path.stem

    def test_every_arm_has_the_controls_models_and_run_settings(self):
        """The arms differ from M3 in the prompt variants and nothing else: the
        difference in a result is then the variant's."""
        control = yaml.safe_load((PROJECT_ROOT / "tests/homebench/variants/"
                                  "homebench.ecai.M3.yaml").read_text(encoding="utf-8"))
        for path in ARM_FILES:
            arm = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert arm.pop("prompt_variants")
            assert arm == control, path.name
