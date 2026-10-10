"""Ambiguous goals: a context hung off each implied variable, and its planning.

A hand-written home, closed under subclass inference as the InteractionSolver
closes its graph. Every room is an hmas workspace whose space has an
environment holding the room's variables:

    Bathroom     dehumidifier  onOff -> affects bathroom humidity; actsUpon onOff
                 humidifier    mist  -> increases bathroom humidity
                 sensor        humidity reading
    Living Room  dehumidifier  onOff -> affects living-room humidity
                 ac            onOff -> affects living-room temperature
    Bedroom      fan_a (title "Bedroom")  /  fan_b (title "Guest Bedroom"):
                 each affects its own room's temperature
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rdflib import Graph

from ami_agents.agents.interaction_solver.behaviours.ambiguous_goal_planning import (
    AmbiguousGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.utils.ambiguous_goal_plan import ambiguous_goal
from ami_agents.agents.interaction_solver.utils.specific_capability_context import (
    DIRECTION_INSTRUCTIONS,
    build_specific_context,
)
from ami_agents.shared.models.goal_structure import GoalSpec, GoalStructure
from ami_agents.shared.utils.vocabulary import close_types

PREFIXES = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .
@prefix ssn: <http://www.w3.org/ns/ssn/> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix tdsosa: <https://example.org/hmas/td-sosa-ext#> .
"""


def _room(key, title, cls, variables):
    """A workspace, its space of `cls`, the space's environment and its variables."""
    vars_ = " , ".join(f"<http://h/{key}#{v}>" for v, _ in variables)
    typed = "".join(f"<http://h/{key}#{v}> a homeont:{c} ; td:title \"{v}\" .\n"
                    for v, c in variables)
    return (f"<http://h/{key}#ws> a hmas:Workspace ; td:title \"{title}\" ; "
            f"homeont:hasSpace <http://h/{key}#space> .\n"
            f"<http://h/{key}#space> a homeont:{cls} ; homeont:hasEnvironment <http://h/{key}#env> .\n"
            f"<http://h/{key}#env> ssn:hasProperty {vars_} .\n" + typed)


def _device(room, name, cls, action, var, direction=None, reading=None):
    """A device in `room` whose `action` acts on the room variable `var`."""
    kind = {"increase": ", tdsosa:IncreasingActuation",
            "decrease": ", tdsosa:DecreasingActuation"}.get(direction, "")
    out = (f"<http://h/{room}#ws> hmas:contains <http://h/{name}#artifact> .\n"
           f"<http://h/{name}#artifact> a hmas:Artifact, homeont:{cls} ; td:title \"{name}\" ;\n"
           f"  td:hasActionAffordance <http://h/{name}/actions/{action}> ;\n"
           f"  td:hasPropertyAffordance <http://h/{name}/properties/{action}> .\n"
           f"<http://h/{name}/actions/{action}> a td:ActionAffordance, homeont:SetOnOffCommand ;\n"
           f"  td:title \"{action}\" ; saref:actsUpon <http://h/{name}/properties/{action}> ;\n"
           f"  td:hasForm [ hctl:hasTarget <http://h/{name}/actions/{action}> ] ;\n"
           f"  td:hasInputSchema [ a js:BooleanSchema ] ;\n"
           f"  tdsosa:hasEffectActuation [ a sosa:Actuation{kind} ; "
           f"sosa:actsOnProperty <http://h/{room}#{var}> ] .\n"
           f"<http://h/{name}/properties/{action}> a td:PropertyAffordance, "
           f"homeont:DimmableLightOnOff ; td:title \"{action}\" ;\n"
           f"  td:hasForm [ hctl:hasTarget <http://h/{name}/properties/{action}> ] .\n")
    if reading:
        out += (f"<http://h/{name}#artifact> td:hasPropertyAffordance <http://h/{name}/properties/{reading}> .\n"
                f"<http://h/{name}/properties/{reading}> a td:PropertyAffordance, homeont:RelativeHumidity ;\n"
                f"  td:title \"{reading}\" ; td:hasForm [ hctl:hasTarget <http://h/{name}/properties/{reading}> ] .\n")
    return out


SENSOR = """
<http://h/bathroom#ws> hmas:contains <http://h/sensor#artifact> .
<http://h/sensor#artifact> a hmas:Artifact, homeont:Dehumidifier ; td:title "sensor" ;
  td:hasPropertyAffordance <http://h/sensor/properties/humidity> .
<http://h/sensor/properties/humidity> a td:PropertyAffordance, homeont:RelativeHumidity ;
  td:title "humidity" ; td:hasForm [ hctl:hasTarget <http://h/sensor/properties/humidity> ] .
"""

HOME = PREFIXES + "".join([
    _room("bathroom", "Bathroom", "Bathroom", [("relative_humidity", "RelativeHumidity")]),
    _room("living", "Living Room", "LivingRoom", [("relative_humidity", "RelativeHumidity"),
                                                  ("air_temperature", "AirTemperature")]),
    _room("bed_a", "Bedroom", "Bedroom", [("air_temperature", "AirTemperature")]),
    _room("bed_b", "Guest Bedroom", "Bedroom", [("air_temperature", "AirTemperature")]),
    _device("bathroom", "bath_dehumidifier", "Dehumidifier", "onOff", "relative_humidity"),
    _device("bathroom", "bath_humidifier", "Humidifier", "mist", "relative_humidity",
            direction="increase"),
    _device("living", "living_dehumidifier", "Dehumidifier", "onOff", "relative_humidity"),
    _device("living", "living_ac", "AirConditioner", "onOff", "air_temperature"),
    _device("bed_a", "fan_a", "Fan", "onOff", "air_temperature"),
    _device("bed_b", "fan_b", "Fan", "onOff", "air_temperature"),
    SENSOR,
])


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


def _goal(*variables, text="the bathroom is so damp"):
    """An ambiguous goal; each variable is (name, class, space class, space name)."""
    return GoalSpec.from_dict({
        "goal_specificity": "ambiguous", "goal_kind": "achievement", "intent_text": text,
        "implied_environment_vars": [
            {"property_name": n, "property_class": c,
             "property_sensed_space": {"space_class": s, "space_name": sn}}
            for n, c, s, sn in variables]})


DAMP_BATHROOM = ("relative_humidity", "homeont:RelativeHumidity", "homeont:Bathroom", "Bathroom")


class TestContext:
    def test_only_what_acts_on_the_rooms_variable(self, graph):
        context = build_specific_context(graph, _goal(DAMP_BATHROOM))
        assert context.artifacts == ["bath_dehumidifier", "bath_humidifier", "sensor"]
        # The living-room dehumidifier acts on the living room's humidity: out.
        assert not any("living" in a["target"] for a in context.affordances)

    def test_actions_their_current_values_and_the_readings(self, graph):
        context = build_specific_context(graph, _goal(DAMP_BATHROOM))
        found = {(a["artifact_name"], a["name"], a["type"]) for a in context.affordances}
        assert found == {
            ("bath_dehumidifier", "onOff", "action_affordance"),
            ("bath_dehumidifier", "onOff", "property_affordance"),   # actsUpon
            ("bath_humidifier", "mist", "action_affordance"),
            ("bath_humidifier", "mist", "property_affordance"),
            ("sensor", "humidity", "property_affordance"),           # the reading
        }

    def test_grouped_by_variable_without_filtering_on_direction(self, graph):
        context = build_specific_context(graph, _goal(DAMP_BATHROOM))
        (result,) = context.observable_property_hints["results"]
        assert result["property_uri"] == ('"relative_humidity" of Bathroom '
                                          '(homeont:RelativeHumidity)')
        assert [(a["artifact_title"], a["direction"]) for a in result["actions"]] == [
            ("bath_dehumidifier", "affects"), ("bath_humidifier", "increase")]
        assert result["readable_property_urls"] == ["http://h/sensor/properties/humidity"]

    def test_the_room_name_narrows_rooms_of_one_class(self, graph):
        guest = ("air_temperature", "homeont:AirTemperature", "homeont:Bedroom", "Guest Bedroom")
        assert build_specific_context(graph, _goal(guest)).artifacts == ["fan_b"]
        unnamed = ("air_temperature", "homeont:AirTemperature", "homeont:Bedroom", None)
        assert build_specific_context(graph, _goal(unnamed)).artifacts == ["fan_a", "fan_b"]

    def test_one_group_per_variable(self, graph):
        stuffy = _goal(
            ("relative_humidity", "homeont:RelativeHumidity", "homeont:LivingRoom", "Living Room"),
            ("air_temperature", "homeont:AirTemperature", "homeont:LivingRoom", "Living Room"))
        results = build_specific_context(graph, stuffy).observable_property_hints["results"]
        assert [[a["artifact_title"] for a in r["actions"]] for r in results] == [
            ["living_dehumidifier"], ["living_ac"]]

    def test_nothing_acting_on_the_variable_has_no_context(self, graph):
        dark = ("illuminance", "homeont:Illuminance", "homeont:Bathroom", "Bathroom")
        assert build_specific_context(graph, _goal(dark)) is None

    def test_the_brief_names_the_variables_and_leaves_the_direction_to_the_model(self, graph):
        brief = build_specific_context(graph, _goal(DAMP_BATHROOM)).goal_brief
        assert '"relative_humidity" (homeont:RelativeHumidity) in Bathroom' in brief
        assert DIRECTION_INSTRUCTIONS in brief


class TestSelection:
    def test_a_simple_ambiguous_achievement_goal_is_selected(self):
        structure = GoalStructure.from_dict(_goal(DAMP_BATHROOM).to_wire_dict(),
                                            intent_text="x", structure="simple")
        assert ambiguous_goal(structure) is structure.goals["G1"]

    def test_without_variables_or_as_maintenance_it_is_not(self):
        for fields in ({"implied_environment_vars": []}, {"goal_kind": "maintenance"}):
            data = {**_goal(DAMP_BATHROOM).to_wire_dict(), **fields}
            structure = GoalStructure.from_dict(data, intent_text="x", structure="simple")
            assert ambiguous_goal(structure) is None


# --------------------------------------------------------------------------
# The behaviour, with the planner faked
# --------------------------------------------------------------------------

MODULE = "ami_agents.agents.interaction_solver.behaviours.ambiguous_goal_planning"


def _agent(result=None, error=None):
    agent = MagicMock()
    agent.model, agent.temperature = "gpt-5-mini", 0.0
    agent.reasoning_effort, agent.max_completion_tokens = "medium", None
    agent.llm_client = "general-client"
    agent.bt_planner = SimpleNamespace(
        generate_bt=AsyncMock(return_value=result, side_effect=error))
    return agent


async def _plan(graph, goal, agent):
    behaviour = AmbiguousGoalPlanningBehaviour([goal])
    behaviour.agent = agent
    with patch(f"{MODULE}.fetch_environment_graph", AsyncMock(return_value=graph)):
        await behaviour.run()
    (entry,) = behaviour.results
    return entry


class TestBehaviour:
    @pytest.mark.asyncio
    async def test_the_general_model_plans_against_the_variable_context(self, graph):
        tree = {"type": "action", "action_url": "http://h/bath_dehumidifier/actions/onOff",
                "parameters": {"value": True}}
        agent = _agent({"tree": tree, "explanation": "dehumidify"})
        entry = await _plan(graph, _goal(DAMP_BATHROOM), agent)
        assert entry == {"tree": tree, "explanation": "dehumidify", "source": "ambiguous_llm"}

        kwargs = agent.bt_planner.generate_bt.await_args.kwargs
        assert kwargs["client"] == "general-client" and kwargs["model"] == "gpt-5-mini"
        assert kwargs["temperature"] is None                  # reasoning model
        assert kwargs["observable_property_hints"]["results"]
        assert {a["artifact_name"] for a in kwargs["affordances"]} == {
            "bath_dehumidifier", "bath_humidifier", "sensor"}
        assert kwargs["intents"] == ["the bathroom is so damp"]

    @pytest.mark.asyncio
    async def test_nothing_to_act_with_is_impossible_without_a_call(self, graph):
        agent = _agent({"tree": {}})
        dark = ("illuminance", "homeont:Illuminance", "homeont:Bathroom", "Bathroom")
        entry = await _plan(graph, _goal(dark, text="the bathroom is too dark"), agent)
        assert entry["impossible"] and "illuminance in Bathroom" in entry["explanation"]
        agent.bt_planner.generate_bt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_planner_failure_is_an_error_entry(self, graph):
        entry = await _plan(graph, _goal(DAMP_BATHROOM), _agent(error=RuntimeError("down")))
        assert entry["error"] == "ambiguous_planning_failed" and entry["tree"] is None
