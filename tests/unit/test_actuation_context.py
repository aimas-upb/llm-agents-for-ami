"""The actuation-only view the GOAL_REQUEST parser is shown, and how it is scoped.

Most tests use a graph built offline from a SimuHome episode -- no simulator, no
SHTD, no LLM -- and skip if the corpus is not checked out beside this repo. A
small hand-written graph covers what SimuHome does not emit: an action taking a
parameterized (object) input, and a home with a single workspace.
"""

import json
import re
from pathlib import Path

import pytest
from rdflib import Graph

from ami_agents.agents.env_explorer.utils.actuation_context import (
    build_actuation_context,
    device_family_labels,
    match_class_groups,
    match_classes,
    render_actuation_context,
    scope_artifacts,
    space_labels,
)
from ami_agents.agents.env_explorer.utils.state_resolution import load_vocabulary
from ami_agents.shared.utils.namespaces import action_types

REPO = Path(__file__).resolve().parents[2]
EPISODE = (REPO.parent / "SimuHome" / "data" / "benchmark"
           / "qt1_feasible_seed_1.json")

needs_corpus = pytest.mark.skipif(
    not EPISODE.is_file(), reason="SimuHome benchmark corpus not available"
)

# Forces scoping regardless of the home's size.
SCOPED = {"max_artifacts": 0, "max_actions": 0}


@pytest.fixture(scope="module")
def graph() -> Graph:
    """Discovered TDs + the vocabulary, as EnvExplorer sees them."""
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD

    config = json.loads(EPISODE.read_text())["initial_home_config"]
    builder = SimuHomeTD("http://localhost:8097", "test", config)

    g = Graph()
    for room in config["rooms"]:
        g += builder.room_workspace(room)
        for device in config["rooms"][room].get("devices", []):
            g += builder.artifact(room, device["device_id"])
    return load_vocabulary(g)


# One workspace, one lamp whose colour action takes an object with a nested
# object inside it, plus a protocol action that must never be shown.
SMALL_HOME = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix websub: <https://purl.org/hmas/websub/> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .

<http://h/office#workspace> a hmas:Workspace ; td:title "Office" ;
    homeont:hasSpace <http://h/office#place> ;
    hmas:contains <http://h/office/lamp#artifact>, <http://h/office/fan#artifact> .
<http://h/office#place> a homeont:Office, homeont:BuildingSpace .

<http://h/office/lamp#artifact> a hmas:Artifact, homeont:DimmableLight, saref:Actuator ;
    td:title "office_lamp" ;
    td:hasActionAffordance [
        a td:ActionAffordance, saref:SetLevelCommand ;
        td:title "colour" ; td:name "colour" ;
        td:hasForm [ hctl:hasTarget <http://h/office/lamp/actions/colour> ] ;
        td:hasInputSchema [ a js:ObjectSchema ;
            js:properties [ a js:IntegerSchema ; js:propertyName "brightness" ;
                            qudt:unit unit:PERCENT ] ,
                          [ a js:ObjectSchema ; js:propertyName "hs" ;
                            js:properties [ a js:NumberSchema ; js:propertyName "hue" ;
                                            js:description "degrees on the colour wheel" ] ] ] ] ,
    [
        a td:ActionAffordance, websub:subscribeToArtifact ;
        td:title "subscribeToArtifact" ; td:name "subscribeToArtifact" ] .

<http://h/office/fan#artifact> a hmas:Artifact, homeont:Fan, saref:Appliance ;
    td:title "office_fan" ;
    td:hasActionAffordance [
        a td:ActionAffordance, homeont:SetOnOffCommand ;
        td:title "onOff" ; td:name "onOff" ;
        td:hasInputSchema [ a js:BooleanSchema ] ] .
"""


@pytest.fixture(scope="module")
def small_home() -> Graph:
    g = Graph()
    g.parse(data=SMALL_HOME, format="turtle")
    return load_vocabulary(g)


def _artifact_titles(context):
    titles = []

    def walk(ws):
        titles.extend(a["title"] for a in ws.get("artifacts", []))
        for sub in ws.get("workspaces", []):
            walk(sub)

    for ws in context["workspaces"]:
        walk(ws)
    return sorted(titles)


# --------------------------------------------------------------------------
# The view
# --------------------------------------------------------------------------

@needs_corpus
class TestView:
    def test_no_uris_reach_the_prompt(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert not re.search(r"https?://", text)

    def test_every_action_is_named_and_typed(self, graph):
        context = build_actuation_context(graph)

        def walk(ws):
            for art in ws.get("artifacts", []):
                assert art["actions"], art["title"]
                for action in art["actions"]:
                    assert action["title"]
                    assert action["command_types"]
            for sub in ws.get("workspaces", []):
                walk(sub)

        for ws in context["workspaces"]:
            walk(ws)

    def test_protocol_actions_and_sensors_are_left_out(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert "subscribeToArtifact" not in text
        assert "unsubscribeFromArtifact" not in text
        # Ambient sensors offer no typed action: nothing to actuate.
        assert "Temperature Sensor" not in text

    def test_rooms_nest_under_the_home_workspace(self, graph):
        context = build_actuation_context(graph)
        assert [ws["title"] for ws in context["workspaces"]] == ["test"]
        rooms = {ws["title"]: ws.get("space_class")
                 for ws in context["workspaces"][0]["workspaces"]}
        assert rooms["Kitchen"] == "homeont:Kitchen"
        assert rooms["Living Room"] == "homeont:LivingRoom"

    def test_an_action_states_what_it_changes_and_moves(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert ('acts upon: property "coolingSetpoint" '
                '(class: homeont:AirConditionerCoolingSetpoint)') in text
        assert ('environment effect: variable "air_temperature" (class: homeont:AirTemperature; '
                'quantity kind: quantitykind:ThermodynamicTemperature; unit: unit:DEG_C) '
                'of environment "Bathroom environment"') in text

    def test_a_device_name_is_never_confused_with_its_class_label(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert re.search(r'Artifact\n\s+name: "bathroom_air_conditioner_1"\n'
                         r'\s+class: homeont:AirConditioner \(class label: "Air Conditioner"\)\n'
                         r'\s+semantic types: homeont:AirConditioner, saref:HVAC, hmas:Artifact\n',
                         text)

    def test_an_action_names_its_title_and_command_class(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert re.search(r'Action\n\s+name: "coolingSetpoint"\n'
                         r'\s+command class: saref:SetAbsoluteLevelCommand\n', text)

    def test_a_direct_value_carries_its_type_unit_and_meaning(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert "input: direct value; type: js:IntegerSchema; unit: unit:DEG_C" in text
        assert re.search(r"input: direct value; type: js:IntegerSchema; description: 0 = Off",
                         text)

    def test_a_bounded_input_states_its_range(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert re.search(r'name: "brightness"\n\s+command class: \S+\n'
                         r'\s+input: direct value; type: js:IntegerSchema; '
                         r'range: \[min: 1, max: 254\]\n', text)

    def test_manufacturer_and_model_are_shown(self, graph):
        text = render_actuation_context(build_actuation_context(graph))
        assert "manufacturer: LG Electronics\n" in text
        assert "model: Air Conditioner\n" in text


class TestParameterizedCall:
    def test_object_input_renders_as_named_nested_parameters(self, small_home):
        text = render_actuation_context(build_actuation_context(small_home))
        assert "input: parameterized call" in text
        assert '- parameter "brightness": type: js:IntegerSchema; unit: unit:PERCENT' in text
        assert '- parameter "hs": object' in text
        assert ('- parameter "hue": type: js:NumberSchema; '
                'description: degrees on the colour wheel') in text

    def test_object_input_structure(self, small_home):
        context = build_actuation_context(small_home)
        lamp = next(a for a in context["workspaces"][0]["artifacts"]
                    if a["title"] == "office_lamp")
        (colour,) = lamp["actions"]
        params = {p["parameter_name"]: p["parameter_schema"]
                  for p in colour["input"]["parameters"]}
        assert params["brightness"] == {"form": "direct value",
                                        "type": "js:IntegerSchema",
                                        "unit": "unit:PERCENT"}
        assert params["hs"]["form"] == "parameterized call"


def test_action_types_keeps_both_vocabularies():
    assert action_types(["td:ActionAffordance", "saref:OnCommand",
                         "http://example.org/homeont/SetModeCommand",
                         "http://example.org/StatusCommand"]) == [
        "saref:OnCommand", "homeont:SetModeCommand"]


# --------------------------------------------------------------------------
# Label matching
# --------------------------------------------------------------------------

class TestMatching:
    def test_a_head_noun_names_the_family(self, small_home):
        labels = device_family_labels(small_home)
        assert match_classes("start washer 1", labels) == ["homeont:LaundryWasher"]
        assert match_classes("start the dryer", labels) == ["homeont:LaundryDryer"]

    def test_a_shared_head_noun_names_every_family_with_it(self, small_home):
        found = match_classes("turn on the light", device_family_labels(small_home))
        assert {"homeont:OnOffLight", "homeont:DimmableLight"} <= set(found)

    def test_alt_labels_and_plurals(self, small_home):
        labels = device_family_labels(small_home)
        assert match_classes("close the blinds", labels) == ["homeont:WindowCoveringController"]
        assert set(match_classes("turn off both lamps", labels)) == {
            "homeont:OnOffLight", "homeont:DimmableLight"}
        assert match_classes("run the washing machine", labels) == ["homeont:LaundryWasher"]

    def test_an_alt_label_contributes_no_head_noun(self, small_home):
        # "washing machine" must not make "machine" a device name.
        assert match_classes("the coffee machine", device_family_labels(small_home)) == []

    def test_a_modifier_names_nothing(self, small_home):
        # "air" qualifies Air Conditioner and Air Purifier; it names neither.
        assert match_classes("clear the air", device_family_labels(small_home)) == []

    def test_whole_words_only(self, small_home):
        # "dishwasher" is not a "washer".
        assert match_classes("run the dishwasher", device_family_labels(small_home)) == [
            "homeont:Dishwasher"]

    def test_one_mention_is_one_group(self, small_home):
        groups = match_class_groups("turn on the light and start the dryer",
                                    device_family_labels(small_home))
        assert len(groups) == 2
        assert {"homeont:OnOffLight", "homeont:DimmableLight"} <= set(groups[0])
        assert groups[1] == ["homeont:LaundryDryer"]

    def test_the_longest_room_name_wins(self, small_home):
        labels = space_labels(small_home)
        assert match_classes("the master bedroom fan", labels) == ["homeont:MasterBedroom"]
        assert match_classes("the bedroom fan", labels) == ["homeont:Bedroom"]
        assert match_classes("the living room lights", labels) == ["homeont:LivingRoom"]


# --------------------------------------------------------------------------
# Scoping
# --------------------------------------------------------------------------

@needs_corpus
class TestScoping:
    def test_a_small_home_is_sent_whole(self, graph):
        artifacts, scope = scope_artifacts(graph, "turn on the kitchen light",
                                           max_artifacts=1000, max_actions=0)
        assert artifacts is None
        assert scope["rule"] == "1-whole"

    def test_either_threshold_is_enough_to_send_it_whole(self, graph):
        artifacts, _ = scope_artifacts(graph, "turn on the kitchen light",
                                       max_artifacts=0, max_actions=1000)
        assert artifacts is None

    def test_a_named_room_qualifies_the_device_it_names(self, graph):
        # Lights stand in every room; "kitchen" says which one is meant.
        artifacts, scope = scope_artifacts(graph, "turn on the kitchen light", **SCOPED)
        assert scope["rule"] == "2.2-union"
        assert scope["matched_rooms"] == ["homeont:Kitchen"]
        assert scope["workspaces"] == ["Kitchen"]
        titles = _artifact_titles(build_actuation_context(graph, artifacts))
        assert titles and all(t.startswith("kitchen_") for t in titles)

    def test_a_device_the_named_rooms_lack_brings_in_its_own_rooms(self, graph):
        # The only heat pump stands in the utility room. A room match alone
        # would have dropped it; the lights stay with the living room.
        goal = ("keep the living room at 22 degrees Celsius while the light is "
                "on, then 2 minutes after that set the heating pump level to 20 degrees")
        artifacts, scope = scope_artifacts(graph, goal, **SCOPED)
        assert scope["rule"] == "2.2-union"
        assert "homeont:HeatPump" in scope["matched_devices"]
        assert scope["workspaces"] == ["Living Room", "Utility Room"]
        titles = _artifact_titles(build_actuation_context(graph, artifacts))
        assert any("heat_pump" in t for t in titles)

    def test_no_room_scopes_to_the_workspaces_holding_the_devices(self, graph):
        # Washers stand in the bathroom and the utility room: both rooms are
        # the scope, with everything else in them -- not the washers alone.
        artifacts, scope = scope_artifacts(graph, "start washer 1", **SCOPED)
        assert scope["rule"] == "2.2-union"
        assert scope["matched_rooms"] == []
        assert scope["matched_devices"] == ["homeont:LaundryWasher"]
        assert scope["workspaces"] == ["Bathroom", "Utility Room"]
        titles = _artifact_titles(build_actuation_context(graph, artifacts))
        assert any("laundry_washer" in t for t in titles)
        assert any("laundry_dryer" in t for t in titles)
        assert all(t.startswith(("bathroom_", "utility_room_")) for t in titles)

    def test_a_room_this_home_lacks_counts_as_no_room(self, graph):
        # There is no master bedroom here; the fans' rooms become the scope.
        _, scope = scope_artifacts(graph, "set the master bedroom fan to low", **SCOPED)
        assert scope["rule"] == "2.2-union"
        assert scope["matched_devices"] == ["homeont:Fan"]
        assert scope["workspaces"] == ["Bathroom", "Kitchen"]

    def test_nothing_named_sends_it_whole(self, graph):
        artifacts, scope = scope_artifacts(graph, "it is far too stuffy", **SCOPED)
        assert artifacts is None
        assert scope["rule"] == "2-no-match-whole"

    def test_scoped_view_is_much_smaller(self, graph):
        whole = render_actuation_context(build_actuation_context(graph))
        artifacts, _ = scope_artifacts(graph, "turn on the kitchen light", **SCOPED)
        scoped = render_actuation_context(build_actuation_context(graph, artifacts))
        assert len(scoped) < len(whole) / 3


class TestSingleWorkspace:
    def test_one_workspace_goes_straight_to_devices(self, small_home):
        # "office" names the only room, but with one workspace rooms are not
        # tried: the device named decides.
        artifacts, scope = scope_artifacts(small_home, "turn on the office fan", **SCOPED)
        assert scope["rule"] == "2.1-device"
        assert _artifact_titles(build_actuation_context(small_home, artifacts)) == [
            "office_fan"]


# --------------------------------------------------------------------------
# Properties and environment (the goal context's two additions)
# --------------------------------------------------------------------------

# Home Assistant's shape: the room has an environment but no room-level
# variables; a reading exists only as a sensor's affordance that is
# `ssn:isPropertyOf` that environment. The sensor offers no action.
SENSOR_HOME = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .
@prefix ssn: <http://www.w3.org/ns/ssn/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .

<http://h/office#workspace> a hmas:Workspace ; td:title "Office" ;
    homeont:hasSpace <http://h/office#place> ;
    hmas:contains <http://h/office/sensor#artifact>, <http://h/office/fan#artifact> .
<http://h/office#place> a homeont:Office, homeont:BuildingSpace ;
    homeont:hasEnvironment <http://h/office#environment> .
<http://h/office#environment> a homeont:Environment ; rdfs:label "Office environment" .

<http://h/office/sensor#artifact> a hmas:Artifact, homeont:TemperatureSensor, saref:Sensor ;
    td:title "office_temperature_sensor" ;
    td:hasPropertyAffordance <http://h/office/sensor/properties/temperature> .
<http://h/office/sensor/properties/temperature> a td:PropertyAffordance, homeont:AirTemperature ;
    td:title "temperature" ;
    ssn:isPropertyOf <http://h/office#environment> ;
    qudt:unit unit:DEG_C ;
    td:hasOutputSchema [ a js:NumberSchema ; qudt:unit unit:DEG_C ] .

<http://h/office/fan#artifact> a hmas:Artifact, homeont:Fan, saref:Appliance ;
    td:title "office_fan" ;
    td:hasActionAffordance [
        a td:ActionAffordance, homeont:SetOnOffCommand ;
        td:title "onOff" ; td:name "onOff" ;
        td:hasInputSchema [ a js:BooleanSchema ] ] .
"""


@pytest.fixture(scope="module")
def sensor_home() -> Graph:
    g = Graph()
    g.parse(data=SENSOR_HOME, format="turtle")
    return load_vocabulary(g)


class TestProperties:
    def test_a_sensor_appears_only_when_properties_are_asked_for(self, sensor_home):
        plain = render_actuation_context(build_actuation_context(sensor_home))
        assert "office_temperature_sensor" not in plain
        full = render_actuation_context(
            build_actuation_context(sensor_home, with_properties=True))
        assert 'name: "office_temperature_sensor"' in full
        assert re.search(r'Property\n\s+name: "temperature"\n'
                         r'\s+property class: homeont:AirTemperature\n', full)
        assert "output: direct value; type: js:NumberSchema; unit: unit:DEG_C" in full

    def test_a_named_sensor_counts_only_with_properties(self, sensor_home):
        goal = "when the temperature sensor reads above 25 turn on the fan"
        without, scope = scope_artifacts(sensor_home, goal, **SCOPED)
        assert scope["rule"] == "2.1-device"        # the fan still matches
        assert len(without) == 1
        with_props, _ = scope_artifacts(sensor_home, goal, with_properties=True,
                                        **SCOPED)
        assert len(with_props) == 2                  # fan and sensor

    @needs_corpus
    def test_device_properties_render_without_uris(self, graph):
        text = render_actuation_context(
            build_actuation_context(graph, with_properties=True))
        assert not re.search(r"https?://", text)
        assert re.search(r'name: "coolingSetpoint"\n'
                         r'\s+property class: homeont:AirConditionerCoolingSetpoint\n'
                         r'\s+kind: actuatable\n'
                         r'\s+values: as Action "coolingSetpoint"\n', text)


class TestEnvironment:
    def test_a_reading_with_no_room_variable_stands_on_its_own(self, sensor_home):
        context = build_actuation_context(sensor_home, with_environment=True)
        (ws,) = context["workspaces"]
        (var,) = ws["environment"]["variables"]
        assert var["class"] == "homeont:AirTemperature"
        assert var["title"] is None
        assert var["observed_by"] == [{"artifact": "office_temperature_sensor",
                                       "affordance": "temperature"}]
        text = render_actuation_context(context)
        assert re.search(r'Environment\n\s+name: "Office environment"\n', text)
        assert re.search(r'Variable\n\s+name: none \(no room-level property\)\n'
                         r'\s+class: homeont:AirTemperature\n'
                         r'\s+unit: unit:DEG_C\n'
                         r'\s+observed by: office_temperature_sensor.temperature', text)

    @needs_corpus
    def test_room_variables_name_the_device_readings_that_observe_them(self, graph):
        text = render_actuation_context(
            build_actuation_context(graph, with_environment=True))
        # The AC's own reading is typed with a subclass of AirTemperature.
        assert re.search(r'name: "air_temperature"\n'
                         r'\s+class: homeont:AirTemperature\n'
                         r'\s+quantity kind: quantitykind:ThermodynamicTemperature\n'
                         r'\s+unit: unit:DEG_C\n'
                         r'\s+observed by: [^\n]*bathroom_air_conditioner_1.temperature', text)

    @needs_corpus
    def test_a_variable_nothing_observes_says_so(self, graph):
        text = render_actuation_context(
            build_actuation_context(graph, with_environment=True))
        assert re.search(r'name: "illuminance"\n(\s+\w[^\n]*\n)*?\s+observed by: none\n', text)


@needs_corpus
class TestWrittenProperties:
    def test_a_property_an_action_writes_points_at_it_instead_of_repeating(self, graph):
        text = render_actuation_context(
            build_actuation_context(graph, with_properties=True))
        line = 'values: as Action "fanMode"'
        assert re.search(r'name: "fanMode"\n\s+property class: homeont:AirConditionerFanMode\n'
                         r'\s+kind: actuatable\n\s+values: as Action "fanMode"\n', text)
        # No output block follows it: the action's input already lists the values.
        following = text.split(line, 1)[1].lstrip("\n").splitlines()[0]
        assert not following.strip().startswith("output:")

    def test_a_capability_keeps_its_type_but_not_its_gloss(self, graph):
        text = render_actuation_context(
            build_actuation_context(graph, with_properties=True))
        assert re.search(r'name: "fanModeSequence"\n\s+property class: \S+\n'
                         r'\s+kind: capability\n'
                         r'\s+output: direct value; type: js:IntegerSchema\n', text)
        assert "OffLowMedHigh" not in text

    def test_a_state_keeps_its_gloss(self, graph):
        text = render_actuation_context(
            build_actuation_context(graph, with_properties=True))
        assert re.search(r'name: "operationalState"\n\s+property class: \S+\n'
                         r'\s+kind: state\n'
                         r'\s+output: direct value; type: \S+; description: ', text)

    def test_an_action_keeps_its_gloss(self, graph):
        text = render_actuation_context(
            build_actuation_context(graph, with_properties=True))
        assert re.search(r'Action\n\s+name: "fanMode"\n.*\n\s+input: direct value; '
                         r'type: js:IntegerSchema; description: 0 = Off', text)
