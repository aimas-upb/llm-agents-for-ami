"""The conditions file and the plan steps it concerns (`utils/planning_conditions.py`).

    HOME        a small TD home, shared with test_condition_guard.py:
        washer_1  LaundryWasher: combined operationalState and onOff commands,
                  each acting upon its property; a spin speed level command
        ha_washer LaundryWasher with separate pause / resume / start commands
                  (Home Assistant style) and a spin speed
        dryer_2   LaundryDryer with a pause command but nothing to resume it
        ac_1      AirConditioner: HVAC mode, cooling setpoint
        light_1   DimmableLight: brightness (power-on only: no condition here)

    loading     the repo file and a file of our own
    matching    by device class, command and acts_upon
    steps       which actions become steps, numbered in tree order; their key
"""

from pathlib import Path

import pytest
from rdflib import Graph, URIRef

from ami_agents.agents.interaction_solver.utils.planning_conditions import (
    conditioned_steps,
    guard_key,
    load_conditions,
    matching,
)
from ami_agents.shared.utils.vocabulary import close_types
from tests.unit.test_dependency_compiler import PREFIXES, _prop, _room


def _device(name, cls):
    return f"<http://h/{name}#artifact> a hmas:Artifact, homeont:{cls} ; td:title \"{name}\" .\n"


def _action(device, name, cls, acts_upon=None, schema="js:IntegerSchema"):
    """An action affordance of ``device`` typed ``cls``, taking a ``schema``
    value (None: no input), acting upon the device's property ``acts_upon``."""
    link = f" ; saref:actsUpon <http://h/{device}/p/{acts_upon}>" if acts_upon else ""
    inp = f" ; td:hasInputSchema [ a {schema} ]" if schema else ""
    return (f"<http://h/{device}#artifact> td:hasActionAffordance <http://h/{device}/a/{name}> .\n"
            f"<http://h/{device}/a/{name}> a td:ActionAffordance, {cls} ; td:title \"{name}\" ;"
            f" td:hasForm [ hctl:hasTarget <http://h/{device}/a/{name}> ]{link}{inp} .\n")


HOME = PREFIXES + "".join([
    _room("utility", "Utility Room", "UtilityRoom", ["washer_1", "ha_washer", "dryer_2"]),
    _room("living", "Living Room", "LivingRoom", ["ac_1", "light_1"]),
    # washer_1: combined commands, as SimuHome exposes them.
    _device("washer_1", "LaundryWasher"),
    _prop("washer_1", "operationalState", "LaundryWasherOperationalState", "js:IntegerSchema"),
    _prop("washer_1", "spinSpeedCurrent", "LaundryWasherSpinSpeedCurrent", "js:IntegerSchema"),
    _prop("washer_1", "onOff", "LaundryWasherOnOff", "js:BooleanSchema"),
    _action("washer_1", "operationalState", "homeont:SetOperationalStateCommand",
            acts_upon="operationalState"),
    _action("washer_1", "spinSpeedCurrent", "saref:SetAbsoluteLevelCommand",
            acts_upon="spinSpeedCurrent"),
    _action("washer_1", "onOff", "homeont:SetOnOffCommand", acts_upon="onOff",
            schema="js:BooleanSchema"),
    # ha_washer: separate lifecycle commands, as Home Assistant exposes them.
    _device("ha_washer", "LaundryWasher"),
    _prop("ha_washer", "operationalState", "LaundryWasherOperationalState", "js:IntegerSchema"),
    _prop("ha_washer", "spinSpeedCurrent", "LaundryWasherSpinSpeedCurrent", "js:IntegerSchema"),
    _action("ha_washer", "pause", "saref:PauseCommand", acts_upon="operationalState", schema=None),
    _action("ha_washer", "start", "saref:StartCommand", acts_upon="operationalState", schema=None),
    _action("ha_washer", "resume", "saref:StartCommand", acts_upon="operationalState", schema=None),
    _action("ha_washer", "spinSpeedCurrent", "saref:SetAbsoluteLevelCommand",
            acts_upon="spinSpeedCurrent"),
    # dryer_2: can pause, cannot resume.
    _device("dryer_2", "LaundryDryer"),
    _prop("dryer_2", "operationalState", "LaundryDryerOperationalState", "js:IntegerSchema"),
    _prop("dryer_2", "selectedDrynessLevel", "LaundryDryerSelectedDrynessLevel", "js:IntegerSchema"),
    _action("dryer_2", "pause", "saref:PauseCommand", acts_upon="operationalState", schema=None),
    _action("dryer_2", "selectedDrynessLevel", "saref:SetAbsoluteLevelCommand",
            acts_upon="selectedDrynessLevel"),
    # ac_1: a setpoint that only acts in a cooling mode.
    _device("ac_1", "AirConditioner"),
    _prop("ac_1", "hvacMode", "AirConditionerHvacMode", "js:IntegerSchema"),
    _prop("ac_1", "coolingSetpoint", "AirConditionerCoolingSetpoint", "js:NumberSchema"),
    _action("ac_1", "hvacMode", "homeont:SetModeCommand", acts_upon="hvacMode"),
    _action("ac_1", "coolingSetpoint", "saref:SetAbsoluteLevelCommand",
            acts_upon="coolingSetpoint", schema="js:NumberSchema"),
    # light_1: brightness; its only condition is power-on, applied elsewhere.
    _device("light_1", "DimmableLight"),
    _prop("light_1", "brightness", "DimmableLightBrightness", "js:IntegerSchema"),
    _action("light_1", "brightness", "saref:SetAbsoluteLevelCommand", acts_upon="brightness"),
])


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


@pytest.fixture(scope="module")
def conditions():
    """The repo's conditions file."""
    return load_conditions(None)


def act(device, name, value=1):
    """A planned action node, as the planners produce it."""
    return {"type": "action", "name": f"{device}.{name}",
            "action_url": f"http://h/{device}/a/{name}", "parameters": {"value": value}}


def _action_node(device, name):
    return URIRef(f"http://h/{device}/a/{name}")


def _artifact(device):
    return URIRef(f"http://h/{device}#artifact")


class TestLoading:
    def test_the_repo_file_has_explicit_conditions_and_no_power_on(self, conditions):
        ids = {c.id for c in conditions}
        assert "power_on_first" not in ids
        # The cycle commands' own rule is a check in code (cycle_guard.py).
        assert "appliance_cycle_commands" not in ids
        assert {"laundry_washer_settings_not_running",
                "thermostat_cooling_setpoint_mode", "rvc_mode_change_when_idle"} <= ids
        assert all(c.commands and c.text for c in conditions)

    def test_a_missing_file_has_no_conditions(self, tmp_path):
        assert load_conditions(str(tmp_path / "none.yaml")) == ()

    def test_an_entry_without_commands_or_text_is_skipped(self, tmp_path):
        """It could match nothing, or say nothing."""
        path = Path(tmp_path / "c.yaml")
        path.write_text("conditions:\n"
                        "  - {id: a, applies_to: {classes: [homeont:Fan]}, text: x}\n"
                        "  - {id: b, applies_to: {classes: [homeont:Fan]},"
                        " commands: [homeont:SetModeCommand]}\n"
                        "  - {id: c, applies_to: {classes: [homeont:Fan]},"
                        " commands: [homeont:SetModeCommand], text: \"  two   words \"}\n",
                        encoding="utf-8")
        (only,) = load_conditions(str(path))
        assert only.id == "c" and only.text == "two words" and only.acts_upon == ()


class TestMatching:
    def _ids(self, graph, conditions, device, name):
        return [c.id for c in matching(graph, _artifact(device), _action_node(device, name),
                                       conditions)]

    def test_a_washer_setting_has_the_settings_condition(self, graph, conditions):
        assert self._ids(graph, conditions, "washer_1", "spinSpeedCurrent") == \
            ["laundry_washer_settings_not_running"]

    def test_the_cycle_command_has_no_condition_for_the_model(self, graph, conditions):
        """Start / pause / resume are checked in code (`cycle_guard.py`)."""
        assert self._ids(graph, conditions, "washer_1", "operationalState") == []

    def test_switching_on_has_none(self, graph, conditions):
        assert self._ids(graph, conditions, "washer_1", "onOff") == []

    def test_acts_upon_narrows_to_the_setpoint(self, graph, conditions):
        """The cooling setpoint has the HVAC-mode condition; the HVAC mode
        itself, a SetModeCommand acting upon another property, has none."""
        assert self._ids(graph, conditions, "ac_1", "coolingSetpoint") == \
            ["thermostat_cooling_setpoint_mode"]
        assert self._ids(graph, conditions, "ac_1", "hvacMode") == []

    def test_a_light_has_none(self, graph, conditions):
        assert self._ids(graph, conditions, "light_1", "brightness") == []


class TestSteps:
    def test_steps_are_numbered_in_tree_order_with_their_paths(self, graph, conditions):
        tree = {"type": "sequence", "name": "s", "children": [
            act("light_1", "brightness", 40), act("ac_1", "coolingSetpoint", 22),
            {"type": "sequence", "name": "inner", "children": [
                act("washer_1", "spinSpeedCurrent", 1200)]}]}
        steps = conditioned_steps(graph, tree, conditions, goal="G2", first=3)
        assert [(s.id, s.path, s.goal) for s in steps] == [("S3", (1,), "G2"),
                                                           ("S4", (2, 0), "G2")]

    def test_the_key_carries_the_value_only_when_it_selects_the_command(self, graph, conditions):
        """A pause and a start of one operationalState action are two keys; a
        spin speed is one key whatever the speed."""
        cycle = guard_key(graph, _artifact("washer_1"), _action_node("washer_1", "operationalState"),
                          {"value": 2})
        (spin,) = conditioned_steps(graph, act("washer_1", "spinSpeedCurrent", 1200), conditions)
        assert cycle["invocation_value"] == "2" and spin.key["invocation_value"] is None
        assert spin.key == {"affordance_name": "spinSpeedCurrent",
                            "affordance_class": "saref:SetAbsoluteLevelCommand",
                            "device_name": "washer_1", "device_class": "homeont:LaundryWasher",
                            "room_class": "homeont:UtilityRoom", "invocation_value": None}

    def test_actions_inside_a_guard_are_skipped(self, graph, conditions):
        """A remembered tree being replayed: its guarded steps were checked."""
        guarded = {"type": "selector", "name": "S1 x: c condition guard",
                   "children": [act("washer_1", "spinSpeedCurrent")]}
        assert conditioned_steps(graph, guarded, conditions) == []
