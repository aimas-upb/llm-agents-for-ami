"""The "power on first" guard: which actions get it, how it is built, where it runs.

    guard_tree / applies / power_on_affordances   on a small TD home:
        dryer_1   LaundryDryer, on/off command + state, dryness level and mode
                  (guarded)
        light_1   DimmableLight, on/off + level -- not listed in applies_to
        washer_1  LaundryWasher, operational state but no on/off command
        acme_1    a Heater made by "Acme", listed by manufacturer
    order                              a device switched on earlier in the
                                       plan's execution order is not guarded
    DependencyGoalPlanningBehaviour    the compiled dependency tree gets the
                                       guard too, in that order
    GoalRequestBehaviour._plan_mixed   the guard step gets every entry except
                                       the dependency ones
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rdflib import Graph

from ami_agents.agents.interaction_solver.behaviours.goal_request import GoalRequestBehaviour
from ami_agents.agents.interaction_solver.utils.power_on_guard import (
    applies,
    guard_tree,
    load_applies_to,
    power_on_affordances,
)
from ami_agents.bt_planning.serialization import validate_tree
from ami_agents.shared.utils.vocabulary import close_types
from tests.unit.test_dependency_compiler import PREFIXES, _prop, _room
from ami_agents.shared.utils.namespaces import (  # noqa: E402
    SCHEMA,
)

APPLIES_TO = {"classes": ["homeont:LaundryDryer", "homeont:LaundryWasher"],
              "manufacturers": ["Acme"]}


def _action(device, name, cls, acts_upon=None):
    """An action affordance of ``device`` typed ``cls``, optionally acting upon
    the device's property ``acts_upon``."""
    link = f" ; saref:actsUpon <http://h/{device}/p/{acts_upon}>" if acts_upon else ""
    return (f"<http://h/{device}#artifact> td:hasActionAffordance <http://h/{device}/a/{name}> .\n"
            f"<http://h/{device}/a/{name}> a td:ActionAffordance, {cls} ; td:title \"{name}\" ;"
            f" td:hasForm [ hctl:hasTarget <http://h/{device}/a/{name}> ]{link} .\n")


def _device(name, cls, extra=""):
    return f"<http://h/{name}#artifact> a hmas:Artifact, homeont:{cls} ; td:title \"{name}\" {extra}.\n"


HOME = PREFIXES + f"@prefix schema: <{SCHEMA}> .\n" + "".join([
    _room("utility", "Utility Room", "UtilityRoom", ["dryer_1", "washer_1"]),
    _room("living", "Living Room", "LivingRoom", ["light_1", "acme_1"]),
    # The dryer: on/off command acting upon its on/off state, and a level command.
    _device("dryer_1", "LaundryDryer"),
    _prop("dryer_1", "onOff", "OnOff", "js:BooleanSchema"),
    _action("dryer_1", "onOff", "homeont:SetOnOffCommand", acts_upon="onOff"),
    _action("dryer_1", "selectedDrynessLevel", "saref:SetAbsoluteLevelCommand"),
    _action("dryer_1", "mode", "homeont:SetModeCommand"),
    # A light: same commands, but its class is not listed.
    _device("light_1", "DimmableLight"),
    _prop("light_1", "onOff", "OnOff", "js:BooleanSchema"),
    _action("light_1", "onOff", "homeont:SetOnOffCommand", acts_upon="onOff"),
    _action("light_1", "level", "saref:SetAbsoluteLevelCommand"),
    # A washer: listed, but no on/off command to switch it on with.
    _device("washer_1", "LaundryWasher"),
    _action("washer_1", "operationalState", "homeont:SetOperationalStateCommand"),
    # A heater listed by its manufacturer only.
    _device("acme_1", "Heater", '; schema:manufacturer "Acme" '),
    _prop("acme_1", "onOff", "OnOff", "js:BooleanSchema"),
    _action("acme_1", "onOff", "homeont:SetOnOffCommand", acts_upon="onOff"),
    _action("acme_1", "mode", "homeont:SetModeCommand"),
])


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


def _act(device, name, value=1):
    """A planned action node, as the planners produce it."""
    return {"type": "action", "name": f"{device}.{name}",
            "action_url": f"http://h/{device}/a/{name}", "parameters": {"value": value}}


def _artifact(graph, name):
    from rdflib import URIRef
    return URIRef(f"http://h/{name}#artifact")


class TestWhichActions:
    def test_a_setting_on_a_listed_device_is_guarded(self, graph):
        """The dryness level of the dryer: switch on first, then set it."""
        tree, notes = guard_tree(graph, _act("dryer_1", "selectedDrynessLevel"), APPLIES_TO)
        assert notes == [] and validate_tree(tree) == []
        # The guard: on already, or switch on; then the planned action, unchanged.
        selector, planned = tree["children"]
        condition, switch_on = selector["children"]
        assert condition["property_url"] == "http://h/dryer_1/p/onOff"
        assert condition["expected_value"] is True
        assert switch_on["action_url"] == "http://h/dryer_1/a/onOff"
        assert switch_on["parameters"] == {"value": True}
        assert planned == _act("dryer_1", "selectedDrynessLevel")

    def test_the_on_off_command_itself_is_not_guarded(self, graph):
        node = _act("dryer_1", "onOff", True)
        assert guard_tree(graph, node, APPLIES_TO) == (node, [])

    def test_a_device_not_listed_is_not_guarded(self, graph):
        node = _act("light_1", "level", 40)
        assert guard_tree(graph, node, APPLIES_TO) == (node, [])

    def test_a_listed_device_without_on_off_stays_unguarded_with_a_note(self, graph):
        node = _act("washer_1", "operationalState")
        tree, notes = guard_tree(graph, node, APPLIES_TO)
        assert tree == node
        assert notes and "washer_1.operationalState" in notes[0] and "no on/off" in notes[0]

    def test_nested_actions_are_found_and_the_input_is_not_changed(self, graph):
        """A sequence of two settings: each is guarded where it stands."""
        original = {"type": "sequence", "name": "s", "children": [
            _act("dryer_1", "selectedDrynessLevel"), _act("light_1", "level", 40)]}
        tree, _ = guard_tree(graph, original, APPLIES_TO)
        assert tree["children"][0]["name"].endswith("power on first")
        assert tree["children"][1] == _act("light_1", "level", 40)
        assert original["children"][0] == _act("dryer_1", "selectedDrynessLevel")

    def test_an_already_guarded_tree_is_not_guarded_again(self, graph):
        """A remembered tree being replayed already carries the guard."""
        once, _ = guard_tree(graph, _act("dryer_1", "selectedDrynessLevel"), APPLIES_TO)
        assert guard_tree(graph, once, APPLIES_TO) == (once, [])


def _guards(tree):
    """The names of the power-on guards in ``tree``, in tree order."""
    out = []

    def walk(node):
        if isinstance(node, dict):
            if str(node.get("name", "")).endswith("power on first"):
                out.append(node["name"])
            for child in node.get("children") or []:
                walk(child)

    walk(tree)
    return out


def _seq(*children):
    return {"type": "sequence", "name": "s", "children": list(children)}


class TestOrder:
    """A device switched on before the action, in execution order, needs no guard."""

    def test_switched_on_earlier_in_a_sequence(self, graph):
        tree = _seq(_act("dryer_1", "onOff", True), _act("dryer_1", "selectedDrynessLevel"))
        assert guard_tree(graph, tree, APPLIES_TO) == (tree, [])

    def test_only_the_first_setting_of_a_device_is_guarded(self, graph):
        """The first guard switches the dryer on; the second setting follows it."""
        tree, _ = guard_tree(graph, _seq(_act("dryer_1", "selectedDrynessLevel"),
                                         _act("dryer_1", "mode", 2)), APPLIES_TO)
        assert len(_guards(tree)) == 1 and _guards(tree["children"][0]) == _guards(tree)

    def test_a_switch_on_in_a_selector_does_not_count_after_it(self, graph):
        """A selector's branch may not have run."""
        tree, _ = guard_tree(graph, _seq(
            {"type": "selector", "name": "maybe", "children": [
                {"type": "condition", "name": "c", "property_url": "http://h/x",
                 "expected_value": True}, _act("dryer_1", "onOff", True)]},
            _act("dryer_1", "selectedDrynessLevel")), APPLIES_TO)
        assert len(_guards(tree)) == 1

    def test_a_parallel_sibling_does_not_run_before(self, graph):
        tree, _ = guard_tree(graph, {"type": "parallel", "name": "p", "children": [
            _act("dryer_1", "onOff", True), _act("dryer_1", "selectedDrynessLevel")]},
            APPLIES_TO)
        assert len(_guards(tree)) == 1

    def test_switching_off_is_not_switching_on(self, graph):
        tree, _ = guard_tree(graph, _seq(_act("dryer_1", "onOff", False),
                                         _act("dryer_1", "selectedDrynessLevel")), APPLIES_TO)
        assert len(_guards(tree)) == 1

    def test_an_existing_guard_counts_as_switching_on(self, graph):
        guarded, _ = guard_tree(graph, _act("dryer_1", "selectedDrynessLevel"), APPLIES_TO)
        tree = _seq(guarded, _act("dryer_1", "mode", 2))
        assert guard_tree(graph, tree, APPLIES_TO) == (tree, [])


class TestWhichDevices:
    def test_by_class(self, graph):
        assert applies(graph, _artifact(graph, "dryer_1"), APPLIES_TO)
        assert not applies(graph, _artifact(graph, "light_1"), APPLIES_TO)

    def test_by_manufacturer_literally(self, graph):
        """"Acme" matches; "acme" does not -- the match is exact."""
        heater = _artifact(graph, "acme_1")
        assert applies(graph, heater, APPLIES_TO)
        assert not applies(graph, heater, {"classes": [], "manufacturers": ["acme"]})
        tree, _ = guard_tree(graph, _act("acme_1", "mode", 2), APPLIES_TO)
        assert tree["name"].endswith("power on first")

    def test_the_on_off_affordances(self, graph):
        found, why = power_on_affordances(graph, _artifact(graph, "dryer_1"))
        assert found == ("http://h/dryer_1/p/onOff", "http://h/dryer_1/a/onOff") and why is None
        assert power_on_affordances(graph, _artifact(graph, "washer_1"))[0] is None


class TestConfig:
    def test_the_repo_file_lists_the_power_on_entry(self):
        """Under `conditions:`; the dead-front appliances and, through Level
        Control's ExecuteIfOff, the dimmable lights (ColorLight among them)."""
        applies_to = load_applies_to(None)
        assert "homeont:LaundryDryer" in applies_to["classes"]
        assert "homeont:DimmableLight" in applies_to["classes"]

    def test_a_missing_file_turns_the_guard_off(self, tmp_path):
        assert load_applies_to(str(tmp_path / "none.yaml")) is None


DEPENDENCY = "ami_agents.agents.interaction_solver.behaviours.dependency_goal_planning"
DRYER = dict(goal_specificity="explicit", goal_kind="achievement", goal_effect="set",
             artifact_name="dryer_1", artifact_class="homeont:LaundryDryer")


class _DryerPlanner:
    """Stands in for the achievement planners: each goal's action on dryer_1,
    named by its affordance."""

    def __init__(self, goals, logger=None, graph=None):
        self.results = [{"tree": _act("dryer_1", g.affordance_name,
                                      True if g.affordance_name == "onOff" else 1),
                         "explanation": "x"} for g in goals]

    async def join(self):
        return None


async def _dependency_plan(graph, goals):
    """One dependency structure through the planner, with ``graph`` as the
    environment and the dryer planner for its steps; returns its entry."""
    from ami_agents.agents.interaction_solver.behaviours.dependency_goal_planning import (
        DependencyGoalPlanningBehaviour,
    )
    from ami_agents.shared.models.goal_structure import GoalStructure
    structure = GoalStructure.from_dict({"goals": goals, "predicates": {}},
                                        intent_text="the request", structure="dependency")
    behaviour = DependencyGoalPlanningBehaviour([structure])
    behaviour.agent = MagicMock()
    behaviour.agent.config = {"planning": {}}
    with patch(f"{DEPENDENCY}.fetch_environment_graph", AsyncMock(return_value=graph)), \
         patch(f"{DEPENDENCY}.ExplicitGoalPlanningBehaviour", _DryerPlanner):
        await behaviour.run()
    (entry,) = behaviour.results
    return entry


class TestDependencyPlans:
    @pytest.mark.asyncio
    async def test_a_step_after_switching_on_is_not_guarded(self, graph):
        """G1 switches the dryer on, G2 sets its dryness after G1: no guard."""
        entry = await _dependency_plan(graph, {
            "G1": {**DRYER, "intent_text": "turn on dryer 1", "affordance_name": "onOff",
                   "affordance_class": "homeont:SetOnOffCommand",
                   "target_value_determined": "true", "timing": {"starts": "now"}},
            "G2": {**DRYER, "intent_text": "set it to Normal", "affordance_name":
                   "selectedDrynessLevel", "affordance_class": "saref:SetAbsoluteLevelCommand",
                   "target_value_determined": "1",
                   "timing": {"starts": "after_goal", "ref_goal": "G1"}}})
        assert entry["tree"] and _guards(entry["tree"]) == []

    @pytest.mark.asyncio
    async def test_a_step_with_nothing_before_it_is_guarded(self, graph):
        entry = await _dependency_plan(graph, {
            "G1": {**DRYER, "intent_text": "set dryer 1 to Normal", "affordance_name":
                   "selectedDrynessLevel", "affordance_class": "saref:SetAbsoluteLevelCommand",
                   "target_value_determined": "1", "timing": {"starts": "now"}}})
        assert _guards(entry["tree"]) == ["dryer_1: power on first"]


MODULE = "ami_agents.agents.interaction_solver.behaviours.goal_request"


class _FakeRoute:
    """Stands in for every route's planner: one tree per goal, named after it."""

    def __init__(self, goals, logger=None):
        self.results = [{"tree": {"type": "action", "name": str(g)}} for g in goals]

    async def join(self):
        return None


class _FakeGuard:
    """Stands in for a guard step: records which step ran, in order, and which
    entries it was given."""

    seen = []
    step = "power_on"

    def __init__(self, entries, logger=None, graph=None):
        type(self).seen.append((self.step, [e["tree"]["name"] for e in entries]))

    async def join(self):
        return None


class _FakeConditionGuard(_FakeGuard):
    step = "condition"


@pytest.mark.asyncio
async def test_the_guard_step_gets_every_entry_except_the_dependency_ones():
    """The condition-guard check runs first, then the power-on guard -- which
    then also covers the condition guards' own actions -- both on the same
    entries: every one except the dependency structure's."""
    # 1. Four routed items, one per route; every route planned by the fake.
    items = [SimpleNamespace(to_wire_dict=lambda i=i: {"i": i}) for i in range(4)]
    behaviour = GoalRequestBehaviour()
    behaviour.agent = SimpleNamespace(add_behaviour=lambda b: None)
    behaviour.logger = SimpleNamespace(info=lambda *a, **k: None)
    _FakeGuard.seen = []
    with patch(f"{MODULE}.route_behaviour", lambda name: _FakeRoute), \
         patch(f"{MODULE}.PowerOnGuardBehaviour", _FakeGuard), \
         patch(f"{MODULE}.ConditionGuardBehaviour", _FakeConditionGuard), \
         patch(f"{MODULE}.fetch_environment_graph", AsyncMock(return_value=None)):
        await behaviour._plan_mixed(items, [(0, "explicit")], [(1, "incomplete")], None,
                                    ambiguous=[(2, "ambiguous")],
                                    dependency=[(3, "dependency")])
    # 2. The two guard steps, in that order, each with the three simple
    #    entries and not the dependency one.
    simple = ["explicit", "incomplete", "ambiguous"]
    assert _FakeGuard.seen == [("condition", simple), ("power_on", simple)]
