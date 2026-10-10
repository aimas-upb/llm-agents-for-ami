"""GoalRequestBehaviour: structured goals in, deterministic or workflow plans out."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ami_agents.agents.interaction_solver.behaviours.goal_request import (
    GoalRequestBehaviour,
)
from ami_agents.agents.interaction_solver.utils.explicit_goal_plan import deterministic_goal
from ami_agents.agents.interaction_solver.utils.incomplete_goal_plan import incomplete_goal
from ami_agents.shared.models.goal_structure import GoalStructure
from ami_agents.shared.models.intents import Intent

MODULE = "ami_agents.agents.interaction_solver.behaviours.goal_request"

EXPLICIT = {"goal_specificity": "explicit", "goal_kind": "achievement",
            "goal_effect": "set", "artifact_name": "kitchen_on_off_light_1",
            "affordance_name": "onOff", "target_value_text": "on",
            "target_value_determined": "true"}


def _structure(text, **fields):
    return GoalStructure.from_dict({**EXPLICIT, **fields}, intent_text=text,
                                   structure="simple")


class _FakeExplicit:
    def __init__(self, goals, logger=None):
        self.results = [{"tree": {"type": "action", "name": g.artifact_name},
                         "explanation": "deterministic"} for g in goals]

    async def join(self):
        return None


class _FakeWorkflow:
    seen = []

    def __init__(self, intents, workspace_id=None, logger=None):
        type(self).seen = [i.to_query_string() for i in intents]
        self.reply_envelope = {"plans": [
            {"tree": {"type": "action", "name": f"llm:{i.to_query_string()}"},
             "intent": i.to_wire_dict()} for i in intents]}

    async def join(self):
        return None


def _behaviour():
    behaviour = GoalRequestBehaviour()
    behaviour.agent = MagicMock()
    return behaviour


class TestParsing:
    def test_structured_goals_are_parsed_as_structures(self):
        msg = MagicMock()
        msg.body = json.dumps({"intents": [_structure("turn on the light").to_wire_dict()]})
        (item,), _ = GoalRequestBehaviour._parse_request(msg)
        assert isinstance(item, GoalStructure)
        assert item.goals["G1"].target_value_determined == "true"


class TestPlanning:
    @pytest.mark.asyncio
    async def test_deterministic_and_workflow_plans_merge_in_request_order(self):
        items = [
            Intent("make it cosy"),                                         # workflow
            _structure("turn on the light"),                                # deterministic
            _structure("brighten it", target_value_determined=None),        # workflow
        ]
        with patch(f"{MODULE}.ExplicitGoalPlanningBehaviour", _FakeExplicit), \
             patch(f"{MODULE}.PlanningWorkflowBehaviour", _FakeWorkflow):
            envelope = await _behaviour()._plan_mixed(
                items, [(1, items[1].goals["G1"])], [], "ws")

        names = [p["tree"]["name"] for p in envelope["plans"]]
        assert names == ["llm:make it cosy", "kitchen_on_off_light_1", "llm:brighten it"]
        # The workflow plans from the words; the entry keeps the structure.
        assert _FakeWorkflow.seen == ["make it cosy", "brighten it"]
        assert envelope["plans"][2]["intent"]["goals"]["G1"]["target_value_text"] == "on"

    @pytest.mark.asyncio
    async def test_all_deterministic_skips_the_workflow(self):
        items = [_structure("turn on the light")]
        _FakeWorkflow.seen = None
        with patch(f"{MODULE}.ExplicitGoalPlanningBehaviour", _FakeExplicit), \
             patch(f"{MODULE}.PlanningWorkflowBehaviour", _FakeWorkflow):
            envelope = await _behaviour()._plan_mixed(
                items, [(0, items[0].goals["G1"])], [], None)
        assert _FakeWorkflow.seen is None
        assert len(envelope["plans"]) == 1


INCOMPLETE = {"goal_specificity": "incomplete", "goal_kind": "achievement",
              "goal_effect": "set", "location_class": "homeont:Kitchen",
              "artifact_class": "homeont:OnOffLight", "artifact_name": None,
              "affordance_class": "homeont:SetOnOffCommand", "affordance_name": "onOff",
              "target_value_text": "on", "quantifier": "all"}


class _FakeIncomplete:
    def __init__(self, goals, logger=None):
        self.results = [{"tree": {"type": "sequence", "name": f"all:{g.artifact_class}"},
                         "explanation": "incomplete"} for g in goals]

    async def join(self):
        return None


class TestIncompleteRouting:
    def test_a_simple_incomplete_achievement_goal_is_selected(self):
        structure = GoalStructure.from_dict(INCOMPLETE, intent_text="turn on the kitchen lights",
                                            structure="simple")
        assert incomplete_goal(structure) is structure.goals["G1"]
        assert deterministic_goal(structure) is None

    def test_dependency_structures_and_maintenance_stay_on_the_workflow(self):
        dependency = GoalStructure.from_dict({"goals": {"G1": INCOMPLETE}},
                                             intent_text="x", structure="dependency")
        dependency.predicates = {"P1": object()}
        assert incomplete_goal(dependency) is None
        maintenance = GoalStructure.from_dict({**INCOMPLETE, "goal_kind": "maintenance"},
                                              intent_text="x", structure="simple")
        assert incomplete_goal(maintenance) is None

    def test_goals_only_the_user_can_resolve_are_not_selected(self):
        unnamed = GoalStructure.from_dict(
            {**INCOMPLETE, "location_class": None, "artifact_class": None},
            intent_text="turn it on", structure="simple")
        assert incomplete_goal(unnamed) is None

    @pytest.mark.asyncio
    async def test_incomplete_goals_skip_the_workflow(self):
        items = [_structure("turn on the light"),
                 GoalStructure.from_dict(INCOMPLETE, intent_text="turn on the kitchen lights",
                                         structure="simple")]
        _FakeWorkflow.seen = None
        with patch(f"{MODULE}.ExplicitGoalPlanningBehaviour", _FakeExplicit), \
             patch(f"{MODULE}.IncompleteGoalPlanningBehaviour", _FakeIncomplete), \
             patch(f"{MODULE}.PlanningWorkflowBehaviour", _FakeWorkflow):
            behaviour = _behaviour()
            envelope = await behaviour._plan_mixed(
                items, behaviour._select(items, deterministic_goal),
                behaviour._select(items, incomplete_goal), None)
        assert _FakeWorkflow.seen is None
        names = [p["tree"]["name"] for p in envelope["plans"]]
        assert names == ["kitchen_on_off_light_1", "all:homeont:OnOffLight"]
        assert envelope["plans"][1]["intent"]["goals"]["G1"]["quantifier"] == "all"


AMBIGUOUS = {"goal_specificity": "ambiguous", "goal_kind": "achievement",
             "intent_text": "the bathroom is so damp",
             "implied_environment_vars": [{
                 "property_name": "relative_humidity",
                 "property_class": "homeont:RelativeHumidity",
                 "property_sensed_space": {"space_class": "homeont:Bathroom",
                                           "space_name": "Bathroom"}}]}


class _FakeAmbiguous:
    def __init__(self, goals, logger=None):
        self.results = [{"tree": {"type": "action", "name": f"llm:{g.intent_text}"}}
                        for g in goals]

    async def join(self):
        return None


class TestAmbiguousRouting:
    @pytest.mark.asyncio
    async def test_ambiguous_goals_skip_the_workflow(self):
        from ami_agents.agents.interaction_solver.utils.ambiguous_goal_plan import (
            ambiguous_goal,
        )
        items = [GoalStructure.from_dict(AMBIGUOUS, intent_text="the bathroom is so damp",
                                         structure="simple")]
        _FakeWorkflow.seen = None
        with patch(f"{MODULE}.AmbiguousGoalPlanningBehaviour", _FakeAmbiguous), \
             patch(f"{MODULE}.PlanningWorkflowBehaviour", _FakeWorkflow):
            behaviour = _behaviour()
            envelope = await behaviour._plan_mixed(
                items, [], [], None, behaviour._select(items, ambiguous_goal))
        assert _FakeWorkflow.seen is None
        assert envelope["plans"][0]["tree"]["name"] == "llm:the bathroom is so damp"
        assert envelope["plans"][0]["intent"]["goals"]["G1"]["implied_environment_vars"]
