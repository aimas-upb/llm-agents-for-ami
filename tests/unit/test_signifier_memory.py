"""Memory on the planning side: EnvExplorer hosts it, the ISA recalls from it.

    EnvExplorer   record / match behaviours over a real MemoryStore
    ISA           recall: local first, the community on a miss
                  explicit LLM path: a hit is replayed, no model call
                  incomplete path: a hit is the whole stored plan
                  ambiguous path: a hit becomes the planner's signifier hints
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rdflib import Graph

from ami_agents.agents.env_explorer.behaviors.signifier_match_behaviour import (
    SignifierMatchBehaviour,
)
from ami_agents.agents.env_explorer.behaviors.signifier_record_behaviour import (
    SignifierRecordBehaviour,
)
from ami_agents.agents.interaction_solver.behaviours.ambiguous_goal_planning import (
    AmbiguousGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.behaviours.explicit_goal_llm_planning import (
    ExplicitGoalLLMPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.behaviours.incomplete_goal_planning import (
    IncompleteGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.utils.signifier_memory import recall, replay_entry
from ami_agents.shared.memory import ensure_engine_on_path
from ami_agents.shared.models.goal_structure import GoalSpec
from ami_agents.shared.utils.goal_signifiers import signifier_key

ensure_engine_on_path()
from src.storage.memory_store import MemoryStore  # noqa: E402

TREE = {"type": "action", "name": "ac.hvacMode", "action_url": "http://h/ac/hvacMode",
        "parameters": {"value": 3}}

EXPLICIT = GoalSpec.from_dict({
    "goal_specificity": "explicit", "goal_kind": "achievement",
    "intent_text": "set the AC to cooling", "artifact_name": "ac",
    "artifact_class": "homeont:AirConditioner", "affordance_class": "homeont:SetModeCommand",
    "affordance_name": "hvacMode", "parameter_name": "NA", "goal_effect": "set",
    "target_value_text": "cooling"})

LOGGER = MagicMock()


def _record(goal, **fields):
    sk = signifier_key(goal)
    return {"kind": sk.kind, "key": sk.key, "intent_text": goal.intent_text, **fields}


# --------------------------------------------------------------------------
# EnvExplorer: the host
# --------------------------------------------------------------------------

def _explorer(tmp_path):
    return SimpleNamespace(_experience_engine_ready=True,
                           _experience_engine_store=MemoryStore(tmp_path), logger=LOGGER)


class TestHost:
    @pytest.mark.asyncio
    async def test_record_then_match_by_exact_key(self, tmp_path):
        agent = _explorer(tmp_path)
        recorder, matcher = SignifierRecordBehaviour(), SignifierMatchBehaviour()
        recorder.agent = matcher.agent = agent

        recorded = await recorder.record({"signifier": _record(EXPLICIT, tree=TREE)})
        assert recorded["ok"] and recorded["signifier_id"]

        sk = signifier_key(EXPLICIT)
        found = await matcher.match({"kind": sk.kind, "key": sk.key})
        assert found["match"]["tree"] == TREE
        missed = await matcher.match({"kind": sk.kind, "key": {**sk.key, "artifact_name": "x"}})
        assert missed == {"ok": True, "match": None}

    @pytest.mark.asyncio
    async def test_malformed_requests_are_refused(self, tmp_path):
        recorder, matcher = SignifierRecordBehaviour(), SignifierMatchBehaviour()
        recorder.agent = matcher.agent = _explorer(tmp_path)
        assert (await recorder.record({"signifiers": []}))["error"] == "malformed_payload"
        assert (await recorder.record({"signifier": {"kind": "vague", "key": {}}}))[
            "error"] == "invalid_signifier"
        assert (await matcher.match({"key": {}}))["error"] == "malformed_payload"


# --------------------------------------------------------------------------
# ISA: recall
# --------------------------------------------------------------------------

def _isa(local=None, community=None):
    """An ISA agent whose EnvExplorer answers `local` and whose community
    answers `community` (no community client when None)."""
    agent = MagicMock()
    agent._query_env_explorer = AsyncMock(return_value=json.dumps({"ok": True, "match": local}))
    agent.community_client = (SimpleNamespace(match_signifiers=AsyncMock(return_value=community))
                              if community is not None else None)
    return agent


class TestRecall:
    @pytest.mark.asyncio
    async def test_a_local_hit(self):
        agent = _isa(local={"signifier_id": "sig-1", "tree": TREE})
        assert (await recall(agent, EXPLICIT, LOGGER))["signifier_id"] == "sig-1"
        body = agent._query_env_explorer.await_args.kwargs["body"]
        assert body == signifier_key(EXPLICIT).to_wire_dict()

    @pytest.mark.asyncio
    async def test_the_community_on_a_local_miss(self):
        agent = _isa(local=None, community={"signifier_id": "sig-c", "tree": TREE})
        assert (await recall(agent, EXPLICIT, LOGGER))["signifier_id"] == "sig-c"

    @pytest.mark.asyncio
    async def test_nothing_anywhere_or_no_key(self):
        assert await recall(_isa(), EXPLICIT, LOGGER) is None
        maintenance = GoalSpec.from_dict({**EXPLICIT.to_wire_dict(), "goal_kind": "maintenance"})
        agent = _isa(local={"signifier_id": "x"})
        assert await recall(agent, maintenance, LOGGER) is None
        agent._query_env_explorer.assert_not_awaited()

    def test_a_record_without_a_tree_is_not_replayed(self):
        assert replay_entry({"signifier_id": "s", "used": {}}) is None
        entry = replay_entry({"signifier_id": "s", "tree": TREE, "intent_text": "x"})
        assert entry["source"] == "memory" and entry["tree"] == TREE


# --------------------------------------------------------------------------
# ISA: the planning paths
# --------------------------------------------------------------------------

class TestExplicitReplay:
    @pytest.mark.asyncio
    async def test_a_hit_is_replayed_without_the_model(self):
        agent = _isa(local={"signifier_id": "sig-1", "tree": TREE, "intent_text": "x"})
        agent.explicit_bt_planner = SimpleNamespace(generate_bt=AsyncMock())
        behaviour = ExplicitGoalLLMPlanningBehaviour([EXPLICIT], Graph())
        behaviour.agent = agent
        await behaviour.run()
        (entry,) = behaviour.results
        assert (entry["source"], entry["tree"], entry["signifier_id"]) == ("memory", TREE, "sig-1")
        agent.explicit_bt_planner.generate_bt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_can_be_left_to_the_caller(self):
        agent = _isa(local={"signifier_id": "sig-1", "tree": TREE})
        behaviour = ExplicitGoalLLMPlanningBehaviour([EXPLICIT], Graph(), recall_memory=False)
        behaviour.agent = agent
        await behaviour.run()
        agent._query_env_explorer.assert_not_awaited()


class TestIncompleteReplay:
    @pytest.mark.asyncio
    async def test_a_hit_replays_the_whole_plan(self):
        goal = GoalSpec.from_dict({
            "goal_specificity": "incomplete", "goal_kind": "achievement",
            "intent_text": "turn on the kitchen lights", "location_class": "homeont:Kitchen",
            "artifact_class": "homeont:OnOffLight",
            "affordance_class": "homeont:SetOnOffCommand", "affordance_name": "onOff",
            "goal_effect": "set", "quantifier": "all"})
        whole = {"type": "sequence", "children": [TREE, TREE]}
        agent = _isa(local={"signifier_id": "sig-2", "tree": whole})
        behaviour = IncompleteGoalPlanningBehaviour([goal])
        behaviour.agent = agent
        module = "ami_agents.agents.interaction_solver.behaviours.incomplete_goal_planning"
        with patch(f"{module}.fetch_environment_graph", AsyncMock(return_value=Graph())), \
             patch(f"{module}.candidates") as candidates:
            await behaviour.run()
        (entry,) = behaviour.results
        assert entry["tree"] == whole and entry["source"] == "memory"
        candidates.assert_not_called()          # no device choosing on a hit


class TestAmbiguousHints:
    @pytest.mark.asyncio
    async def test_a_hit_becomes_the_planners_hints(self):
        goal = GoalSpec.from_dict({
            "goal_specificity": "ambiguous", "goal_kind": "achievement",
            "intent_text": "the bathroom is so damp",
            "implied_environment_vars": [{
                "property_name": "relative_humidity",
                "property_class": "homeont:RelativeHumidity",
                "property_sensed_space": {"space_class": "homeont:Bathroom",
                                          "space_name": "Bathroom"},
                "property_change_direction": "decrease"}]})
        used = {"actions": [{"url": "http://h/d/onOff", "artifact": "d", "action": "onOff",
                             "parameters": {"value": True}}], "properties": []}
        agent = _isa(local={"signifier_id": "sig-3", "used": used,
                            "intent_text": "the bathroom is so damp"})
        agent.model, agent.temperature = "gpt-5-mini", 0.0
        agent.reasoning_effort, agent.max_completion_tokens = "medium", None
        agent.bt_planner = SimpleNamespace(generate_bt=AsyncMock(
            return_value={"tree": TREE, "explanation": "x"}))
        context = SimpleNamespace(affordances=[{"name": "onOff"}], artifacts=["d"],
                                  goal_brief="b", observable_property_hints={"results": []},
                                  goal_context={"variables": []})
        behaviour = AmbiguousGoalPlanningBehaviour([goal])
        behaviour.agent = agent
        module = "ami_agents.agents.interaction_solver.behaviours.ambiguous_goal_planning"
        with patch(f"{module}.fetch_environment_graph", AsyncMock(return_value=Graph())), \
             patch(f"{module}.build_specific_context", return_value=context):
            await behaviour.run()

        kwargs = agent.bt_planner.generate_bt.await_args.kwargs
        assert kwargs["signifier_hints"]["used"] == used
        assert kwargs["affordances"] == context.affordances     # the context still goes
        (entry,) = behaviour.results
        assert entry["signifier_id"] == "sig-3" and entry["llm"] is True
