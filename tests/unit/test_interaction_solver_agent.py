import json
import logging
from unittest.mock import AsyncMock

import pytest

from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.models import Intent


def _bare_solver() -> InteractionSolverAgent:
    """An InteractionSolverAgent without SPADE initialization."""
    agent = object.__new__(InteractionSolverAgent)
    agent.logger = logging.getLogger("test-solver")
    agent.td_sosa_supported = False
    agent.semantic_capabilities = {"td_sosa_supported": False}
    agent._cached_affordances = []
    agent._cached_state_payload = {}
    agent._cached_affordance_by_id = {}
    agent._gather_signifier_matches = AsyncMock(return_value={})
    return agent


AFF_LIGHT = {
    "artifact_id": "http://h/workspaces/home/artifacts/light1#artifact",
    "workspace_id": "http://h/workspaces/home#workspace",
    "affordance_id": "http://h/workspaces/home/artifacts/light1/turnOn",
    "affordance_type": "action",
    "action_name": "turnOn",
    "target": "http://h/workspaces/home/artifacts/light1/turnOn",
}


def _env_query_mock():
    caps = json.dumps({"affordances": [AFF_LIGHT], "semantic_capabilities": {"td_sosa_supported": False}})
    state = json.dumps({"artifacts": {}})

    async def query(message_type, body, expect_type):
        return caps if "capabilities" in message_type.lower() else state

    return AsyncMock(side_effect=query)


class TestWorkspaceScopingFallback:
    @pytest.mark.asyncio
    async def test_matching_workspace_keeps_scoped_affordances(self):
        agent = _bare_solver()
        agent._query_env_explorer = _env_query_mock()
        context = await agent._gather_planning_context(["turn on the light"], workspace_id="home")
        assert context["affordances"] == [AFF_LIGHT]

    @pytest.mark.asyncio
    async def test_unmatched_workspace_falls_back_to_unscoped(self):
        # Extraction often returns a room name ("utility_room") that is not a
        # workspace; scoping to it must not empty the planning context.
        agent = _bare_solver()
        agent._query_env_explorer = _env_query_mock()
        context = await agent._gather_planning_context(
            ["make it cooler"], workspace_id="utility_room"
        )
        assert context["affordances"] == [AFF_LIGHT]


def test_infer_observable_properties_accepts_serialized_structured_intents():
    properties = InteractionSolverAgent._infer_observable_properties_from_intents(
        [
            {
                "action": "modify",
                "artifact": "light",
                "parameter": "brightness",
                "intent_text": "reduce glare",
            }
        ]
    )

    assert "glare" in properties
    assert "luminosity" in properties


def test_infer_observable_properties_still_accepts_intent_objects():
    properties = InteractionSolverAgent._infer_observable_properties_from_intents(
        [
            Intent(
                action="set",
                artifact="window",
                parameter="open_close",
                value=True,
                intent_text="the air is stuffy in here",
            )
        ]
    )

    assert "air_quality" in properties


class TestPlanModeSelection:
    """output_format (agents.yaml planning.llm_planning) picks the planner class."""

    def _agent(self, monkeypatch, output_format=None):
        from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
        from ami_agents.bt_planning.planning.code_planner import AsyncCodeBTPlanner

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = {}
        if output_format is not None:
            config = {"planning": {"llm_planning": {"output_format": output_format}}}
        return InteractionSolverAgent("test@localhost", "pw", config)

    def test_python_code_selects_code_planner(self, monkeypatch):
        from ami_agents.bt_planning.planning.code_planner import AsyncCodeBTPlanner

        agent = self._agent(monkeypatch, "python_code")
        assert agent.plan_mode == "python_code"
        assert isinstance(agent.bt_planner, AsyncCodeBTPlanner)

    def test_default_is_behavior_tree(self, monkeypatch):
        from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
        from ami_agents.bt_planning.planning.code_planner import AsyncCodeBTPlanner

        agent = self._agent(monkeypatch)
        assert agent.plan_mode == "behavior_tree"
        assert isinstance(agent.bt_planner, AsyncBTPlanner)
        assert not isinstance(agent.bt_planner, AsyncCodeBTPlanner)

    def test_garbage_value_falls_back_to_behavior_tree(self, monkeypatch):
        agent = self._agent(monkeypatch, "garbage")
        assert agent.plan_mode == "behavior_tree"


class TestGeneratePlanEnvelope:
    """The plan envelope must carry plan_mode/generated_code and keep the
    intent chain (intents + intent_type) intact."""

    def _solver_for_generate(self):
        agent = _bare_solver()
        agent.plan_mode = "python_code"
        agent._resolve_workspace_id = AsyncMock(return_value="home")
        agent._try_build_plan_from_signifiers = AsyncMock(return_value=None)
        agent._gather_planning_context = AsyncMock(
            return_value={"affordances": [AFF_LIGHT], "state": {}, "signifier_matches": {}}
        )
        agent.bt_planner = AsyncMock()
        agent.bt_planner.generate_bt = AsyncMock(
            return_value={
                "tree": {"name": "N", "type": "action", "action_url": AFF_LIGHT["target"]},
                "explanation": "ok",
                "impossible": False,
                "intents": ["turn on the light"],
                "plan_mode": "python_code",
                "generated_code": "tree = action('light1/turnOn')",
            }
        )
        agent.llm_client = object()
        agent.model = "gpt-4o-mini"
        agent.temperature = 0.5
        agent.reasoning_effort = None
        agent.max_completion_tokens = None
        return agent

    @pytest.mark.asyncio
    async def test_envelope_carries_mode_code_and_intents(self):
        agent = self._solver_for_generate()
        intent = Intent(
            action="set", artifact="light1", parameter="state", value="on",
            intent_text="turn on the light",
        )
        raw = await agent._generate_plan([intent], workspace_id="home", intent_type="EXPLICIT")
        envelope = json.loads(raw)

        assert envelope["plan_type"] == "behavior_tree"
        assert envelope["plan_mode"] == "python_code"
        assert envelope["generated_code"] == "tree = action('light1/turnOn')"
        assert envelope["intent_type"] == "EXPLICIT"
        assert envelope["intents"][0]["intent_text"] == "turn on the light"
        assert envelope["tree"]["action_url"] == AFF_LIGHT["target"]

    @pytest.mark.asyncio
    async def test_error_envelope_carries_mode(self):
        agent = self._solver_for_generate()
        agent._gather_planning_context = AsyncMock(side_effect=RuntimeError("boom"))
        intent = Intent(
            action="set", artifact="light1", parameter="state", value="on",
            intent_text="turn on the light",
        )
        raw = await agent._generate_plan([intent], workspace_id="home", intent_type="IMPLICIT")
        envelope = json.loads(raw)

        assert envelope["error"] == "context_gathering_failed"
        assert envelope["plan_mode"] == "python_code"
        assert envelope["intent_type"] == "IMPLICIT"
