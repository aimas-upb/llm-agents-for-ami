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
