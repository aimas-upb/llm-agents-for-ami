"""
Live-LLM A/B smoke for the two plan-generation modes.

Requires OPENAI_API_KEY. Runs the same planning task through AsyncBTPlanner
(JSON IR tool call) and AsyncCodeBTPlanner (builder-DSL script) and asserts
both return a validated, resolved IR tree — an end-to-end mode comparison
that needs no SimuHome/Ollama infrastructure.

    conda run -n ami pytest tests/integration/test_code_planner_live.py -q
"""

import os

import pytest
import pytest_asyncio

from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.bt_planner_direct import DirectCodeBTPlanner
from ami_agents.bt_planning.planning.code_planner import AsyncCodeBTPlanner

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
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

STATE = {"artifacts": {"light308": {"brightness": 10, "state": "on"}}}


@pytest_asyncio.fixture
async def openai_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture
def test_model():
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _collect_action_urls(spec: dict) -> list[str]:
    urls = []
    if not isinstance(spec, dict):
        return urls
    if spec.get("action_url"):
        urls.append(spec["action_url"])
    for child in spec.get("children", []):
        urls.extend(_collect_action_urls(child))
    return urls


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("planner_cls", [AsyncBTPlanner, AsyncCodeBTPlanner, DirectCodeBTPlanner])
async def test_all_modes_generate_a_plan(planner_cls, openai_client, test_model):
    planner = planner_cls(max_attempts=3)
    result = await planner.generate_bt(
        intents=["increase the brightness of the light to 80"],
        affordances=[LIGHT_ACTION, LIGHT_PROPERTY],
        state=STATE,
        client=openai_client,
        model=test_model,
        temperature=0.2,
    )

    assert result["impossible"] is False, result["explanation"]

    if planner_cls is DirectCodeBTPlanner:
        # py_trees_code mode: no IR tree; the payload is executable code that
        # must reference the real action URL and define the tree contract.
        assert result["plan_mode"] == "py_trees_code"
        code = result["generated_code"]
        assert code.strip(), f"empty code: {result['explanation']}"
        assert LIGHT_ACTION["target"] in code
        assert "tree" in code
    else:
        tree = result["tree"]
        assert tree, f"empty tree: {result['explanation']}"
        assert LIGHT_ACTION["target"] in _collect_action_urls(tree)
        expected_mode = (
            "python_code" if planner_cls is AsyncCodeBTPlanner else "behavior_tree"
        )
        assert result["plan_mode"] == expected_mode
        if planner_cls is AsyncCodeBTPlanner:
            assert result["generated_code"].strip()
