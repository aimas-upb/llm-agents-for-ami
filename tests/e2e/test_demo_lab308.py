"""
E2E test: Lab308 demo scenario.

Scenario: "It's too dark in here" -> BT generated -> BT executed -> signifiers recorded.

Requirements:
- OPENAI_API_KEY environment variable
- Yggdrasil running at http://localhost:8080 with Lab308 workspace
- Community signifier API running at http://localhost:8085 (optional)
"""

import json
import os

import pytest
import pytest_asyncio

from ami_agents.shared.utils.config_loader import ConfigLoader

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
    pytest.mark.e2e,
]


@pytest_asyncio.fixture
async def openai_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture
def planner():
    from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner

    return AsyncBTPlanner(max_attempts=3)


@pytest.fixture
def lab308_affordances():
    """Full Lab308 affordances (mirrors real Yggdrasil discovery)."""
    return [
        {
            "action_name": "turn_on",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308",
            "method": "POST",
            "input_schema": None,
        },
        {
            "action_name": "turn_off",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_off",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308",
            "method": "POST",
            "input_schema": None,
        },
        {
            "action_name": "set_brightness",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/set_brightness",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308",
            "method": "POST",
            "input_schema": {
                "type": "object",
                "properties": {"brightness": {"type": "integer", "minimum": 0, "maximum": 100}},
            },
        },
        {
            "action_name": "set_position",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/blinds308/set_position",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/blinds308",
            "method": "POST",
            "input_schema": {
                "type": "object",
                "properties": {"position": {"type": "integer", "minimum": 0, "maximum": 100}},
            },
        },
    ]


@pytest.fixture
def lab308_state():
    """Lab308 current state (dark room)."""
    return {
        "http://localhost:8080/workspaces/lab308/artifacts/light308": {
            "state": "off",
            "brightness": 0,
        },
        "http://localhost:8080/workspaces/lab308/artifacts/blinds308": {
            "position": 0,
        },
    }


@pytest.fixture
def test_model():
    """Get model from configuration, with environment variable override."""
    try:
        # Load configuration
        config = ConfigLoader.merge_configs(
            ConfigLoader.load_with_env_vars("ami_agents/config/agents.yaml"),
            ConfigLoader.load_with_env_vars("ami_agents/config/environment.yaml")
        )

        llm_config = config.get("llm", {})
        provider_name = llm_config.get("default_provider", "openai")
        provider_cfg = llm_config.get("providers", {}).get(provider_name, {})

        # Use environment override or config value or fallback
        model = os.getenv("OPENAI_MODEL") or provider_cfg.get("model", "gpt-4")
        return model
    except:
        # Fallback if config loading fails
        return os.getenv("OPENAI_MODEL", "gpt-4")


class TestLab308ImplicitToDark:
    """
    Demo Scenario 1: User says "It's too dark in here" in Lab308.

    Expected:
    1. InteractionSolver generates BT with light + blinds actions
    2. BT execution succeeds (or validates if no live environment)
    3. Signifiers extracted from executed BT
    """

    @pytest.mark.asyncio
    async def test_bt_generation_for_dark_room(
        self, planner, openai_client, lab308_affordances, lab308_state, test_model
    ):
        """Generate BT for implicit 'too dark' request."""
        result = await planner.generate_bt(
            intents=["turn on light308", "set light308 brightness to 100", "set blinds308 position to 100"],
            affordances=lab308_affordances,
            state=lab308_state,
            client=openai_client,
            model=test_model,
        )

        assert isinstance(result, dict)
        assert not result.get("impossible"), f"Goal should be possible: {result.get('explanation')}"

        tree = result.get("tree", {})
        assert tree, "Tree should be non-empty for a possible goal"

        # Validate tree structure
        validation_errors = planner._validate_tree(tree)
        assert validation_errors == [], f"Validation errors: {validation_errors}"

        # Count action nodes - should have at least 2 (light + blinds)
        def count_actions(node):
            if not isinstance(node, dict):
                return 0
            c = 1 if node.get("type") == "action" else 0
            for child in node.get("children", []):
                c += count_actions(child)
            return c

        action_count = count_actions(tree)
        assert action_count >= 2, f"Expected at least 2 action nodes, got {action_count}"

    @pytest.mark.asyncio
    async def test_signifier_extraction_from_bt(
        self, planner, openai_client, lab308_affordances, lab308_state, test_model
    ):
        """Extract signifiers from generated BT."""
        from ami_agents.bt_planning.signifier_bridge import extract_signifiers_from_bt

        result = await planner.generate_bt(
            intents=["turn on light308", "set light308 brightness to 100"],
            affordances=lab308_affordances,
            state=lab308_state,
            client=openai_client,
            model=test_model,
        )

        tree = result.get("tree", {})
        if not tree:
            pytest.skip("No tree generated")

        signifiers = extract_signifiers_from_bt(
            tree_spec=tree,
            intents=result.get("intents", []),
            was_successful=True,
        )

        assert len(signifiers) >= 1, "Should extract at least 1 signifier"

        for sig in signifiers:
            assert "action_url" in sig or "affordance_uri" in sig, f"Signifier missing URL: {sig}"
            assert "intent" in sig, f"Signifier missing intent: {sig}"

    @pytest.mark.asyncio
    async def test_bt_execution_dry_run(
        self, planner, openai_client, lab308_affordances, lab308_state, test_model
    ):
        """Validate generated BT with IRExecutor (dry run, no live environment)."""
        from ami_agents.bt_planning.execution.ir_executor import IRExecutor

        result = await planner.generate_bt(
            intents=["turn on light308"],
            affordances=lab308_affordances,
            state=lab308_state,
            client=openai_client,
            model=test_model,
        )

        tree = result.get("tree", {})
        if not tree:
            pytest.skip("No tree generated")

        executor = IRExecutor(max_ticks=50)
        validation_errors = executor.validate_tree(tree)
        assert validation_errors == [], f"IRExecutor validation errors: {validation_errors}"
