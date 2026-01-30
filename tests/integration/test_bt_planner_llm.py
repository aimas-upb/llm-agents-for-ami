"""
Integration tests for AsyncBTPlanner with real LLM.

These tests require OPENAI_API_KEY to be set in the environment.
They verify that the planner produces valid BT JSON IR from natural language intents.
"""

import json
import os

import pytest
import pytest_asyncio

from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.schema import TREE_PARAMETER_SCHEMA

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest_asyncio.fixture
async def openai_client():
    """Real AsyncOpenAI client."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture
def planner():
    return AsyncBTPlanner(max_attempts=2)


@pytest.fixture
def lab308_affordances():
    """Affordances for Lab308 environment."""
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


def _validate_tree_structure(tree: dict, errors: list, path: str = "tree") -> None:
    """Recursively validate a BT JSON IR tree."""
    assert isinstance(tree, dict), f"{path}: expected dict, got {type(tree).__name__}"
    assert "name" in tree, f"{path}: missing 'name'"
    assert "type" in tree, f"{path}: missing 'type'"

    valid_types = {"sequence", "selector", "parallel", "action", "condition"}
    assert tree["type"] in valid_types, f"{path}: invalid type '{tree['type']}'"

    if tree["type"] in ("sequence", "selector", "parallel"):
        assert "children" in tree, f"{path}: composite node missing 'children'"
        assert isinstance(tree["children"], list), f"{path}: children must be a list"
        assert len(tree["children"]) > 0, f"{path}: children must be non-empty"
        for i, child in enumerate(tree["children"]):
            _validate_tree_structure(child, errors, f"{path}.children[{i}]")

    elif tree["type"] == "action":
        assert "action_url" in tree, f"{path}: action missing 'action_url'"
        assert isinstance(tree["action_url"], str), f"{path}: action_url must be string"

    elif tree["type"] == "condition":
        assert "property_url" in tree, f"{path}: condition missing 'property_url'"
        assert "expected_value" in tree, f"{path}: condition missing 'expected_value'"


class TestBTGeneration:
    """Test BT generation with real LLM calls."""

    @pytest.mark.asyncio
    async def test_generate_bt_single_action(self, planner, openai_client, lab308_affordances):
        """Single intent should produce a valid BT with at least one action node."""
        result = await planner.generate_bt(
            intents=["turn on light308"],
            affordances=lab308_affordances,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        assert "tree" in result
        assert "explanation" in result
        assert result.get("impossible") is not True

        tree = result["tree"]
        assert isinstance(tree, dict)
        assert tree  # non-empty

        errors = []
        _validate_tree_structure(tree, errors)

    @pytest.mark.asyncio
    async def test_generate_bt_multiple_actions(self, planner, openai_client, lab308_affordances):
        """Multiple intents should produce a BT with multiple action nodes."""
        result = await planner.generate_bt(
            intents=["turn on light308", "set light308 brightness to 80"],
            affordances=lab308_affordances,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        tree = result["tree"]
        assert isinstance(tree, dict)
        assert tree

        # Count action nodes
        def count_actions(node):
            if not isinstance(node, dict):
                return 0
            c = 1 if node.get("type") == "action" else 0
            for child in node.get("children", []):
                c += count_actions(child)
            return c

        assert count_actions(tree) >= 2, "Should have at least 2 action nodes for 2 intents"

    @pytest.mark.asyncio
    async def test_generate_bt_impossible_request(self, planner, openai_client, lab308_affordances):
        """Request for non-existent artifact should be marked impossible."""
        result = await planner.generate_bt(
            intents=["turn on the heater"],
            affordances=lab308_affordances,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        # Either impossible=true or the tree is empty
        if not result.get("impossible"):
            # If LLM didn't mark impossible, tree should at minimum be empty or have no heater actions
            tree = result.get("tree", {})
            if tree:
                # Check no action references a heater URL
                def check_no_heater(node):
                    if not isinstance(node, dict):
                        return
                    url = node.get("action_url", "")
                    assert "heater" not in url.lower(), f"Found heater URL: {url}"
                    for child in node.get("children", []):
                        check_no_heater(child)

                check_no_heater(tree)

    @pytest.mark.asyncio
    async def test_generated_bt_validates(self, planner, openai_client, lab308_affordances):
        """Generated BT should pass the planner's own validation."""
        result = await planner.generate_bt(
            intents=["turn on light308", "open blinds308 fully"],
            affordances=lab308_affordances,
            client=openai_client,
            model="gpt-4o-mini",
        )

        tree = result.get("tree", {})
        if tree and not result.get("impossible"):
            validation_errors = planner._validate_tree(tree)
            assert validation_errors == [], f"Validation errors: {validation_errors}"

    @pytest.mark.asyncio
    async def test_generated_bt_uses_provided_urls(self, planner, openai_client, lab308_affordances):
        """Generated BT action URLs should come from the provided affordances."""
        result = await planner.generate_bt(
            intents=["turn on light308"],
            affordances=lab308_affordances,
            client=openai_client,
            model="gpt-4o-mini",
        )

        tree = result.get("tree", {})
        if not tree or result.get("impossible"):
            pytest.skip("No tree generated")

        valid_urls = {a["affordance_uri"] for a in lab308_affordances}

        def check_urls(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "action":
                url = node.get("action_url", "")
                assert url in valid_urls, f"action_url {url} not in provided affordances"
            for child in node.get("children", []):
                check_urls(child)

        check_urls(tree)


class TestBTPlannerWithSignifiers:
    """Test BT planning with signifier hints injected."""

    @pytest.fixture
    def signifier_hints(self):
        return {
            "turn on light308": {
                "matches": [
                    {
                        "signifier_id": "sig-001",
                        "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
                        "payload_hint": {},
                        "intent_similarity": 0.95,
                        "source": "lab308",
                    },
                ],
                "final_matches": ["sig-001"],
            },
        }

    @pytest.mark.asyncio
    async def test_signifier_hint_influences_plan(
        self, planner, openai_client, lab308_affordances, signifier_hints
    ):
        """Providing signifier hints should produce a valid plan that uses the hinted affordance."""
        result = await planner.generate_bt(
            intents=["turn on light308"],
            affordances=lab308_affordances,
            signifier_hints=signifier_hints,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        tree = result.get("tree", {})
        assert tree
        assert not result.get("impossible")

        # Verify the hinted URL was used
        def find_urls(node, urls=None):
            if urls is None:
                urls = []
            if not isinstance(node, dict):
                return urls
            if node.get("type") == "action" and node.get("action_url"):
                urls.append(node["action_url"])
            for child in node.get("children", []):
                find_urls(child, urls)
            return urls

        urls = find_urls(tree)
        hinted_url = "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on"
        assert hinted_url in urls, f"Expected hinted URL in action nodes, got: {urls}"

    @pytest.mark.asyncio
    async def test_no_signifier_plans_independently(
        self, planner, openai_client, lab308_affordances
    ):
        """Without signifier hints, planner should still generate a valid plan."""
        result = await planner.generate_bt(
            intents=["turn on light308"],
            affordances=lab308_affordances,
            signifier_hints=None,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        tree = result.get("tree", {})
        assert tree
        assert not result.get("impossible")

    @pytest.mark.asyncio
    async def test_signifier_from_different_env(
        self, planner, openai_client
    ):
        """
        Signifier hint from Lab308, affordances from HomeBench Home 17.
        BT should use HomeBench URLs guided by Lab308 hint.
        """
        homebench_affordances = [
            {
                "action_name": "turn_on",
                "affordance_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight/turn_on",
                "artifact_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight",
                "method": "POST",
                "input_schema": None,
            },
            {
                "action_name": "turn_off",
                "affordance_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight/turn_off",
                "artifact_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight",
                "method": "POST",
                "input_schema": None,
            },
            {
                "action_name": "set_color",
                "affordance_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight/set_color",
                "artifact_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight",
                "method": "POST",
                "input_schema": {"type": "object", "properties": {"color": {"type": "string"}}},
            },
        ]

        # Signifier from Lab308 (different env, but similar intent)
        lab308_hints = {
            "increase light level": {
                "matches": [
                    {
                        "signifier_id": "sig-lab308-01",
                        "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
                        "payload_hint": {},
                        "intent_similarity": 0.9,
                        "source": "lab308_community",
                    },
                ],
                "final_matches": ["sig-lab308-01"],
            },
        }

        result = await planner.generate_bt(
            intents=["increase the light in the study room"],
            affordances=homebench_affordances,
            signifier_hints=lab308_hints,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        tree = result.get("tree", {})
        if not tree or result.get("impossible"):
            pytest.skip("No tree generated for cross-env scenario")

        # Verify URLs come from HomeBench, not Lab308
        homebench_urls = {a["affordance_uri"] for a in homebench_affordances}

        def check_urls(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "action":
                url = node.get("action_url", "")
                assert url in homebench_urls, (
                    f"action_url {url} should be from HomeBench, not Lab308"
                )
            for child in node.get("children", []):
                check_urls(child)

        check_urls(tree)
