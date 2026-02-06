"""
E2E test: HomeBench Home 17 demo scenario.

Scenario: "I can't read anything at my desk" -> BT generated for Home 17 -> executed.

Requirements:
- OPENAI_API_KEY environment variable
- HomeBench FastAPI simulator running at http://localhost:8090 (optional)
"""

import json
import os

import pytest
import pytest_asyncio

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
def homebench_affordances():
    """Affordances for HomeBench Home 17 study room."""
    return [
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
            "input_schema": {
                "type": "object",
                "properties": {"color": {"type": "string"}},
            },
        },
    ]


@pytest.fixture
def homebench_state():
    """Home 17 current state."""
    return {
        "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight": {
            "state": "off",
        },
    }


class TestHomeBenchImplicitCantRead:
    """
    Demo Scenario 2: User says "I can't read anything at my desk" in HomeBench Home 17.

    Expected:
    1. InteractionSolver generates BT for studyRoomLight
    2. BT uses turn_on and/or set_color (no setBrightness in Home 17)
    3. This demonstrates different affordances for similar intents
    """

    @pytest.mark.asyncio
    async def test_bt_generation_for_cant_read(
        self, planner, openai_client, homebench_affordances, homebench_state
    ):
        """Generate BT for implicit 'can't read' request."""
        result = await planner.generate_bt(
            intents=["turn on studyRoomLight"],
            affordances=homebench_affordances,
            state=homebench_state,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        assert not result.get("impossible"), f"Goal should be possible: {result.get('explanation')}"

        tree = result.get("tree", {})
        assert tree, "Tree should be non-empty"

        validation_errors = planner._validate_tree(tree)
        assert validation_errors == [], f"Validation errors: {validation_errors}"

        # Verify URLs are from HomeBench, not Lab308
        valid_urls = {a["affordance_uri"] for a in homebench_affordances}

        def check_urls(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "action":
                url = node.get("action_url", "")
                assert url in valid_urls, f"URL {url} not from HomeBench affordances"
            for child in node.get("children", []):
                check_urls(child)

        check_urls(tree)

    @pytest.mark.asyncio
    async def test_bt_with_cross_env_signifier_hints(
        self, planner, openai_client, homebench_affordances, homebench_state
    ):
        """
        Generate BT for Home 17 with signifier hints from Lab308.

        Lab308 had setBrightness, Home 17 only has turnOn/setColor.
        The hint should guide intent-level transfer, not exact affordance copying.
        """
        lab308_signifier_hints = {
            "increase light level": {
                "matches": [
                    {
                        "signifier_id": "sig-lab308-brightness",
                        "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/set_brightness",
                        "payload_hint": {"brightness": 100},
                        "intent_similarity": 0.88,
                        "source": "lab308_community",
                    },
                    {
                        "signifier_id": "sig-lab308-turnon",
                        "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
                        "payload_hint": {},
                        "intent_similarity": 0.92,
                        "source": "lab308_community",
                    },
                ],
                "final_matches": ["sig-lab308-turnon", "sig-lab308-brightness"],
            },
        }

        result = await planner.generate_bt(
            intents=["turn on studyRoomLight"],
            affordances=homebench_affordances,
            state=homebench_state,
            signifier_hints=lab308_signifier_hints,
            client=openai_client,
            model="gpt-4o-mini",
        )

        assert isinstance(result, dict)
        tree = result.get("tree", {})
        assert tree, "Should generate a tree even with cross-env hints"

        # URLs must come from HomeBench, not Lab308
        valid_urls = {a["affordance_uri"] for a in homebench_affordances}

        def check_urls(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "action":
                url = node.get("action_url", "")
                assert url in valid_urls, (
                    f"URL {url} should be from HomeBench. "
                    f"Lab308 hints should not override available affordances."
                )
            for child in node.get("children", []):
                check_urls(child)

        check_urls(tree)


class TestHomeBenchSignifierExtraction:
    """Test signifier extraction from HomeBench BTs."""

    @pytest.mark.asyncio
    async def test_extract_signifiers_homebench(
        self, planner, openai_client, homebench_affordances, homebench_state
    ):
        """Signifiers extracted from HomeBench BT should reference Home 17 URLs."""
        from ami_agents.bt_planning.signifier_bridge import extract_signifiers_from_bt

        result = await planner.generate_bt(
            intents=["turn on studyRoomLight"],
            affordances=homebench_affordances,
            state=homebench_state,
            client=openai_client,
            model="gpt-4o-mini",
        )

        tree = result.get("tree", {})
        if not tree:
            pytest.skip("No tree generated")

        signifiers = extract_signifiers_from_bt(
            tree_spec=tree,
            intents=result.get("intents", []),
            was_successful=True,
        )

        assert len(signifiers) >= 1

        for sig in signifiers:
            url = sig.get("action_url") or sig.get("affordance_uri", "")
            assert "localhost:8090" in url or "home17" in url or "studyRoomLight" in url, (
                f"Signifier URL should reference HomeBench: {url}"
            )
