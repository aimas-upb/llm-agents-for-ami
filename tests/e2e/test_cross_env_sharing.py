"""
E2E test: Cross-environment signifier sharing.

Runs Lab308 scenario first, then HomeBench scenario.
Verifies that signifiers from Lab308 influence HomeBench planning.

Requirements:
- OPENAI_API_KEY environment variable
- Community signifier API running at http://localhost:8085 (optional for full flow)
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
def lab308_affordances():
    return [
        {
            "action_name": "turn_on",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308",
            "method": "POST",
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
def homebench_affordances():
    return [
        {
            "action_name": "turn_on",
            "affordance_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight/turn_on",
            "artifact_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight",
            "method": "POST",
        },
        {
            "action_name": "turn_off",
            "affordance_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight/turn_off",
            "artifact_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight",
            "method": "POST",
        },
        {
            "action_name": "set_color",
            "affordance_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight/set_color",
            "artifact_uri": "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight",
            "method": "POST",
            "input_schema": {"type": "object", "properties": {"color": {"type": "string"}}},
        },
    ]


class TestCrossEnvironmentSignifierSharing:
    """
    Full cross-environment scenario:
    1. Lab308: "too dark" -> generate BT -> extract signifiers
    2. HomeBench: inject Lab308 signifiers as hints -> generate BT
    3. Verify HomeBench BT uses its own affordances guided by Lab308 hints
    """

    @pytest.mark.asyncio
    async def test_lab308_then_homebench_sharing(
        self, planner, openai_client, lab308_affordances, homebench_affordances
    ):
        """Full cross-environment flow."""
        from ami_agents.bt_planning.signifier_bridge import extract_signifiers_from_bt

        # --- Step 1: Lab308 scenario ---
        lab308_state = {
            "http://localhost:8080/workspaces/lab308/artifacts/light308": {
                "state": "off",
                "brightness": 0,
            },
        }

        lab308_result = await planner.generate_bt(
            intents=["turn on light308", "set light308 brightness to 100"],
            affordances=lab308_affordances,
            state=lab308_state,
            client=openai_client,
            model="gpt-4o-mini",
        )

        lab308_tree = lab308_result.get("tree", {})
        assert lab308_tree, "Lab308 should produce a tree"
        assert not lab308_result.get("impossible")

        # Extract signifiers from Lab308 BT
        lab308_signifiers = extract_signifiers_from_bt(
            tree_spec=lab308_tree,
            intents=lab308_result.get("intents", []),
            was_successful=True,
        )
        assert len(lab308_signifiers) >= 1, "Should extract signifiers from Lab308 BT"

        # --- Step 2: Convert Lab308 signifiers to hints for HomeBench ---
        signifier_hints = {}
        for sig in lab308_signifiers:
            intent = sig.get("intent", "increase light level")
            if intent not in signifier_hints:
                signifier_hints[intent] = {"matches": [], "final_matches": []}
            match = {
                "signifier_id": sig.get("signifier_id", f"sig-lab308-{len(signifier_hints[intent]['matches'])}"),
                "affordance_uri": sig.get("action_url") or sig.get("affordance_uri", ""),
                "payload_hint": sig.get("parameters", {}),
                "intent_similarity": 0.85,
                "source": "lab308_community",
            }
            signifier_hints[intent]["matches"].append(match)
            signifier_hints[intent]["final_matches"].append(match["signifier_id"])

        # --- Step 3: HomeBench scenario with Lab308 hints ---
        homebench_state = {
            "http://localhost:8090/workspaces/home17/artifacts/studyRoomLight": {
                "state": "off",
            },
        }

        homebench_result = await planner.generate_bt(
            intents=["turn on studyRoomLight"],
            affordances=homebench_affordances,
            state=homebench_state,
            signifier_hints=signifier_hints,
            client=openai_client,
            model="gpt-4o-mini",
        )

        homebench_tree = homebench_result.get("tree", {})
        assert homebench_tree, "HomeBench should produce a tree"
        assert not homebench_result.get("impossible")

        # --- Step 4: Verify HomeBench uses its own URLs ---
        valid_homebench_urls = {a["affordance_uri"] for a in homebench_affordances}

        def collect_action_urls(node, urls=None):
            if urls is None:
                urls = []
            if not isinstance(node, dict):
                return urls
            if node.get("type") == "action" and node.get("action_url"):
                urls.append(node["action_url"])
            for child in node.get("children", []):
                collect_action_urls(child, urls)
            return urls

        homebench_urls = collect_action_urls(homebench_tree)
        assert len(homebench_urls) >= 1, "HomeBench BT should have action nodes"

        for url in homebench_urls:
            assert url in valid_homebench_urls, (
                f"HomeBench BT used URL {url} which is not in HomeBench affordances. "
                f"Lab308 signifier hints should guide intent, not override URLs."
            )

        # --- Step 5: Verify both environments produced valid trees ---
        lab308_errors = planner._validate_tree(lab308_tree)
        homebench_errors = planner._validate_tree(homebench_tree)
        assert lab308_errors == [], f"Lab308 validation errors: {lab308_errors}"
        assert homebench_errors == [], f"HomeBench validation errors: {homebench_errors}"

    @pytest.mark.asyncio
    async def test_signifier_intent_transfer(
        self, planner, openai_client, lab308_affordances, homebench_affordances
    ):
        """
        Verify intent-level transfer: Lab308 has setBrightness, Home 17 has turnOn/setColor.
        The signifier hint conveys 'increase light' concept even though affordances differ.
        """
        # Lab308 signifier for brightness adjustment
        lab308_hints = {
            "make it brighter": {
                "matches": [
                    {
                        "signifier_id": "sig-brightness",
                        "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/set_brightness",
                        "payload_hint": {"brightness": 100},
                        "intent_similarity": 0.9,
                        "source": "lab308_community",
                    },
                ],
                "final_matches": ["sig-brightness"],
            },
        }

        # HomeBench has no setBrightness - only turnOn and setColor
        result = await planner.generate_bt(
            intents=["make it brighter in the study room"],
            affordances=homebench_affordances,
            signifier_hints=lab308_hints,
            client=openai_client,
            model="gpt-4o-mini",
        )

        tree = result.get("tree", {})
        if not tree or result.get("impossible"):
            # It's acceptable for the LLM to mark this as partially impossible
            # since there's no brightness control in Home 17
            return

        # If a tree was generated, it should use HomeBench URLs
        valid_urls = {a["affordance_uri"] for a in homebench_affordances}

        def check_urls(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "action":
                url = node.get("action_url", "")
                assert url in valid_urls, (
                    f"URL {url} not in HomeBench affordances - "
                    f"signifier hint should not inject Lab308 URLs"
                )
            for child in node.get("children", []):
                check_urls(child)

        check_urls(tree)
