"""
Smoke test for atomic intent segmentation.

Tests the LLM-based segmentation without requiring a running simulator.
Validates that the segmentation prompt correctly identifies atomic intents
and their categories based on a set of known test cases.

Run with:
    conda run -n ami python tests/test_atomic_segmentation.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path (go up from tests/unit -> tests -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load .env
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)

from ami_agents.agents.user_assistant.utils.llm_client import (
    build_behaviour_llm_client,
    build_llm_call_kwargs,
)
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.agents.user_assistant.prompts import ATOMIC_SEGMENTATION_SYSTEM_PROMPT
from ami_agents.agents.user_assistant.models import AtomicIntent


async def test_atomic_segmentation():
    """Test atomic segmentation with known inputs."""

    # Use the same agents.yaml settings the live UA reads, so this exercises the
    # model the segmentation prompt is actually tuned for rather than a default.
    agents_config = ConfigLoader.load_with_env_vars(
        str(PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml"))
    ua_config = agents_config.get("user_assistant", {}) or {}
    llm_cfg = build_behaviour_llm_client(ua_config, "atomic_segmentation")
    llm_client = llm_cfg.client
    print(f"Model: {llm_cfg.model}")

    # Test cases: (description, input, expected_count, optional_check)
    # optional_check is a callable that validates specific spans
    def check_living_room_in_goal(intents):
        """Verify the goal intent includes 'living room' context."""
        goal_intents = [i for i in intents if i.type == "GOAL_REQUEST"]
        if not goal_intents:
            return False, "No GOAL_REQUEST found"
        goal_text = goal_intents[0].text.lower()
        if "living room" in goal_text:
            return True, f"Goal text includes 'living room': {goal_intents[0].text!r}"
        else:
            return False, f"Goal text missing 'living room': {goal_intents[0].text!r}"

    test_cases = [
        (
            "Single goal request",
            "Turn on the kitchen light",
            1,
            None,
        ),
        (
            "Mixed categories (state + goal)",
            "Is the bedroom light on, and turn on the living room lights",
            2,
            None,
        ),
        (
            "Multi-symptom collapse (single GOAL_REQUEST)",
            "The bathroom humidity is so high, my skin feels clammy and the towels are probably still damp",
            1,
            None,
        ),
        (
            "Multi-intent with conditional (state + conditional goal with implicit context)",
            "Tell me what the temperature in the living room is and make sure to keep the AC temperature at 24 Celsius whenever the fan is set to low",
            2,
            check_living_room_in_goal,
        ),
    ]

    capabilities_ctx = "(empty capabilities for testing)"

    print("\n" + "=" * 70)
    print("ATOMIC INTENT SEGMENTATION SMOKE TEST")
    print("=" * 70 + "\n")

    all_passed = True

    for i, (description, user_input, expected_count, optional_check) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        print(f"  Input: {user_input!r}")
        print(f"  Expected intent count: {expected_count}")

        # Call segmentation
        prompt = ATOMIC_SEGMENTATION_SYSTEM_PROMPT.format(capabilities=capabilities_ctx)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            response = await llm_client.chat.completions.create(
                model=llm_cfg.model,
                messages=messages,
                **build_llm_call_kwargs(llm_cfg),
            )

            raw_response = (response.choices[0].message.content or "").strip()

            # Parse JSON response
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")

            # Extract intents
            intents_data = parsed.get("intents", [])
            if not isinstance(intents_data, list):
                raise ValueError(f"Expected 'intents' to be a list, got {type(intents_data)}")

            # Validate each intent
            atomic_intents = []
            for item in intents_data:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "").strip()
                intent_type = item.get("type", "").strip()
                reason = item.get("reason", "").strip()
                raw_qualifiers = item.get("qualifiers")
                qualifiers = [
                    q.strip() for q in raw_qualifiers
                    if isinstance(q, str) and q.strip()
                ] if isinstance(raw_qualifiers, list) else []

                if text and intent_type in ("GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST") and reason:
                    atomic_intents.append(AtomicIntent(
                        text=text, type=intent_type, reason=reason,
                        qualifiers=qualifiers))

            # Check result
            actual_count = len(atomic_intents)
            if actual_count == expected_count:
                print(f"  ✓ PASS: Got {actual_count} intent(s) as expected")
                for j, intent in enumerate(atomic_intents, 1):
                    print(f"    [{j}] {intent.type}: {intent.text!r} {intent.qualifiers}")

                # Run optional validation check
                if optional_check:
                    check_passed, check_msg = optional_check(atomic_intents)
                    if check_passed:
                        print(f"  ✓ EXTRA CHECK PASSED: {check_msg}")
                    else:
                        print(f"  ✗ EXTRA CHECK FAILED: {check_msg}")
                        all_passed = False
            else:
                print(f"  ✗ FAIL: Got {actual_count} intent(s), expected {expected_count}")
                for j, intent in enumerate(atomic_intents, 1):
                    print(f"    [{j}] {intent.type}: {intent.text!r} {intent.qualifiers}")
                all_passed = False

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False

        print()

    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70 + "\n")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    exit_code = asyncio.run(test_atomic_segmentation())
    sys.exit(exit_code)
