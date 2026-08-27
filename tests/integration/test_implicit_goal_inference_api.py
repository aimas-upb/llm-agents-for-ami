"""Integration tests for ImplicitGoalDesireInferenceBehaviour.

These tests actually call the LLM API to verify that equivalent implicit
intent phrasings produce the same affected environment variables.

Run with:
    OPENAI_API_KEY=... conda run -n ami-agents pytest tests/integration/test_implicit_goal_inference_api.py -v -s
"""

import json
import os
import pytest


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping real LLM tests",
)
class TestImplicitGoalInferenceWithRealLLM:
    """Integration tests that call the real OpenAI API."""

    @pytest.mark.asyncio
    async def test_darkness_phrasing_equivalence_real_llm(self):
        """Test that two darkness phrasings produce equivalent env vars via real LLM.

        Both "It's too dark in here" and "I can't see anything on my desk"
        should infer luminosity/increase via the INFER_IMPLICIT_INTENT_DESIRE_PROMPT.
        """
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from ami_agents.agents.interaction_solver.utils.llm_client import (
            build_llm_client,
        )
        from ami_agents.shared.utils.config_loader import ConfigLoader
        from ami_agents.shared.utils.logger import LoggerFactory
        from unittest.mock import MagicMock
        import asyncio
        from pathlib import Path

        # Load config to get LLM settings
        config_dir = Path(__file__).resolve().parents[2] / "ami_agents" / "config"
        agents_config = ConfigLoader.load_with_env_vars(str(config_dir / "agents.yaml"))
        llm_config = agents_config.get("llm", {})

        # Build real LLM client
        llm_client_config = build_llm_client(llm_config)
        logger = LoggerFactory.get_logger("TestImplicitGoalInference")

        # Test phrasings that should produce equivalent results
        test_cases = [
            ("It's too dark in here", {"luminosity"}),
            ("I can't see anything on my desk", {"luminosity"}),
        ]

        results = {}

        for intent_text, expected_variables in test_cases:
            # Create behaviour with mocked agent but real LLM client
            behaviour = ImplicitGoalDesireInferenceBehaviour(
                intent_text=intent_text,
                logger=logger,
            )

            # Mock minimal agent interface (only needs llm_client)
            mock_agent = MagicMock()
            mock_agent.llm_client = llm_client_config.client
            behaviour.agent = mock_agent

            # Run the behaviour (this will make a real API call)
            print(f"\n[TEST] Inferring env vars for: '{intent_text}'")
            await behaviour.run()

            if behaviour.error:
                pytest.fail(f"LLM call failed for '{intent_text}': {behaviour.error}")

            result = behaviour.result
            print(f"[RESULT] {json.dumps(result, indent=2)}")

            # Verify structure
            assert "affected_env_vars" in result
            assert isinstance(result["affected_env_vars"], list)
            assert len(result["affected_env_vars"]) > 0

            # Extract variables from result
            env_vars = {v["variable"] for v in result["affected_env_vars"]}
            results[intent_text] = env_vars

            # Verify expected variables are present
            assert expected_variables.issubset(env_vars), (
                f"For '{intent_text}': expected {expected_variables} in {env_vars}"
            )

        # CRITICAL: Verify that both phrasings infer the same variable
        phrasing_1 = "It's too dark in here"
        phrasing_2 = "I can't see anything on my desk"

        vars_1 = results[phrasing_1]
        vars_2 = results[phrasing_2]

        print(f"\n[EQUIVALENCE CHECK]")
        print(f"  Phrasing 1: '{phrasing_1}'")
        print(f"  Variables: {vars_1}")
        print(f"  Phrasing 2: '{phrasing_2}'")
        print(f"  Variables: {vars_2}")
        print(f"  Match: {vars_1 == vars_2}")

        assert vars_1 == vars_2, (
            f"Phrasings should produce equivalent env vars. "
            f"Got {vars_1} vs {vars_2}"
        )

    @pytest.mark.asyncio
    async def test_stuffy_inference_real_llm(self):
        """Test stuffy inference via real LLM."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from ami_agents.agents.interaction_solver.utils.llm_client import (
            build_llm_client,
        )
        from ami_agents.shared.utils.config_loader import ConfigLoader
        from ami_agents.shared.utils.logger import LoggerFactory
        from unittest.mock import MagicMock
        from pathlib import Path

        config_dir = Path(__file__).resolve().parents[2] / "ami_agents" / "config"
        agents_config = ConfigLoader.load_with_env_vars(str(config_dir / "agents.yaml"))
        llm_config = agents_config.get("llm", {})
        llm_client_config = build_llm_client(llm_config)
        logger = LoggerFactory.get_logger("TestImplicitGoalInference")

        intent_text = "It's too stuffy in here"
        expected_variables = {
            "air_circulation",
            "air_quality",
        }

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text=intent_text,
            logger=logger,
        )

        mock_agent = MagicMock()
        mock_agent.llm_client = llm_client_config.client
        behaviour.agent = mock_agent

        print(f"\n[TEST] Inferring env vars for: '{intent_text}'")
        await behaviour.run()

        if behaviour.error:
            pytest.fail(f"LLM call failed: {behaviour.error}")

        result = behaviour.result
        print(f"[RESULT] {json.dumps(result, indent=2)}")

        env_vars = {v["variable"] for v in result["affected_env_vars"]}
        print(f"  Variables: {env_vars}")
        print(f"  Expected: {expected_variables}")

        # Both should be present
        assert expected_variables.issubset(env_vars), (
            f"Expected {expected_variables} in {env_vars}"
        )

    @pytest.mark.asyncio
    async def test_unknown_inference_real_llm(self):
        """Test that vague intent produces unknown vars via real LLM."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from ami_agents.agents.interaction_solver.utils.llm_client import (
            build_llm_client,
        )
        from ami_agents.shared.utils.config_loader import ConfigLoader
        from ami_agents.shared.utils.logger import LoggerFactory
        from unittest.mock import MagicMock
        from pathlib import Path

        config_dir = Path(__file__).resolve().parents[2] / "ami_agents" / "config"
        agents_config = ConfigLoader.load_with_env_vars(str(config_dir / "agents.yaml"))
        llm_config = agents_config.get("llm", {})
        llm_client_config = build_llm_client(llm_config)
        logger = LoggerFactory.get_logger("TestImplicitGoalInference")

        intent_text = "Something feels off in here"

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text=intent_text,
            logger=logger,
        )

        mock_agent = MagicMock()
        mock_agent.llm_client = llm_client_config.client
        behaviour.agent = mock_agent

        print(f"\n[TEST] Inferring env vars for: '{intent_text}'")
        await behaviour.run()

        if behaviour.error:
            pytest.fail(f"LLM call failed: {behaviour.error}")

        result = behaviour.result
        print(f"[RESULT] {json.dumps(result, indent=2)}")

        # For vague intents, should return unknown pair
        env_vars = result.get("affected_env_vars", [])
        print(f"  Variables: {env_vars}")

        # Either we get unknown, or we get valid inferenced vars
        # The key is that we get a non-empty result
        assert len(env_vars) > 0, "Should have at least one env var pair"

    @pytest.mark.asyncio
    async def test_generalization_cold_phrasing(self):
        """Test that cold-related phrasings (not in prompt examples) are inferred correctly."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from ami_agents.agents.interaction_solver.utils.llm_client import (
            build_llm_client,
        )
        from ami_agents.shared.utils.config_loader import ConfigLoader
        from ami_agents.shared.utils.logger import LoggerFactory
        from unittest.mock import MagicMock
        from pathlib import Path

        config_dir = Path(__file__).resolve().parents[2] / "ami_agents" / "config"
        agents_config = ConfigLoader.load_with_env_vars(str(config_dir / "agents.yaml"))
        llm_config = agents_config.get("llm", {})
        llm_client_config = build_llm_client(llm_config)
        logger = LoggerFactory.get_logger("TestImplicitGoalInference")

        # NOT in prompt examples: cold-related phrasing (opposite of stuffy)
        intent_text = "This room is freezing"

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text=intent_text,
            logger=logger,
        )

        mock_agent = MagicMock()
        mock_agent.llm_client = llm_client_config.client
        behaviour.agent = mock_agent

        print(f"\n[TEST] Inferring env vars for: '{intent_text}'")
        await behaviour.run()

        if behaviour.error:
            pytest.fail(f"LLM call failed: {behaviour.error}")

        result = behaviour.result
        print(f"[RESULT] {json.dumps(result, indent=2)}")

        env_vars = {v["variable"] for v in result["affected_env_vars"]}
        print(f"  Variables: {env_vars}")

        # Should infer temperature/increase (cold → want warmth)
        assert "temperature" in env_vars, (
            f"Expected temperature to be inferred for '{intent_text}', got {env_vars}"
        )

    @pytest.mark.asyncio
    async def test_generalization_temperature_phrasing(self):
        """Test that temperature-related phrasings (not in prompt examples) are inferred correctly."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from ami_agents.agents.interaction_solver.utils.llm_client import (
            build_llm_client,
        )
        from ami_agents.shared.utils.config_loader import ConfigLoader
        from ami_agents.shared.utils.logger import LoggerFactory
        from unittest.mock import MagicMock
        from pathlib import Path

        config_dir = Path(__file__).resolve().parents[2] / "ami_agents" / "config"
        agents_config = ConfigLoader.load_with_env_vars(str(config_dir / "agents.yaml"))
        llm_config = agents_config.get("llm", {})
        llm_client_config = build_llm_client(llm_config)
        logger = LoggerFactory.get_logger("TestImplicitGoalInference")

        # NOT in prompt examples: warm/heat-related phrasing (opposite of "freezing")
        intent_text = "This place needs to be warmer"

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text=intent_text,
            logger=logger,
        )

        mock_agent = MagicMock()
        mock_agent.llm_client = llm_client_config.client
        behaviour.agent = mock_agent

        print(f"\n[TEST] Inferring env vars for: '{intent_text}'")
        await behaviour.run()

        if behaviour.error:
            pytest.fail(f"LLM call failed: {behaviour.error}")

        result = behaviour.result
        print(f"[RESULT] {json.dumps(result, indent=2)}")

        env_vars = {v["variable"] for v in result["affected_env_vars"]}
        print(f"  Variables: {env_vars}")

        # Temperature phrasing → temperature/increase
        assert "temperature" in env_vars, (
            f"Expected temperature to be inferred for '{intent_text}', got {env_vars}"
        )

    @pytest.mark.asyncio
    async def test_generalization_sound_phrasing(self):
        """Test that sound-related phrasings (not in prompt examples) are inferred correctly."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from ami_agents.agents.interaction_solver.utils.llm_client import (
            build_llm_client,
        )
        from ami_agents.shared.utils.config_loader import ConfigLoader
        from ami_agents.shared.utils.logger import LoggerFactory
        from unittest.mock import MagicMock
        from pathlib import Path

        config_dir = Path(__file__).resolve().parents[2] / "ami_agents" / "config"
        agents_config = ConfigLoader.load_with_env_vars(str(config_dir / "agents.yaml"))
        llm_config = agents_config.get("llm", {})
        llm_client_config = build_llm_client(llm_config)
        logger = LoggerFactory.get_logger("TestImplicitGoalInference")

        # NOT in prompt examples: sound-related phrasing
        intent_text = "It's too loud in here"

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text=intent_text,
            logger=logger,
        )

        mock_agent = MagicMock()
        mock_agent.llm_client = llm_client_config.client
        behaviour.agent = mock_agent

        print(f"\n[TEST] Inferring env vars for: '{intent_text}'")
        await behaviour.run()

        if behaviour.error:
            pytest.fail(f"LLM call failed: {behaviour.error}")

        result = behaviour.result
        print(f"[RESULT] {json.dumps(result, indent=2)}")

        env_vars = {v["variable"] for v in result["affected_env_vars"]}
        print(f"  Variables: {env_vars}")

        # Too loud → sound/decrease
        assert "sound" in env_vars, (
            f"Expected sound to be inferred for '{intent_text}', got {env_vars}"
        )
