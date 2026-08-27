"""Unit tests for ImplicitGoalDesireInferenceBehaviour.

Tests that the LLM-backed inference of affected environment variables
works correctly for various implicit intent phrasings, especially
equivalent expressions that should produce the same env vars.
"""

import pytest


class TestImplicitGoalDesireInference:
    """Tests for ImplicitGoalDesireInferenceBehaviour."""

    @pytest.mark.asyncio
    async def test_darkness_phrasing_equivalence(self):
        """Test that two different darkness phrasings produce equivalent env vars.

        Both "It's too dark in here" and "I can't see anything on my desk"
        should infer: luminosity / increase
        """
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from unittest.mock import AsyncMock, MagicMock
        import json

        # Test data: two equivalent darkness phrasings
        test_cases = [
            {
                "intent": "It's too dark in here",
                "expected_vars": {("luminosity", "increase")},
            },
            {
                "intent": "I can't see anything on my desk",
                "expected_vars": {("luminosity", "increase")},
            },
        ]

        for test_case in test_cases:
            intent_text = test_case["intent"]
            expected_vars = test_case["expected_vars"]

            # Create behaviour
            behaviour = ImplicitGoalDesireInferenceBehaviour(
                intent_text=intent_text,
                logger=MagicMock(),
            )

            # Mock the agent and its LLM client (OpenAI format)
            mock_agent = MagicMock()
            behaviour.agent = mock_agent

            # Mock OpenAI response format
            response_json = {
                "intent_text": intent_text,
                "affected_env_vars": [
                    {"variable": "luminosity", "direction": "increase"}
                ],
            }
            mock_message = MagicMock()
            mock_message.content = json.dumps(response_json)

            mock_choice = MagicMock()
            mock_choice.message = mock_message

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Make the LLM client async (OpenAI API format)
            mock_agent.llm_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # Run the behaviour
            await behaviour.run()

            # Verify no error occurred
            assert behaviour.error is None, f"Error for '{intent_text}': {behaviour.error}"

            # Verify result has correct structure
            assert "affected_env_vars" in behaviour.result
            assert isinstance(behaviour.result["affected_env_vars"], list)
            assert len(behaviour.result["affected_env_vars"]) > 0

            # Extract and compare env vars
            actual_vars = {
                (v["variable"], v["direction"])
                for v in behaviour.result["affected_env_vars"]
            }
            assert (
                actual_vars == expected_vars
            ), f"For '{intent_text}': expected {expected_vars}, got {actual_vars}"

    @pytest.mark.asyncio
    async def test_stuffy_inference(self):
        """Test that 'stuffy' inference produces air circulation and quality vars."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from unittest.mock import AsyncMock, MagicMock
        import json

        intent_text = "It's too stuffy in here"
        expected_vars = {
            ("air_circulation", "increase"),
            ("air_quality", "increase"),
        }

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text=intent_text,
            logger=MagicMock(),
        )

        mock_agent = MagicMock()
        behaviour.agent = mock_agent

        response_json = {
            "intent_text": intent_text,
            "affected_env_vars": [
                {"variable": "air_circulation", "direction": "increase"},
                {"variable": "air_quality", "direction": "increase"},
            ],
        }
        mock_message = MagicMock()
        mock_message.content = json.dumps(response_json)

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_agent.llm_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await behaviour.run()

        assert behaviour.error is None
        actual_vars = {
            (v["variable"], v["direction"]) for v in behaviour.result["affected_env_vars"]
        }
        assert actual_vars == expected_vars

    @pytest.mark.asyncio
    async def test_unknown_inference(self):
        """Test that vague intent produces unknown env vars."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from unittest.mock import AsyncMock, MagicMock
        import json

        intent_text = "Something feels off in here"
        expected_vars = {("unknown", "unknown")}

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text=intent_text,
            logger=MagicMock(),
        )

        mock_agent = MagicMock()
        behaviour.agent = mock_agent

        response_json = {
            "intent_text": intent_text,
            "affected_env_vars": [{"variable": "unknown", "direction": "unknown"}],
        }
        mock_message = MagicMock()
        mock_message.content = json.dumps(response_json)

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_agent.llm_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await behaviour.run()

        assert behaviour.error is None
        actual_vars = {
            (v["variable"], v["direction"]) for v in behaviour.result["affected_env_vars"]
        }
        assert actual_vars == expected_vars

    @pytest.mark.asyncio
    async def test_llm_failure_handling(self):
        """Test that LLM failures are handled gracefully."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from unittest.mock import AsyncMock, MagicMock

        mock_agent = MagicMock()

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text="test intent",
            logger=MagicMock(),
        )

        behaviour.agent = mock_agent

        # Simulate LLM returning empty response
        mock_agent.llm_client.agenerate = AsyncMock(return_value=None)

        await behaviour.run()

        # Should have error set
        assert behaviour.error is not None
        assert behaviour.result == {}

    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test that malformed JSON responses are handled gracefully."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from unittest.mock import AsyncMock, MagicMock

        mock_agent = MagicMock()
        mock_llm_response = MagicMock()
        mock_generation = MagicMock()

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text="test intent",
            logger=MagicMock(),
        )

        behaviour.agent = mock_agent

        # Return malformed JSON
        mock_generation.text = "not valid json {broken"
        mock_llm_response.generations = [mock_generation]
        mock_agent.llm_client.agenerate = AsyncMock(return_value=mock_llm_response)

        await behaviour.run()

        # Should have error set
        assert behaviour.error is not None

    @pytest.mark.asyncio
    async def test_missing_fields_handling(self):
        """Test that responses with missing required fields are rejected."""
        from ami_agents.agents.interaction_solver.behaviours.implicit_goal_desire import (
            ImplicitGoalDesireInferenceBehaviour,
        )
        from unittest.mock import AsyncMock, MagicMock
        import json

        mock_agent = MagicMock()
        mock_llm_response = MagicMock()
        mock_generation = MagicMock()

        behaviour = ImplicitGoalDesireInferenceBehaviour(
            intent_text="test intent",
            logger=MagicMock(),
        )

        behaviour.agent = mock_agent

        # Missing 'affected_env_vars' field
        response_json = {"intent_text": "test"}
        mock_generation.text = json.dumps(response_json)
        mock_llm_response.generations = [mock_generation]
        mock_agent.llm_client.agenerate = AsyncMock(return_value=mock_llm_response)

        await behaviour.run()

        # Should have error set
        assert behaviour.error is not None
