"""End-to-end integration tests for RD5 workflow.

This module tests the complete workflow with mocked external services.
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Ensure mock sandbox is used
os.environ["RD5_USE_DOCKER_SANDBOX"] = "false"


class TestEndToEndWorkflow:
    """End-to-end tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_mocked_llm(self):
        """Test complete workflow with mocked LLM responses."""
        from rd5.workflow.state import create_initial_state, WorkflowStatus
        from rd5.workflow.graph import create_workflow

        # Create initial state
        state = create_initial_state(
            user_request="Turn on the living room lights",
            request_id="test-e2e-001",
        )

        # Mock LLM client
        mock_llm = MagicMock()
        mock_llm.extract_intent = AsyncMock(return_value={
            "intent": "turn on lights in living room",
            "action_verb": "turn on",
            "target_objects": ["lights"],
            "parameters": {},
            "location": "living room",
            "conditions": {"time": None, "trigger": None},
        })
        mock_llm.generate_plan_code = AsyncMock(return_value='''
import json
import asyncio

async def main():
    # Simulated light control
    result = {"success": True, "message": "Lights turned on"}
    print(json.dumps(result))
    return result
''')
        mock_llm.summarize_execution = AsyncMock(
            return_value="Successfully turned on the living room lights."
        )

        # Mock RD4 client
        mock_rd4_health = AsyncMock(return_value=False)

        # Mock Weaviate client
        mock_weaviate = MagicMock()
        mock_weaviate.connect = MagicMock()
        mock_weaviate.close = MagicMock()
        mock_weaviate.ensure_schema = MagicMock()
        mock_weaviate.store_plan = MagicMock(return_value="plan-123")
        mock_weaviate.search_similar_plans = MagicMock(return_value=[])

        with patch("rd5.workflow.nodes.intent_extraction.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.WeaviateClient", return_value=mock_weaviate), \
             patch("rd5.workflow.nodes.result_feedback.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.plan_storage.WeaviateClient", return_value=mock_weaviate), \
             patch("rd5.integration.rd4_client.RD4Client.health_check", mock_rd4_health):

            # Create and compile workflow
            workflow = create_workflow()
            compiled = workflow.compile()

            # Run workflow
            final_state = await compiled.ainvoke(state)

            # Verify results
            assert final_state["extracted_intent"]["intent"] == "turn on lights in living room"
            assert final_state["extracted_intent"]["action_verb"] == "turn on"
            assert len(final_state["matched_affordances"]) > 0
            assert final_state["validation_result"]["is_valid"] is True
            assert final_state["execution_result"]["success"] is True
            assert final_state["workflow_status"] == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_workflow_with_validation_failure_and_retry(self):
        """Test workflow retries code generation on validation failure."""
        from rd5.workflow.state import create_initial_state, WorkflowStatus
        from rd5.workflow.graph import create_workflow

        state = create_initial_state(
            user_request="Set thermostat to 72 degrees",
            request_id="test-e2e-002",
            max_retries=2,
        )

        # Track generation attempts
        generation_attempts = [0]

        async def mock_generate_code(*args, **kwargs):
            generation_attempts[0] += 1
            if generation_attempts[0] == 1:
                # First attempt: unsafe code
                return '''
import os
os.system("rm -rf /")
'''
            else:
                # Second attempt: safe code
                return '''
import json
import asyncio

async def main():
    result = {"success": True, "temperature": 72}
    print(json.dumps(result))
    return result
'''

        mock_llm = MagicMock()
        mock_llm.extract_intent = AsyncMock(return_value={
            "intent": "set thermostat to 72 degrees",
            "action_verb": "set",
            "target_objects": ["thermostat"],
            "parameters": {"temperature": 72},
        })
        mock_llm.generate_plan_code = AsyncMock(side_effect=mock_generate_code)
        mock_llm.summarize_execution = AsyncMock(return_value="Thermostat set to 72°F")

        mock_weaviate = MagicMock()
        mock_weaviate.connect = MagicMock()
        mock_weaviate.close = MagicMock()
        mock_weaviate.ensure_schema = MagicMock()
        mock_weaviate.store_plan = MagicMock(return_value="plan-456")
        mock_weaviate.search_similar_plans = MagicMock(return_value=[])

        with patch("rd5.workflow.nodes.intent_extraction.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.WeaviateClient", return_value=mock_weaviate), \
             patch("rd5.workflow.nodes.result_feedback.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.plan_storage.WeaviateClient", return_value=mock_weaviate):

            workflow = create_workflow()
            compiled = workflow.compile()
            final_state = await compiled.ainvoke(state)

            # Should have retried and succeeded
            assert generation_attempts[0] == 2
            assert final_state["validation_result"]["is_valid"] is True
            assert final_state["execution_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_workflow_with_execution_failure(self):
        """Test workflow handles execution failure gracefully."""
        from rd5.workflow.state import create_initial_state, WorkflowStatus
        from rd5.workflow.graph import create_workflow

        state = create_initial_state(
            user_request="Do something that fails",
            request_id="test-e2e-003",
        )

        mock_llm = MagicMock()
        mock_llm.extract_intent = AsyncMock(return_value={
            "intent": "do something that fails",
            "action_verb": "do",
            "target_objects": [],
        })
        mock_llm.generate_plan_code = AsyncMock(return_value='''
import json

# This will raise an error immediately when executed
result = 1 / 0
''')
        mock_llm.summarize_execution = AsyncMock(return_value="Execution failed")

        mock_weaviate = MagicMock()
        mock_weaviate.search_similar_plans = MagicMock(return_value=[])

        with patch("rd5.workflow.nodes.intent_extraction.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.WeaviateClient", return_value=mock_weaviate), \
             patch("rd5.workflow.nodes.result_feedback.get_llm_client", return_value=mock_llm):

            workflow = create_workflow()
            compiled = workflow.compile()
            final_state = await compiled.ainvoke(state)

            # Should fail but complete workflow
            assert final_state["execution_result"]["success"] is False
            assert final_state["workflow_status"] == WorkflowStatus.FAILED
            assert final_state["feedback_sent"] is True


class TestWorkflowRunFunction:
    """Test the run_plan_generation convenience function."""

    @pytest.mark.asyncio
    async def test_run_plan_generation(self):
        """Test the run_plan_generation function."""
        from rd5.workflow.graph import run_plan_generation

        mock_llm = MagicMock()
        mock_llm.extract_intent = AsyncMock(return_value={
            "intent": "test intent",
            "action_verb": "test",
            "target_objects": [],
        })
        mock_llm.generate_plan_code = AsyncMock(return_value='''
import json
print(json.dumps({"success": True}))
''')
        mock_llm.summarize_execution = AsyncMock(return_value="Done")

        mock_weaviate = MagicMock()
        mock_weaviate.connect = MagicMock()
        mock_weaviate.close = MagicMock()
        mock_weaviate.ensure_schema = MagicMock()
        mock_weaviate.store_plan = MagicMock(return_value="plan-789")
        mock_weaviate.search_similar_plans = MagicMock(return_value=[])

        with patch("rd5.workflow.nodes.intent_extraction.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.WeaviateClient", return_value=mock_weaviate), \
             patch("rd5.workflow.nodes.result_feedback.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.plan_storage.WeaviateClient", return_value=mock_weaviate):

            result = await run_plan_generation(
                user_request="Test request",
                request_id="test-run-001",
            )

            assert result is not None
            assert "execution_result" in result


class TestAffordanceIntegration:
    """Test affordance matching integration."""

    @pytest.mark.asyncio
    async def test_affordance_matching_filters_correctly(self):
        """Test that affordances are correctly filtered by intent."""
        from rd5.workflow.nodes.affordance_match import (
            affordance_match_node,
            _get_mock_affordances,
        )
        from rd5.workflow.state import create_initial_state

        state = create_initial_state(user_request="Adjust the blinds")
        state["extracted_intent"] = {
            "intent": "adjust blinds",
            "action_verb": "adjust",
            "target_objects": ["blinds"],
            "location": "living room",
        }

        result = await affordance_match_node(state)

        # Should match blinds affordances
        matched = result["matched_affordances"]
        blinds_affordances = [
            a for a in matched
            if "blind" in a.get("name", "").lower() or "blind" in a.get("artifact_name", "").lower()
        ]
        assert len(blinds_affordances) > 0

    @pytest.mark.asyncio
    async def test_affordance_matching_returns_all_when_no_match(self):
        """Test that all affordances are returned when no specific match."""
        from rd5.workflow.nodes.affordance_match import affordance_match_node
        from rd5.workflow.state import create_initial_state

        state = create_initial_state(user_request="Do something generic")
        state["extracted_intent"] = {
            "intent": "do something",
            "action_verb": "do",
            "target_objects": [],  # No specific targets
        }

        result = await affordance_match_node(state)

        # Should return all mock affordances
        assert len(result["matched_affordances"]) == 4  # All mock affordances


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
