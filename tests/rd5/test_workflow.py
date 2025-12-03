"""Tests for the LangGraph workflow and nodes.

This module tests the workflow nodes with mocked external dependencies.
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rd5.workflow.state import (
    PlanState,
    create_initial_state,
    WorkflowStatus,
    ValidationStatus,
    ExecutionStatus,
)


class TestPlanState:
    """Test cases for PlanState and state creation."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state(
            user_request="Turn on the lights",
            request_id="test-123",
        )

        assert state["user_request"] == "Turn on the lights"
        assert state["request_id"] == "test-123"
        assert state["workflow_status"] == WorkflowStatus.PENDING
        assert state["retry_count"] == 0
        assert state["max_retries"] == 3
        assert state["extracted_intent"] == {}
        assert state["matched_signifiers"] == []
        assert state["matched_affordances"] == []

    def test_create_initial_state_auto_request_id(self):
        """Test that request_id is auto-generated if not provided."""
        state = create_initial_state(user_request="Test request")

        assert state["request_id"] is not None
        assert len(state["request_id"]) == 36  # UUID format


class TestIntentExtractionNode:
    """Test cases for intent extraction node."""

    @pytest.mark.asyncio
    async def test_intent_extraction_no_request(self):
        """Test intent extraction with empty request."""
        from rd5.workflow.nodes.intent_extraction import intent_extraction_node

        state = create_initial_state(user_request="")

        result = await intent_extraction_node(state)

        assert result["extracted_intent"] == {}
        assert result["intent_extraction_error"] == "No user request provided"
        assert result["workflow_status"] == WorkflowStatus.FAILED

    @pytest.mark.asyncio
    async def test_intent_extraction_llm_failure(self):
        """Test intent extraction handles LLM failure gracefully."""
        from rd5.workflow.nodes.intent_extraction import intent_extraction_node

        state = create_initial_state(user_request="Turn on the lights")

        # Mock LLM client to raise exception
        with patch("rd5.workflow.nodes.intent_extraction.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.extract_intent = AsyncMock(side_effect=Exception("API error"))
            mock_llm.return_value = mock_client

            result = await intent_extraction_node(state)

        # Should return fallback intent
        assert result["extracted_intent"]["intent"] == "Turn on the lights"
        assert result["extracted_intent"]["action_verb"] == "unknown"
        assert "API error" in result["intent_extraction_error"]

    @pytest.mark.asyncio
    async def test_intent_extraction_success(self):
        """Test successful intent extraction."""
        from rd5.workflow.nodes.intent_extraction import intent_extraction_node

        state = create_initial_state(user_request="Turn on the living room lights")

        # Mock successful LLM response
        with patch("rd5.workflow.nodes.intent_extraction.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.extract_intent = AsyncMock(return_value={
                "intent": "turn on lights in living room",
                "action_verb": "turn on",
                "target_objects": ["lights"],
                "parameters": {},
                "location": "living room",
                "conditions": {"time": None, "trigger": None},
            })
            mock_llm.return_value = mock_client

            result = await intent_extraction_node(state)

        assert result["extracted_intent"]["intent"] == "turn on lights in living room"
        assert result["extracted_intent"]["action_verb"] == "turn on"
        assert "lights" in result["extracted_intent"]["target_objects"]
        assert result["intent_extraction_error"] is None


class TestAffordanceMatchNode:
    """Test cases for affordance matching node."""

    @pytest.mark.asyncio
    async def test_affordance_match_returns_mock_affordances(self):
        """Test that affordance matching returns mock affordances."""
        from rd5.workflow.nodes.affordance_match import affordance_match_node

        state = create_initial_state(user_request="Turn on lights")
        state["extracted_intent"] = {
            "intent": "turn on lights",
            "action_verb": "turn on",
            "target_objects": ["lights"],
        }

        result = await affordance_match_node(state)

        assert len(result["matched_affordances"]) > 0
        assert result["current_node"] == "affordance_match"

    @pytest.mark.asyncio
    async def test_affordance_match_filters_by_target(self):
        """Test that affordances are filtered by target objects."""
        from rd5.workflow.nodes.affordance_match import match_affordances_to_intent

        affordances = [
            {"name": "toggle_lights", "artifact_name": "Living Room Lights"},
            {"name": "set_temperature", "artifact_name": "Thermostat"},
            {"name": "set_blinds", "artifact_name": "Blinds"},
        ]

        intent = {
            "target_objects": ["lights"],
            "action_verb": "toggle",
        }

        matched = match_affordances_to_intent(affordances, intent)

        assert len(matched) == 1
        assert matched[0]["name"] == "toggle_lights"


class TestCodeValidationNode:
    """Test cases for code validation node."""

    @pytest.mark.asyncio
    async def test_validation_passes_for_safe_code(self):
        """Test validation passes for safe code."""
        from rd5.workflow.nodes.code_validation import code_validation_node

        state = create_initial_state(user_request="Test")
        state["generated_code"] = """
import json
import asyncio

async def main():
    return {"success": True}
"""

        result = await code_validation_node(state)

        assert result["validation_result"]["is_valid"] is True
        assert result["validation_status"] == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_validation_fails_for_unsafe_code(self):
        """Test validation fails for unsafe code."""
        from rd5.workflow.nodes.code_validation import code_validation_node

        state = create_initial_state(user_request="Test")
        state["generated_code"] = """
import os
import subprocess

def dangerous():
    subprocess.run(["rm", "-rf", "/"])
"""

        result = await code_validation_node(state)

        assert result["validation_result"]["is_valid"] is False
        assert result["validation_status"] == ValidationStatus.FAILED
        assert len(result["validation_result"]["errors"]) > 0

    @pytest.mark.asyncio
    async def test_validation_empty_code(self):
        """Test validation handles empty code."""
        from rd5.workflow.nodes.code_validation import code_validation_node

        state = create_initial_state(user_request="Test")
        state["generated_code"] = ""

        result = await code_validation_node(state)

        assert result["validation_result"]["is_valid"] is False
        assert "No code provided" in result["validation_result"]["errors"][0]


class TestSandboxedExecutionNode:
    """Test cases for sandboxed execution node."""

    @pytest.mark.asyncio
    async def test_execution_skipped_for_invalid_code(self):
        """Test execution is skipped for invalid code."""
        from rd5.workflow.nodes.sandboxed_execution import sandboxed_execution_node

        state = create_initial_state(user_request="Test")
        state["generated_code"] = "import os"
        state["validation_result"] = {"is_valid": False}

        result = await sandboxed_execution_node(state)

        assert result["execution_result"]["success"] is False
        assert result["execution_status"] == ExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execution_simulated_success(self):
        """Test simulated successful execution."""
        from rd5.workflow.nodes.sandboxed_execution import sandboxed_execution_node

        state = create_initial_state(user_request="Test")
        state["generated_code"] = "print('hello')"
        state["validation_result"] = {"is_valid": True}

        result = await sandboxed_execution_node(state)

        assert result["execution_result"]["success"] is True
        assert result["execution_status"] == ExecutionStatus.SUCCESS


class TestWorkflowGraph:
    """Test cases for workflow graph creation."""

    def test_workflow_creation(self):
        """Test that workflow can be created."""
        from rd5.workflow.graph import create_workflow

        workflow = create_workflow()

        assert workflow is not None
        # Check nodes are registered
        assert "intent_extraction" in workflow.nodes
        assert "signifier_lookup" in workflow.nodes
        assert "affordance_match" in workflow.nodes
        assert "code_generation" in workflow.nodes
        assert "code_validation" in workflow.nodes

    def test_workflow_compilation(self):
        """Test that workflow can be compiled."""
        from rd5.workflow.graph import compile_workflow

        compiled = compile_workflow()

        assert compiled is not None

    def test_should_retry_generation_passes(self):
        """Test retry logic when validation passes."""
        from rd5.workflow.graph import should_retry_generation

        state = {
            "validation_result": {"is_valid": True},
            "retry_count": 0,
            "max_retries": 3,
        }

        result = should_retry_generation(state)

        assert result == "sandboxed_execution"

    def test_should_retry_generation_retries(self):
        """Test retry logic when validation fails and retries available."""
        from rd5.workflow.graph import should_retry_generation

        state = {
            "validation_result": {"is_valid": False},
            "retry_count": 1,
            "max_retries": 3,
        }

        result = should_retry_generation(state)

        assert result == "code_generation"

    def test_should_retry_generation_max_exceeded(self):
        """Test retry logic when max retries exceeded."""
        from rd5.workflow.graph import should_retry_generation

        state = {
            "validation_result": {"is_valid": False},
            "retry_count": 3,
            "max_retries": 3,
        }

        result = should_retry_generation(state)

        assert result == "sandboxed_execution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
