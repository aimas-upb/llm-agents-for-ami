"""
Unit tests for IRExecutor - compilation and execution with mocked HTTP.
"""

import pytest
from unittest.mock import patch, MagicMock

import py_trees
from py_trees.common import Status

from ami_agents.bt_planning.execution.ir_executor import IRExecutor
from ami_agents.bt_planning.execution.base import ExecutionResult
from ami_agents.bt_planning.nodes.affordance_nodes import (
    ActionAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    WaitPropertyConditionNode,
)


@pytest.fixture
def executor():
    return IRExecutor(max_ticks=10)


class TestCompilation:
    """Tests for IRExecutor._compile()."""

    def test_compile_sequence_node(self, executor, sample_sequence_spec):
        tree = executor._compile(sample_sequence_spec)
        assert isinstance(tree, py_trees.composites.Sequence)
        assert tree.name == "IncreaseLightSequence"
        assert len(tree.children) == 3

    def test_compile_selector_node(self, executor, sample_selector_spec):
        tree = executor._compile(sample_selector_spec)
        assert isinstance(tree, py_trees.composites.Selector)
        assert tree.name == "EnsureLightOn"
        assert len(tree.children) == 2

    def test_compile_parallel_node(self, executor, sample_parallel_spec):
        tree = executor._compile(sample_parallel_spec)
        assert isinstance(tree, py_trees.composites.Parallel)
        assert tree.name == "ParallelActions"
        assert len(tree.children) == 2

    def test_compile_action_node(self, executor, sample_action_spec):
        tree = executor._compile(sample_action_spec)
        assert isinstance(tree, ActionAffordanceNode)
        assert tree.name == "TurnOnLight"
        assert tree.action_url.endswith("/turn_on")

    def test_compile_condition_node(self, executor, sample_condition_spec):
        tree = executor._compile(sample_condition_spec)
        assert isinstance(tree, PropertyConditionNode)
        assert tree.name == "IsLightOn"
        assert tree.expected_value == "on"

    def test_compile_condition_with_operator(self, executor):
        spec = {
            "name": "TempCheck",
            "type": "condition",
            "property_url": "http://localhost:8080/props/temp",
            "expected_value": 25,
            "operator": ">",
        }
        tree = executor._compile(spec)
        assert isinstance(tree, ComparisonPropertyConditionNode)

    def test_compile_wait_condition_node(self, executor):
        spec = {
            "name": "WaitForGlare",
            "type": "wait_condition",
            "property_url": "http://localhost:8080/props/glare",
            "expected_value": 50,
            "operator": "<=",
            "timeout_seconds": 20,
            "poll_interval_seconds": 0,
        }
        tree = executor._compile(spec)
        assert isinstance(tree, WaitPropertyConditionNode)
        assert tree.timeout_seconds == 20.0
        assert tree.poll_interval_seconds == 0.0

    def test_compile_nested_tree(self, executor, sample_nested_spec):
        tree = executor._compile(sample_nested_spec)
        assert isinstance(tree, py_trees.composites.Sequence)
        assert len(tree.children) == 2
        assert isinstance(tree.children[0], py_trees.composites.Selector)
        assert isinstance(tree.children[1], ActionAffordanceNode)

    def test_compile_unknown_type_raises(self, executor):
        spec = {"name": "Bad", "type": "banana"}
        with pytest.raises(ValueError, match="Unknown node type"):
            executor._compile(spec)

    def test_compile_action_missing_url_raises(self, executor):
        spec = {"name": "NoUrl", "type": "action"}
        with pytest.raises(KeyError):
            executor._compile(spec)


class TestExecution:
    """Tests for IRExecutor.execute_from_spec() with mocked HTTP."""

    def test_execute_empty_spec(self, executor):
        result = executor.execute_from_spec({})
        assert result.success is True
        assert result.ticks == 0
        assert "No behavior tree" in result.final_status

    def test_execute_none_spec(self, executor):
        result = executor.execute_from_spec(None)
        assert result.success is True
        assert result.ticks == 0

    @patch("ami_agents.bt_planning.nodes.affordance_nodes.ActionAffordanceNode.setup")
    @patch("ami_agents.bt_planning.nodes.affordance_nodes.ActionAffordanceNode.update")
    def test_execute_single_action_success(self, mock_update, mock_setup, executor, sample_action_spec):
        mock_update.return_value = Status.SUCCESS
        result = executor.execute_from_spec(sample_action_spec)
        assert result.success is True
        assert result.ticks == 1
        assert result.final_status == "SUCCESS"

    @patch("ami_agents.bt_planning.nodes.affordance_nodes.ActionAffordanceNode.setup")
    @patch("ami_agents.bt_planning.nodes.affordance_nodes.ActionAffordanceNode.update")
    def test_execute_single_action_failure(self, mock_update, mock_setup, executor, sample_action_spec):
        mock_update.return_value = Status.FAILURE
        result = executor.execute_from_spec(sample_action_spec)
        assert result.success is False
        assert result.final_status == "FAILURE"

    def test_execute_invalid_spec_returns_error(self, executor):
        spec = {"name": "Bad", "type": "nonexistent"}
        result = executor.execute_from_spec(spec)
        assert result.success is False
        assert result.error is not None

    def test_execution_result_to_dict(self):
        result = ExecutionResult(
            success=True,
            tree_name="TestTree",
            ticks=3,
            final_status="SUCCESS",
            tick_history=["RUNNING", "RUNNING", "SUCCESS"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["tree_name"] == "TestTree"
        assert d["ticks"] == 3
        assert len(d["tick_history"]) == 3

    @patch("ami_agents.bt_planning.nodes.affordance_nodes.HTTPClient")
    def test_execute_wait_condition_polls_until_success(self, mock_client_cls):
        from ami_agents.bt_planning.nodes.http_client import HTTPResponse

        mock_client = MagicMock()
        mock_client.get.side_effect = [
            HTTPResponse(200, 75, {}, "http://localhost/props/glare", 0.0),
            HTTPResponse(200, 45, {}, "http://localhost/props/glare", 0.0),
        ]
        mock_client_cls.return_value = mock_client

        spec = {
            "name": "WaitForGlare",
            "type": "wait_condition",
            "property_url": "http://localhost/props/glare",
            "expected_value": 50,
            "operator": "<=",
            "timeout_seconds": 5,
            "poll_interval_seconds": 0,
        }
        result = IRExecutor(max_ticks=3).execute_from_spec(spec)

        assert result.success is True
        assert result.ticks == 2
        assert result.tick_history == ["RUNNING", "SUCCESS"]
        assert mock_client.get.call_count == 2
