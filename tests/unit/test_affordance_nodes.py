"""
Unit tests for custom py_trees affordance nodes with mocked HTTP.
"""

import pytest
from unittest.mock import patch, MagicMock

from py_trees.common import Status

from ami_agents.bt_planning.nodes.affordance_nodes import (
    ActionAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    WaitPropertyConditionNode,
    ComparisonOperator,
    ActionResult,
    PropertyValue,
)
from ami_agents.bt_planning.nodes.http_client import HTTPResponse, HTTPError


def _make_http_response(status_code=200, body=None):
    """Helper to create mock HTTPResponse."""
    return HTTPResponse(
        status_code=status_code,
        body=body or {"status": "success"},
        headers={"content-type": "application/json"},
        url="http://localhost:8080/test",
        elapsed_time=0.05,
    )


class TestActionAffordanceNode:
    """Tests for ActionAffordanceNode."""

    def _make_node(self, **kwargs):
        defaults = {
            "name": "TestAction",
            "action_url": "http://localhost:8080/artifacts/light/turn_on",
        }
        defaults.update(kwargs)
        return ActionAffordanceNode(**defaults)

    @patch.object(ActionAffordanceNode, "setup")
    def test_action_node_success(self, mock_setup):
        node = self._make_node()
        node._http_client = MagicMock()
        node._http_client.post.return_value = _make_http_response(200)
        node.initialise()

        status = node.update()
        assert status == Status.SUCCESS
        assert node.last_result is not None
        assert node.last_result.success is True

    @patch.object(ActionAffordanceNode, "setup")
    def test_action_node_failure_status(self, mock_setup):
        node = self._make_node()
        node._http_client = MagicMock()
        node._http_client.post.return_value = _make_http_response(500, {"error": "fail"})
        node.initialise()

        status = node.update()
        assert status == Status.FAILURE

    @patch.object(ActionAffordanceNode, "setup")
    def test_action_node_http_error(self, mock_setup):
        node = self._make_node()
        node._http_client = MagicMock()
        node._http_client.post.side_effect = HTTPError(
            url="http://test", status_code=500, message="Server error"
        )
        node.initialise()

        status = node.update()
        assert status == Status.FAILURE
        assert node.last_result.error_message == "Server error"

    @patch.object(ActionAffordanceNode, "setup")
    def test_action_node_with_parameters(self, mock_setup):
        node = self._make_node(parameters={"brightness": 75})
        node._http_client = MagicMock()
        node._http_client.post.return_value = _make_http_response(200)
        node.initialise()

        node.update()
        call_args = node._http_client.post.call_args
        assert call_args[1]["payload"]["brightness"] == 75


class TestPropertyConditionNode:
    """Tests for PropertyConditionNode."""

    def _make_node(self, **kwargs):
        defaults = {
            "name": "TestCondition",
            "property_url": "http://localhost:8080/artifacts/light/properties/state",
            "expected_value": "on",
        }
        defaults.update(kwargs)
        return PropertyConditionNode(**defaults)

    @patch.object(PropertyConditionNode, "setup")
    def test_condition_matches(self, mock_setup):
        node = self._make_node(expected_value="on")
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, "on")
        node.initialise()

        status = node.update()
        assert status == Status.SUCCESS

    @patch.object(PropertyConditionNode, "setup")
    def test_condition_mismatch(self, mock_setup):
        node = self._make_node(expected_value="on")
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, "off")
        node.initialise()

        status = node.update()
        assert status == Status.FAILURE

    @patch.object(PropertyConditionNode, "setup")
    def test_condition_http_error(self, mock_setup):
        node = self._make_node()
        node._http_client = MagicMock()
        node._http_client.get.side_effect = HTTPError(
            url="http://test", status_code=500, message="Error"
        )
        node.initialise()

        status = node.update()
        assert status == Status.FAILURE

    @patch.object(PropertyConditionNode, "setup")
    def test_condition_with_value_path(self, mock_setup):
        node = self._make_node(
            expected_value="empty",
            value_path=["hand"],
        )
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(
            200, {"hand": "empty", "blocks": ["a", "b"]}
        )
        node.initialise()

        status = node.update()
        assert status == Status.SUCCESS


class TestComparisonPropertyConditionNode:
    """Tests for ComparisonPropertyConditionNode."""

    def _make_node(self, **kwargs):
        defaults = {
            "name": "TestComparison",
            "property_url": "http://localhost:8080/props/temp",
            "expected_value": 25,
            "operator": ComparisonOperator.GREATER_THAN,
        }
        defaults.update(kwargs)
        return ComparisonPropertyConditionNode(**defaults)

    @patch.object(ComparisonPropertyConditionNode, "setup")
    def test_greater_than_true(self, mock_setup):
        node = self._make_node(expected_value=25, operator=ComparisonOperator.GREATER_THAN)
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, 30)
        node.initialise()

        assert node.update() == Status.SUCCESS

    @patch.object(ComparisonPropertyConditionNode, "setup")
    def test_greater_than_false(self, mock_setup):
        node = self._make_node(expected_value=25, operator=ComparisonOperator.GREATER_THAN)
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, 20)
        node.initialise()

        assert node.update() == Status.FAILURE

    @patch.object(ComparisonPropertyConditionNode, "setup")
    def test_less_than(self, mock_setup):
        node = self._make_node(expected_value=25, operator=ComparisonOperator.LESS_THAN)
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, 20)
        node.initialise()

        assert node.update() == Status.SUCCESS

    @patch.object(ComparisonPropertyConditionNode, "setup")
    def test_not_equal(self, mock_setup):
        node = self._make_node(expected_value="off", operator=ComparisonOperator.NOT_EQUAL)
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, "on")
        node.initialise()

        assert node.update() == Status.SUCCESS

    @patch.object(ComparisonPropertyConditionNode, "setup")
    def test_in_operator(self, mock_setup):
        node = self._make_node(
            expected_value=["cool", "auto"],
            operator=ComparisonOperator.IN,
        )
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, "cool")
        node.initialise()

        assert node.update() == Status.SUCCESS

    @patch.object(ComparisonPropertyConditionNode, "setup")
    def test_greater_equal(self, mock_setup):
        node = self._make_node(expected_value=25, operator=ComparisonOperator.GREATER_THAN_OR_EQUAL)
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, 25)
        node.initialise()

        assert node.update() == Status.SUCCESS


class TestWaitPropertyConditionNode:
    """Tests for WaitPropertyConditionNode."""

    @patch.object(WaitPropertyConditionNode, "setup")
    def test_wait_condition_runs_until_match(self, mock_setup):
        node = WaitPropertyConditionNode(
            name="WaitForTemp",
            property_url="http://localhost:8080/props/temp",
            expected_value=24,
            operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
            timeout_seconds=10,
            poll_interval_seconds=0,
        )
        node._http_client = MagicMock()
        node._http_client.get.side_effect = [
            _make_http_response(200, 26),
            _make_http_response(200, 24),
        ]
        node.initialise()

        assert node.update() == Status.RUNNING
        assert node.update() == Status.SUCCESS

    @patch.object(WaitPropertyConditionNode, "setup")
    def test_wait_condition_times_out(self, mock_setup):
        node = WaitPropertyConditionNode(
            name="WaitForTemp",
            property_url="http://localhost:8080/props/temp",
            expected_value=24,
            operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
            timeout_seconds=0,
            poll_interval_seconds=0,
        )
        node._http_client = MagicMock()
        node._http_client.get.return_value = _make_http_response(200, 26)
        node.initialise()

        assert node.update() == Status.FAILURE
