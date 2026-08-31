"""Node I/O must not block the tick.

The point of `FuturePollingMixin`: a leaf waiting on HTTP reports RUNNING and
is asked again, instead of holding the agent for the length of a round trip.
"""

import time

import py_trees
import pytest
from py_trees.common import Status
from unittest.mock import MagicMock, patch

from ami_agents.bt_planning.nodes.affordance_nodes import (
    ActionAffordanceNode,
    ComparisonOperator,
    ComparisonPropertyConditionNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
    SettlingTimeWaitNode,
    WaitPropertyConditionNode,
)
from ami_agents.bt_planning.nodes.async_support import (
    FuturePollingMixin,
    get_executor,
)
from ami_agents.bt_planning.nodes.http_client import HTTPError, HTTPResponse


SLOW = 0.15  # long enough that a blocking implementation would be obvious


def slow_response(delay=SLOW, status_code=200, body=None):
    def call(*args, **kwargs):
        time.sleep(delay)
        return HTTPResponse(status_code=status_code,
                            body=body if body is not None else {"ok": True},
                            headers={}, url="http://localhost/x", elapsed_time=delay)
    return call


def drain(node, limit=500):
    """Tick until the node reaches a decision; count the RUNNING ticks."""
    running = 0
    for _ in range(limit):
        status = node.update()
        if status is not Status.RUNNING:
            return status, running
        running += 1
        if not node._io_in_flight and node._future is None:
            return status, running
        time.sleep(0.005)
    raise AssertionError("node never settled")


class TestMixin:
    def test_start_is_idempotent_per_attempt(self):
        calls = []

        class Node(FuturePollingMixin):
            pass

        node = Node()
        for _ in range(5):
            node._start(lambda: calls.append(1))
        node._future.result(timeout=5)
        assert len(calls) == 1

    def test_poll_does_not_block_and_reports_completion(self):
        class Node(FuturePollingMixin):
            pass

        node = Node()
        node._start(lambda: time.sleep(SLOW) or "done")

        started = time.monotonic()
        done, result, exc = node._poll()
        assert time.monotonic() - started < SLOW / 2  # it returned immediately
        assert (done, result, exc) == (False, None, None)

        node._future.result(timeout=5)
        done, result, exc = node._poll()
        assert done is True and result == "done" and exc is None

    def test_exception_is_returned_not_raised(self):
        class Node(FuturePollingMixin):
            pass

        node = Node()

        def boom():
            raise ValueError("nope")

        node._start(boom)
        node._future.exception(timeout=5)
        done, result, exc = node._poll()
        assert done is True
        assert result is None
        assert isinstance(exc, ValueError)

    def test_reset_allows_a_fresh_attempt(self):
        calls = []

        class Node(FuturePollingMixin):
            pass

        node = Node()
        node._start(lambda: calls.append(1))
        node._future.result(timeout=5)
        node._reset()
        node._start(lambda: calls.append(1))
        node._future.result(timeout=5)
        assert len(calls) == 2

    def test_executor_is_a_reusable_singleton(self):
        # Deliberately does not call shutdown_executor(): the pool is a module
        # singleton shared with every other test in the run, and tearing it
        # down here would leave them submitting to a dead executor.
        assert get_executor() is get_executor()


class TestNodesYieldWhileWaiting:
    """Every I/O node must report RUNNING at least once, then decide."""

    @patch.object(ActionAffordanceNode, "setup")
    def test_action_node(self, _setup):
        node = ActionAffordanceNode(name="A", action_url="http://localhost/a")
        node._http_client = MagicMock()
        node._http_client.post.side_effect = slow_response()
        node.initialise()

        status, running = drain(node)
        assert status is Status.SUCCESS
        assert running >= 1, "the tick blocked instead of yielding"

    @patch.object(PropertyAffordanceNode, "setup")
    def test_property_node(self, _setup):
        node = PropertyAffordanceNode(name="P", property_url="http://localhost/p")
        node._http_client = MagicMock()
        node._http_client.get.side_effect = slow_response(body=42)
        node.initialise()

        status, running = drain(node)
        assert status is Status.SUCCESS
        assert running >= 1

    @patch.object(PropertyConditionNode, "setup")
    def test_condition_node(self, _setup):
        node = PropertyConditionNode(name="C", property_url="http://localhost/p",
                                     expected_value=42)
        node._http_client = MagicMock()
        node._http_client.get.side_effect = slow_response(body=42)
        node.initialise()

        status, running = drain(node)
        assert status is Status.SUCCESS
        assert running >= 1

    @patch.object(ComparisonPropertyConditionNode, "setup")
    def test_comparison_condition_node(self, _setup):
        node = ComparisonPropertyConditionNode(
            name="CC", property_url="http://localhost/p", expected_value=50,
            operator=ComparisonOperator.LESS_THAN_OR_EQUAL)
        node._http_client = MagicMock()
        node._http_client.get.side_effect = slow_response(body=45)
        node.initialise()

        status, running = drain(node)
        assert status is Status.SUCCESS
        assert running >= 1

    @patch.object(WaitPropertyConditionNode, "setup")
    def test_wait_condition_node(self, _setup):
        node = WaitPropertyConditionNode(
            name="W", property_url="http://localhost/p", expected_value=50,
            operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
            timeout_seconds=10, poll_interval_seconds=0)
        node._http_client = MagicMock()
        node._http_client.get.side_effect = slow_response(body=45)
        node.initialise()

        status, running = drain(node)
        assert status is Status.SUCCESS
        assert running >= 1

    @patch.object(ActionAffordanceNode, "setup")
    def test_failure_still_reached_through_the_poll(self, _setup):
        node = ActionAffordanceNode(name="A", action_url="http://localhost/a")
        node._http_client = MagicMock()
        node._http_client.post.side_effect = slow_response(status_code=500)
        node.initialise()

        status, running = drain(node)
        assert status is Status.FAILURE
        assert running >= 1

    @patch.object(ActionAffordanceNode, "setup")
    def test_http_error_surfaces_as_failure(self, _setup):
        node = ActionAffordanceNode(name="A", action_url="http://localhost/a")
        node._http_client = MagicMock()

        def boom(*args, **kwargs):
            time.sleep(SLOW)
            raise HTTPError("boom", status_code=503)

        node._http_client.post.side_effect = boom
        node.initialise()

        status, running = drain(node)
        assert status is Status.FAILURE
        assert running >= 1
        assert node.last_result is not None
        assert node.last_result.success is False


class TestSettlingTimeWaitNode:
    def test_runs_until_the_span_elapses(self):
        node = SettlingTimeWaitNode(name="settle", settling_seconds=0.12)
        node.initialise()

        assert node.update() is Status.RUNNING
        started = time.monotonic()
        while node.update() is Status.RUNNING:
            if time.monotonic() - started > 5:
                pytest.fail("settling node never completed")
            time.sleep(0.005)
        assert time.monotonic() - started >= 0.05

    def test_zero_settles_immediately(self):
        node = SettlingTimeWaitNode(name="settle", settling_seconds=0)
        node.initialise()
        assert node.update() is Status.SUCCESS

    def test_does_not_block_the_tick(self):
        node = SettlingTimeWaitNode(name="settle", settling_seconds=2.0)
        node.initialise()
        started = time.monotonic()
        node.update()
        assert time.monotonic() - started < 0.05

    def test_reinitialise_restarts_the_span(self):
        node = SettlingTimeWaitNode(name="settle", settling_seconds=0.05)
        node.initialise()
        while node.update() is Status.RUNNING:
            time.sleep(0.005)
        node.initialise()
        assert node.update() is Status.RUNNING


class TestCompiledPairing:
    """`_compile_action` emits the settling wait, so the ISA needs no change."""

    def _compile(self, spec):
        from ami_agents.bt_planning.nodes.registry import compile_node
        return compile_node(spec, lambda child: self._compile(child))

    def test_plain_action_compiles_to_a_bare_node(self):
        tree = self._compile({"name": "A", "type": "action",
                              "action_url": "http://localhost/a"})
        assert isinstance(tree, ActionAffordanceNode)

    def test_annotated_action_gains_a_settling_sibling(self):
        tree = self._compile({"name": "A", "type": "action",
                              "action_url": "http://localhost/a",
                              "settling_time_seconds": 3})
        assert isinstance(tree, py_trees.composites.Sequence)
        assert isinstance(tree.children[0], ActionAffordanceNode)
        assert isinstance(tree.children[1], SettlingTimeWaitNode)
        assert tree.children[1].settling_seconds == 3.0

    def test_zero_annotation_is_not_paired(self):
        tree = self._compile({"name": "A", "type": "action",
                              "action_url": "http://localhost/a",
                              "settling_time_seconds": 0})
        assert isinstance(tree, ActionAffordanceNode)

    def test_settle_is_a_registered_node_type(self):
        tree = self._compile({"name": "S", "type": "settle",
                              "settling_seconds": 1.5})
        assert isinstance(tree, SettlingTimeWaitNode)
        assert tree.settling_seconds == 1.5

    def test_settle_validation_rejects_a_missing_span(self):
        from ami_agents.bt_planning.nodes.registry import validate_node
        errors = validate_node({"name": "S", "type": "settle"}, "root")
        assert errors and "settling_seconds" in errors[0]
