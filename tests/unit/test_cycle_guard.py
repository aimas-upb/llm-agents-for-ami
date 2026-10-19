"""Cycle checks: a start, pause, resume or stop is sent only from a state that allows it.

On washer_1 of test_planning_conditions.py (an operationalState action acting
upon its operationalState property, plus other actions):

    the tree       which actions are checked, the check per value sent, valid
                   IR, a check already there kept, the input not modified
    at run time    the checked tree ticked against each washer state: the
                   command sent, skipped as already done, or the step failed
                   without sending (HTTP faked, node I/O inline)
    where it runs  before the power-on guard, in the simple-plan behaviour
"""

from unittest.mock import AsyncMock, MagicMock

import py_trees
import pytest

from ami_agents.agents.interaction_solver.behaviours.power_on_guard import PowerOnGuardBehaviour
from ami_agents.agents.interaction_solver.utils.cycle_guard import (
    CHECK_NAME,
    PAUSED,
    RUNNING,
    STOPPED,
    guard_cycles,
    is_cycle_check,
)
from ami_agents.bt_planning.nodes import async_support
from ami_agents.bt_planning.nodes.http_client import HTTPClient, HTTPResponse
from ami_agents.bt_planning.nodes.registry import compile_node
from ami_agents.bt_planning.serialization import validate_tree
from tests.simuhome.bench.scheduling import InlineExecutor
from tests.unit.test_planning_conditions import act, graph  # noqa: F401 - fixture

STATE_URL = "http://h/washer_1/p/operationalState"


def _compile(spec):
    return compile_node(spec, _compile)


class TestTheTree:
    @pytest.mark.parametrize("sends, operator", [(PAUSED, "=="), (RUNNING, "!=")])
    def test_pause_and_start_check_the_state_they_are_sent_from(self, graph, sends, operator):
        tree, notes = guard_cycles(graph, act("washer_1", "operationalState", sends))
        assert notes == [] and is_cycle_check(tree) and validate_tree(tree) == []
        done, send = tree["children"]
        assert (done["operator"], done["expected_value"]) == ("==", sends)
        allowed, action = send["children"]
        assert allowed["property_url"] == STATE_URL and allowed["operator"] == operator
        assert action == act("washer_1", "operationalState", sends)

    def test_a_stop_is_skipped_when_already_stopped_else_sent(self, graph):
        tree, _ = guard_cycles(graph, act("washer_1", "operationalState", STOPPED))
        done, action = tree["children"]
        assert done["expected_value"] == STOPPED and action["type"] == "action"

    def test_a_named_state_is_read_as_matter_names_it(self, graph):
        tree, _ = guard_cycles(graph, act("washer_1", "operationalState", "Paused"))
        assert tree["name"].endswith(f"Paused: {CHECK_NAME}")

    def test_other_actions_are_left_alone_and_a_check_is_not_doubled(self, graph):
        spin = act("washer_1", "spinSpeedCurrent", 1200)
        assert guard_cycles(graph, spin) == (spin, [])
        once, _ = guard_cycles(graph, act("washer_1", "operationalState", PAUSED))
        assert guard_cycles(graph, once)[0] == once

    def test_nested_commands_are_found_and_the_input_kept(self, graph):
        original = {"type": "sequence", "name": "s", "children": [
            act("washer_1", "onOff", True), act("washer_1", "operationalState", RUNNING)]}
        tree, _ = guard_cycles(graph, original, goal="G1")
        assert tree["children"][0] == original["children"][0]
        assert tree["children"][1]["name"].startswith("G1 washer_1.operationalState")
        assert original["children"][1]["type"] == "action"


class TestAtRunTime:
    """The checked tree ticked with the washer in a given state."""

    def _run(self, graph, monkeypatch, sends, state):
        """(the tree's status, the actions it posted)."""
        posts = []
        monkeypatch.setattr(HTTPClient, "get",
                            lambda client, url, headers=None: HTTPResponse(200, state, {}, url, 0))
        monkeypatch.setattr(HTTPClient, "post", lambda client, url, payload=None, headers=None:
                            posts.append(payload) or HTTPResponse(200, {"ok": True}, {}, url, 0))
        monkeypatch.setattr(async_support, "_executor", InlineExecutor())
        py_trees.blackboard.Blackboard.clear()
        tree, _ = guard_cycles(graph, act("washer_1", "operationalState", sends))
        root = _compile(tree)
        root.setup_with_descendants()
        for _ in range(5):
            root.tick_once()
            if root.status != py_trees.common.Status.RUNNING:
                break
        return root.status.name, posts

    @pytest.mark.parametrize("sends, state, status, sent", [
        (PAUSED, RUNNING, "SUCCESS", True),        # pause a running cycle
        (PAUSED, PAUSED, "SUCCESS", False),        # already paused: nothing to send
        (PAUSED, STOPPED, "FAILURE", False),       # no cycle to pause: refused here
        (RUNNING, PAUSED, "SUCCESS", True),        # resume
        (RUNNING, STOPPED, "SUCCESS", True),       # start
        (RUNNING, RUNNING, "SUCCESS", False),      # already running
        (RUNNING, 3, "FAILURE", False),            # error: neither start nor resume
        (STOPPED, RUNNING, "SUCCESS", True),       # stop
        (STOPPED, STOPPED, "SUCCESS", False),      # already stopped
    ])
    def test_the_command_is_sent_skipped_or_refused(self, graph, monkeypatch,
                                                    sends, state, status, sent):
        result, posts = self._run(graph, monkeypatch, sends, state)
        assert result == status and bool(posts) == sent


@pytest.mark.asyncio
async def test_simple_plans_get_the_check_before_the_power_on_guard(graph):
    """No power-on file: the cycle check still applies."""
    entry = {"tree": act("washer_1", "operationalState", PAUSED)}
    behaviour = PowerOnGuardBehaviour([entry], logger=MagicMock(), graph=graph)
    behaviour.agent = MagicMock()
    behaviour.agent.config = {"planning": {"guidelines": {"path": "/nonexistent.yaml"}}}
    await behaviour.run()
    assert is_cycle_check(entry["tree"])
