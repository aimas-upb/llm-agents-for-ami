"""The BT planner accepts `ignore_failure`: a step that only helps does not
stop the steps after it when a device rejects it."""

import time

from py_trees.common import Status
from unittest.mock import patch

from ami_agents.bt_planning.nodes.http_client import HTTPClient, HTTPResponse
from ami_agents.bt_planning.nodes.registry import compile_node
from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.schema import TREE_PARAMETER_SCHEMA

URL = "http://h/ws/room/artifacts/ac_1/actions/"


def _action(name):
    return {"type": "action", "name": name, "action_url": URL + name,
            "parameters": {"value": 1}}


TREE = {"type": "sequence", "name": "cool", "children": [
    _action("hvacMode"),
    {"type": "ignore_failure", "name": "setpoint helps", "children": [_action("coolingSetpoint")]},
    _action("fanSpeed")]}


def test_the_tool_schema_offers_it():
    assert "ignore_failure" in TREE_PARAMETER_SCHEMA["properties"]["type"]["enum"]


def test_the_planner_validates_it():
    planner = AsyncBTPlanner()
    assert planner._validate_tree(TREE) == []
    two = {"type": "ignore_failure", "name": "x", "children": [_action("a"), _action("b")]}
    assert planner._validate_tree(two)


def test_a_rejected_helping_step_does_not_stop_the_next():
    def post(self, url, payload=None, headers=None):
        code = 422 if url.endswith("coolingSetpoint") else 200
        return HTTPResponse(code, {}, {}, url, 0.0)

    def compile_child(spec):
        return compile_node(spec, compile_child)

    tree = compile_child(TREE)
    tree.setup_with_descendants()
    with patch.object(HTTPClient, "post", post):
        for _ in range(200):                     # requests run on a thread pool
            tree.tick_once()
            if tree.status != Status.RUNNING:
                break
            time.sleep(0.01)
    assert tree.status == Status.SUCCESS                 # the fan step still ran
