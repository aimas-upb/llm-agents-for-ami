"""The read -> compute -> set tree, run by IRExecutor against a fake device.

Covers the IR additions it needs: `read` (the existing PropertyAffordanceNode),
action `parameter_keys` (the existing blackboard parameters), the
`change_clamped` compute op, and `ignore_failure` around each device.
"""

from unittest.mock import MagicMock, patch

import pytest

from ami_agents.bt_planning.execution.ir_executor import IRExecutor
from ami_agents.bt_planning.nodes.affordance_nodes import PropertyValue
from ami_agents.bt_planning.nodes.compute_node import COMPUTE_OPS
from ami_agents.bt_planning.nodes.http_client import HTTPResponse
from ami_agents.bt_planning.serialization import validate_tree


def _modify(device, amount, mode="add", **args):
    """The tree `modify_goal_plan.modify_tree` builds, for one device."""
    current, new = f"modify/{device}/level/current", f"modify/{device}/level/new"
    return {"type": "sequence", "name": f"{device}: change", "children": [
        {"type": "read", "name": f"{device}: read",
         "property_url": f"http://h/{device}/properties/level", "output": current},
        {"type": "compute", "name": f"{device}: compute", "op": "change_clamped",
         "inputs": [current], "output": new,
         "args": {"amount": amount, "mode": mode, **args}},
        {"type": "action", "name": device, "action_url": f"http://h/{device}/actions/level",
         "parameter_keys": {"value": new}},
    ]}


class TestChangeClamped:
    op = staticmethod(COMPUTE_OPS["change_clamped"])

    def test_add_and_scale(self):
        assert self.op([20], {"amount": -2, "mode": "add"}) == 18
        assert self.op([100], {"amount": -20, "mode": "scale"}) == 80

    def test_a_read_value_is_unwrapped(self):
        assert self.op([PropertyValue(success=True, value=50)],
                       {"amount": 10, "mode": "add"}) == 60

    def test_the_result_is_kept_in_range_and_rounded(self):
        assert self.op([5], {"amount": -10, "mode": "add", "min": 1}) == 1
        assert self.op([250], {"amount": 20, "mode": "scale", "max": 254}) == 254
        assert self.op([101], {"amount": -20, "mode": "scale", "integer": True}) == 81

    def test_a_non_numeric_reading_raises(self):
        with pytest.raises(ValueError):
            self.op(["on"], {"amount": 1, "mode": "add"})


class TestIR:
    def test_the_tree_validates(self):
        tree = {"type": "sequence", "name": "all", "children": [
            {"type": "ignore_failure", "name": "a", "children": [_modify("a", -20, "scale")]}]}
        assert validate_tree(tree) == []

    @pytest.mark.parametrize("node, error", [
        ({"type": "read", "name": "r", "output": "k"}, "property_url"),
        ({"type": "read", "name": "r", "property_url": "http://h/p"}, "output"),
        ({"type": "ignore_failure", "name": "i", "children": []}, "exactly one"),
        ({"type": "action", "name": "a", "action_url": "http://h/a",
          "parameter_keys": {"value": 3}}, "parameter_keys"),
    ])
    def test_malformed_nodes_are_rejected(self, node, error):
        assert any(error in e for e in validate_tree(node))


def _fake_device(readings):
    """An HTTP client: GET returns the reading for the URL (or a 500), POST
    records the body and succeeds."""
    client = MagicMock()
    posted = {}

    def get(url, headers=None):
        if url in readings:
            return HTTPResponse(200, readings[url], {}, url, 0.0)
        return HTTPResponse(500, "error", {}, url, 0.0)

    def post(url, payload=None, headers=None):
        posted[url] = payload
        return HTTPResponse(200, {}, {}, url, 0.0)

    client.get.side_effect = get
    client.post.side_effect = post
    return client, posted


@patch("ami_agents.bt_planning.nodes.affordance_nodes.HTTPClient")
class TestExecution:
    def test_the_new_value_is_read_computed_and_set(self, client_cls):
        client, posted = _fake_device({"http://h/light/properties/level": 100})
        client_cls.return_value = client
        result = IRExecutor(max_ticks=50).execute_from_spec(
            _modify("light", -20, "scale", min=1, max=254, integer=True))
        assert result.success is True
        assert posted == {"http://h/light/actions/level": {"value": 80}}

    def test_one_device_failing_does_not_stop_the_next(self, client_cls):
        # The first device cannot be read; the second is changed all the same.
        client, posted = _fake_device({"http://h/b/properties/level": 10})
        client_cls.return_value = client
        tree = {"type": "sequence", "name": "every matching device", "children": [
            {"type": "ignore_failure", "name": "a", "children": [_modify("a", 5)]},
            {"type": "ignore_failure", "name": "b", "children": [_modify("b", 5)]},
        ]}
        result = IRExecutor(max_ticks=50).execute_from_spec(tree)
        assert result.success is True
        assert posted == {"http://h/b/actions/level": {"value": 15}}
