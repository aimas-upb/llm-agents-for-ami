"""
Unit tests for the BT node registry + JSON serialization (issue #22).

Covers:
- the registry knows the built-ins plus the custom ``compute`` type
- custom compute nodes validate, round-trip as JSON, and compile/execute
- unknown node types fail validation and compilation
- third parties can register their own node types
"""

import py_trees
import pytest
from py_trees.common import Status

from ami_agents.bt_planning.execution.ir_executor import IRExecutor
from ami_agents.bt_planning.nodes import registry
from ami_agents.bt_planning.nodes.affordance_nodes import WaitPropertyConditionNode
from ami_agents.bt_planning.nodes.compute_node import BlackboardComputeNode
from ami_agents.bt_planning.serialization import (
    deserialize_tree,
    serialize_tree,
    to_bytes,
    validate_tree,
)


# --------------------------------------------------------------------------- #
# Registry contents
# --------------------------------------------------------------------------- #
def test_registry_contains_builtins_and_compute():
    types = set(registry.registered_types())
    assert {"sequence", "selector", "parallel", "action", "condition"} <= types
    assert "compute" in types


# --------------------------------------------------------------------------- #
# Custom compute node — validation
# --------------------------------------------------------------------------- #
def _compute_spec(**overrides):
    spec = {
        "name": "AggregatePresence",
        "type": "compute",
        "op": "any",
        "inputs": ["sensor/hall", "sensor/kitchen"],
        "output": "presence/anywhere",
    }
    spec.update(overrides)
    return spec


def test_compute_node_validates_ok():
    assert validate_tree(_compute_spec()) == []


def test_compute_node_missing_op():
    spec = _compute_spec()
    del spec["op"]
    errors = validate_tree(spec)
    assert any("op" in e for e in errors)


def test_compute_node_unknown_op():
    errors = validate_tree(_compute_spec(op="definitely_not_an_op"))
    assert any("unknown compute op" in e for e in errors)


def test_compute_node_missing_output():
    spec = _compute_spec()
    del spec["output"]
    errors = validate_tree(spec)
    assert any("output" in e for e in errors)


# --------------------------------------------------------------------------- #
# Unknown node types
# --------------------------------------------------------------------------- #
def test_unknown_type_fails_validation():
    errors = validate_tree({"name": "X", "type": "banana"})
    assert any("type" in e for e in errors)


def test_unknown_type_fails_compile():
    executor = IRExecutor(max_ticks=3)
    with pytest.raises(ValueError, match="Unknown node type"):
        executor._compile({"name": "X", "type": "banana"})


# --------------------------------------------------------------------------- #
# Serialization round-trip (the #22 contract)
# --------------------------------------------------------------------------- #
def test_json_roundtrip_with_compute_node():
    tree = {
        "name": "Root",
        "type": "sequence",
        "children": [
            _compute_spec(),
            {
                "name": "TurnOn",
                "type": "action",
                "action_url": "http://localhost:8080/light/turn_on",
            },
        ],
    }
    assert deserialize_tree(serialize_tree(tree)) == tree


def test_bytes_roundtrip():
    tree = _compute_spec()
    payload = to_bytes(tree)
    assert isinstance(payload, bytes)
    assert deserialize_tree(payload) == tree


def test_deserialize_rejects_non_object():
    with pytest.raises(ValueError):
        deserialize_tree("[1, 2, 3]")


# --------------------------------------------------------------------------- #
# Compute node execution: reads inputs, writes output to the blackboard
# --------------------------------------------------------------------------- #
def test_compute_node_reads_and_writes_blackboard():
    node = BlackboardComputeNode(
        name="SumTwo",
        op="sum",
        inputs=["t/registry/a", "t/registry/b"],
        output="t/registry/out",
    )

    writer = py_trees.blackboard.Client(name="writer")
    writer.register_key(key="t/registry/a", access=py_trees.common.Access.WRITE)
    writer.register_key(key="t/registry/b", access=py_trees.common.Access.WRITE)
    writer.set("t/registry/a", 2)
    writer.set("t/registry/b", 3)

    node.setup()
    node.initialise()
    assert node.update() == Status.SUCCESS

    reader = py_trees.blackboard.Client(name="reader")
    reader.register_key(key="t/registry/out", access=py_trees.common.Access.READ)
    assert reader.get("t/registry/out") == 5


def test_executor_runs_tree_with_compute_node():
    tree = {
        "name": "ComputeOnly",
        "type": "sequence",
        "children": [_compute_spec(op="any", output="t/registry/exec_out")],
    }
    result = IRExecutor(max_ticks=3).execute_from_spec(tree)
    assert result.success is True


# --------------------------------------------------------------------------- #
# Third-party node-type registration
# --------------------------------------------------------------------------- #
def test_register_custom_node_type():
    def _compile_always_succeed(spec, compile_child, ctx):
        return py_trees.behaviours.Success(name=spec.get("name", "ok"))

    def _validate_always_succeed(spec, path, validate_child):
        return []

    registry.register_node_type(
        "always_succeed",
        compile=_compile_always_succeed,
        validate=_validate_always_succeed,
    )
    try:
        spec = {"name": "Y", "type": "always_succeed"}
        assert validate_tree(spec) == []
        result = IRExecutor(max_ticks=2).execute_from_spec(spec)
        assert result.success is True
    finally:
        registry.NODE_REGISTRY.pop("always_succeed", None)


# --------------------------------------------------------------------------- #
# wait_condition registration (EACL2027 addition over the code_cleanup port)
# --------------------------------------------------------------------------- #
def _wait_condition_spec(**overrides):
    spec = {
        "name": "WaitCool",
        "type": "wait_condition",
        "property_url": "http://localhost:8080/room/temperature",
        "expected_value": 21.0,
        "operator": "<=",
    }
    spec.update(overrides)
    return spec


def test_wait_condition_is_registered():
    assert "wait_condition" in registry.registered_types()


def test_wait_condition_compiles_with_defaults():
    node = registry.compile_node(_wait_condition_spec(), lambda c: None)
    assert isinstance(node, WaitPropertyConditionNode)
    assert node.timeout_seconds == 30.0
    assert node.poll_interval_seconds == 1.0


def test_wait_condition_compiles_with_explicit_timing():
    node = registry.compile_node(
        _wait_condition_spec(timeout_seconds=95.0, poll_interval_seconds=2.5),
        lambda c: None,
    )
    assert node.timeout_seconds == 95.0
    assert node.poll_interval_seconds == 2.5


def test_wait_condition_validator_rejects_non_numeric_timeout():
    errors = validate_tree(_wait_condition_spec(timeout_seconds="soon"))
    assert any("timeout_seconds must be numeric" in e for e in errors)


def test_wait_condition_validator_rejects_negative_poll_interval():
    errors = validate_tree(_wait_condition_spec(poll_interval_seconds=-1))
    assert any("poll_interval_seconds" in e for e in errors)


def test_wait_condition_validator_requires_property_url():
    spec = _wait_condition_spec()
    del spec["property_url"]
    errors = validate_tree(spec)
    assert any("property_url" in e for e in errors)


# --------------------------------------------------------------------------- #
# Compile context: settling-time resolution for action nodes
# --------------------------------------------------------------------------- #
def _action_spec(**overrides):
    spec = {
        "name": "CoolDown",
        "type": "action",
        "action_url": "/workspaces/lab/artifacts/ac_unit/ha/climate/set_temperature",
    }
    spec.update(overrides)
    return spec


class _StubCtx:
    def __init__(self, settling):
        self._settling = settling
        self.calls = []

    def _settling_time_for_action(self, action_url, explicit_seconds=None):
        self.calls.append((action_url, explicit_seconds))
        return self._settling


def test_compile_action_uses_ctx_settling_resolution():
    ctx = _StubCtx(settling=42.0)
    node = registry.compile_node(_action_spec(), lambda c: None, ctx)
    assert node.settling_time_seconds == 42.0
    assert ctx.calls == [(_action_spec()["action_url"], None)]


def test_compile_action_passes_spec_annotation_to_ctx():
    ctx = _StubCtx(settling=90.0)
    registry.compile_node(_action_spec(settling_time_seconds=90.0), lambda c: None, ctx)
    assert ctx.calls[0][1] == 90.0


def test_compile_action_without_ctx_uses_spec_annotation():
    node = registry.compile_node(
        _action_spec(settling_time_seconds=12.0), lambda c: None, None
    )
    assert node.settling_time_seconds == 12.0


def test_compile_action_without_ctx_defaults_to_zero():
    node = registry.compile_node(_action_spec(), lambda c: None)
    assert node.settling_time_seconds == 0.0


def test_executor_compile_resolves_settling_from_env(monkeypatch):
    settling_key = "entity:climate.ac_unit:action:climate.set_temperature"
    monkeypatch.setenv("TD_SOSA_SETTLING_TIMES", f'{{"{settling_key}": 60.0}}')
    executor = IRExecutor(max_ticks=1)
    node = executor._compile(_action_spec())
    assert node.settling_time_seconds == 60.0
