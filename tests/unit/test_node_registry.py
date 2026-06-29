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
    def _compile_always_succeed(spec, compile_child):
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
