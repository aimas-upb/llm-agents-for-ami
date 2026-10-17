"""
Node-type registry — the extensible compile/validate dispatch for BT JSON IR.

This decouples ``IRExecutor`` from a hard-coded ``if node_type == ...`` ladder.
Every node type (the five built-ins plus any custom compute node) registers:

- a ``compile`` factory: ``(spec, compile_child) -> py_trees.behaviour.Behaviour``
- a ``validate`` function: ``(spec, path, validate_child) -> list[str]``

Because a node is identified in the IR purely by its ``"type"`` string and a
JSON-serialisable config, custom nodes round-trip as plain JSON. Reconstruction
("load") is done here by name; persistence ("save") is just ``json.dumps`` of
the IR. This is what lets custom compute nodes cross the ISA -> UA wire without
``pickle`` — see issue #22 and ``compute_node.py``.

To add a new node type, call :func:`register_node_type` (typically at import
time of the module that defines the node class).
"""

from typing import Any, Callable, Dict, List, Optional

import py_trees

from .affordance_nodes import (
    ActionAffordanceNode,
    ComparisonOperator,
    ComparisonPropertyConditionNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
    SettlingTimeWaitNode,
    WaitPropertyConditionNode,
)
from .compute_node import BlackboardComputeNode, COMPUTE_OPS
from .dependency_nodes import (
    COMBINERS,
    OPERATORS,
    ClockTimer,
    CompareBlackboardKeys,
    StampTime,
    WaitUntilTime,
)

# A compiler turns a single (child) spec dict into a runnable py_trees node.
ChildCompiler = Callable[[dict], py_trees.behaviour.Behaviour]
# A child validator returns errors for a (child) spec at a given path.
ChildValidator = Callable[[dict, str], List[str]]

CompileFn = Callable[[dict, ChildCompiler], py_trees.behaviour.Behaviour]
ValidateFn = Callable[[dict, str, ChildValidator], List[str]]


OPERATOR_MAP = {
    "==": ComparisonOperator.EQUAL,
    "!=": ComparisonOperator.NOT_EQUAL,
    ">": ComparisonOperator.GREATER_THAN,
    ">=": ComparisonOperator.GREATER_THAN_OR_EQUAL,
    "<": ComparisonOperator.LESS_THAN,
    "<=": ComparisonOperator.LESS_THAN_OR_EQUAL,
    "in": ComparisonOperator.IN,
    "not_in": ComparisonOperator.NOT_IN,
    "contains": ComparisonOperator.CONTAINS,
}


class NodeType:
    """A registered node type: how to compile it and how to validate it."""

    __slots__ = ("name", "compile", "validate")

    def __init__(self, name: str, compile: CompileFn, validate: ValidateFn):
        self.name = name
        self.compile = compile
        self.validate = validate


NODE_REGISTRY: Dict[str, NodeType] = {}


def register_node_type(name: str, *, compile: CompileFn, validate: ValidateFn) -> None:
    """Register (or override) a node type by its IR ``"type"`` name."""
    NODE_REGISTRY[name] = NodeType(name, compile, validate)


def registered_types() -> List[str]:
    return sorted(NODE_REGISTRY)


def compile_node(spec: dict, compile_child: ChildCompiler) -> py_trees.behaviour.Behaviour:
    """Compile one IR node to a py_trees behaviour, dispatching via the registry."""
    node_type = spec.get("type")
    entry = NODE_REGISTRY.get(node_type)
    if entry is None:
        raise ValueError(
            f"Unknown node type: {node_type!r}. Registered: {registered_types()}"
        )
    return entry.compile(spec, compile_child)


def validate_node(spec: Any, path: str = "tree") -> List[str]:
    """Validate one IR node (recursively) via the registry. Returns error strings."""
    if not spec:
        return [f"{path}: tree is empty"]
    if not isinstance(spec, dict):
        return [f"{path}: expected object, got {type(spec).__name__}"]

    entry = NODE_REGISTRY.get(spec.get("type"))
    if entry is None:
        return [f"{path}: missing or invalid 'type'"]

    return entry.validate(spec, path, validate_node)


# ---------------------------------------------------------------------------
# Built-in node types
# ---------------------------------------------------------------------------
def _name(spec: dict) -> str:
    return spec.get("name", "unnamed")


def _compile_sequence(spec, compile_child):
    # `memory` (default True): resume at the running child; False re-checks the
    # earlier children every tick.
    children = [compile_child(c) for c in spec.get("children", [])]
    return py_trees.composites.Sequence(name=_name(spec), memory=bool(spec.get("memory", True)),
                                        children=children)


def _compile_selector(spec, compile_child):
    # `memory` (default False): re-try the earlier children every tick.
    children = [compile_child(c) for c in spec.get("children", [])]
    return py_trees.composites.Selector(name=_name(spec), memory=bool(spec.get("memory", False)),
                                        children=children)


def _compile_parallel(spec, compile_child):
    children = [compile_child(c) for c in spec.get("children", [])]
    if spec.get("policy") == "success_on_one":
        policy = py_trees.common.ParallelPolicy.SuccessOnOne()
    else:
        # `synchronise`: a child that has succeeded is not ticked again while
        # the others finish.
        policy = py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=bool(spec.get("synchronise", False)))
    return py_trees.composites.Parallel(name=_name(spec), policy=policy, children=children)


def _compile_action(spec, compile_child):
    """An action, plus a settling wait when the planner asked for one.

    The planner already annotates actions with `settling_time_seconds` where an
    effect takes time to appear. That used to be a `time.sleep` inside the
    action node, blocking the tick; here it becomes a sibling that returns
    RUNNING until the span has passed. The ISA emits the same IR either way --
    the pairing is a compile-time detail.
    """
    action = ActionAffordanceNode(
        name=_name(spec),
        action_url=spec["action_url"],
        parameters=spec.get("parameters", {}),
        # A parameter whose value is computed while the tree runs (a read, then
        # a compute node) names the blackboard key holding it.
        parameter_keys=spec.get("parameter_keys", {}),
    )
    settling = spec.get("settling_time_seconds")
    if not isinstance(settling, (int, float)) or settling <= 0:
        return action

    sequence = py_trees.composites.Sequence(
        name=f"{_name(spec)}+settle", memory=True)
    sequence.add_children([
        action,
        SettlingTimeWaitNode(name=f"{_name(spec)}/settle",
                             settling_seconds=float(settling)),
    ])
    return sequence


def _compile_settle(spec, compile_child):
    return SettlingTimeWaitNode(
        name=_name(spec),
        settling_seconds=spec.get("settling_seconds",
                                  spec.get("settling_time_seconds", 0.0)),
    )


def _compile_condition(spec, compile_child):
    operator = spec.get("operator")
    if operator and operator != "==":
        return ComparisonPropertyConditionNode(
            name=_name(spec),
            property_url=spec["property_url"],
            expected_value=spec["expected_value"],
            operator=OPERATOR_MAP.get(operator, ComparisonOperator.EQUAL),
            value_path=spec.get("value_path"),
        )
    return PropertyConditionNode(
        name=_name(spec),
        property_url=spec["property_url"],
        expected_value=spec["expected_value"],
        value_path=spec.get("value_path"),
    )


def _compile_wait_condition(spec, compile_child):
    """A condition that keeps looking rather than failing on the first read.

    Same shape as `condition`, plus the two knobs that make it a wait: how long
    to keep trying and how often. Both are optional -- the node's own defaults
    (30s / 1s) apply when the IR omits them.
    """
    node = WaitPropertyConditionNode(
        name=_name(spec),
        property_url=spec["property_url"],
        expected_value=spec["expected_value"],
        operator=OPERATOR_MAP.get(spec.get("operator"), ComparisonOperator.EQUAL),
        value_path=spec.get("value_path"),
    )
    if spec.get("timeout_seconds") is not None:
        node.timeout_seconds = max(0.0, float(spec["timeout_seconds"]))
    if spec.get("poll_interval_seconds") is not None:
        node.poll_interval_seconds = max(0.0, float(spec["poll_interval_seconds"]))
    return node


def _compile_read(spec, compile_child):
    """Read a property and leave its value on the blackboard, under `output`.

    The existing `PropertyAffordanceNode`: it stores the reading as a
    `PropertyValue`, which a compute node unwraps.
    """
    return PropertyAffordanceNode(
        name=_name(spec),
        property_url=spec["property_url"],
        result_key=spec["output"],
    )


def _compile_ignore_failure(spec, compile_child):
    """Run the one child; report SUCCESS even when it fails.

    py_trees' own `FailureIsSuccess` decorator. Used so that one device failing
    in a plan over several devices does not stop the others.
    """
    (child,) = spec["children"]
    return py_trees.decorators.FailureIsSuccess(
        name=_name(spec), child=compile_child(child))


def _compile_compute(spec, compile_child):
    return BlackboardComputeNode(
        name=_name(spec),
        op=spec["op"],
        inputs=spec.get("inputs", []),
        output=spec.get("output"),
        args=spec.get("args", {}),
    )


def _validate_composite(spec, path, validate_child):
    errors: List[str] = []
    children = spec.get("children")
    if not isinstance(children, list) or not children:
        errors.append(f"{path}: composite nodes require non-empty 'children'")
    else:
        for idx, child in enumerate(children):
            errors.extend(validate_child(child, f"{path}.children[{idx}]"))
    return errors


def _validate_action(spec, path, validate_child):
    url = spec.get("action_url")
    if not url or not isinstance(url, str):
        return [f"{path}: action nodes require 'action_url'"]
    keys = spec.get("parameter_keys", {})
    if not isinstance(keys, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in keys.items()):
        return [f"{path}: action 'parameter_keys' must map parameter names "
                f"to blackboard keys"]
    return []


def _validate_read(spec, path, validate_child):
    errors: List[str] = []
    url = spec.get("property_url")
    if not url or not isinstance(url, str):
        errors.append(f"{path}: read nodes require 'property_url'")
    if not spec.get("output") or not isinstance(spec.get("output"), str):
        errors.append(f"{path}: read nodes require an 'output' blackboard key")
    return errors


def _validate_ignore_failure(spec, path, validate_child):
    # `children` with exactly one entry, like a composite: every tree walker
    # (auto-confirmation, node counts, action URL collection) recurses on
    # `children`, so the actions inside stay visible to them.
    children = spec.get("children")
    if not isinstance(children, list) or len(children) != 1:
        return [f"{path}: ignore_failure nodes require exactly one entry in 'children'"]
    return validate_child(children[0], f"{path}.children[0]")


def _validate_condition(spec, path, validate_child):
    errors: List[str] = []
    url = spec.get("property_url")
    if not url or not isinstance(url, str):
        errors.append(f"{path}: condition nodes require 'property_url'")
    if "expected_value" not in spec:
        errors.append(f"{path}: condition nodes require 'expected_value'")
    return errors


def _validate_wait_condition(spec, path, validate_child):
    """Mirrors bt_planner's own wait_condition validation (bt_planner.py:743-757)."""
    errors = _validate_condition(spec, path, validate_child)
    timeout_seconds = spec.get("timeout_seconds")
    if timeout_seconds is not None and not isinstance(timeout_seconds, (int, float)):
        errors.append(f"{path}: wait_condition timeout_seconds must be numeric")
    poll_interval = spec.get("poll_interval_seconds")
    if poll_interval is not None and (
        not isinstance(poll_interval, (int, float)) or poll_interval < 0
    ):
        errors.append(f"{path}: wait_condition poll_interval_seconds must be >= 0")
    return errors


def _validate_settle(spec, path, validate_child):
    seconds = spec.get("settling_seconds", spec.get("settling_time_seconds"))
    if seconds is None:
        return [f"{path}: settle nodes require 'settling_seconds'"]
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return [f"{path}: settle 'settling_seconds' must be a number >= 0"]
    return []


def _validate_compute(spec, path, validate_child):
    errors: List[str] = []
    op = spec.get("op")
    if not op or not isinstance(op, str):
        errors.append(f"{path}: compute nodes require 'op'")
    elif op not in COMPUTE_OPS:
        errors.append(f"{path}: unknown compute op '{op}' (registered: {sorted(COMPUTE_OPS)})")
    if not spec.get("output") or not isinstance(spec.get("output"), str):
        errors.append(f"{path}: compute nodes require an 'output' blackboard key")
    inputs = spec.get("inputs", [])
    if not isinstance(inputs, list):
        errors.append(f"{path}: compute 'inputs' must be a list of blackboard keys")
    return errors


register_node_type("sequence", compile=_compile_sequence, validate=_validate_composite)
register_node_type("selector", compile=_compile_selector, validate=_validate_composite)
register_node_type("parallel", compile=_compile_parallel, validate=_validate_composite)
register_node_type("action", compile=_compile_action, validate=_validate_action)
register_node_type("condition", compile=_compile_condition, validate=_validate_condition)
register_node_type("wait_condition", compile=_compile_wait_condition,
                   validate=_validate_wait_condition)
register_node_type("compute", compile=_compile_compute, validate=_validate_compute)
register_node_type("settle", compile=_compile_settle, validate=_validate_settle)
register_node_type("read", compile=_compile_read, validate=_validate_read)

# ---------------------------------------------------------------------------
# Dependency-plan types (`interaction_solver/utils/dependency_compiler.py`).
# Each compiles to a py_trees built-in, or to one of the few nodes py_trees
# lacks (`dependency_nodes.py`). Decorators take their one child in
# `children`, like composites, so every tree walker still finds it. Operators
# travel by name: "eq ne gt ge lt le" and "and_ or_".
# ---------------------------------------------------------------------------

def _expressions(spec):
    return [py_trees.common.ComparisonExpression(c["key"], c.get("value"), OPERATORS[c["op"]])
            for c in spec["checks"]]


def _compile_check(spec, compile_child):
    """SUCCESS / FAILURE now: one key, or several combined with and_ / or_."""
    expressions = _expressions(spec)
    if len(expressions) == 1:
        return py_trees.behaviours.CheckBlackboardVariableValue(
            name=_name(spec), check=expressions[0])
    return py_trees.behaviours.CheckBlackboardVariableValues(
        name=_name(spec), checks=expressions,
        operator=COMBINERS[spec.get("combine", "and_")])


def _compile_wait_for(spec, compile_child):
    """RUNNING until the check holds (never FAILURE)."""
    expressions = _expressions(spec)
    if len(expressions) == 1:
        return py_trees.behaviours.WaitForBlackboardVariableValue(
            name=_name(spec), check=expressions[0])
    return py_trees.decorators.Condition(
        name=_name(spec), status=py_trees.common.Status.SUCCESS,
        child=_compile_check(spec, compile_child))


def _compile_compare(spec, compile_child):
    return CompareBlackboardKeys(_name(spec), spec["key1"], spec["key2"],
                                 OPERATORS[spec["op"]])


def _only_child(spec, compile_child):
    (child,) = spec["children"]
    return compile_child(child)


def _compile_await(spec, compile_child):
    """Turn any check into a wait: RUNNING until the child succeeds."""
    return py_trees.decorators.Condition(name=_name(spec), status=py_trees.common.Status.SUCCESS,
                                         child=_only_child(spec, compile_child))


def _compile_repeat(spec, compile_child):
    """Re-run the child for ever; ends only when the child fails (or is stopped)."""
    return py_trees.decorators.Repeat(name=_name(spec), num_success=-1,
                                      child=_only_child(spec, compile_child))


def _compile_inverter(spec, compile_child):
    return py_trees.decorators.Inverter(name=_name(spec),
                                        child=_only_child(spec, compile_child))


def _compile_running(spec, compile_child):
    return py_trees.behaviours.Running(name=_name(spec))


def _compile_set_variable(spec, compile_child):
    return py_trees.behaviours.SetBlackboardVariable(
        name=_name(spec), variable_name=spec["key"], variable_value=spec.get("value"),
        overwrite=True)


def _compile_timer(spec, compile_child):
    return ClockTimer(_name(spec), spec["seconds"])


def _compile_wait_until(spec, compile_child):
    return WaitUntilTime(_name(spec), spec["time_of_day"])


def _compile_stamp(spec, compile_child):
    return StampTime(_name(spec), spec["key"])


def _validate_checks(spec, path, validate_child):
    checks = spec.get("checks")
    if not isinstance(checks, list) or not checks:
        return [f"{path}: needs a non-empty 'checks' list"]
    errors = [f"{path}.checks[{i}]: needs 'key' and an 'op' in {sorted(OPERATORS)}"
              for i, c in enumerate(checks)
              if not isinstance(c, dict) or not c.get("key") or c.get("op") not in OPERATORS]
    if len(checks) > 1 and spec.get("combine", "and_") not in COMBINERS:
        errors.append(f"{path}: 'combine' must be one of {sorted(COMBINERS)}")
    return errors


def _validate_compare(spec, path, validate_child):
    if not spec.get("key1") or not spec.get("key2") or spec.get("op") not in OPERATORS:
        return [f"{path}: compare needs 'key1', 'key2' and an 'op' in {sorted(OPERATORS)}"]
    return []


def _validate_one_child(spec, path, validate_child):
    children = spec.get("children")
    if not isinstance(children, list) or len(children) != 1:
        return [f"{path}: needs exactly one entry in 'children'"]
    return validate_child(children[0], f"{path}.children[0]")


def _validate_nothing(spec, path, validate_child):
    return []


def _validate_key(spec, path, validate_child):
    return [] if spec.get("key") else [f"{path}: needs a blackboard 'key'"]


def _validate_timer(spec, path, validate_child):
    seconds = spec.get("seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
        return [f"{path}: timer needs 'seconds' >= 0"]
    return []


def _validate_wait_until(spec, path, validate_child):
    try:
        parts = [int(p) for p in str(spec.get("time_of_day")).split(":")]
        ok = len(parts) in (2, 3) and 0 <= parts[0] < 24 and 0 <= parts[1] < 60
    except ValueError:
        ok = False
    return [] if ok else [f"{path}: wait_until needs 'time_of_day' as HH:MM[:SS]"]


for _type, _compile, _validate in (
        ("check", _compile_check, _validate_checks),
        ("wait_for", _compile_wait_for, _validate_checks),
        ("compare", _compile_compare, _validate_compare),
        ("await", _compile_await, _validate_one_child),
        ("repeat", _compile_repeat, _validate_one_child),
        ("inverter", _compile_inverter, _validate_one_child),
        ("running", _compile_running, _validate_nothing),
        ("set_variable", _compile_set_variable, _validate_key),
        ("timer", _compile_timer, _validate_timer),
        ("wait_until", _compile_wait_until, _validate_wait_until),
        ("stamp", _compile_stamp, _validate_key)):
    register_node_type(_type, compile=_compile, validate=_validate)
register_node_type("ignore_failure", compile=_compile_ignore_failure,
                   validate=_validate_ignore_failure)
