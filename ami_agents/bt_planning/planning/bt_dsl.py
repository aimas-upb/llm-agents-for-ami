"""
BT-builder DSL for direct-code plan generation (``python_code`` mode).

In direct-code mode the LLM emits a Python script instead of a JSON tool
call. The script composes a behaviour tree through the small builder API
below; every builder returns the exact JSON IR dict the ``behavior_tree``
mode produces, so the whole downstream pipeline (affordance-ref resolution,
normalization, settling-time annotation, validation, ``IRExecutor``) is
shared verbatim between the two generation modes.

Free-form Python over the ``env`` snapshot replaces the IR mode's declarative
``compute`` nodes: thresholds and aggregations are computed natively at
generation time. That representational difference — named compute nodes in
data vs. arbitrary code — is exactly what the EACL 2027 experiment measures.

The script runs through :func:`run_bt_code` under a restricted ``exec``:
an AST gate rejects imports and dunder access, and the namespace exposes
only whitelisted builtins, the builders, ``env`` and ``mark_impossible``.
The threat model is buggy SLM output, not adversarial code — the planner
side only ever builds JSON; generated code never reaches the executor.
"""

import ast
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from .prompts import _flatten_state_props, _short_state_key

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Builder API — each returns a JSON IR node dict
# --------------------------------------------------------------------------- #
def _composite(node_type: str, name: Any, children: tuple) -> dict:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{node_type}() requires a non-empty name as its first argument")
    if not children:
        raise ValueError(f"{node_type}('{name}') requires at least one child node")
    for child in children:
        if not isinstance(child, dict) or "type" not in child:
            raise ValueError(
                f"{node_type}('{name}') children must be nodes built with the "
                f"provided functions (got {type(child).__name__})"
            )
    return {"type": node_type, "name": name, "children": list(children)}


def sequence(name: str, *children: dict) -> dict:
    """Ordered steps; fails on the first failing child."""
    return _composite("sequence", name, children)


def selector(name: str, *children: dict) -> dict:
    """Alternatives; succeeds on the first succeeding child."""
    return _composite("selector", name, children)


def parallel(name: str, *children: dict, policy: str = "success_on_all") -> dict:
    """Concurrent children; policy is success_on_all or success_on_one."""
    if policy not in ("success_on_all", "success_on_one"):
        raise ValueError(
            f"parallel('{name}') policy must be 'success_on_all' or 'success_on_one'"
        )
    node = _composite("parallel", name, children)
    node["policy"] = policy
    return node


def action(affordance_id: str, parameters: Optional[dict] = None, name: Optional[str] = None) -> dict:
    """Invoke an action affordance, optionally with JSON parameters."""
    if not isinstance(affordance_id, str) or not affordance_id:
        raise ValueError("action() requires an affordance_id string from the affordances list")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError(f"action('{affordance_id}') parameters must be a dict")
    node: dict = {"type": "action", "affordance_id": affordance_id}
    if parameters:
        node["parameters"] = parameters
    if name:
        node["name"] = name
    return node


def condition(
    affordance_id: str,
    expected: Any,
    op: str = "==",
    name: Optional[str] = None,
    value_path: Optional[str] = None,
) -> dict:
    """Check a property value once."""
    return _condition_node("condition", affordance_id, expected, op, name, value_path)


def wait_condition(
    affordance_id: str,
    expected: Any,
    op: str = "==",
    timeout_seconds: Optional[float] = None,
    poll_interval_seconds: Optional[float] = None,
    name: Optional[str] = None,
    value_path: Optional[str] = None,
) -> dict:
    """Poll a property value until it matches or the timeout expires."""
    node = _condition_node("wait_condition", affordance_id, expected, op, name, value_path)
    if timeout_seconds is not None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds < 0:
            raise ValueError(f"wait_condition('{affordance_id}') timeout_seconds must be a number >= 0")
        node["timeout_seconds"] = timeout_seconds
    if poll_interval_seconds is not None:
        if not isinstance(poll_interval_seconds, (int, float)) or poll_interval_seconds <= 0:
            raise ValueError(f"wait_condition('{affordance_id}') poll_interval_seconds must be a number > 0")
        node["poll_interval_seconds"] = poll_interval_seconds
    return node


_VALID_OPERATORS = ("==", "!=", ">", "<", ">=", "<=")


def _condition_node(node_type, affordance_id, expected, op, name, value_path) -> dict:
    if not isinstance(affordance_id, str) or not affordance_id:
        raise ValueError(
            f"{node_type}() requires an affordance_id string (a [property] id or readable property URL)"
        )
    if expected is None:
        raise ValueError(f"{node_type}('{affordance_id}') requires an expected value")
    if op not in _VALID_OPERATORS:
        raise ValueError(
            f"{node_type}('{affordance_id}') op must be one of {', '.join(_VALID_OPERATORS)}"
        )
    node: dict = {"type": node_type, "affordance_id": affordance_id, "expected_value": expected}
    if op != "==":
        node["operator"] = op
    if name:
        node["name"] = name
    if value_path:
        node["value_path"] = value_path
    return node


BUILDERS = {
    "sequence": sequence,
    "selector": selector,
    "parallel": parallel,
    "action": action,
    "condition": condition,
    "wait_condition": wait_condition,
}


# --------------------------------------------------------------------------- #
# Restricted execution
# --------------------------------------------------------------------------- #
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "True": True,
    "False": False,
    "None": None,
}


@dataclass
class DslResult:
    """Outcome of executing one generated build script."""

    tree: Optional[dict] = None
    explanation: str = ""
    impossible: bool = False
    errors: list = field(default_factory=list)


def _gate_ast(code: str) -> list[str]:
    """Static checks before exec: no imports, no dunder access, no global escape."""
    try:
        parsed = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

    errors: list[str] = []
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(
                f"import statements are not allowed (line {node.lineno}); "
                "use only the provided builder functions and env"
            )
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            errors.append(f"dunder names are not allowed: {node.id} (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder attributes are not allowed: .{node.attr} (line {node.lineno})")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append(f"global/nonlocal statements are not allowed (line {node.lineno})")
    return errors


def _format_runtime_error(exc: Exception) -> str:
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<generated_plan>":
            line = frame.lineno
    location = f" (line {line})" if line is not None else ""
    return f"{type(exc).__name__}: {exc}{location}"


def run_bt_code(code: str, env: dict) -> DslResult:
    """
    Execute one generated build script under the restricted namespace.

    The script must assign the final behaviour tree (built with the builder
    functions) to a top-level variable named ``tree``, or call
    ``mark_impossible(reason)`` when the goal cannot be achieved.
    """
    result = DslResult()

    if not code or not code.strip():
        result.errors.append("the generated code was empty")
        return result

    gate_errors = _gate_ast(code)
    if gate_errors:
        result.errors.extend(gate_errors)
        return result

    def mark_impossible(reason: str = "") -> None:
        result.impossible = True
        if reason:
            result.explanation = str(reason)

    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        **BUILDERS,
        "env": env,
        "mark_impossible": mark_impossible,
    }

    try:
        compiled = compile(code, "<generated_plan>", "exec")
        exec(compiled, namespace)  # noqa: S102 — gated, whitelisted namespace
    except Exception as exc:  # noqa: BLE001 — every script bug becomes retry feedback
        result.errors.append(_format_runtime_error(exc))
        return result

    if result.impossible:
        return result

    tree = namespace.get("tree")
    if not isinstance(tree, dict) or "type" not in tree:
        result.errors.append(
            "the code must assign the final behaviour tree (built with sequence/"
            "selector/parallel/action/condition/wait_condition) to a variable named 'tree'"
        )
        return result

    result.tree = tree
    explanation = namespace.get("explanation")
    if isinstance(explanation, str):
        result.explanation = explanation
    return result


# --------------------------------------------------------------------------- #
# env snapshot
# --------------------------------------------------------------------------- #
def build_env_snapshot(state: Optional[dict]) -> dict:
    """
    Build the read-only ``env`` dict handed to generated code.

    Mirrors the "Current State" rendering in ``format_capability_context`` so
    the keys the model reads in the prompt are the keys that exist in ``env``:
    ``{artifact_short_id: {property_name: value}}``.
    """
    if not isinstance(state, dict) or not state:
        return {}

    state_items = state
    if "artifacts" in state and isinstance(state["artifacts"], dict):
        state_items = state["artifacts"]
    if not isinstance(state_items, dict):
        return {}

    snapshot: dict = {}
    for artifact_uri, props in state_items.items():
        artifact_key = _short_state_key(artifact_uri)
        if not isinstance(props, dict):
            snapshot[artifact_key] = props
            continue
        flat = _flatten_state_props(props)
        if flat:
            snapshot[artifact_key] = flat
    return snapshot
