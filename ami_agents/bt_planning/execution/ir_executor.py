"""
IR Executor - compiles JSON IR to py_trees and executes.

Ported from behaviortree-planning-for-ami-agents with adjusted imports
for the AAMAS 2026 demo multi-agent system.
"""

import json
import logging
import os
import time
from typing import Optional

import py_trees
from py_trees.common import Status

from ..nodes.affordance_nodes import (
    ActionAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    WaitPropertyConditionNode,
    ComparisonOperator,
)
from ..nodes.compute_node import BlackboardComputeNode
from .base import ExecutionResult

logger = logging.getLogger(__name__)


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


class IRExecutor:
    """
    Executor for JSON IR behavior trees.

    Compiles JSON specification to py_trees objects and executes.
    """

    def __init__(self, max_ticks: int = None, config: dict = None):
        # Get max_ticks from config if not explicitly provided
        if max_ticks is None and config:
            max_ticks = config.get("bt_execution", {}).get("max_ticks", {}).get("default", 10)
        elif max_ticks is None:
            max_ticks = 10

        self.max_ticks = max_ticks
        self._settling_times = self._load_settling_times()

    def execute_from_spec(self, tree_spec: dict) -> ExecutionResult:
        """
        Execute a raw JSON IR tree specification.

        This is the primary entry point for the AAMAS integration,
        accepting a raw dict instead of a Plan object.

        Args:
            tree_spec: JSON IR tree specification dict

        Returns:
            ExecutionResult with execution details
        """
        if not tree_spec:
            logger.info("No behavior tree to execute")
            return ExecutionResult(
                success=True,
                final_status="No behavior tree to execute.",
                ticks=0,
                tick_history=[],
            )

        try:
            logger.info("Compiling JSON IR to py_trees")
            tree = self._compile(tree_spec)
            logger.info(f"Compiled tree: {tree.name}")

            logger.info("Executing behavior tree")
            result = self._execute_tree(tree)
            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
            )

    def _compile(self, spec: dict) -> py_trees.behaviour.Behaviour:
        """
        Compile a JSON specification to py_trees.

        Args:
            spec: JSON tree specification

        Returns:
            py_trees behavior
        """
        node_type = spec.get("type")
        name = spec.get("name", "unnamed")

        if node_type == "sequence":
            children = [self._compile(child) for child in spec.get("children", [])]
            return py_trees.composites.Sequence(name=name, memory=True, children=children)

        elif node_type == "selector":
            children = [self._compile(child) for child in spec.get("children", [])]
            return py_trees.composites.Selector(name=name, memory=False, children=children)

        elif node_type == "parallel":
            children = [self._compile(child) for child in spec.get("children", [])]
            policy_name = spec.get("policy", "success_on_all")
            if policy_name == "success_on_one":
                policy = py_trees.common.ParallelPolicy.SuccessOnOne()
            else:
                policy = py_trees.common.ParallelPolicy.SuccessOnAll()
            return py_trees.composites.Parallel(name=name, policy=policy, children=children)

        elif node_type == "action":
            return ActionAffordanceNode(
                name=name,
                action_url=spec["action_url"],
                parameters=spec.get("parameters", {}),
                settling_time_seconds=self._settling_time_for_action(
                    spec["action_url"],
                    spec.get("settling_time_seconds"),
                ),
            )

        elif node_type == "condition":
            operator = spec.get("operator")
            if operator and operator != "==":
                return ComparisonPropertyConditionNode(
                    name=name,
                    property_url=spec["property_url"],
                    expected_value=spec["expected_value"],
                    operator=OPERATOR_MAP.get(operator, ComparisonOperator.EQUAL),
                    value_path=spec.get("value_path"),
                )
            else:
                return PropertyConditionNode(
                    name=name,
                    property_url=spec["property_url"],
                    expected_value=spec["expected_value"],
                    value_path=spec.get("value_path"),
                )

        elif node_type == "wait_condition":
            return WaitPropertyConditionNode(
                name=name,
                property_url=spec["property_url"],
                expected_value=spec["expected_value"],
                operator=OPERATOR_MAP.get(spec.get("operator", "=="), ComparisonOperator.EQUAL),
                value_path=spec.get("value_path"),
                timeout_seconds=spec.get("timeout_seconds", 30.0),
                poll_interval_seconds=spec.get("poll_interval_seconds", 1.0),
            )

        elif node_type == "compute":
            return BlackboardComputeNode(
                name=name,
                op=spec["op"],
                inputs=spec.get("inputs", []),
                output=spec.get("output"),
                args=spec.get("args", {}),
            )

        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _load_settling_times(self) -> dict[str, float]:
        """Load entity/action settling times from the lab TD-SOSA env mapping."""
        raw = os.getenv("TD_SOSA_SETTLING_TIMES", "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Ignoring invalid TD_SOSA_SETTLING_TIMES JSON: %s", exc)
            return {}
        if not isinstance(payload, dict):
            logger.warning("Ignoring TD_SOSA_SETTLING_TIMES because it is not an object")
            return {}

        settling_times: dict[str, float] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, (int, float)):
                continue
            seconds = float(value)
            if seconds > 0:
                settling_times[key] = seconds
        return settling_times

    def _settling_time_for_action(
        self,
        action_url: str,
        explicit_seconds: object = None,
    ) -> float:
        """Resolve settling time from the tree annotation, then the env mapping."""
        if isinstance(explicit_seconds, (int, float)) and explicit_seconds > 0:
            return float(explicit_seconds)

        key = self._settling_key_for_action_url(action_url)
        if key is None:
            return 0.0
        return self._settling_times.get(key, 0.0)

    @staticmethod
    def _settling_key_for_action_url(action_url: str) -> Optional[str]:
        """
        Convert HASP action URLs to TD-SOSA settling-time keys.

        Example:
        /workspaces/lab308e/artifacts/blackout_blinds_308e_cover/ha/cover/close_cover
        -> entity:cover.blackout_blinds_308e_cover:action:cover.close_cover
        """
        if not isinstance(action_url, str):
            return None

        parts = [part for part in action_url.rstrip("/").split("/") if part]
        try:
            artifacts_index = parts.index("artifacts")
            ha_index = parts.index("ha", artifacts_index + 1)
            artifact = parts[artifacts_index + 1]
            domain = parts[ha_index + 1]
            service = parts[ha_index + 2]
        except (ValueError, IndexError):
            return None

        return f"entity:{domain}.{artifact}:action:{domain}.{service}"

    def _execute_tree(self, tree: py_trees.behaviour.Behaviour) -> ExecutionResult:
        """
        Execute a compiled behavior tree.

        Args:
            tree: Compiled py_trees behavior

        Returns:
            ExecutionResult with execution details
        """
        tree.setup_with_descendants()

        result = ExecutionResult(
            success=False,
            tree_name=tree.name,
            ticks=0,
            tick_history=[],
        )

        for tick in range(self.max_ticks):
            result.ticks = tick + 1
            tree.tick_once()

            status_name = tree.status.name
            result.tick_history.append(status_name)
            logger.debug(f"Tick {tick + 1}: {status_name}")

            if tree.status == Status.SUCCESS:
                result.final_status = "SUCCESS"
                result.success = True
                break
            elif tree.status == Status.FAILURE:
                result.final_status = "FAILURE"
                result.success = False
                break
            elif tree.status == Status.RUNNING:
                poll_interval = self._min_poll_interval(tree)
                if poll_interval > 0:
                    time.sleep(poll_interval)
        else:
            result.final_status = "RUNNING (max ticks reached)"

        # Harvest the set of action leaves that actually invoked their HTTP
        # endpoint successfully. Anything not in this list was short-circuited
        # (e.g. by a sibling condition under a Selector) and must not be
        # treated as a causal action by signifier extraction.
        executed: list[tuple[str, str]] = []
        for node in tree.iterate():
            if isinstance(node, ActionAffordanceNode) and getattr(node, "_executed_successfully", False):
                executed.append((node.name, node.action_url))
        result.executed_actions = executed

        tree.shutdown()
        logger.info(
            f"Execution complete: {result.final_status} after {result.ticks} ticks "
            f"(executed_actions={len(executed)})"
        )
        return result

    def validate_tree(self, spec: dict) -> list[str]:
        """
        Public validation method for tree specs.

        Args:
            spec: JSON tree specification

        Returns:
            List of validation error strings (empty if valid)
        """
        return self._validate_tree(spec)

    def _validate_tree(self, spec: dict, path: str = "tree") -> list[str]:
        """Validate that the tree spec is non-empty and structurally sound."""
        errors: list[str] = []

        if not spec:
            errors.append(f"{path}: tree is empty")
            return errors

        if not isinstance(spec, dict):
            errors.append(f"{path}: expected object, got {type(spec).__name__}")
            return errors

        # 'name' is optional -- the compiler defaults to "unnamed" and the
        # AsyncBTPlanner normalizer fills in sensible defaults.  We only warn
        # (do NOT add to errors) so that trees without names still pass.
        node_type = spec.get("type")
        valid_types = {"sequence", "selector", "parallel", "action", "condition", "wait_condition", "compute"}
        if node_type not in valid_types:
            errors.append(f"{path}: missing or invalid 'type'")

        # Composite nodes
        if node_type in {"sequence", "selector", "parallel"}:
            children = spec.get("children")
            if not isinstance(children, list) or not children:
                errors.append(f"{path}: composite nodes require non-empty 'children'")
            else:
                for idx, child in enumerate(children):
                    errors.extend(
                        self._validate_tree(child, path=f"{path}.children[{idx}]")
                    )
        # Action nodes
        elif node_type == "action":
            action_url = spec.get("action_url")
            if not action_url or not isinstance(action_url, str):
                errors.append(f"{path}: action nodes require 'action_url'")
        # Condition nodes
        elif node_type in {"condition", "wait_condition"}:
            property_url = spec.get("property_url")
            if not property_url or not isinstance(property_url, str):
                errors.append(f"{path}: {node_type} nodes require 'property_url'")
            if "expected_value" not in spec:
                errors.append(f"{path}: {node_type} nodes require 'expected_value'")
            if node_type == "wait_condition":
                timeout_seconds = spec.get("timeout_seconds")
                if timeout_seconds is not None and not isinstance(timeout_seconds, (int, float)):
                    errors.append(f"{path}: wait_condition timeout_seconds must be numeric")
                poll_interval = spec.get("poll_interval_seconds")
                if poll_interval is not None and (
                    not isinstance(poll_interval, (int, float)) or poll_interval < 0
                ):
                    errors.append(f"{path}: wait_condition poll_interval_seconds must be >= 0")
        # Compute nodes
        elif node_type == "compute":
            op = spec.get("op")
            if not op or not isinstance(op, str):
                errors.append(f"{path}: compute nodes require a string 'op'")
            inputs = spec.get("inputs")
            if inputs is not None and not isinstance(inputs, list):
                errors.append(f"{path}: compute 'inputs' must be a list of blackboard keys")

        return errors

    def _min_poll_interval(self, tree: py_trees.behaviour.Behaviour) -> float:
        intervals = []
        for node in tree.iterate():
            if isinstance(node, WaitPropertyConditionNode) and node.status == Status.RUNNING:
                interval = getattr(node, "poll_interval_seconds", None)
                if interval is not None:
                    intervals.append(float(interval))
        return min(intervals) if intervals else 0.0
