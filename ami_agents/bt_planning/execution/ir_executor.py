"""
IR Executor - compiles JSON IR to py_trees and executes.

Ported from behaviortree-planning-for-ami-agents with adjusted imports
for the AAMAS 2026 demo multi-agent system.
"""

import logging
from typing import Optional

import py_trees
from py_trees.common import Status

from ..nodes.affordance_nodes import (
    ActionAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    ComparisonOperator,
)
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
            max_ticks_raw = config.get("bt_execution", {}).get("max_ticks", {}).get("default", 10)
            max_ticks = int(max_ticks_raw)
        elif max_ticks is None:
            max_ticks = 10

        self.max_ticks = max_ticks

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
            import json
            logger.info(f"[BT JSON IR] {json.dumps(tree_spec, indent=2)}")
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

        else:
            raise ValueError(f"Unknown node type: {node_type}")

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
        else:
            result.final_status = "RUNNING (max ticks reached)"

        tree.shutdown()
        logger.info(f"Execution complete: {result.final_status} after {result.ticks} ticks")
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
        valid_types = {"sequence", "selector", "parallel", "action", "condition"}
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
        elif node_type == "condition":
            property_url = spec.get("property_url")
            if not property_url or not isinstance(property_url, str):
                errors.append(f"{path}: condition nodes require 'property_url'")
            if "expected_value" not in spec:
                errors.append(f"{path}: condition nodes require 'expected_value'")

        return errors
