"""
IR Executor - compiles JSON IR to py_trees and executes.

Ported from behaviortree-planning-for-ami-agents with adjusted imports
for the AAMAS 2026 demo multi-agent system.
"""

import logging

import py_trees
from py_trees.common import Status

from ..nodes.registry import compile_node, validate_node
from .base import ExecutionResult

logger = logging.getLogger(__name__)


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

        Dispatches each node through the node registry (``nodes/registry.py``),
        so custom node types (e.g. ``compute``) compile the same way the five
        built-ins do.

        Args:
            spec: JSON tree specification

        Returns:
            py_trees behavior
        """
        return compile_node(spec, self._compile)

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
        """
        Validate that the tree spec is non-empty and structurally sound.

        Delegates per-node validation to the node registry so built-in and
        custom node types share one source of truth. ``name`` stays optional —
        the compiler defaults it to "unnamed".
        """
        return validate_node(spec, path)
