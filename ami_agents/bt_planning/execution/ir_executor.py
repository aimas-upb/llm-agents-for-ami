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

from ..nodes.affordance_nodes import ActionAffordanceNode
from ..nodes.registry import compile_node, validate_node
from .base import ExecutionResult

logger = logging.getLogger(__name__)


# Fallback when yaml is missing the key, per the repo convention of keeping
# defaults as module constants so unit tests stay self-contained.
_DEFAULT_MIN_TICK_YIELD_S = 0.01


class IRExecutor:
    """
    Executor for JSON IR behavior trees.

    Compiles JSON specification to py_trees objects and executes.
    """

    def __init__(self, max_ticks: int = None, config: dict = None,
                 min_tick_yield: float = None):
        # Get max_ticks from config if not explicitly provided
        if max_ticks is None and config:
            max_ticks_raw = config.get("bt_execution", {}).get("max_ticks", {}).get("default", 10)
            max_ticks = int(max_ticks_raw)
        elif max_ticks is None:
            max_ticks = 10

        if min_tick_yield is None:
            min_tick_yield = float(
                (config or {}).get("bt_execution", {}).get(
                    "min_tick_yield", _DEFAULT_MIN_TICK_YIELD_S)
            )

        self.max_ticks = max_ticks
        self.min_tick_yield = max(0.0, float(min_tick_yield))
        self._settling_times = self._load_settling_times()

    def _min_poll_interval(self, tree) -> float:
        """How long this synchronous harness may sleep between ticks.

        A RUNNING tree is waiting on something, and the only nodes that know how
        long are the waiting nodes themselves. Take the smallest interval any of
        them asks for, so the most impatient node still gets polled on time; a
        tree with no waiting nodes (or one asking for 0) does not sleep at all.

        This is the blocking-loop path only. Once execution moves into a SPADE
        behaviour that ticks once per `run()`, the sleep belongs to the
        behaviour's own cadence and this goes away with it.
        """
        intervals = [
            float(interval)
            for node in tree.iterate()
            for interval in (getattr(node, "poll_interval_seconds", None),)
            if isinstance(interval, (int, float))
        ]
        return min(intervals) if intervals else 0.0

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

        Dispatches each node through the node registry (``nodes/registry.py``),
        so custom node types (e.g. ``compute``) compile the same way the five
        built-ins do.

        Args:
            spec: JSON tree specification

        Returns:
            py_trees behavior
        """
        return compile_node(spec, self._compile)

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
                # Always yield a little. Node I/O runs on a thread pool, so a
                # loop that never sleeps just re-ticks faster than the request
                # can come back and burns the whole tick budget before the
                # first response lands. A node asking for interval 0 means
                # "poll as fast as sensible", not "spin".
                time.sleep(max(self._min_poll_interval(tree), self.min_tick_yield))
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
        """
        Validate that the tree spec is non-empty and structurally sound.

        Delegates per-node validation to the node registry so built-in and
        custom node types share one source of truth. ``name`` stays optional —
        the compiler defaults it to "unnamed".
        """
        return validate_node(spec, path)
