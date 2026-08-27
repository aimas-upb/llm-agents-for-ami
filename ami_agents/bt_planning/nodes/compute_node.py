"""
Blackboard compute node — the canonical "custom compute node" template.

This is the node class issue #22 is about: a node that runs custom logic
reading from and writing to the py-trees blackboard. The key design choice
that keeps it JSON-serialisable (and avoids ``pickle`` across agents) is that
the custom logic is **referenced by name**, not embedded as code.

A compute node's JSON IR is pure data::

    {
        "name": "AggregatePresence",
        "type": "compute",
        "op": "any",                       # registered op name, NOT a lambda
        "inputs": ["sensor/hall", "sensor/kitchen"],   # blackboard keys to read
        "output": "presence/anywhere",     # blackboard key to write
        "args": {}                          # optional op arguments
    }

The actual Python implementation of ``op`` lives in :data:`COMPUTE_OPS`, which
is shared code present in both the InteractionSolver and the UserAssistant
(they both import ``ami_agents``). Only the *name* crosses the wire, so a new
environment can reconstruct the node as long as it has the same shared package
— exactly like the built-in action/condition nodes are reconstructed by
``IRExecutor`` today.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import py_trees
from py_trees.common import Status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compute-op registry: named, pure, JSON-config-driven functions.
# Each op takes (values, args) where `values` are the blackboard reads (in the
# order of `inputs`) and `args` is the node's optional `args` dict.
# ---------------------------------------------------------------------------
ComputeOp = Callable[[List[Any], Dict[str, Any]], Any]

COMPUTE_OPS: Dict[str, ComputeOp] = {}


def register_compute_op(name: str, fn: ComputeOp) -> None:
    """Register a named compute op usable from a ``compute`` node's ``op`` field."""
    COMPUTE_OPS[name] = fn


def registered_compute_ops() -> List[str]:
    return sorted(COMPUTE_OPS)


# Built-in ops. Kept deliberately small and side-effect free.
register_compute_op("all", lambda values, args: all(values))
register_compute_op("any", lambda values, args: any(values))
register_compute_op("not", lambda values, args: not (values[0] if values else None))
register_compute_op("sum", lambda values, args: sum(v for v in values if isinstance(v, (int, float))))
register_compute_op("max", lambda values, args: max((v for v in values if v is not None), default=None))
register_compute_op("min", lambda values, args: min((v for v in values if v is not None), default=None))


class BlackboardComputeNode(py_trees.behaviour.Behaviour):
    """
    A custom node that reads inputs from the blackboard, applies a registered
    op, and writes the result back to the blackboard.

    Returns ``SUCCESS`` whenever the op runs without raising; the computed
    value is left on the blackboard for downstream nodes to consume. Returns
    ``FAILURE`` only on an unknown op or a raised exception. (A future variant
    could gate on the truthiness of the result; kept simple here.)
    """

    def __init__(
        self,
        name: str,
        op: str,
        inputs: Optional[List[str]] = None,
        output: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name)
        self.op = op
        self.inputs = list(inputs or [])
        self.output = output
        self.args = dict(args or {})

        self.blackboard = self.attach_blackboard_client(name=self.name)
        for key in self.inputs:
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)
        if self.output:
            self.blackboard.register_key(key=self.output, access=py_trees.common.Access.WRITE)

    def update(self) -> Status:
        fn = COMPUTE_OPS.get(self.op)
        if fn is None:
            self.feedback_message = f"unknown compute op: {self.op!r}"
            logger.warning("[%s] %s", self.name, self.feedback_message)
            return Status.FAILURE

        values: List[Any] = []
        for key in self.inputs:
            try:
                values.append(self.blackboard.get(key))
            except KeyError:
                values.append(None)

        try:
            result = fn(values, self.args)
        except Exception as exc:  # pragma: no cover - defensive
            self.feedback_message = f"compute op {self.op!r} raised: {exc}"
            logger.warning("[%s] %s", self.name, self.feedback_message)
            return Status.FAILURE

        if self.output:
            self.blackboard.set(self.output, result)
        self.feedback_message = f"{self.op}({self.inputs}) -> {result!r}"
        return Status.SUCCESS
