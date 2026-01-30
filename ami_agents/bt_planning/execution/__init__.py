"""
BT execution module - JSON IR compilation and tick-loop execution.
"""

from .base import ExecutionResult
from .ir_executor import IRExecutor

__all__ = [
    "ExecutionResult",
    "IRExecutor",
]
