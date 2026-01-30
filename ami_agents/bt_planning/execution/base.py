"""
Base classes for BT execution.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of executing a behavior tree."""
    success: bool
    tree_name: str = ""
    ticks: int = 0
    final_status: str = ""
    tick_history: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for tracing."""
        return {
            "success": self.success,
            "tree_name": self.tree_name,
            "ticks": self.ticks,
            "final_status": self.final_status,
            "tick_history": self.tick_history,
            "error": self.error,
        }
