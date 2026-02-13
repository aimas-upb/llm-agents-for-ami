"""
Data models for the User Assistant agent.

Defines structured intents, conversation state machine, and per-conversation state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConversationPhase(Enum):
    """State machine phases for a single conversation."""

    IDLE = "idle"
    EXTRACTING_INTENTS = "extracting_intents"
    AWAITING_PLAN = "awaiting_plan"
    SUMMARIZING_PLAN = "summarizing_plan"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"


@dataclass
class Intent:
    """Structured intent extracted from a user message.

    Canonical action types:
        - ``turn_on``   → "turn on <artifact>"
        - ``turn_off``  → "turn off <artifact>"
        - ``set``       → "set <artifact> <parameter> to <value>"
        - ``check_status`` → "check status of <artifact>"
    """

    action: str
    artifact: str
    parameter: Optional[str] = None
    value: Optional[Any] = None

    def to_canonical_string(self) -> str:
        """Convert to the canonical string form expected by InteractionSolver.

        Follows the templates from section VII of the original prompt.
        """
        if self.action == "turn_on":
            return f"turn on {self.artifact}"
        elif self.action == "turn_off":
            return f"turn off {self.artifact}"
        elif self.action == "set" and self.parameter is not None and self.value is not None:
            return f"set {self.artifact} {self.parameter} to {self.value}"
        elif self.action == "check_status":
            return f"check status of {self.artifact}"
        # Fallback for non-standard actions
        parts = [self.action, self.artifact]
        if self.parameter:
            parts.append(self.parameter)
        if self.value is not None:
            parts.append(f"to {self.value}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        d: Dict[str, Any] = {"action": self.action, "artifact": self.artifact}
        if self.parameter is not None:
            d["parameter"] = self.parameter
        if self.value is not None:
            d["value"] = self.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intent":
        """Deserialize from a dict (e.g. LLM JSON output)."""
        return cls(
            action=data.get("action", ""),
            artifact=data.get("artifact", ""),
            parameter=data.get("parameter"),
            value=data.get("value"),
        )


# Tokens treated as user confirmation / rejection by the confirmation handler.
CONFIRM_TOKENS = frozenset(
    {"yes", "ok", "okay", "proceed", "continue", "do it", "go ahead", "sure", "yep", "yeah"}
)
REJECT_TOKENS = frozenset(
    {"no", "cancel", "discard", "reject", "nope", "nah", "stop"}
)


@dataclass
class ConversationState:
    """Per-conversation state tracked by UserAssistantAgent."""

    phase: ConversationPhase = ConversationPhase.IDLE
    user_message: str = ""
    intents: List[Intent] = field(default_factory=list)
    workspace_id: Optional[str] = None
    plan_json: Optional[str] = None
    plan_hash: Optional[str] = None
    plan_summary: Optional[str] = None
    is_query: bool = False

    def clear_plan(self) -> None:
        """Reset plan-related fields."""
        self.plan_json = None
        self.plan_hash = None
        self.plan_summary = None
        self.intents = []
        self.workspace_id = None
