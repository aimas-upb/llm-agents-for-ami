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
        - ``check``  → "check <artifact>" or "check <artifact> <parameter>"
        - ``set``    → "set <artifact> <parameter> to <value>"
                       Includes boolean actions: turn on/off, open/close
                       (e.g. set on_off to true, set open_close to false).
        - ``modify`` → "modify <artifact> <parameter> by <value>"
                       or "modify <artifact> <parameter>" when the delta
                       is unspecified (BT planner determines the amount).
    """

    action: str
    artifact: str
    parameter: Optional[str] = None
    value: Optional[Any] = None
    intent_text: Optional[str] = None

    def to_canonical_string(self) -> str:
        """Convert to the canonical string form expected by InteractionSolver."""
        if self.action == "check":
            if self.parameter:
                return f"check {self.artifact} {self.parameter}"
            return f"check {self.artifact}"
        elif self.action == "set" and self.parameter is not None and self.value is not None:
            return f"set {self.artifact} {self.parameter} to {self.value}"
        elif self.action == "modify" and self.parameter is not None:
            if self.value is not None:
                return f"modify {self.artifact} {self.parameter} by {self.value}"
            return f"modify {self.artifact} {self.parameter}"
        # Fallback for non-standard actions or incomplete fields
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
        if self.intent_text is not None:
            d["intent_text"] = self.intent_text
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intent":
        """Deserialize from a dict (e.g. LLM JSON output)."""
        return cls(
            action=data.get("action", ""),
            artifact=data.get("artifact", ""),
            parameter=data.get("parameter"),
            value=data.get("value"),
            intent_text=data.get("intent_text"),
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
