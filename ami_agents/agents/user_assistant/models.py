"""
Data models for the User Assistant agent.

Defines structured intents, conversation state machine, and per-conversation state.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Import intent types from shared models. `Intent` is re-exported here because
# several modules still import it from this location; it lives in shared because
# the InteractionSolver uses it too.
from ...shared.models.intents import (  # noqa: F401
    ExplicitGoalIntent,
    ImplicitGoalIntent,
    Intent,
)


class ConversationPhase(Enum):
    """State machine phases for a single conversation."""

    IDLE = "idle"
    SEGMENTING = "segmenting"
    EXTRACTING_INTENTS = "extracting_intents"
    AWAITING_PLAN = "awaiting_plan"
    SUMMARIZING_PLAN = "summarizing_plan"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"


@dataclass
class AtomicIntent:
    """Atomic intent segment from the LLM segmentation stage.

    Represents one indivisible request as identified by the atomic segmenter.
    """

    span: str  # verbatim text from user input for this intent
    category: str  # "GOAL_REQUEST" | "ENV_STATE_REQUEST" | "ENV_CAPABILITIES_REQUEST"
    reason: str  # LLM justification for the categorization


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
    # The request currently being handled on this thread. Carried here while
    # request handling still lives in the receiver; it moves onto the per-request
    # FSM once that lands, and the plans this request produces keep it as
    # provenance either way.
    request_id: Optional[str] = None
    intents: List[Union[ImplicitGoalIntent, ExplicitGoalIntent]] = field(default_factory=list)
    workspace_id: Optional[str] = None
    plan_json: Optional[str] = None
    plan_hash: Optional[str] = None
    plan_summary: Optional[str] = None
    plan_count: int = 1  # Number of independent plans (for multi-plan responses)
    is_query: bool = False

    def clear_plan(self) -> None:
        """Reset plan-related fields."""
        self.plan_json = None
        self.plan_hash = None
        self.plan_summary = None
        self.plan_count = 1
        self.intents = []
        self.workspace_id = None
