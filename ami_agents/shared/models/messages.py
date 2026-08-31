"""
Message models for inter-agent communication.

## Wire protocol (Phase 1)
All inter-agent XMPP messages MUST use:
- metadata["type"] == one of MessageType values (string)
- metadata["correlation_id"] for request/response correlation (string UUID)
- XMPP "thread" to carry conversation_id when available

Bodies SHOULD be JSON (agent-to-agent). During the transition we keep backwards
compatibility with some plain-text bodies, but new code should prefer dict/list
payloads serialized to JSON.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4


# --- Standard metadata keys used in SPADE messages (wire protocol) ---
META_TYPE = "type"
META_CORRELATION_ID = "correlation_id"
META_CONVERSATION_ID = "conversation_id"
# Routing keys. SPADE's `dispatch` hands a copy of a message to EVERY behaviour
# whose Template matches, so a per-request or per-plan behaviour addresses
# itself by carrying its own id in a Template rather than by claiming messages
# from a shared queue. Both are plain strings so a registry can persist them.
META_REQUEST_ID = "active_request_id"
META_PLAN_ID = "ami_plan_id"


def new_correlation_id() -> str:
    """Create a new correlation id for request/response pairs."""
    return str(uuid4())


def new_request_id() -> str:
    """Create a new id for one user utterance and everything it spawns."""
    return f"req-{uuid4().hex[:12]}"


def new_plan_id() -> str:
    """Create a new id for one execution of one plan.

    An instance key, not a content key: running the same plan text twice yields
    two plan ids and one plan hash.
    """
    return f"plan-{uuid4().hex[:12]}"


def ensure_correlation_id(metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Ensure metadata contains a correlation id; create one if missing.
    Returns the correlation id as a string.
    """
    if metadata is None:
        return new_correlation_id()
    existing = metadata.get(META_CORRELATION_ID)
    if existing is None or existing == "":
        existing = new_correlation_id()
        metadata[META_CORRELATION_ID] = existing
    return str(existing)


def extract_conversation_id(conversation_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Resolve a conversation id from explicit field or metadata. Intended to be
    mapped to XMPP message.thread.
    """
    if conversation_id:
        return str(conversation_id)
    if metadata:
        cid = metadata.get(META_CONVERSATION_ID)
        return str(cid) if cid else None
    return None


def serialize_body(content: Any) -> str:
    """
    Serialize message body for inter-agent messages.

    Transition behaviour:
    - dict/list/number/bool/None -> JSON
    - string -> returned as-is (legacy); prefer sending dict/list going forward
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except TypeError:
        # Last resort: string fallback
        return str(content)


class MessageType(Enum):
    """Types of messages exchanged between agents."""
    # User-facing messages
    USER_QUERY = "user_query"
    USER_RESPONSE = "user_response"

    # Environment-related messages
    ENV_DISCOVERY_COMPLETE = "env_discovery_complete"
    ENV_CHANGE_NOTIFICATION = "env_change_notification"
    ENV_STATE_REQUEST = "env_state_request"
    ENV_STATE_RESPONSE = "env_state_response"
    ENV_CAPABILITIES_REQUEST = "env_capabilities_request"
    ENV_CAPABILITIES_RESPONSE = "env_capabilities_response"
    ENV_SEMANTIC_QUERY_REQUEST = "env_semantic_query_request"
    ENV_SEMANTIC_QUERY_RESPONSE = "env_semantic_query_response"

    # Goal and planning messages
    GOAL_REQUEST = "goal_request"
    GOAL_RESPONSE = "goal_response"
    PLAN_CREATED = "plan_created"
    PLAN_EXECUTION_STATUS = "plan_execution_status"
    PLAN_COMPLETED = "plan_completed"
    PLAN_FAILED = "plan_failed"

    # Affordance matching
    AFFORDANCE_MATCH_REQUEST = "affordance_match_request"
    AFFORDANCE_MATCH_RESPONSE = "affordance_match_response"

    # Plan management
    # A confirmed plan is handed to plan management as a message rather than
    # executed in place, so the behaviour that took the request is free to end
    # and execution becomes independently addressable.
    PLAN_EXECUTE_REQUEST = "plan_execute_request"
    PLAN_STATUS_REQUEST = "plan_status_request"
    PLAN_EXPLAIN_REQUEST = "plan_explain_request"
    # Ask a maintenance plan to start a burst now instead of waiting out its
    # interval. The condition that decides this lives outside the plan.
    PLAN_TRIGGER_REQUEST = "plan_trigger_request"
    PLAN_EXPLAIN_RESPONSE = "plan_explain_response"
    PLAN_CANCEL_REQUEST = "plan_cancel_request"
    PLAN_ALTER_REQUEST = "plan_alter_request"
    PLAN_REPEAT_REQUEST = "plan_repeat_request"
    PLAN_IMPACT_NOTIFICATION = "plan_impact_notification"

    # Preference management
    PREFERENCE_STATEMENT = "preference_statement"

    # Signifier engine (RD4 memory, embedded in EnvExplorer)
    SIGNIFIER_MATCH_REQUEST = "signifier_match_request"
    SIGNIFIER_MATCH_RESPONSE = "signifier_match_response"
    SIGNIFIER_RECORD_EXECUTION_REQUEST = "signifier_record_execution_request"
    SIGNIFIER_RECORD_EXECUTION_RESPONSE = "signifier_record_execution_response"
    SIGNIFIER_LIST_REQUEST = "signifier_list_request"
    SIGNIFIER_LIST_RESPONSE = "signifier_list_response"


class MessageClassification(Enum):
    """Classification categories for user messages."""
    ENV_CAPABILITIES = "env_capabilities"
    ENV_STATE = "env_state"
    GOAL_REQUEST = "goal_request"
    PLAN_MANAGEMENT = "plan_management"
    PREFERENCE_STATEMENT = "preference_statement"


@dataclass
class Message:
    """Base message class for inter-agent communication."""
    message_type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalRequest:
    """Represents a goal request extracted from user input."""
    intent: str  # Logical, concise, environment-related intent
    original_message: str
    conversation_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    is_maintenance_goal: bool = False  # True for persistent goals


@dataclass
class AffordanceMatchRequest:
    """Request for matching affordances to a goal."""
    goal: GoalRequest
    requester_id: str
    max_affordances: int = 10
    prefer_signifiers: bool = True


@dataclass
class AffordanceMatchResponse:
    """Response containing matched affordances for a goal."""
    request_id: str
    affordances: List[Dict[str, Any]]
    signifiers: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
