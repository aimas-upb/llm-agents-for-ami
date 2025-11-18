"""
Message models for inter-agent communication.
Defines the message types and structures used across the AMI agent system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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
    PLAN_CANCEL_REQUEST = "plan_cancel_request"
    PLAN_ALTER_REQUEST = "plan_alter_request"
    PLAN_REPEAT_REQUEST = "plan_repeat_request"
    PLAN_IMPACT_NOTIFICATION = "plan_impact_notification"

    # Preference management
    PREFERENCE_STATEMENT = "preference_statement"


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
