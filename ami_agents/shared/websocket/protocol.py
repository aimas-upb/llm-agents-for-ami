"""
WebSocket message protocol definitions for manual mode dialog interface.

Defines message types, data structures, and validation utilities for
bi-directional communication between the web dialog interface and
the AMI agent system.
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict


class WebSocketMessageType(str, Enum):
    """Enumeration of WebSocket message types."""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    PLAN_PROPOSAL = "plan_proposal"
    PLAN_CONFIRMATION = "plan_confirmation"
    EXECUTION_STATUS = "execution_status"
    SYSTEM_STATUS = "system_status"
    CLEAR_SIGNIFIERS = "clear_signifiers"


class ExecutionStatusLevel(str, Enum):
    """Enumeration of execution status levels."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SystemStatusLevel(str, Enum):
    """Enumeration of system status levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class UserMessage:
    """Message sent from client to server containing user input."""
    type: str = WebSocketMessageType.USER_MESSAGE.value
    content: str = ""
    thread_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class AssistantMessage:
    """Message sent from server to client containing assistant response."""
    type: str = WebSocketMessageType.ASSISTANT_MESSAGE.value
    content: str = ""
    thread_id: str = ""
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class PlanProposal:
    """Message sent from server to client containing a proposed plan."""
    type: str = WebSocketMessageType.PLAN_PROPOSAL.value
    content: str = ""  # Human-readable plan summary
    plan: Dict[str, Any] = None  # BT JSON structure
    thread_id: str = ""
    plan_id: str = ""
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.plan_id == "":
            self.plan_id = str(uuid.uuid4())
        if self.plan is None:
            self.plan = {}


@dataclass
class PlanConfirmation:
    """Message sent from client to server for plan approval/rejection."""
    type: str = WebSocketMessageType.PLAN_CONFIRMATION.value
    approved: bool = False
    thread_id: str = ""
    plan_id: str = ""
    message: Optional[str] = None  # Optional user comment
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionStatus:
    """Message sent from server to client with execution progress."""
    type: str = WebSocketMessageType.EXECUTION_STATUS.value
    content: str = ""
    status: str = ExecutionStatusLevel.RUNNING.value
    progress: Optional[int] = None  # 0-100 percentage
    thread_id: str = ""
    plan_id: str = ""
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.progress is not None:
            self.progress = max(0, min(100, int(self.progress)))


@dataclass
class SystemStatus:
    """Message sent from server to client with system information."""
    type: str = WebSocketMessageType.SYSTEM_STATUS.value
    content: str = ""
    level: str = SystemStatusLevel.INFO.value
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ClearSignifiers:
    """Message sent from client to server to clear signifier memory."""
    type: str = WebSocketMessageType.CLEAR_SIGNIFIERS.value
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# Type alias for all message types
WebSocketMessage = Union[
    UserMessage,
    AssistantMessage,
    PlanProposal,
    PlanConfirmation,
    ExecutionStatus,
    SystemStatus,
    ClearSignifiers,
]


def validate_message(data: Dict[str, Any]) -> bool:
    """
    Validate that a dictionary contains a valid WebSocket message.

    Args:
        data: Dictionary to validate

    Returns:
        True if the data represents a valid message type
    """
    if not isinstance(data, dict):
        return False

    message_type = data.get("type")
    if not message_type or message_type not in [t.value for t in WebSocketMessageType]:
        return False

    # Basic validation for required fields
    if message_type == WebSocketMessageType.USER_MESSAGE.value:
        return "content" in data
    elif message_type == WebSocketMessageType.ASSISTANT_MESSAGE.value:
        return "content" in data and "thread_id" in data
    elif message_type == WebSocketMessageType.PLAN_PROPOSAL.value:
        return all(field in data for field in ["content", "thread_id", "plan"])
    elif message_type == WebSocketMessageType.PLAN_CONFIRMATION.value:
        return all(field in data for field in ["approved", "thread_id", "plan_id"])
    elif message_type == WebSocketMessageType.EXECUTION_STATUS.value:
        return all(field in data for field in ["content", "status", "thread_id", "plan_id"])
    elif message_type == WebSocketMessageType.SYSTEM_STATUS.value:
        return "content" in data and "level" in data
    elif message_type == WebSocketMessageType.CLEAR_SIGNIFIERS.value:
        return True  # No required fields beyond type

    return False


def serialize_message(message: WebSocketMessage) -> str:
    """
    Serialize a WebSocket message to JSON string.

    Args:
        message: Message object to serialize

    Returns:
        JSON string representation of the message
    """
    try:
        return json.dumps(asdict(message))
    except Exception as e:
        raise ValueError(f"Failed to serialize message: {e}")


def deserialize_message(data: Union[str, Dict[str, Any]]) -> WebSocketMessage:
    """
    Deserialize a JSON string or dict to a WebSocket message object.

    Args:
        data: JSON string or dictionary to deserialize

    Returns:
        Appropriate message object based on type

    Raises:
        ValueError: If data is invalid or message type is unknown
    """
    if isinstance(data, str):
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    elif isinstance(data, dict):
        parsed_data = data
    else:
        raise ValueError("Data must be JSON string or dictionary")

    if not validate_message(parsed_data):
        raise ValueError("Invalid message format")

    message_type = parsed_data["type"]

    # Remove type field to avoid duplication in constructor
    message_data = {k: v for k, v in parsed_data.items() if k != "type"}

    if message_type == WebSocketMessageType.USER_MESSAGE.value:
        return UserMessage(**message_data)
    elif message_type == WebSocketMessageType.ASSISTANT_MESSAGE.value:
        return AssistantMessage(**message_data)
    elif message_type == WebSocketMessageType.PLAN_PROPOSAL.value:
        return PlanProposal(**message_data)
    elif message_type == WebSocketMessageType.PLAN_CONFIRMATION.value:
        return PlanConfirmation(**message_data)
    elif message_type == WebSocketMessageType.EXECUTION_STATUS.value:
        return ExecutionStatus(**message_data)
    elif message_type == WebSocketMessageType.SYSTEM_STATUS.value:
        return SystemStatus(**message_data)
    elif message_type == WebSocketMessageType.CLEAR_SIGNIFIERS.value:
        return ClearSignifiers(**message_data)
    else:
        raise ValueError(f"Unknown message type: {message_type}")


def create_error_response(error_message: str, thread_id: str = "") -> SystemStatus:
    """
    Create a standardized error response message.

    Args:
        error_message: Error description
        thread_id: Optional thread ID for context

    Returns:
        SystemStatus message with error level
    """
    return SystemStatus(
        content=error_message,
        level=SystemStatusLevel.ERROR.value,
    )


def create_info_response(info_message: str, thread_id: str = "") -> SystemStatus:
    """
    Create a standardized info response message.

    Args:
        info_message: Information description
        thread_id: Optional thread ID for context

    Returns:
        SystemStatus message with info level
    """
    return SystemStatus(
        content=info_message,
        level=SystemStatusLevel.INFO.value,
    )