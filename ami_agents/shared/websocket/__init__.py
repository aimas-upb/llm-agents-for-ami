"""
WebSocket communication utilities for manual mode dialog interface.

This package provides:
- Message type definitions for WebSocket communication
- Protocol utilities for serialization/deserialization
- WebSocket connection management helpers
"""

from .protocol import (
    WebSocketMessageType,
    WebSocketMessage,
    UserMessage,
    AssistantMessage,
    PlanProposal,
    PlanConfirmation,
    ExecutionStatus,
    SystemStatus,
    ClearSignifiers,
    validate_message,
    serialize_message,
    deserialize_message,
    create_error_response,
    create_info_response,
)

from .utils import (
    WebSocketConnectionManager,
    generate_thread_id,
    format_timestamp,
)

__all__ = [
    "WebSocketMessageType",
    "WebSocketMessage",
    "UserMessage",
    "AssistantMessage",
    "PlanProposal",
    "PlanConfirmation",
    "ExecutionStatus",
    "SystemStatus",
    "ClearSignifiers",
    "validate_message",
    "serialize_message",
    "deserialize_message",
    "create_error_response",
    "create_info_response",
    "WebSocketConnectionManager",
    "generate_thread_id",
    "format_timestamp",
]