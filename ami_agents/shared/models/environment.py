"""
Environment models for representing smart environment components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkspaceType(Enum):
    """Types of workspaces in the HMAS environment."""
    ROOT = "root"  # Top-level workspace (e.g., "home")
    FLOOR = "floor"  # Floor-level workspace
    AREA = "area"  # Area-level workspace (room)
    LOGICAL_AREA = "logical_area"  # Logical grouping (e.g., "kids corner")
    EXTERNAL_SERVICES = "external_services"  # Workspace for external services


class ArtifactType(Enum):
    """Types of artifacts in the environment."""
    PHYSICAL_DEVICE = "physical_device"
    VIRTUAL_DEVICE = "virtual_device"
    SERVICE = "service"


class ChangeEventType(Enum):
    """Types of environment change events."""
    ARTIFACT_ADDED = "artifact_added"
    ARTIFACT_REMOVED = "artifact_removed"
    CAPABILITY_CHANGED = "capability_changed"
    STATE_CHANGED = "state_changed"


@dataclass
class ThingDescription:
    """W3C WoT Thing Description representation."""
    id: str
    title: str
    description: str

    # Interaction affordances
    properties: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Links and forms
    links: List[Dict[str, Any]] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Artifact:
    """Represents an artifact in the HMAS environment."""
    artifact_id: str
    artifact_type: ArtifactType
    name: str
    workspace_id: str

    # Thing Description
    thing_description: ThingDescription

    # Current state
    current_state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp_added: datetime = field(default_factory=datetime.now)


@dataclass
class Workspace:
    """Represents a workspace in the HMAS environment."""
    workspace_id: str
    workspace_type: WorkspaceType
    name: str
    parent_workspace_id: Optional[str] = None

    # Contained artifacts
    artifacts: List[str] = field(default_factory=list)  # artifact IDs

    # Sub-workspaces
    sub_workspaces: List[str] = field(default_factory=list)  # workspace IDs

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Affordance:
    """Represents an affordance (capability) of an artifact."""
    affordance_id: str
    affordance_type: str  # property, action, event
    name: str
    description: str

    # Parent artifact
    artifact_id: str

    # Endpoint information
    uri: str

    # Schema information
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signifier:
    """
    Represents a usage experience (signifier) for an affordance.

    A signifier is a record of how an affordance was used for a specific
    intent in a given context.
    """
    signifier_id: str
    affordance_id: str
    artifact_id: str

    # Usage information
    intent: str  # What goal was this used for
    context: Dict[str, Any]  # Context of use (time, conditions, etc.)

    # Parameters used
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Success information
    was_successful: bool = True
    outcome: Optional[str] = None

    # Timestamps
    timestamp_created: datetime = field(default_factory=datetime.now)
    usage_count: int = 1


@dataclass
class ChangeEvent:
    """Represents a change event in the environment."""
    event_id: str
    event_type: ChangeEventType
    timestamp: datetime = field(default_factory=datetime.now)

    # Affected entities
    workspace_id: Optional[str] = None
    artifact_id: Optional[str] = None
    affordance_id: Optional[str] = None

    # Change details
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
