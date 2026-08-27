"""
Environment models for representing smart environment components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkspaceCategory(Enum):
    """Categories of workspaces in the HMAS environment."""
    ROOT = "root"  # Top-level workspace (e.g., "home")
    FLOOR = "floor"  # Floor-level workspace
    AREA = "area"  # Area-level workspace (room)
    LOGICAL_AREA = "logical_area"  # Logical grouping (e.g., "kids corner")
    EXTERNAL_SERVICES = "external_services"  # Workspace for external services


class ArtifactCategory(Enum):
    """Categories of artifacts in the environment."""
    PHYSICAL_DEVICE = "physical_device"
    VIRTUAL_DEVICE = "virtual_device"
    SERVICE = "service"


class ChangeEventType(Enum):
    """Types of environment change events."""
    ARTIFACT_ADDED = "artifact_added"
    ARTIFACT_REMOVED = "artifact_removed"
    CAPABILITY_CHANGED = "capability_changed"
    STATE_CHANGED = "state_changed"


class AffordanceType(Enum):
    """Types of affordances for artifacts."""
    PROPERTY = "property"
    ACTION = "action"
    EVENT = "event"


@dataclass
class ThingDescription:
    """W3C WoT Thing Description representation."""
    id: str
    title: str
    description: str

    # RDF serialization - this will always hold the full TD in RDF/Turtle format 
    # as obtained from the Integration Engine
    rdf: str

    # Interaction affordances
    properties: List[Dict[str, "Affordance"]] = field(default_factory=list)
    actions: List[Dict[str, "Affordance"]] = field(default_factory=list)
    events: List[Dict[str, "Affordance"]] = field(default_factory=list)

    # Additional metadata will include other TD fields related to security, context, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """Represents an artifact in the HMAS environment."""
    artifact_id: str
    artifact_type: ArtifactCategory
    name: str
    workspace_id: str

    # Thing Description
    thing_description: ThingDescription

    # Current state
    current_state: Dict[str, Any] = field(default_factory=dict)

    # Semantic types extracted from RDF (e.g., "ex:Light", "ex:Blinds")
    semantic_types: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp_added: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Workspace:
    """Represents a workspace in the HMAS environment."""
    workspace_id: str
    workspace_type: WorkspaceCategory
    name: str
    parent_workspace_id: Optional[str] = None

    # RDF representation of the workspace - this will always hold the full RDF/Turtle serialization
    # as obtained from the Integration Engine
    rdf: str = None

    # Semantic types extracted from RDF (e.g., "ex:Kitchen", "ex:LivingRoom")
    semantic_types: List[str] = field(default_factory=list)

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
    affordance_type: AffordanceType  # property, action, event
    name: str
    description: str

    # Parent artifact
    artifact_id: str

    # RDF serialization - this will always hold the full affordance description in RDF/Turtle format 
    # as obtained from the Integration Engine
    rdf: str

    # form details: includes protocol, method type, content type, and target URI
    form: "AffordanceForm" = field(default_factory="AffordanceForm")

    # Semantic types
    semantic_types: List[str] = field(default_factory=list)  # List of RDF types (IRIs)

    # Schema information
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AffordanceForm:
    """Represents the form details of an affordance."""
    href: str  # The URI of the affordance
    content_type: Optional[str] = None  # e.g., application/json
    method: Optional[str] = None  # e.g., GET, POST
    operation_type: Optional[str] = None  # e.g., td:observeProperty, td:invokeAction, td:subscribeEvent
    additional_fields: Dict[str, Any] = field(default_factory=dict)  # Any other form fields


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
