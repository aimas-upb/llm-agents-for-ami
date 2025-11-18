"""
Plan models for representing goals and execution plans.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class PlanType(Enum):
    """Types of plans in the system."""
    IMMEDIATE = "immediate"  # One-time execution
    MAINTENANCE = "maintenance"  # Persistent, triggered goals


class PlanStatus(Enum):
    """Status of plan execution."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """Status of individual nodes in a behavior tree."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class NodeTemplate:
    """Template for a node in a behavior tree plan."""
    node_id: str
    node_type: str  # action, condition, sequence, selector, parallel, etc.
    name: str
    description: str

    # Status tracking
    status: NodeStatus = NodeStatus.PENDING

    # Execution metadata
    expected_duration: Optional[float] = None  # seconds
    actual_duration: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

    # Action-specific (for leaf nodes)
    affordance_uri: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)

    # Children (for composite nodes)
    children: List['NodeTemplate'] = field(default_factory=list)

    # Error handling
    error_message: Optional[str] = None
    fallback_node: Optional[str] = None


@dataclass
class BehaviorTreePlan:
    """Represents a plan as a behavior tree."""
    plan_id: str
    root_node: NodeTemplate
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_signifiers(self) -> List[Dict[str, Any]]:
        """
        Extract all signifiers (leaf action nodes) from the plan.

        Returns:
            List of signifiers with affordance usage information.
        """
        # TODO: Implement recursive extraction of leaf action nodes
        pass

    def get_node_by_id(self, node_id: str) -> Optional[NodeTemplate]:
        """
        Retrieve a node by its ID.

        Args:
            node_id: The unique identifier of the node.

        Returns:
            The node template if found, None otherwise.
        """
        # TODO: Implement recursive node search
        pass

    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """
        Update the status of a node.

        Args:
            node_id: The unique identifier of the node.
            status: The new status.

        Returns:
            True if updated successfully, False otherwise.
        """
        # TODO: Implement node status update
        pass


@dataclass
class Plan:
    """High-level plan representation with metadata."""
    plan_id: str
    plan_type: PlanType
    status: PlanStatus

    # Goal information
    goal_description: str  # Natural language phrasing
    goal_intent: str  # Concise logical intent

    # Conversation context
    conversation_id: str

    # Execution plan
    behavior_tree: BehaviorTreePlan

    # Timestamps
    timestamp_created: datetime = field(default_factory=datetime.now)
    timestamp_last_executed: Optional[datetime] = None
    execution_count: int = 0

    # Triggering conditions (for maintenance goals)
    triggering_conditions: List[Dict[str, Any]] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_human_readable_summary(self) -> str:
        """
        Generate a human-readable summary of the plan.

        Returns:
            A natural language description of the plan.
        """
        # TODO: Use LLM to generate summary from behavior tree
        pass

    def is_triggered(self, environment_state: Dict[str, Any]) -> bool:
        """
        Check if the plan's triggering conditions are met.

        Args:
            environment_state: Current state of the environment.

        Returns:
            True if conditions are met, False otherwise.
        """
        # TODO: Evaluate triggering conditions against environment state
        pass

    def matches_intent(self, intent: str, context: Dict[str, Any],
                      similarity_threshold: float = 0.85) -> bool:
        """
        Check if this plan matches a given intent and context.

        Args:
            intent: The intent to match against.
            context: The context to match against.
            similarity_threshold: Minimum similarity score for a match.

        Returns:
            True if the plan matches, False otherwise.
        """
        # TODO: Implement intent and context matching using LLM embeddings
        pass
