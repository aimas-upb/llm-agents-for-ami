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
    """
    Represents a plan as a behavior tree.

    Supports two representations:
    - Legacy: NodeTemplate-based tree (root_node)
    - BT JSON IR: Dict-based tree spec (tree_spec) used by IRExecutor
    """
    plan_id: str
    root_node: Optional[NodeTemplate] = None
    tree_spec: Optional[Dict[str, Any]] = None  # BT JSON IR dict
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json_ir(cls, plan_id: str, plan_dict: Dict[str, Any]) -> "BehaviorTreePlan":
        """Create a BehaviorTreePlan from a BT JSON IR plan dict."""
        return cls(
            plan_id=plan_id,
            tree_spec=plan_dict.get("tree"),
            metadata={
                "plan_type": plan_dict.get("plan_type", "behavior_tree"),
                "explanation": plan_dict.get("explanation"),
                "intents": plan_dict.get("intents", []),
                "signifier_reuse": plan_dict.get("signifier_reuse", False),
            },
        )

    def get_all_signifiers(self) -> List[Dict[str, Any]]:
        """
        Extract all signifiers (leaf action nodes) from the plan.

        Returns:
            List of signifiers with affordance usage information.
        """
        signifiers: List[Dict[str, Any]] = []
        tree = self.tree_spec or (self._node_to_dict(self.root_node) if self.root_node else None)
        if not tree:
            return signifiers
        self._collect_action_nodes(tree, signifiers)
        return signifiers

    def _collect_action_nodes(self, node: Dict[str, Any], result: List[Dict[str, Any]]) -> None:
        """Recursively collect action nodes from a BT JSON IR tree."""
        if not isinstance(node, dict):
            return
        if node.get("type") == "action":
            result.append({
                "name": node.get("name", ""),
                "action_url": node.get("action_url", ""),
                "parameters": node.get("parameters", {}),
            })
        for child in node.get("children", []):
            self._collect_action_nodes(child, result)

    @staticmethod
    def _node_to_dict(node: "NodeTemplate") -> Dict[str, Any]:
        """Convert a NodeTemplate to a dict for uniform traversal."""
        d: Dict[str, Any] = {
            "type": node.node_type,
            "name": node.name,
        }
        if node.affordance_uri:
            d["action_url"] = node.affordance_uri
        if node.action_params:
            d["parameters"] = node.action_params
        if node.children:
            d["children"] = [BehaviorTreePlan._node_to_dict(c) for c in node.children]
        return d

    def get_node_by_id(self, node_id: str) -> Optional[NodeTemplate]:
        """
        Retrieve a node by its ID (legacy NodeTemplate tree only).

        Args:
            node_id: The unique identifier of the node.

        Returns:
            The node template if found, None otherwise.
        """
        if not self.root_node:
            return None
        return self._find_node(self.root_node, node_id)

    @staticmethod
    def _find_node(node: "NodeTemplate", node_id: str) -> Optional["NodeTemplate"]:
        if node.node_id == node_id:
            return node
        for child in node.children:
            found = BehaviorTreePlan._find_node(child, node_id)
            if found:
                return found
        return None

    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """
        Update the status of a node (legacy NodeTemplate tree only).

        Args:
            node_id: The unique identifier of the node.
            status: The new status.

        Returns:
            True if updated successfully, False otherwise.
        """
        node = self.get_node_by_id(node_id)
        if node:
            node.status = status
            return True
        return False


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

    # Provenance: which user request produced this plan. `goal_description`
    # holds the user's own phrasing, so a running plan can always be traced
    # back to the utterance that asked for it, and explained in those words.
    request_id: Optional[str] = None

    # Content key, alongside `plan_id`'s instance key: the same plan text run
    # twice gives two ids and one hash.
    plan_hash: Optional[str] = None

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
