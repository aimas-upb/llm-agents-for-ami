"""Goal planning status tracking helpers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import asyncio
import time


class PlanningPhase(Enum):
    """Phases of plan construction."""

    INITIATED = "initiated"
    GATHERING_REUSED_PLAN = "gathering_reused_plan"
    GENERATING_LOCAL_PLAN = "generating_local_plan"
    QUERYING_COMMUNITY = "querying_community"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_FAILURE = "completed_failure"


@dataclass
class GoalStatus:
    """Track planning status and related metadata for a single goal."""

    goal_id: str
    intents: List[str]
    workspace_id: Optional[str] = None
    phase: PlanningPhase = PlanningPhase.INITIATED
    created_at: float = field(default_factory=time.monotonic)
    last_updated: float = field(default_factory=time.monotonic)

    reused_plan: Optional[Dict[str, Any]] = None
    reused_plan_source: Optional[str] = None

    local_plan: Optional[Dict[str, Any]] = None
    local_plan_complete: bool = False

    best_plan: Optional[Dict[str, Any]] = None
    best_plan_source: Optional[str] = None

    relevant_communities: List[str] = field(default_factory=list)
    community_responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    community_expected_responses: int = 0
    continue_triggered: bool = False
    community_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    error: Optional[str] = None
    error_detail: Optional[str] = None

    def update_status(self, phase: Optional[PlanningPhase] = None, **kwargs) -> None:
        if phase is not None:
            self.phase = phase
        self.last_updated = time.monotonic()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
