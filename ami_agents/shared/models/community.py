import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime


class AffordanceMatch:
    def __init__(self, description: str):
        self.description = description

    def matches_context(self, context: Dict[str, Any]) -> float:
        return 1.0


class Community:
    def __init__(
        self,
        affordance_match: AffordanceMatch,
        community_id: Optional[str] = None,
        member_ids: Optional[List[str]] = None,
        created_at: Optional[datetime] = None,
    ):
        self.affordance_match = affordance_match
        self.community_id = community_id if community_id else uuid.uuid4().hex[:8]
        self.member_ids = member_ids if member_ids is not None else []
        self.created_at = created_at if created_at else datetime.now()

    @property
    def member_count(self) -> int:
        return len(self.member_ids)

    def add_member(self, agent_id: str) -> bool:
        if agent_id in self.member_ids:
            return False
        self.member_ids.append(agent_id)
        return True

    def remove_member(self, agent_id: str) -> bool:
        if agent_id not in self.member_ids:
            return False
        self.member_ids.remove(agent_id)
        return True

    def has_member(self, agent_id: str) -> bool:
        return agent_id in self.member_ids

    def matches_context(self, context: Dict[str, Any]) -> float:
        if self.affordance_match:
            return self.affordance_match.matches_context(context)
        return 0.0

