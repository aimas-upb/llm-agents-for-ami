"""
Who asked for what, and what is running because of it.

Two registries hang off the UserAssistant agent:

- `RequestRegistry` — one record per user utterance, from arrival to the point
  the request is answered or a plan is handed to plan management.
- `PlanRegistry` — one record per *execution* of a plan, which is what makes a
  running plan addressable: something to query the status of, cancel, or
  explain while it is still ticking.

Both are in-memory but persistence-shaped. Every mutation goes through a
method, and `to_jsonl()` is the seam: swapping to SQLite later is a change to
these two classes and nothing that calls them. The ids are non-opaque strings
(`req-…`, `plan-…`) rather than object references, because they travel as SPADE
message metadata and will eventually be foreign keys for the Experience Engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from ...shared.models.plan import Plan, PlanStatus, PlanType


@dataclass
class RequestRecord:
    """One user utterance and what became of it."""

    request_id: str
    conversation_id: str
    user_text: str

    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None

    # What segmentation made of it, and what it turned into.
    atomic_intents: List[Dict[str, Any]] = field(default_factory=list)
    plan_ids: List[str] = field(default_factory=list)
    outcome: Optional[str] = None  # answered | planned | rejected | failed

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


class RequestRegistry:
    """The open and recent user requests, keyed by request id."""

    def __init__(self) -> None:
        self._records: Dict[str, RequestRecord] = {}

    def open(self, request_id: str, conversation_id: str,
             user_text: str, **metadata: Any) -> RequestRecord:
        record = RequestRecord(
            request_id=request_id,
            conversation_id=conversation_id,
            user_text=user_text,
            metadata=dict(metadata),
        )
        self._records[request_id] = record
        return record

    def get(self, request_id: str) -> Optional[RequestRecord]:
        return self._records.get(request_id)

    def note_intents(self, request_id: str,
                     intents: List[Dict[str, Any]]) -> None:
        record = self._records.get(request_id)
        if record is not None:
            record.atomic_intents = list(intents)

    def note_plan(self, request_id: str, plan_id: str) -> None:
        """Link a plan to the request that produced it."""
        record = self._records.get(request_id)
        if record is not None and plan_id not in record.plan_ids:
            record.plan_ids.append(plan_id)

    def close(self, request_id: str, outcome: str) -> Optional[RequestRecord]:
        record = self._records.get(request_id)
        if record is not None and record.is_open:
            record.closed_at = datetime.now()
            record.outcome = outcome
        return record

    def open_requests(self) -> List[RequestRecord]:
        return [r for r in self._records.values() if r.is_open]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RequestRecord]:
        return iter(self._records.values())

    def to_jsonl(self) -> str:
        """The persistence seam — one JSON object per record."""
        return "\n".join(
            json.dumps(_jsonable(asdict(record)), default=str)
            for record in self._records.values()
        )


@dataclass
class PlanRecord:
    """One execution of one plan, and how far it has got."""

    plan_id: str
    plan: Plan

    # Tick accounting. `ticks` counts this run; `execution_count` counts how
    # many times a maintenance plan has re-armed and run again.
    ticks: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_status: Optional[str] = None
    error: Optional[str] = None

    @property
    def request_id(self) -> Optional[str]:
        return self.plan.request_id

    @property
    def goal_description(self) -> str:
        """The user's own words — the back-trace an explanation is built on."""
        return self.plan.goal_description

    @property
    def is_maintenance(self) -> bool:
        return self.plan.plan_type is PlanType.MAINTENANCE

    @property
    def is_active(self) -> bool:
        return self.plan.status in (PlanStatus.CREATED, PlanStatus.RUNNING,
                                    PlanStatus.PAUSED)

    def snapshot(self) -> Dict[str, Any]:
        """What a status query answers with."""
        return {
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "status": self.plan.status.value,
            "plan_type": self.plan.plan_type.value,
            "ticks": self.ticks,
            "execution_count": self.plan.execution_count,
            "goal_description": self.goal_description,
            "goal_intent": self.plan.goal_intent,
            "conversation_id": self.plan.conversation_id,
            "plan_hash": self.plan.plan_hash,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
        }


class PlanRegistry:
    """Every plan handed to plan management, running or finished."""

    def __init__(self) -> None:
        self._records: Dict[str, PlanRecord] = {}

    def register(self, plan: Plan) -> PlanRecord:
        record = PlanRecord(plan_id=plan.plan_id, plan=plan)
        self._records[plan.plan_id] = record
        return record

    def get(self, plan_id: str) -> Optional[PlanRecord]:
        return self._records.get(plan_id)

    def mark_started(self, plan_id: str) -> None:
        record = self._records.get(plan_id)
        if record is None:
            return
        record.started_at = record.started_at or datetime.now()
        record.plan.status = PlanStatus.RUNNING

    def mark_waiting(self, plan_id: str) -> None:
        """A maintenance plan between bursts: still live, not currently ticking."""
        record = self._records.get(plan_id)
        if record is None:
            return
        record.plan.status = PlanStatus.PAUSED
        record.ticks = 0

    def bump_tick(self, plan_id: str, status: Optional[str] = None) -> int:
        """Count one tick. Returns the new tick count."""
        record = self._records.get(plan_id)
        if record is None:
            return 0
        record.ticks += 1
        if status is not None:
            record.last_status = status
        return record.ticks

    def bump_execution_count(self, plan_id: str) -> int:
        """A maintenance plan completed a round and re-armed."""
        record = self._records.get(plan_id)
        if record is None:
            return 0
        record.plan.execution_count += 1
        record.plan.timestamp_last_executed = datetime.now()
        record.ticks = 0
        return record.plan.execution_count

    def finish(self, plan_id: str, status: PlanStatus,
               error: Optional[str] = None) -> Optional[PlanRecord]:
        record = self._records.get(plan_id)
        if record is None:
            return None
        record.plan.status = status
        record.finished_at = datetime.now()
        record.error = error
        if status is PlanStatus.COMPLETED:
            record.plan.execution_count += 1
            record.plan.timestamp_last_executed = record.finished_at
        return record

    def active(self) -> List[PlanRecord]:
        """What is running right now — the answer to "what are you doing?"."""
        return [r for r in self._records.values() if r.is_active]

    def for_request(self, request_id: str) -> List[PlanRecord]:
        return [r for r in self._records.values() if r.request_id == request_id]

    def for_conversation(self, conversation_id: str) -> List[PlanRecord]:
        return [r for r in self._records.values()
                if r.plan.conversation_id == conversation_id]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[PlanRecord]:
        return iter(self._records.values())

    def to_jsonl(self) -> str:
        """The persistence seam — one JSON object per record."""
        return "\n".join(json.dumps(record.snapshot(), default=str)
                         for record in self._records.values())


def _jsonable(value: Any) -> Any:
    """Datetimes and enums out of a dataclass dump, so json.dumps succeeds."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value
    return value
