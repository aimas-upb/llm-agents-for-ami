"""The signifier record: what a successful goal execution is remembered as.

A goal is remembered under an exact key built from its structure (the agents
build it: `ami_agents/shared/utils/goal_signifiers.py`). The record's content
depends on the kind of goal:

    explicit, incomplete,   the whole executed plan (`tree`): a goal that needed
    incomplete_text         an LLM is replayed on an exact match, without one

    ambiguous               the context the plan was made in (`context`: the
                            affordances acting on and reading each environment
                            variable, the properties those actions act upon, with
                            their values at execution start) and the affordances
                            the plan actually used (`used`) -- given back to the
                            planner as hints on an exact match

`shacl_shapes` may hold shapes over the stored context, for the SHACL context
validation (`src/validation`), which is kept but not used for matching yet.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SignifierKind = Literal["explicit", "incomplete", "incomplete_text", "ambiguous"]
SIGNIFIER_KINDS = ("explicit", "incomplete", "incomplete_text", "ambiguous")


def key_digest(kind: str, key: Dict[str, Any]) -> str:
    """The index of a key: the same kind and fields always give the same digest."""
    canonical = json.dumps([kind, key], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class GoalSignifier(BaseModel):
    """One remembered goal execution."""

    model_config = ConfigDict(extra="forbid")

    signifier_id: str = Field(..., min_length=1)
    kind: SignifierKind
    key: Dict[str, Any]
    key_digest: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    workspace_id: Optional[str] = None
    intent_text: Optional[str] = None
    # The goal as structured (GoalSpec wire dict), for reading the record back.
    goal: Dict[str, Any] = Field(default_factory=dict)

    # explicit / incomplete / incomplete_text: the plan to replay.
    tree: Optional[Dict[str, Any]] = None

    # ambiguous: where the plan was made, and what it used.
    context: Optional[Dict[str, Any]] = None
    used: Optional[Dict[str, Any]] = None

    # Optional SHACL shapes over the stored context (validation, not matching).
    shacl_shapes: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        # The digest always follows from the kind and key.
        self.key_digest = key_digest(self.kind, self.key)

    def to_json_doc(self) -> Dict[str, Any]:
        """A JSON-serialisable dict, for file storage and the wire."""
        return self.model_dump(mode="json")
