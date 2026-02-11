"""Signifier data models (CASHMERE-inspired).

These models are used by the RD4 Signifier System to:
- parse and normalize signifiers from RDF (Turtle)
- store signifiers as JSON documents
- support intent matching and context validation (SHACL)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field


class SignifierStatus(str, Enum):
    """Lifecycle status for a signifier."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"


class ValueCondition(BaseModel):
    """A single value constraint for a context property."""

    model_config = ConfigDict(extra="forbid")

    operator: str = Field(default="equals", min_length=1)
    value: Any = None
    datatype: Optional[str] = None


class StructuredCondition(BaseModel):
    """A structured context condition (artifact + property + constraints)."""

    model_config = ConfigDict(extra="forbid")

    artifact: str = Field(default="", min_length=1)
    property_affordance: str = Field(default="", min_length=1)
    value_conditions: List[ValueCondition] = Field(default_factory=list)


class IntentContext(BaseModel):
    """Recommended context for successful affordance usage."""

    model_config = ConfigDict(extra="forbid")

    nl_description: Optional[str] = None
    structured_conditions: List[StructuredCondition] = Field(default_factory=list)
    shacl_shapes: Optional[str] = None


class IntentionDescription(BaseModel):
    """Natural language + optional structured intent representation."""

    model_config = ConfigDict(extra="forbid")

    nl_text: str = Field(default="", min_length=1)
    structured: Optional[Dict[str, Any]] = None


class Provenance(BaseModel):
    """Provenance metadata for a signifier."""

    model_config = ConfigDict(extra="forbid")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(default="system", min_length=1)
    source: Optional[str] = None


class Signifier(BaseModel):
    """Canonical signifier record stored in the RD4 memory store."""

    model_config = ConfigDict(extra="forbid")

    signifier_id: str = Field(..., min_length=1)
    version: int = Field(default=1, ge=1)
    status: SignifierStatus = SignifierStatus.ACTIVE

    intent: IntentionDescription
    context: IntentContext

    affordance_uri: str = Field(..., min_length=1)
    intent_type: Optional[str] = Field(default=None, pattern="^(EXPLICIT|IMPLICIT)$")
    provenance: Optional[Provenance] = None

    def to_json_doc(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict suitable for file storage."""

        return self.model_dump(mode="json")

    def get_property_keys(self) -> Set[Tuple[str, str]]:
        """Return (artifact_uri, property_uri) keys referenced by this signifier."""

        keys: Set[Tuple[str, str]] = set()
        for condition in self.context.structured_conditions:
            if condition.artifact and condition.property_affordance:
                keys.add((condition.artifact, condition.property_affordance))
        return keys

