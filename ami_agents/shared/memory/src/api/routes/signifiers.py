"""The memory API: record goal signifiers, match them by exact key, list them.

Served as the community memory (`ami_agents/shared/community/community_client.py`
talks to it). The same `MemoryStore` the EnvExplorer keeps locally, behind HTTP:

    POST   /signifiers        record one (a GoalSignifier dict)
    GET    /signifiers        every record, newest first
    DELETE /signifiers        forget everything
    POST   /matching/match    {kind, key} -> {match: record | null}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.config import get_settings
from src.models.signifier import SIGNIFIER_KINDS
from src.storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["signifiers"])

_store: Optional[MemoryStore] = None


def store() -> MemoryStore:
    """The service's store, opened on first use under the configured storage dir."""
    global _store
    if _store is None:
        _store = MemoryStore(get_settings().storage_dir)
    return _store


class MatchRequest(BaseModel):
    kind: str
    key: Dict[str, Any]


class MatchResponse(BaseModel):
    match: Optional[Dict[str, Any]] = None


class ListResponse(BaseModel):
    signifiers: List[Dict[str, Any]]
    total_count: int


@router.post("/signifiers", status_code=status.HTTP_201_CREATED)
async def record_signifier(signifier: Dict[str, Any]) -> Dict[str, Any]:
    """Record one goal signifier; it becomes its key's answer."""
    try:
        record = store().record(signifier)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    logger.info("Recorded %s signifier %s", record.kind, record.signifier_id)
    return {"ok": True, "signifier_id": record.signifier_id, "key_digest": record.key_digest}


@router.get("/signifiers", response_model=ListResponse)
async def list_signifiers() -> ListResponse:
    records = [r.to_json_doc() for r in store().list()]
    return ListResponse(signifiers=records, total_count=len(records))


@router.delete("/signifiers")
async def delete_signifiers() -> Dict[str, Any]:
    return {"ok": True, "deleted_count": store().clear()}


@router.post("/matching/match", response_model=MatchResponse)
async def match_signifier(request: MatchRequest) -> MatchResponse:
    """The record stored under exactly this kind and key, if any."""
    if request.kind not in SIGNIFIER_KINDS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"unknown kind {request.kind!r}")
    record = store().match(request.kind, request.key)
    return MatchResponse(match=record.to_json_doc() if record else None)
