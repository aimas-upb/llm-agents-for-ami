"""API routes for intent matching operations.

This module implements the Phase 3 Intent Matching APIs:
- POST /match/intent (match intent query)
- GET /match/versions (list matcher versions)
- GET /match/info (get matcher information)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.matching import IntentMatcherRegistry
from src.storage.registry import SignifierRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/match", tags=["intent-matching"])

settings = get_settings()
matcher_registry = IntentMatcherRegistry(default_version="v0")
signifier_registry = SignifierRegistry(storage_dir=settings.storage_dir)


class MatchIntentRequest(BaseModel):
    """Request for intent matching.

    Args:
        intent_query: Natural language intent query
        k: Number of top results to return
        version: Matcher version to use (optional)
        parameters: Algorithm-specific parameters
    """

    intent_query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    version: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class MatchResultResponse(BaseModel):
    """Single match result.

    Args:
        signifier_id: Signifier identifier
        similarity: Similarity score (0.0 to 1.0)
        metadata: Optional metadata
    """

    signifier_id: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None


class MatchIntentResponse(BaseModel):
    """Response for intent matching.

    Args:
        results: List of match results
        total_signifiers: Total signifiers in database
        matcher_version: Version of matcher used
        query: Original query
    """

    results: List[MatchResultResponse]
    total_signifiers: int
    matcher_version: str
    query: str


class MatcherInfoResponse(BaseModel):
    """Information about a matcher version.

    Args:
        version: Version identifier
        name: Matcher name
        description: Description
        parameters: Available parameters
        latency_budget_ms: Latency budget
    """

    version: str
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    latency_budget_ms: Optional[int] = None


class MatcherVersionsResponse(BaseModel):
    """List of available matcher versions.

    Args:
        versions: List of version identifiers
        default_version: Default version
        matchers: Information about each matcher
    """

    versions: List[str]
    default_version: str
    matchers: Dict[str, MatcherInfoResponse]


@router.post("/intent", response_model=MatchIntentResponse, status_code=status.HTTP_200_OK)
async def match_intent(request: MatchIntentRequest) -> MatchIntentResponse:
    """Match intent query against signifiers.

    Args:
        request: Intent matching request

    Returns:
        Matching results with similarity scores

    Raises:
        HTTPException: If matching fails
    """
    try:
        all_signifiers = signifier_registry.list_signifiers(limit=10000)

        signifier_dicts = [s.model_dump() for s in all_signifiers]

        params = request.parameters or {}
        results = matcher_registry.match(
            intent_query=request.intent_query,
            signifiers=signifier_dicts,
            k=request.k,
            version=request.version,
            **params,
        )

        matcher_version = (
            request.version or matcher_registry.get_default_version()
        )

        logger.info(
            f"Intent matching: query='{request.intent_query}', "
            f"version={matcher_version}, results={len(results)}"
        )

        return MatchIntentResponse(
            results=[
                MatchResultResponse(
                    signifier_id=r.signifier_id,
                    similarity=r.similarity,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total_signifiers=len(all_signifiers),
            matcher_version=matcher_version,
            query=request.intent_query,
        )

    except ValueError as e:
        logger.error(f"Intent matching failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in intent matching: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/versions", response_model=MatcherVersionsResponse)
async def list_matcher_versions() -> MatcherVersionsResponse:
    """List all available matcher versions.

    Returns:
        List of matcher versions and their information
    """
    versions = matcher_registry.list_versions()
    default_version = matcher_registry.get_default_version()
    all_info = matcher_registry.get_all_info()

    matchers_info = {}
    for version, info in all_info.items():
        matchers_info[version] = MatcherInfoResponse(
            version=info.get("version", version),
            name=info.get("name", "Unknown"),
            description=info.get("description", ""),
            parameters=info.get("parameters"),
            latency_budget_ms=info.get("latency_budget_ms"),
        )

    logger.debug(f"Listed {len(versions)} matcher versions")

    return MatcherVersionsResponse(
        versions=versions,
        default_version=default_version,
        matchers=matchers_info,
    )


@router.get("/info/{version}", response_model=MatcherInfoResponse)
async def get_matcher_info(version: str) -> MatcherInfoResponse:
    """Get information about a specific matcher version.

    Args:
        version: Matcher version identifier

    Returns:
        Matcher information

    Raises:
        HTTPException: If version not found
    """
    try:
        matcher = matcher_registry.get_matcher(version)
        info = matcher.get_info()

        return MatcherInfoResponse(
            version=info.get("version", version),
            name=info.get("name", "Unknown"),
            description=info.get("description", ""),
            parameters=info.get("parameters"),
            latency_budget_ms=info.get("latency_budget_ms"),
        )

    except ValueError as e:
        logger.error(f"Matcher version not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/default-version")
async def set_default_matcher_version(
    version: str = Query(..., description="Version to set as default")
) -> dict:
    """Set the default matcher version.

    Args:
        version: Version identifier

    Returns:
        Success message

    Raises:
        HTTPException: If version is invalid
    """
    try:
        matcher_registry.set_default_version(version)
        logger.info(f"Set default matcher version to: {version}")

        return {
            "message": f"Default matcher version set to {version}",
            "version": version,
        }

    except ValueError as e:
        logger.error(f"Failed to set default version: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
