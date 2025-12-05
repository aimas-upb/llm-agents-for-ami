"""Simplified API routes for signifier operations.

This module provides simplified endpoints:
- GET /signifiers - List all signifiers
- POST /signifiers - Create signifier from RDF
- DELETE /signifiers - Delete all signifiers (clear memory)
- GET /signifiers/match - Match query with intent and context
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.config import get_settings
from src.matching.registry import IntentMatcherRegistry
from src.storage.registry import SignifierRegistry
from src.validation.context_builder import ContextGraphBuilder
from src.validation.shacl_validator import SHACLValidator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["simple-signifiers"])

settings = get_settings()
registry = SignifierRegistry(
    storage_dir=settings.storage_dir,
    enable_authoring_validation=False,
)
matcher_registry = IntentMatcherRegistry(default_version="v1")
context_builder = ContextGraphBuilder()
shacl_validator = SHACLValidator(enable_caching=False)


class SignifierResponse(BaseModel):
    """Response model for a single signifier.

    Args:
        signifier_id: Unique identifier for the signifier
        version: Version number
        status: Active or deprecated
        intent: Natural language intent description
        affordance_uri: URI of the signified affordance
    """

    signifier_id: str
    version: int
    status: str
    intent: str
    affordance_uri: str


class SignifierListResponse(BaseModel):
    """Response model for list of signifiers.

    Args:
        signifiers: List of signifiers
        total: Total count
    """

    signifiers: List[SignifierResponse]
    total: int


class CreateSignifierRequest(BaseModel):
    """Request to create a signifier from RDF data.

    Args:
        rdf_data: RDF data in Turtle format
    """

    rdf_data: str = Field(..., min_length=1, description="RDF data in Turtle format")


class CreateSignifierResponse(BaseModel):
    """Response after creating a signifier.

    Args:
        signifier_id: Created signifier ID
        message: Success message
    """

    signifier_id: str
    message: str


class DeleteAllResponse(BaseModel):
    """Response after deleting all signifiers.

    Args:
        success: Whether operation succeeded
        message: Status message
        deleted_count: Number of signifiers deleted
    """

    success: bool
    message: str
    deleted_count: int


class MatchQuery(BaseModel):
    """Query for matching signifiers.

    Args:
        intent: Natural language intent query
        context: Context conditions (artifact URIs with properties and values)
    """

    intent: str = Field(..., min_length=1, description="Natural language intent")
    context: Dict[str, Dict[str, Any]] = Field(
        default={},
        description="Context as nested dict: {artifact_uri: {property_uri: value}}"
    )


class MatchResult(BaseModel):
    """Single match result.

    Args:
        signifier_id: Matched signifier ID
        intent_similarity: Intent matching similarity score (0.0 to 1.0)
        shacl_conforms: Whether SHACL validation passed
        shacl_violations: List of SHACL violation messages
    """

    signifier_id: str
    intent_similarity: float
    shacl_conforms: bool
    shacl_violations: List[str]


class MatchResponse(BaseModel):
    """Response for match query.

    Args:
        matches: List of all matches (including those that failed SHACL)
        final_matches: List of signifier IDs that passed both intent and SHACL
        total_signifiers: Total signifiers in memory
    """

    matches: List[MatchResult]
    final_matches: List[str]
    total_signifiers: int


@router.get("/signifiers", response_model=SignifierListResponse)
async def list_all_signifiers() -> SignifierListResponse:
    """Get list of all signifiers in memory.

    Returns:
        List of all signifiers with basic information
    """
    try:
        all_signifiers = registry.list_signifiers(limit=10000)

        signifier_list = [
            SignifierResponse(
                signifier_id=s.signifier_id,
                version=s.version,
                status=s.status.value,
                intent=s.intent.nl_text or "",
                affordance_uri=s.affordance_uri,
            )
            for s in all_signifiers
        ]

        logger.info(f"Listed {len(signifier_list)} signifiers")

        return SignifierListResponse(
            signifiers=signifier_list,
            total=len(signifier_list)
        )

    except Exception as e:
        logger.error(f"Error listing signifiers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list signifiers: {str(e)}"
        )


@router.post("/signifiers", response_model=CreateSignifierResponse, status_code=status.HTTP_201_CREATED)
async def create_signifier_from_rdf(request: CreateSignifierRequest) -> CreateSignifierResponse:
    """Create a signifier from RDF (Turtle) data.

    Args:
        request: Request containing RDF data

    Returns:
        Created signifier ID and message

    Raises:
        HTTPException: If RDF parsing or creation fails
    """
    try:
        signifier = registry.create_from_rdf(request.rdf_data, format="turtle")

        logger.info(f"Created signifier: {signifier.signifier_id}")

        return CreateSignifierResponse(
            signifier_id=signifier.signifier_id,
            message=f"Signifier {signifier.signifier_id} created successfully"
        )

    except ValueError as e:
        logger.error(f"Failed to create signifier from RDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid RDF data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating signifier: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create signifier: {str(e)}"
        )


@router.delete("/signifiers", response_model=DeleteAllResponse)
async def delete_all_signifiers() -> DeleteAllResponse:
    """Delete all signifiers from memory (clear storage).

    Returns:
        Success status and count of deleted signifiers
    """
    global registry

    try:
        all_signifiers = registry.list_signifiers(limit=10000)
        deleted_count = len(all_signifiers)

        storage_dir = Path(settings.storage_dir)
        if storage_dir.exists():
            for subdir in ["rdf", "json", "indexes"]:
                subdir_path = storage_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)

        registry = SignifierRegistry(
            storage_dir=settings.storage_dir,
            enable_authoring_validation=False,
        )

        logger.info(f"Deleted all signifiers (count: {deleted_count})")

        return DeleteAllResponse(
            success=True,
            message=f"Successfully deleted all signifiers from memory",
            deleted_count=deleted_count
        )

    except Exception as e:
        logger.error(f"Error deleting all signifiers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete signifiers: {str(e)}"
        )


@router.get("/signifiers/match", response_model=MatchResponse)
async def match_signifiers(
    intent: str = Query(..., description="Natural language intent query"),
    context: Optional[str] = Query(None, description="JSON string of context (optional)")
) -> MatchResponse:
    """Match signifiers based on intent and context.

    This endpoint performs two-phase matching:
    1. Intent matching using semantic similarity
    2. SHACL validation against context constraints

    Args:
        intent: Natural language intent query
        context: Optional JSON string of context data

    Returns:
        Matching results with similarity scores and validation status
    """
    import json

    try:
        context_dict = {}
        if context:
            try:
                context_dict = json.loads(context)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON in context parameter: {str(e)}"
                )

        all_signifiers = registry.list_signifiers(limit=10000)
        signifier_dicts = [s.model_dump() for s in all_signifiers]

        logger.info(f"Matching intent: '{intent}' against {len(signifier_dicts)} signifiers")

        match_results = matcher_registry.match(
            intent_query=intent,
            signifiers=signifier_dicts,
            k=10,
            version="v1"
        )

        context_graph, _ = context_builder.normalize_context(context_dict)

        matches = []
        for match in match_results:
            signifier = registry.get(match.signifier_id)
            if not signifier:
                continue

            shacl_result = {"conforms": True, "violations": []}

            if signifier.context.shacl_shapes:
                validation = shacl_validator.validate_signifier_context(
                    context_graph,
                    signifier.context.shacl_shapes,
                    format="turtle"
                )
                shacl_result = {
                    "conforms": validation.conforms,
                    "violations": [v.message for v in validation.violations]
                }

            match_info = MatchResult(
                signifier_id=match.signifier_id,
                intent_similarity=round(match.similarity, 4),
                shacl_conforms=shacl_result["conforms"],
                shacl_violations=shacl_result["violations"]
            )
            matches.append(match_info)

        final_matches = [m.signifier_id for m in matches if m.shacl_conforms]

        logger.info(f"Found {len(matches)} intent matches, {len(final_matches)} passed SHACL")

        return MatchResponse(
            matches=matches,
            final_matches=final_matches,
            total_signifiers=len(signifier_dicts)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during matching: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}"
        )
