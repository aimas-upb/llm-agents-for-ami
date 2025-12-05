"""API routes for signifier storage operations.

This module implements the Phase 1 Storage APIs:
- POST /signifiers (create)
- PUT /signifiers/{id} (update)
- GET /signifiers/{id} (retrieve)
- GET /signifiers (list with filters)
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.config import get_settings
from src.models.signifier import Signifier, SignifierStatus
from src.storage.registry import SignifierRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/signifiers", tags=["signifiers"])

settings = get_settings()
registry = SignifierRegistry(
    storage_dir=settings.storage_dir,
    enable_authoring_validation=settings.enable_authoring_validation,
)


class CreateSignifierRequest(BaseModel):
    """Request body for creating a signifier.

    Args:
        signifier: Signifier data
        rdf_data: Optional RDF representation
    """

    signifier: Signifier
    rdf_data: Optional[str] = None


class CreateFromRDFRequest(BaseModel):
    """Request body for creating signifier from RDF.

    Args:
        rdf_data: RDF data in Turtle format
        format: RDF serialization format
    """

    rdf_data: str = Field(..., min_length=1)
    format: str = Field(default="turtle")


class UpdateSignifierRequest(BaseModel):
    """Request body for updating a signifier.

    Args:
        signifier: Updated signifier data
        create_new_version: Whether to create a new version
    """

    signifier: Signifier
    create_new_version: bool = Field(default=False)


class SignifierResponse(BaseModel):
    """Response containing a single signifier.

    Args:
        signifier: Signifier data
        message: Optional message
    """

    signifier: Signifier
    message: Optional[str] = None


class SignifierListResponse(BaseModel):
    """Response containing list of signifiers.

    Args:
        signifiers: List of signifiers
        total: Total count before pagination
        limit: Page limit
        offset: Page offset
    """

    signifiers: List[Signifier]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    """Response for delete operation.

    Args:
        success: Whether deletion succeeded
        message: Status message
    """

    success: bool
    message: str


@router.post("", response_model=SignifierResponse, status_code=status.HTTP_201_CREATED)
async def create_signifier(request: CreateSignifierRequest) -> SignifierResponse:
    """Create a new signifier.

    Args:
        request: Create signifier request

    Returns:
        Created signifier

    Raises:
        HTTPException: If signifier already exists or validation fails
    """
    try:
        signifier = registry.create(request.signifier, request.rdf_data)
        logger.info(f"Created signifier via API: {signifier.signifier_id}")
        return SignifierResponse(
            signifier=signifier,
            message=f"Signifier {signifier.signifier_id} created successfully",
        )
    except ValueError as e:
        logger.error(f"Failed to create signifier: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error creating signifier: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/from-rdf", response_model=SignifierResponse, status_code=status.HTTP_201_CREATED)
async def create_signifier_from_rdf(
    request: CreateFromRDFRequest,
) -> SignifierResponse:
    """Create signifier from RDF data.

    Args:
        request: RDF data request

    Returns:
        Created signifier

    Raises:
        HTTPException: If RDF parsing or creation fails
    """
    try:
        signifier = registry.create_from_rdf(request.rdf_data, request.format)
        logger.info(f"Created signifier from RDF via API: {signifier.signifier_id}")
        return SignifierResponse(
            signifier=signifier,
            message=f"Signifier {signifier.signifier_id} created from RDF",
        )
    except ValueError as e:
        logger.error(f"Failed to create signifier from RDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error creating signifier from RDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/{signifier_id}", response_model=SignifierResponse)
async def get_signifier(signifier_id: str) -> SignifierResponse:
    """Retrieve a signifier by ID.

    Args:
        signifier_id: Signifier identifier

    Returns:
        Signifier data

    Raises:
        HTTPException: If signifier not found
    """
    signifier = registry.get(signifier_id)
    if not signifier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signifier {signifier_id} not found",
        )

    logger.debug(f"Retrieved signifier via API: {signifier_id}")
    return SignifierResponse(signifier=signifier)


@router.get("/{signifier_id}/rdf")
async def get_signifier_rdf(
    signifier_id: str,
    version: Optional[int] = Query(None, description="Specific version to retrieve"),
) -> dict:
    """Retrieve RDF representation of a signifier.

    Args:
        signifier_id: Signifier identifier
        version: Optional specific version

    Returns:
        RDF data in Turtle format

    Raises:
        HTTPException: If signifier or RDF not found
    """
    rdf_data = registry.get_rdf_representation(signifier_id, version)
    if not rdf_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RDF for signifier {signifier_id} not found",
        )

    logger.debug(f"Retrieved RDF via API: {signifier_id}")
    return {"signifier_id": signifier_id, "version": version, "rdf_data": rdf_data}


@router.put("/{signifier_id}", response_model=SignifierResponse)
async def update_signifier(
    signifier_id: str, request: UpdateSignifierRequest
) -> SignifierResponse:
    """Update an existing signifier.

    Args:
        signifier_id: Signifier identifier
        request: Update request

    Returns:
        Updated signifier

    Raises:
        HTTPException: If signifier not found or update fails
    """
    if request.signifier.signifier_id != signifier_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signifier ID mismatch",
        )

    try:
        signifier = registry.update(request.signifier, request.create_new_version)
        logger.info(f"Updated signifier via API: {signifier_id}")
        return SignifierResponse(
            signifier=signifier,
            message=f"Signifier {signifier_id} updated successfully",
        )
    except ValueError as e:
        logger.error(f"Failed to update signifier: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error updating signifier: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.delete("/{signifier_id}", response_model=DeleteResponse)
async def delete_signifier(signifier_id: str) -> DeleteResponse:
    """Delete a signifier and all its versions.

    Args:
        signifier_id: Signifier identifier

    Returns:
        Delete response

    Raises:
        HTTPException: If deletion fails
    """
    success = registry.delete(signifier_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signifier {signifier_id} not found",
        )

    logger.info(f"Deleted signifier via API: {signifier_id}")
    return DeleteResponse(
        success=True,
        message=f"Signifier {signifier_id} deleted successfully",
    )


@router.get("", response_model=SignifierListResponse)
async def list_signifiers(
    status: Optional[SignifierStatus] = Query(
        None, description="Filter by status (active or deprecated)"
    ),
    affordance_uri: Optional[str] = Query(
        None, description="Filter by affordance URI"
    ),
    q: Optional[str] = Query(None, description="Search query (not implemented in Phase 1)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> SignifierListResponse:
    """List signifiers with optional filtering.

    Args:
        status: Filter by status
        affordance_uri: Filter by affordance URI
        q: Search query (placeholder for future)
        limit: Page size
        offset: Page offset

    Returns:
        List of signifiers
    """
    signifiers = registry.list_signifiers(
        status=status,
        affordance_uri=affordance_uri,
        limit=limit,
        offset=offset,
    )

    all_signifiers = registry.list_signifiers()
    total = len(all_signifiers)

    logger.debug(f"Listed {len(signifiers)} signifiers via API")
    return SignifierListResponse(
        signifiers=signifiers,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.patch("/{signifier_id}/status", response_model=SignifierResponse)
async def update_signifier_status(
    signifier_id: str,
    status: SignifierStatus = Query(..., description="New status"),
) -> SignifierResponse:
    """Update signifier status.

    Args:
        signifier_id: Signifier identifier
        status: New status

    Returns:
        Updated signifier

    Raises:
        HTTPException: If signifier not found
    """
    signifier = registry.update_status(signifier_id, status)
    if not signifier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signifier {signifier_id} not found",
        )

    logger.info(f"Updated status via API: {signifier_id} -> {status}")
    return SignifierResponse(
        signifier=signifier,
        message=f"Status updated to {status.value}",
    )
