"""API routes for validation operations.

This module implements the Phase 2 Validation APIs:
- POST /validate/shacl (batch validation)
- POST /signifiers/validate-authoring (authoring validation)
- POST /context/normalize (context normalization)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.models.signifier import Signifier
from src.validation import (
    AuthoringValidationError,
    AuthoringValidator,
    ContextGraphBuilder,
    SHACLValidator,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["validation"])

shacl_validator = SHACLValidator(enable_caching=True)
authoring_validator = AuthoringValidator(strict_mode=False)
context_builder = ContextGraphBuilder()


class ValidateAuthoringRequest(BaseModel):
    """Request for authoring validation.

    Args:
        signifier: Signifier to validate
        strict_mode: Enable strict validation
    """

    signifier: Signifier
    strict_mode: bool = Field(default=False)


class ValidateAuthoringResponse(BaseModel):
    """Response for authoring validation.

    Args:
        valid: Whether signifier is valid
        errors: List of validation errors
        signifier_id: Signifier identifier
    """

    valid: bool
    errors: List[str]
    signifier_id: str


class ValidateSHACLRequest(BaseModel):
    """Request for batch SHACL validation.

    Args:
        context: Context features (KV map or nested dict)
        shapes: SHACL shapes in Turtle format
        artifact_types: Optional artifact type information
    """

    context: Dict[str, Any]
    shapes: str = Field(..., min_length=1)
    artifact_types: Optional[Dict[str, str]] = None


class ValidateSHACLResponse(BaseModel):
    """Response for SHACL validation.

    Args:
        conforms: Whether context conforms to shapes
        violations: List of violations
        violation_count: Number of violations
    """

    conforms: bool
    violations: List[Dict[str, Any]]
    violation_count: int


class NormalizeContextRequest(BaseModel):
    """Request for context normalization.

    Args:
        context: Context in various formats
        artifact_types: Optional artifact type information
    """

    context: Dict[str, Any]
    artifact_types: Optional[Dict[str, str]] = None


class NormalizeContextResponse(BaseModel):
    """Response for context normalization.

    Args:
        rdf_turtle: Normalized context as Turtle
        feature_count: Number of extracted features
        features: Extracted (artifact, property) features
    """

    rdf_turtle: str
    feature_count: int
    features: Dict[str, Dict[str, Any]]


class BatchValidateSHACLRequest(BaseModel):
    """Request for validating multiple signifiers against one context.

    Args:
        context: Context features
        signifier_ids: List of signifier IDs to validate
        artifact_types: Optional artifact type information
    """

    context: Dict[str, Any]
    signifier_ids: List[str]
    artifact_types: Optional[Dict[str, str]] = None


class BatchValidationResult(BaseModel):
    """Result for one signifier in batch validation.

    Args:
        signifier_id: Signifier identifier
        conforms: Whether context conforms
        violations: List of violations
    """

    signifier_id: str
    conforms: bool
    violations: List[Dict[str, Any]]


class BatchValidateSHACLResponse(BaseModel):
    """Response for batch SHACL validation.

    Args:
        results: List of validation results per signifier
        total_count: Total signifiers validated
        conforming_count: Number of conforming signifiers
    """

    results: List[BatchValidationResult]
    total_count: int
    conforming_count: int


@router.post(
    "/signifiers/validate-authoring",
    response_model=ValidateAuthoringResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_authoring(
    request: ValidateAuthoringRequest,
) -> ValidateAuthoringResponse:
    """Validate signifier structure for authoring.

    Args:
        request: Authoring validation request

    Returns:
        Validation result with errors if any

    Raises:
        HTTPException: If validation execution fails
    """
    try:
        validator = AuthoringValidator(strict_mode=request.strict_mode)
        errors = validator.validate_signifier(request.signifier)

        logger.info(
            f"Authoring validation for {request.signifier.signifier_id}: "
            f"valid={len(errors)==0}, errors={len(errors)}"
        )

        return ValidateAuthoringResponse(
            valid=len(errors) == 0,
            errors=errors,
            signifier_id=request.signifier.signifier_id,
        )

    except Exception as e:
        logger.error(f"Authoring validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation execution failed: {str(e)}",
        )


@router.post(
    "/validate/shacl",
    response_model=ValidateSHACLResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_shacl(
    request: ValidateSHACLRequest,
) -> ValidateSHACLResponse:
    """Validate context graph against SHACL shapes.

    Args:
        request: SHACL validation request

    Returns:
        Validation result with violations

    Raises:
        HTTPException: If validation fails
    """
    try:
        context_graph, _ = context_builder.normalize_context(request.context)

        if request.artifact_types:
            context_graph = context_builder.add_type_information(
                context_graph, request.artifact_types
            )

        result = shacl_validator.validate_signifier_context(
            context_graph, request.shapes
        )

        logger.info(
            f"SHACL validation: conforms={result.conforms}, "
            f"violations={len(result.violations)}"
        )

        return ValidateSHACLResponse(
            conforms=result.conforms,
            violations=[v.to_dict() for v in result.violations],
            violation_count=len(result.violations),
        )

    except ValueError as e:
        logger.error(f"SHACL validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in SHACL validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/context/normalize",
    response_model=NormalizeContextResponse,
    status_code=status.HTTP_200_OK,
)
async def normalize_context(
    request: NormalizeContextRequest,
) -> NormalizeContextResponse:
    """Normalize context input to RDF graph.

    Args:
        request: Context normalization request

    Returns:
        Normalized context as RDF Turtle

    Raises:
        HTTPException: If normalization fails
    """
    try:
        context_graph, features = context_builder.normalize_context(
            request.context
        )

        if request.artifact_types:
            context_graph = context_builder.add_type_information(
                context_graph, request.artifact_types
            )

        rdf_turtle = context_graph.serialize(format="turtle")

        features_dict = {}
        for (artifact, prop), value in features.items():
            if artifact not in features_dict:
                features_dict[artifact] = {}
            features_dict[artifact][prop] = value

        logger.info(
            f"Normalized context: {len(features)} features, "
            f"{len(context_graph)} triples"
        )

        return NormalizeContextResponse(
            rdf_turtle=rdf_turtle,
            feature_count=len(features),
            features=features_dict,
        )

    except ValueError as e:
        logger.error(f"Context normalization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in context normalization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/validate/cache-stats")
async def get_cache_stats() -> dict:
    """Get SHACL validation cache statistics.

    Returns:
        Cache statistics
    """
    stats = shacl_validator.get_cache_stats()
    logger.debug(f"Cache stats: {stats}")
    return stats


@router.post("/validate/clear-cache")
async def clear_cache() -> dict:
    """Clear SHACL validation cache.

    Returns:
        Success message
    """
    shacl_validator.clear_cache()
    logger.info("Validation cache cleared")
    return {"message": "Cache cleared successfully"}
