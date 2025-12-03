"""FastAPI application for RD5 Plan Orchestrator.

This module provides the REST API for plan generation,
execution, and management.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rd5.config.settings import get_settings
from rd5.workflow.graph import run_plan_generation, get_workflow
from rd5.workflow.state import WorkflowStatus, ExecutionStatus

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)8s] %(message)s",
)
logger = logging.getLogger(__name__)


# Request/Response Models
class PlanRequest(BaseModel):
    """Request to generate and execute a plan."""

    user_request: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language request for the smart home assistant",
        examples=["Turn on the living room lights", "Set thermostat to 72 degrees"],
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional request identifier for tracking",
    )
    execute: bool = Field(
        True,
        description="Whether to execute the plan after generation",
    )


class IntentResult(BaseModel):
    """Extracted intent information."""

    intent: str
    action_verb: str
    target_objects: List[str]
    parameters: Dict[str, Any]
    location: Optional[str] = None


class ValidationResult(BaseModel):
    """Code validation result."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    imports_found: List[str]


class ExecutionResult(BaseModel):
    """Code execution result."""

    success: bool
    stdout: str
    stderr: str
    return_value: Optional[Dict[str, Any]]
    execution_time_ms: int
    error_type: Optional[str] = None


class PlanResponse(BaseModel):
    """Response from plan generation."""

    request_id: str
    status: str
    extracted_intent: Optional[IntentResult] = None
    matched_affordances_count: int = 0
    generated_code: Optional[str] = None
    validation: Optional[ValidationResult] = None
    execution: Optional[ExecutionResult] = None
    plan_id: Optional[str] = None
    summary: Optional[str] = None
    errors: List[str] = []


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: Dict[str, str]


class StatusResponse(BaseModel):
    """System status response."""

    app_name: str
    version: str
    workflow_nodes: List[str]
    settings: Dict[str, Any]


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"Weaviate: {settings.weaviate_host}:{settings.weaviate_port}")
    logger.info(f"RD4 API: {settings.rd4_api_url}")

    # Pre-compile workflow
    try:
        workflow = get_workflow()
        logger.info("Workflow compiled successfully")
    except Exception as e:
        logger.error(f"Failed to compile workflow: {e}")

    yield

    logger.info(f"Shutting down {settings.app_name}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="RD5 Plan Orchestrator - LangGraph-based plan generation and execution",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {}

    # Check Weaviate
    try:
        from rd5.storage.weaviate_client import WeaviateClient
        client = WeaviateClient()
        client.connect()
        client.close()
        services["weaviate"] = "healthy"
    except Exception as e:
        services["weaviate"] = f"unhealthy: {str(e)[:50]}"

    # Check RD4 API
    try:
        from rd5.integration.rd4_client import RD4Client
        async with RD4Client() as rd4:
            if await rd4.health_check():
                services["rd4_api"] = "healthy"
            else:
                services["rd4_api"] = "unhealthy"
    except Exception as e:
        services["rd4_api"] = f"unhealthy: {str(e)[:50]}"

    # Check OpenAI
    if settings.openai_api_key:
        services["openai"] = "configured"
    else:
        services["openai"] = "not configured"

    # Overall status
    overall_status = "healthy" if all(
        "healthy" in v or "configured" in v
        for v in services.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.version,
        services=services,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and configuration."""
    workflow = get_workflow()

    return StatusResponse(
        app_name=settings.app_name,
        version=settings.version,
        workflow_nodes=list(workflow.nodes.keys()) if hasattr(workflow, 'nodes') else [],
        settings={
            "openai_model": settings.openai_model,
            "openai_temperature": settings.openai_temperature,
            "weaviate_host": settings.weaviate_host,
            "weaviate_port": settings.weaviate_port,
            "rd4_api_url": settings.rd4_api_url,
            "sandbox_timeout": settings.sandbox_timeout,
            "max_retries": settings.max_code_generation_retries,
        },
    )


@app.post("/plans", response_model=PlanResponse, status_code=status.HTTP_201_CREATED)
async def generate_plan(request: PlanRequest):
    """Generate and optionally execute a plan.

    This endpoint:
    1. Extracts intent from the user request
    2. Looks up similar signifiers from RD4
    3. Matches device affordances
    4. Generates Python code using LLM
    5. Validates code for safety
    6. Executes in sandbox (if enabled)
    7. Stores successful plans
    """
    logger.info(f"Received plan request: {request.user_request[:50]}...")

    try:
        # Run the workflow
        final_state = await run_plan_generation(
            user_request=request.user_request,
            request_id=request.request_id,
        )

        # Build response
        extracted_intent = final_state.get("extracted_intent", {})
        validation_result = final_state.get("validation_result", {})
        execution_result = final_state.get("execution_result", {})

        response = PlanResponse(
            request_id=final_state.get("request_id", ""),
            status=final_state.get("workflow_status", WorkflowStatus.FAILED).value,
            extracted_intent=IntentResult(
                intent=extracted_intent.get("intent", ""),
                action_verb=extracted_intent.get("action_verb", ""),
                target_objects=extracted_intent.get("target_objects", []),
                parameters=extracted_intent.get("parameters", {}),
                location=extracted_intent.get("location"),
            ) if extracted_intent else None,
            matched_affordances_count=len(final_state.get("matched_affordances", [])),
            generated_code=final_state.get("generated_code") if settings.debug else None,
            validation=ValidationResult(
                is_valid=validation_result.get("is_valid", False),
                errors=validation_result.get("errors", []),
                warnings=validation_result.get("warnings", []),
                imports_found=validation_result.get("imports_found", []),
            ) if validation_result else None,
            execution=ExecutionResult(
                success=execution_result.get("success", False),
                stdout=execution_result.get("stdout", ""),
                stderr=execution_result.get("stderr", ""),
                return_value=execution_result.get("return_value"),
                execution_time_ms=execution_result.get("execution_time_ms", 0),
                error_type=execution_result.get("error_type"),
            ) if execution_result else None,
            plan_id=final_state.get("plan_id"),
            summary=final_state.get("execution_summary"),
            errors=final_state.get("errors", []),
        )

        return response

    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan generation failed: {str(e)}",
        )


@app.get("/plans/{plan_id}", response_model=Dict[str, Any])
async def get_plan(plan_id: str):
    """Retrieve a stored plan by ID."""
    try:
        from rd5.storage.weaviate_client import WeaviateClient

        client = WeaviateClient()
        client.connect()

        try:
            plan = client.get_plan_by_id(plan_id)
            if not plan:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Plan not found: {plan_id}",
                )
            return plan
        finally:
            client.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve plan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve plan: {str(e)}",
        )


@app.get("/plans", response_model=Dict[str, Any])
async def search_plans(
    query: str,
    limit: int = 5,
    min_certainty: float = 0.7,
):
    """Search for similar plans using semantic search."""
    try:
        from rd5.storage.weaviate_client import WeaviateClient

        client = WeaviateClient()
        client.connect()

        try:
            plans = client.search_similar_plans(
                query=query,
                limit=limit,
                min_certainty=min_certainty,
            )
            return {
                "query": query,
                "count": len(plans),
                "plans": plans,
            }
        finally:
            client.close()

    except Exception as e:
        logger.error(f"Plan search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan search failed: {str(e)}",
        )


@app.post("/validate", response_model=ValidationResult)
async def validate_code(code: str):
    """Validate Python code without execution."""
    from rd5.execution.validator import validate_code as run_validation

    result = run_validation(code)

    return ValidationResult(
        is_valid=result.is_valid,
        errors=list(result.errors),
        warnings=list(result.warnings),
        imports_found=list(result.imports_found),
    )


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rd5.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level=settings.log_level.lower(),
    )
