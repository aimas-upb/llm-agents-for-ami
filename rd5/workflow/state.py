"""LangGraph workflow state definitions.

This module defines the state schema for the RD5 plan generation
workflow using TypedDict for type safety.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class WorkflowStatus(str, Enum):
    """Status of the workflow execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ValidationStatus(str, Enum):
    """Status of code validation."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class ExecutionStatus(str, Enum):
    """Status of code execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ExtractedIntent(TypedDict, total=False):
    """Structured intent extracted from user request."""

    intent: str
    action_verb: str
    target_objects: List[str]
    parameters: Dict[str, Any]
    location: Optional[str]
    conditions: Dict[str, Optional[str]]


class MatchedSignifier(TypedDict, total=False):
    """Signifier matched from RD4."""

    signifier_id: str
    intent_similarity: float
    shacl_conforms: bool
    affordance_uri: str


class MatchedAffordance(TypedDict, total=False):
    """Affordance matched from environment."""

    affordance_uri: str
    name: str
    affordance_type: str
    artifact_id: str
    form: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]]


class ValidationResult(TypedDict, total=False):
    """Result of code validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    imports_found: List[str]
    blocked_patterns_found: List[str]


class ExecutionResult(TypedDict, total=False):
    """Result of sandboxed code execution."""

    success: bool
    stdout: str
    stderr: str
    return_value: Optional[Dict[str, Any]]
    execution_time_ms: int
    error_type: Optional[str]


class PlanState(TypedDict, total=False):
    """Complete state for the plan generation workflow.

    This TypedDict defines all fields that flow through the LangGraph
    workflow nodes.
    """

    # Input
    user_request: str
    request_id: str

    # Intent extraction
    extracted_intent: ExtractedIntent
    intent_extraction_error: Optional[str]

    # Signifier lookup (RD4)
    matched_signifiers: List[MatchedSignifier]
    signifier_lookup_error: Optional[str]

    # Affordance matching
    matched_affordances: List[MatchedAffordance]
    affordance_match_error: Optional[str]

    # Code generation
    generated_code: str
    code_generation_attempts: int
    code_generation_error: Optional[str]

    # Code validation
    validation_result: ValidationResult
    validation_status: ValidationStatus

    # Execution
    execution_result: ExecutionResult
    execution_status: ExecutionStatus

    # Storage
    plan_id: Optional[str]
    plan_stored: bool
    storage_error: Optional[str]

    # Feedback
    feedback_sent: bool
    execution_summary: Optional[str]

    # Workflow metadata
    workflow_status: WorkflowStatus
    current_node: str
    errors: List[str]
    retry_count: int
    max_retries: int


def create_initial_state(
    user_request: str,
    request_id: Optional[str] = None,
    max_retries: int = 3,
) -> PlanState:
    """Create initial workflow state from user request.

    Args:
        user_request: Natural language user request.
        request_id: Optional request identifier.
        max_retries: Maximum code generation retries.

    Returns:
        Initial PlanState for workflow execution.
    """
    import uuid

    return PlanState(
        user_request=user_request,
        request_id=request_id or str(uuid.uuid4()),
        extracted_intent={},
        intent_extraction_error=None,
        matched_signifiers=[],
        signifier_lookup_error=None,
        matched_affordances=[],
        affordance_match_error=None,
        generated_code="",
        code_generation_attempts=0,
        code_generation_error=None,
        validation_result={},
        validation_status=ValidationStatus.PENDING,
        execution_result={},
        execution_status=ExecutionStatus.PENDING,
        plan_id=None,
        plan_stored=False,
        storage_error=None,
        feedback_sent=False,
        execution_summary=None,
        workflow_status=WorkflowStatus.PENDING,
        current_node="start",
        errors=[],
        retry_count=0,
        max_retries=max_retries,
    )


@dataclass
class WorkflowContext:
    """Additional context for workflow execution.

    Not part of the state graph, but passed to nodes for
    external service access.
    """

    rd4_api_url: str = ""
    hmas_base_url: str = ""
    weaviate_host: str = ""
    weaviate_port: int = 8081
    sandbox_timeout: int = 30
    available_device_types: List[str] = field(default_factory=list)
