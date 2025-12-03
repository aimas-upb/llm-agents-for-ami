"""Sandboxed execution workflow node.

This node executes validated Python code in a Docker sandbox
with strict resource and network isolation.
"""

import logging
import os
from typing import Any, Dict

from rd5.execution.sandbox import get_sandbox
from rd5.workflow.state import ExecutionStatus, PlanState, WorkflowStatus

logger = logging.getLogger(__name__)

# Environment variable to control sandbox mode
USE_DOCKER_SANDBOX = os.environ.get("RD5_USE_DOCKER_SANDBOX", "true").lower() == "true"


async def sandboxed_execution_node(state: PlanState) -> Dict[str, Any]:
    """Execute validated code in a Docker sandbox.

    Args:
        state: Current workflow state with generated_code.

    Returns:
        State updates with execution_result.
    """
    request_id = state.get("request_id", "unknown")
    generated_code = state.get("generated_code", "")
    validation_result = state.get("validation_result", {})

    logger.info(f"[{request_id}] Executing code in sandbox...")

    # Check if validation passed
    if not validation_result.get("is_valid", False):
        logger.error(f"[{request_id}] Cannot execute invalid code")
        return {
            "execution_result": {
                "success": False,
                "stdout": "",
                "stderr": "Code validation failed",
                "return_value": None,
                "execution_time_ms": 0,
                "error_type": "validation_failed",
            },
            "execution_status": ExecutionStatus.FAILED,
            "workflow_status": WorkflowStatus.FAILED,
            "current_node": "sandboxed_execution",
        }

    if not generated_code:
        logger.error(f"[{request_id}] No code to execute")
        return {
            "execution_result": {
                "success": False,
                "stdout": "",
                "stderr": "No code provided",
                "return_value": None,
                "execution_time_ms": 0,
                "error_type": "no_code",
            },
            "execution_status": ExecutionStatus.FAILED,
            "workflow_status": WorkflowStatus.FAILED,
            "current_node": "sandboxed_execution",
        }

    try:
        # Get sandbox (Docker or Mock based on availability)
        sandbox = get_sandbox(use_docker=USE_DOCKER_SANDBOX)
        sandbox_type = type(sandbox).__name__

        logger.info(f"[{request_id}] Using {sandbox_type} for execution")

        # Execute code in sandbox
        result = await sandbox.execute(generated_code)

        logger.info(
            f"[{request_id}] Execution completed: "
            f"success={result.success}, "
            f"time={result.execution_time_ms}ms, "
            f"exit_code={result.exit_code}"
        )

        if result.stderr:
            logger.warning(f"[{request_id}] Stderr: {result.stderr[:200]}")

        # Determine execution status
        if result.success:
            execution_status = ExecutionStatus.SUCCESS
            workflow_status = WorkflowStatus.IN_PROGRESS
        elif result.error_type == "timeout":
            execution_status = ExecutionStatus.TIMEOUT
            workflow_status = WorkflowStatus.FAILED
        else:
            execution_status = ExecutionStatus.FAILED
            workflow_status = WorkflowStatus.FAILED

        return {
            "execution_result": {
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_value": result.return_value,
                "execution_time_ms": result.execution_time_ms,
                "error_type": result.error_type,
            },
            "execution_status": execution_status,
            "workflow_status": workflow_status,
            "current_node": "sandboxed_execution",
        }

    except Exception as e:
        error_msg = f"Sandbox execution failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        return {
            "execution_result": {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "return_value": None,
                "execution_time_ms": 0,
                "error_type": "sandbox_error",
            },
            "execution_status": ExecutionStatus.FAILED,
            "workflow_status": WorkflowStatus.FAILED,
            "current_node": "sandboxed_execution",
            "errors": state.get("errors", []) + [error_msg],
        }
