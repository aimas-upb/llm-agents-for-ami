"""Code validation workflow node.

This node validates generated Python code for safety using
AST-based analysis before execution.
"""

import logging
from typing import Any, Dict

from rd5.execution.validator import validate_code
from rd5.workflow.state import PlanState, ValidationStatus

logger = logging.getLogger(__name__)


async def code_validation_node(state: PlanState) -> Dict[str, Any]:
    """Validate generated code for safety.

    Args:
        state: Current workflow state with generated_code.

    Returns:
        State updates with validation_result.
    """
    request_id = state.get("request_id", "unknown")
    generated_code = state.get("generated_code", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    logger.info(f"[{request_id}] Validating generated code...")

    if not generated_code:
        logger.error(f"[{request_id}] No code to validate")
        return {
            "validation_result": {
                "is_valid": False,
                "errors": ["No code provided for validation"],
                "warnings": [],
                "imports_found": [],
                "blocked_patterns_found": [],
            },
            "validation_status": ValidationStatus.FAILED,
            "current_node": "code_validation",
        }

    try:
        # Run AST-based validation
        result = validate_code(generated_code)

        # Convert sets to lists for JSON serialization
        validation_result = {
            "is_valid": result.is_valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "imports_found": list(result.imports_found),
            "blocked_patterns_found": list(result.blocked_patterns_found),
        }

        if result.is_valid:
            logger.info(
                f"[{request_id}] Code validation passed. "
                f"Imports: {result.imports_found}"
            )
            validation_status = ValidationStatus.PASSED
        else:
            logger.warning(
                f"[{request_id}] Code validation failed. "
                f"Errors: {result.errors}"
            )
            validation_status = ValidationStatus.FAILED

            # Increment retry count for conditional edge
            if retry_count < max_retries:
                return {
                    "validation_result": validation_result,
                    "validation_status": validation_status,
                    "retry_count": retry_count + 1,
                    "current_node": "code_validation",
                }

        return {
            "validation_result": validation_result,
            "validation_status": validation_status,
            "current_node": "code_validation",
        }

    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        return {
            "validation_result": {
                "is_valid": False,
                "errors": [error_msg],
                "warnings": [],
                "imports_found": [],
                "blocked_patterns_found": [],
            },
            "validation_status": ValidationStatus.FAILED,
            "current_node": "code_validation",
            "errors": state.get("errors", []) + [error_msg],
        }
