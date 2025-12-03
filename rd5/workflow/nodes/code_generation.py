"""Code generation workflow node.

This node uses the LLM to generate Python code that executes
the plan by calling device affordances.
"""

import logging
from typing import Any, Dict, List

from rd5.llm.client import get_llm_client
from rd5.storage.weaviate_client import WeaviateClient
from rd5.workflow.state import PlanState, WorkflowStatus

logger = logging.getLogger(__name__)


async def fetch_similar_plans(intent: str) -> List[Dict[str, Any]]:
    """Fetch similar plans from Weaviate for reference.

    Args:
        intent: Intent text to search for.

    Returns:
        List of similar plan dictionaries.
    """
    try:
        client = WeaviateClient()
        client.connect()
        try:
            plans = client.search_similar_plans(
                query=intent,
                limit=3,
                min_certainty=0.7,
            )
            return plans
        finally:
            client.close()
    except Exception as e:
        logger.warning(f"Could not fetch similar plans: {e}")
        return []


async def code_generation_node(state: PlanState) -> Dict[str, Any]:
    """Generate Python code to execute the plan.

    Args:
        state: Current workflow state with intent and affordances.

    Returns:
        State updates with generated_code.
    """
    request_id = state.get("request_id", "unknown")
    extracted_intent = state.get("extracted_intent", {})
    matched_affordances = state.get("matched_affordances", [])
    retry_count = state.get("retry_count", 0)
    validation_result = state.get("validation_result", {})

    intent_text = extracted_intent.get("intent", state.get("user_request", ""))

    logger.info(
        f"[{request_id}] Generating code for: {intent_text[:50]}... "
        f"(attempt {retry_count + 1})"
    )

    if not intent_text:
        logger.error(f"[{request_id}] No intent text for code generation")
        return {
            "generated_code": "",
            "code_generation_error": "No intent text provided",
            "code_generation_attempts": retry_count + 1,
            "workflow_status": WorkflowStatus.FAILED,
            "current_node": "code_generation",
        }

    # Build retry feedback if this is a retry
    retry_feedback = None
    if retry_count > 0 and validation_result:
        errors = validation_result.get("errors", [])
        if errors:
            retry_feedback = "Validation errors from previous attempt:\n" + "\n".join(
                f"- {err}" for err in errors
            )

    try:
        llm_client = get_llm_client()

        # Try to fetch similar plans for reference
        similar_plans = await fetch_similar_plans(intent_text)

        # Generate code using LLM
        generated_code = await llm_client.generate_plan_code(
            intent=intent_text,
            affordances=matched_affordances,
            similar_plans=similar_plans,
            retry_feedback=retry_feedback,
        )

        logger.info(
            f"[{request_id}] Generated code ({len(generated_code)} chars)"
        )

        return {
            "generated_code": generated_code,
            "code_generation_error": None,
            "code_generation_attempts": retry_count + 1,
            "retry_count": retry_count + 1 if retry_count > 0 else retry_count,
            "current_node": "code_generation",
        }

    except Exception as e:
        error_msg = f"Code generation failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        return {
            "generated_code": "",
            "code_generation_error": error_msg,
            "code_generation_attempts": retry_count + 1,
            "workflow_status": WorkflowStatus.FAILED,
            "current_node": "code_generation",
            "errors": state.get("errors", []) + [error_msg],
        }
