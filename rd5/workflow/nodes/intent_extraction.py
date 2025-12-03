"""Intent extraction workflow node.

This node uses the LLM to extract structured intent from
natural language user requests.
"""

import logging
from typing import Any, Dict

from rd5.llm.client import get_llm_client
from rd5.workflow.state import PlanState, WorkflowStatus

logger = logging.getLogger(__name__)


async def intent_extraction_node(state: PlanState) -> Dict[str, Any]:
    """Extract structured intent from user request.

    Args:
        state: Current workflow state with user_request.

    Returns:
        State updates with extracted_intent.
    """
    user_request = state.get("user_request", "")
    request_id = state.get("request_id", "unknown")

    logger.info(f"[{request_id}] Extracting intent from: {user_request[:50]}...")

    if not user_request:
        logger.error(f"[{request_id}] No user request provided")
        return {
            "extracted_intent": {},
            "intent_extraction_error": "No user request provided",
            "workflow_status": WorkflowStatus.FAILED,
            "current_node": "intent_extraction",
            "errors": state.get("errors", []) + ["No user request provided"],
        }

    try:
        llm_client = get_llm_client()

        # Extract intent using LLM
        extracted = await llm_client.extract_intent(
            user_request=user_request,
            available_device_types=None,  # Will be populated by affordance_match
        )

        logger.info(
            f"[{request_id}] Extracted intent: {extracted.get('intent', 'unknown')}"
        )

        return {
            "extracted_intent": extracted,
            "intent_extraction_error": None,
            "workflow_status": WorkflowStatus.IN_PROGRESS,
            "current_node": "intent_extraction",
        }

    except Exception as e:
        error_msg = f"Intent extraction failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        # Return fallback intent
        return {
            "extracted_intent": {
                "intent": user_request,
                "action_verb": "unknown",
                "target_objects": [],
                "parameters": {},
                "location": None,
                "conditions": {"time": None, "trigger": None},
            },
            "intent_extraction_error": error_msg,
            "workflow_status": WorkflowStatus.IN_PROGRESS,
            "current_node": "intent_extraction",
            "errors": state.get("errors", []) + [error_msg],
        }
