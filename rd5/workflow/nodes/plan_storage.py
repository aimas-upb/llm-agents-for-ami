"""Plan storage workflow node.

This node stores successful plans in Weaviate for future retrieval
and learning.
"""

import logging
from typing import Any, Dict

from rd5.storage.weaviate_client import WeaviateClient
from rd5.workflow.state import PlanState

logger = logging.getLogger(__name__)


async def plan_storage_node(state: PlanState) -> Dict[str, Any]:
    """Store successful plan in Weaviate.

    Args:
        state: Current workflow state with execution results.

    Returns:
        State updates with plan_id.
    """
    request_id = state.get("request_id", "unknown")
    user_request = state.get("user_request", "")
    extracted_intent = state.get("extracted_intent", {})
    generated_code = state.get("generated_code", "")
    matched_affordances = state.get("matched_affordances", [])
    execution_result = state.get("execution_result", {})

    logger.info(f"[{request_id}] Storing plan in Weaviate...")

    # Only store successful plans
    if not execution_result.get("success", False):
        logger.warning(f"[{request_id}] Skipping storage for failed plan")
        return {
            "plan_stored": False,
            "storage_error": "Plan execution failed",
            "current_node": "plan_storage",
        }

    try:
        client = WeaviateClient()
        client.connect()

        try:
            # Ensure schema exists
            client.ensure_schema()

            # Extract affordance URIs
            affordance_uris = [
                aff.get("affordance_uri", "")
                for aff in matched_affordances
            ]

            # Extract device types
            device_types = list(set(
                aff.get("artifact_name", "").split()[0]
                for aff in matched_affordances
                if aff.get("artifact_name")
            ))

            # Store the plan
            plan_id = client.store_plan(
                intent_text=user_request,
                extracted_intent=extracted_intent.get("intent", user_request),
                plan_code=generated_code,
                plan_explanation=f"Automated plan for: {extracted_intent.get('intent', user_request)}",
                affordance_uris=affordance_uris,
                device_types=device_types,
                execution_context={
                    "request_id": request_id,
                    "parameters": extracted_intent.get("parameters", {}),
                    "location": extracted_intent.get("location"),
                },
            )

            logger.info(f"[{request_id}] Stored plan with ID: {plan_id}")

            return {
                "plan_id": plan_id,
                "plan_stored": True,
                "storage_error": None,
                "current_node": "plan_storage",
            }

        finally:
            client.close()

    except Exception as e:
        error_msg = f"Plan storage failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        return {
            "plan_id": None,
            "plan_stored": False,
            "storage_error": error_msg,
            "current_node": "plan_storage",
        }
