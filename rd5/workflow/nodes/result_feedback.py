"""Result feedback workflow node.

This node summarizes execution results and optionally creates
signifiers in RD4 for successful plans.
"""

import logging
from typing import Any, Dict

from rd5.integration.rd4_client import RD4Client, generate_signifier_rdf
from rd5.llm.client import get_llm_client
from rd5.workflow.state import PlanState, WorkflowStatus

logger = logging.getLogger(__name__)


async def result_feedback_node(state: PlanState) -> Dict[str, Any]:
    """Generate execution summary and feedback.

    Args:
        state: Current workflow state with execution results.

    Returns:
        State updates with execution_summary.
    """
    request_id = state.get("request_id", "unknown")
    user_request = state.get("user_request", "")
    extracted_intent = state.get("extracted_intent", {})
    generated_code = state.get("generated_code", "")
    execution_result = state.get("execution_result", {})
    plan_id = state.get("plan_id")
    matched_affordances = state.get("matched_affordances", [])

    logger.info(f"[{request_id}] Generating execution feedback...")

    success = execution_result.get("success", False)

    try:
        # Generate summary using LLM
        llm_client = get_llm_client()
        summary = await llm_client.summarize_execution(
            intent=extracted_intent.get("intent", user_request),
            code=generated_code,
            execution_result=execution_result,
        )

        logger.info(f"[{request_id}] Execution summary: {summary[:100]}...")

        # If successful and we have affordances, create signifier in RD4
        if success and matched_affordances:
            await _create_signifier(
                request_id=request_id,
                intent=extracted_intent.get("intent", user_request),
                affordances=matched_affordances,
            )

        return {
            "execution_summary": summary,
            "feedback_sent": True,
            "workflow_status": WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED,
            "current_node": "result_feedback",
        }

    except Exception as e:
        error_msg = f"Feedback generation failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        # Generate simple fallback summary
        simple_summary = (
            f"Plan {'succeeded' if success else 'failed'} for: "
            f"{extracted_intent.get('intent', user_request)}"
        )

        return {
            "execution_summary": simple_summary,
            "feedback_sent": True,
            "workflow_status": WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED,
            "current_node": "result_feedback",
        }


async def _create_signifier(
    request_id: str,
    intent: str,
    affordances: list,
) -> None:
    """Create a signifier in RD4 for a successful plan.

    Args:
        request_id: Request identifier.
        intent: Intent text.
        affordances: List of used affordances.
    """
    try:
        # Get the first affordance URI
        if not affordances:
            return

        affordance_uri = affordances[0].get("affordance_uri", "")
        if not affordance_uri:
            return

        # Generate signifier ID from intent
        import hashlib
        signifier_id = hashlib.md5(intent.encode()).hexdigest()[:12]

        # Generate RDF
        rdf_data = generate_signifier_rdf(
            signifier_id=f"auto-{signifier_id}",
            intent_text=intent,
            affordance_uri=affordance_uri,
            created_by="rd5-orchestrator",
        )

        # Create in RD4
        async with RD4Client() as rd4:
            if await rd4.health_check():
                await rd4.create_signifier(rdf_data)
                logger.info(f"[{request_id}] Created signifier in RD4")

    except Exception as e:
        logger.warning(f"[{request_id}] Failed to create signifier: {e}")
        # Non-fatal - don't fail the workflow
