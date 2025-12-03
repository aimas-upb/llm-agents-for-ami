"""LangGraph workflow definition for RD5 plan generation.

This module defines the complete workflow graph for generating
and executing smart home plans.
"""

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from rd5.config.settings import get_settings
from rd5.workflow.state import PlanState, ValidationStatus, WorkflowStatus

logger = logging.getLogger(__name__)


def should_retry_generation(state: PlanState) -> Literal["code_generation", "sandboxed_execution"]:
    """Determine if code generation should be retried after validation failure.

    Args:
        state: Current workflow state.

    Returns:
        Next node name based on validation result.
    """
    validation = state.get("validation_result", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if validation.get("is_valid", False):
        logger.info("Code validation passed, proceeding to execution")
        return "sandboxed_execution"

    if retry_count < max_retries:
        logger.warning(
            f"Code validation failed, retrying ({retry_count + 1}/{max_retries})"
        )
        return "code_generation"

    logger.error("Max retries exceeded, skipping execution")
    return "sandboxed_execution"  # Will fail gracefully


def should_store_plan(state: PlanState) -> Literal["plan_storage", "result_feedback"]:
    """Determine if plan should be stored based on execution result.

    Args:
        state: Current workflow state.

    Returns:
        Next node name.
    """
    execution = state.get("execution_result", {})

    if execution.get("success", False):
        logger.info("Execution successful, storing plan")
        return "plan_storage"

    logger.warning("Execution failed, skipping storage")
    return "result_feedback"


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow for plan generation.

    Returns:
        Configured StateGraph instance.
    """
    from rd5.workflow.nodes.intent_extraction import intent_extraction_node
    from rd5.workflow.nodes.signifier_lookup import signifier_lookup_node
    from rd5.workflow.nodes.affordance_match import affordance_match_node
    from rd5.workflow.nodes.code_generation import code_generation_node
    from rd5.workflow.nodes.code_validation import code_validation_node
    from rd5.workflow.nodes.sandboxed_execution import sandboxed_execution_node
    from rd5.workflow.nodes.plan_storage import plan_storage_node
    from rd5.workflow.nodes.result_feedback import result_feedback_node

    # Create the graph with PlanState
    workflow = StateGraph(PlanState)

    # Add all nodes
    workflow.add_node("intent_extraction", intent_extraction_node)
    workflow.add_node("signifier_lookup", signifier_lookup_node)
    workflow.add_node("affordance_match", affordance_match_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("code_validation", code_validation_node)
    workflow.add_node("sandboxed_execution", sandboxed_execution_node)
    workflow.add_node("plan_storage", plan_storage_node)
    workflow.add_node("result_feedback", result_feedback_node)

    # Define edges (linear flow with conditional retry)
    workflow.set_entry_point("intent_extraction")

    workflow.add_edge("intent_extraction", "signifier_lookup")
    workflow.add_edge("signifier_lookup", "affordance_match")
    workflow.add_edge("affordance_match", "code_generation")
    workflow.add_edge("code_generation", "code_validation")

    # Conditional edge: retry or proceed based on validation
    workflow.add_conditional_edges(
        "code_validation",
        should_retry_generation,
        {
            "code_generation": "code_generation",
            "sandboxed_execution": "sandboxed_execution",
        },
    )

    # Conditional edge: store only on success
    workflow.add_conditional_edges(
        "sandboxed_execution",
        should_store_plan,
        {
            "plan_storage": "plan_storage",
            "result_feedback": "result_feedback",
        },
    )

    workflow.add_edge("plan_storage", "result_feedback")
    workflow.add_edge("result_feedback", END)

    logger.info("Created LangGraph workflow with 8 nodes")
    return workflow


def compile_workflow():
    """Compile the workflow for execution.

    Returns:
        Compiled workflow ready for invocation.
    """
    workflow = create_workflow()
    return workflow.compile()


# Pre-compiled workflow instance
_compiled_workflow = None


def get_workflow():
    """Get or create the compiled workflow.

    Returns:
        Compiled LangGraph workflow.
    """
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = compile_workflow()
    return _compiled_workflow


async def run_plan_generation(user_request: str, request_id: str = None) -> PlanState:
    """Run the complete plan generation workflow.

    Args:
        user_request: Natural language user request.
        request_id: Optional request identifier.

    Returns:
        Final workflow state with results.
    """
    from rd5.workflow.state import create_initial_state

    settings = get_settings()

    # Create initial state
    initial_state = create_initial_state(
        user_request=user_request,
        request_id=request_id,
        max_retries=settings.max_code_generation_retries,
    )

    logger.info(f"Starting plan generation for request: {request_id or 'anonymous'}")

    # Get compiled workflow and run
    workflow = get_workflow()
    final_state = await workflow.ainvoke(initial_state)

    logger.info(
        f"Plan generation completed. Status: {final_state.get('workflow_status')}"
    )

    return final_state
