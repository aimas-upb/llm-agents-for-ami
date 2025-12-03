"""Signifier lookup workflow node.

This node queries the RD4 Signifier API to find similar past intents
that can inform the code generation.
"""

import logging
from typing import Any, Dict

from rd5.integration.rd4_client import RD4Client
from rd5.workflow.state import PlanState, WorkflowStatus

logger = logging.getLogger(__name__)


async def signifier_lookup_node(state: PlanState) -> Dict[str, Any]:
    """Look up matching signifiers from RD4.

    Args:
        state: Current workflow state with extracted_intent.

    Returns:
        State updates with matched_signifiers.
    """
    request_id = state.get("request_id", "unknown")
    extracted_intent = state.get("extracted_intent", {})

    intent_text = extracted_intent.get("intent", state.get("user_request", ""))

    logger.info(f"[{request_id}] Looking up signifiers for: {intent_text[:50]}...")

    if not intent_text:
        logger.warning(f"[{request_id}] No intent text for signifier lookup")
        return {
            "matched_signifiers": [],
            "signifier_lookup_error": None,
            "current_node": "signifier_lookup",
        }

    try:
        async with RD4Client() as rd4:
            # Check if RD4 is healthy
            if not await rd4.health_check():
                logger.warning(f"[{request_id}] RD4 API not available")
                return {
                    "matched_signifiers": [],
                    "signifier_lookup_error": "RD4 API not available",
                    "current_node": "signifier_lookup",
                }

            # Query for matching signifiers
            matches = await rd4.match_intent(
                intent=intent_text,
                context=None,  # Could add context from extracted_intent
            )

            # Convert to state format
            matched_signifiers = [
                {
                    "signifier_id": m.signifier_id,
                    "intent_similarity": m.intent_similarity,
                    "shacl_conforms": m.shacl_conforms,
                    "affordance_uri": "",  # Would need additional lookup
                }
                for m in matches
                if m.shacl_conforms  # Only include valid matches
            ]

            logger.info(
                f"[{request_id}] Found {len(matched_signifiers)} matching signifiers"
            )

            return {
                "matched_signifiers": matched_signifiers,
                "signifier_lookup_error": None,
                "current_node": "signifier_lookup",
            }

    except Exception as e:
        error_msg = f"Signifier lookup failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        # Non-fatal error - continue without signifiers
        return {
            "matched_signifiers": [],
            "signifier_lookup_error": error_msg,
            "current_node": "signifier_lookup",
        }
