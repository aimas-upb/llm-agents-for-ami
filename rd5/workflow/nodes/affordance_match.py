"""Affordance matching workflow node.

This node finds device affordances from the HMAS environment
that can fulfill the extracted intent.
"""

import logging
from typing import Any, Dict, List

import httpx

from rd5.config.settings import get_settings
from rd5.workflow.state import PlanState

logger = logging.getLogger(__name__)


async def fetch_environment_affordances(base_url: str) -> List[Dict[str, Any]]:
    """Fetch available affordances from HMAS environment.

    Args:
        base_url: HMAS platform base URL.

    Returns:
        List of affordance dictionaries.
    """
    # This would interface with Yggdrasil/HMAS platform
    # For now, return mock affordances for development
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try to fetch from HMAS workspaces endpoint
            response = await client.get(f"{base_url}/workspaces")
            if response.status_code == 200:
                data = response.json()
                # Parse HMAS response and extract affordances
                return _parse_hmas_affordances(data)
    except Exception as e:
        logger.warning(f"Could not fetch from HMAS: {e}")

    # Return mock affordances for development/testing
    return _get_mock_affordances()


def _parse_hmas_affordances(hmas_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse HMAS response to extract affordances.

    Args:
        hmas_data: Raw HMAS API response.

    Returns:
        List of normalized affordance dictionaries.
    """
    affordances = []

    # Parse workspaces -> artifacts -> affordances
    for workspace in hmas_data.get("workspaces", []):
        for artifact in workspace.get("artifacts", []):
            for aff in artifact.get("affordances", []):
                affordances.append({
                    "affordance_uri": aff.get("uri", ""),
                    "name": aff.get("name", ""),
                    "affordance_type": aff.get("type", "action"),
                    "artifact_id": artifact.get("id", ""),
                    "artifact_name": artifact.get("name", ""),
                    "form": {
                        "href": aff.get("form", {}).get("href", ""),
                        "method": aff.get("form", {}).get("method", "POST"),
                        "content_type": aff.get("form", {}).get("contentType", "application/json"),
                    },
                    "input_schema": aff.get("input", {}),
                    "output_schema": aff.get("output", {}),
                })

    return affordances


def _get_mock_affordances() -> List[Dict[str, Any]]:
    """Get mock affordances for development.

    Returns:
        List of mock affordance dictionaries.
    """
    return [
        {
            "affordance_uri": "http://localhost:8090/workspaces/living-room/artifacts/lights/affordances/toggle",
            "name": "toggle_lights",
            "affordance_type": "action",
            "artifact_id": "lights",
            "artifact_name": "Living Room Lights",
            "form": {
                "href": "http://localhost:8090/workspaces/living-room/artifacts/lights/toggle",
                "method": "POST",
                "content_type": "application/json",
            },
            "input_schema": {"type": "object", "properties": {"state": {"type": "boolean"}}},
        },
        {
            "affordance_uri": "http://localhost:8090/workspaces/living-room/artifacts/lights/affordances/set-brightness",
            "name": "set_brightness",
            "affordance_type": "action",
            "artifact_id": "lights",
            "artifact_name": "Living Room Lights",
            "form": {
                "href": "http://localhost:8090/workspaces/living-room/artifacts/lights/brightness",
                "method": "PUT",
                "content_type": "application/json",
            },
            "input_schema": {"type": "object", "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}}},
        },
        {
            "affordance_uri": "http://localhost:8090/workspaces/living-room/artifacts/thermostat/affordances/set-temperature",
            "name": "set_temperature",
            "affordance_type": "action",
            "artifact_id": "thermostat",
            "artifact_name": "Living Room Thermostat",
            "form": {
                "href": "http://localhost:8090/workspaces/living-room/artifacts/thermostat/temperature",
                "method": "PUT",
                "content_type": "application/json",
            },
            "input_schema": {"type": "object", "properties": {"temperature": {"type": "number"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}},
        },
        {
            "affordance_uri": "http://localhost:8090/workspaces/living-room/artifacts/blinds/affordances/set-position",
            "name": "set_blinds_position",
            "affordance_type": "action",
            "artifact_id": "blinds",
            "artifact_name": "Living Room Blinds",
            "form": {
                "href": "http://localhost:8090/workspaces/living-room/artifacts/blinds/position",
                "method": "PUT",
                "content_type": "application/json",
            },
            "input_schema": {"type": "object", "properties": {"position": {"type": "integer", "minimum": 0, "maximum": 100}}},
        },
    ]


def match_affordances_to_intent(
    affordances: List[Dict[str, Any]],
    extracted_intent: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Match available affordances to extracted intent.

    Args:
        affordances: List of available affordances.
        extracted_intent: Extracted intent with target_objects.

    Returns:
        Filtered list of relevant affordances.
    """
    target_objects = extracted_intent.get("target_objects", [])
    action_verb = extracted_intent.get("action_verb", "").lower()
    location = extracted_intent.get("location", "")

    if not target_objects and not action_verb:
        # Return all if no specific targets
        return affordances

    matched = []
    for aff in affordances:
        aff_name = aff.get("name", "").lower()
        artifact_name = aff.get("artifact_name", "").lower()

        # Check if affordance matches target objects
        for target in target_objects:
            target_lower = target.lower()
            if target_lower in aff_name or target_lower in artifact_name:
                matched.append(aff)
                break

        # Check if affordance matches action verb
        if action_verb in aff_name:
            if aff not in matched:
                matched.append(aff)

    return matched if matched else affordances  # Return all if no matches


async def affordance_match_node(state: PlanState) -> Dict[str, Any]:
    """Match environment affordances to extracted intent.

    Args:
        state: Current workflow state with extracted_intent.

    Returns:
        State updates with matched_affordances.
    """
    request_id = state.get("request_id", "unknown")
    extracted_intent = state.get("extracted_intent", {})

    logger.info(f"[{request_id}] Matching affordances to intent...")

    settings = get_settings()

    try:
        # Fetch available affordances from environment
        all_affordances = await fetch_environment_affordances(settings.hmas_base_url)

        logger.info(f"[{request_id}] Found {len(all_affordances)} total affordances")

        # Match to intent
        matched = match_affordances_to_intent(all_affordances, extracted_intent)

        logger.info(f"[{request_id}] Matched {len(matched)} affordances to intent")

        return {
            "matched_affordances": matched,
            "affordance_match_error": None,
            "current_node": "affordance_match",
        }

    except Exception as e:
        error_msg = f"Affordance matching failed: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")

        # Return mock affordances as fallback
        return {
            "matched_affordances": _get_mock_affordances(),
            "affordance_match_error": error_msg,
            "current_node": "affordance_match",
        }
