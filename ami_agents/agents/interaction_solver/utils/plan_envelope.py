"""JSON envelope builders for the BT plan responses returned to the UA.

The same envelope shape is used for success replies, error replies, and the
signifier-only fast-path response so downstream consumers (UserAssistant,
tests, the demo dashboard) can rely on a stable contract.
"""

from typing import Any, Dict, List, Optional, Union

from ....shared.models.intents import Intent
from ....shared.models.intents import ImplicitGoalIntent, ExplicitGoalIntent


PLAN_TYPE_BT = "behavior_tree"


def envelope_missing_intents(intent_type: Optional[str]) -> Dict[str, Any]:
    """Reply when the GOAL_REQUEST carried no usable intents."""
    return {
        "plan_type": PLAN_TYPE_BT,
        "error": "missing_intents",
        "detail": "No intents provided.",
        "tree": None,
        "intents": [],
        "intent_type": intent_type,
    }


def envelope_error(
    error_code: str,
    detail: str,
    intents: List[Union[ImplicitGoalIntent, ExplicitGoalIntent, Intent]],
) -> Dict[str, Any]:
    """Reply when a planning stage fails (context or LLM)."""
    return {
        "plan_type": PLAN_TYPE_BT,
        "error": error_code,
        "detail": detail,
        "tree": None,
        "intents": [
            i.to_wire_dict() if hasattr(i, 'to_wire_dict') else i.to_dict()
            for i in intents
        ],
    }


def envelope_llm_plan(
    result: Dict[str, Any],
    intents: List[Union[ImplicitGoalIntent, ExplicitGoalIntent, Intent]],
    workspace_id: Optional[str],
) -> Dict[str, Any]:
    """Reply for an LLM-generated plan (BTPlanGenerationBehaviour output)."""
    out: Dict[str, Any] = {
        "plan_type": PLAN_TYPE_BT,
        "tree": result.get("tree") or None,
        "explanation": result.get("explanation", ""),
        "intents": [
            i.to_wire_dict() if hasattr(i, 'to_wire_dict') else i.to_dict()
            for i in intents
        ],
        "workspace_id": workspace_id,
    }
    if result.get("impossible"):
        out["impossible"] = True
    return out


def envelope_signifier_reuse(
    tree: Dict[str, Any],
    signifier_ids: List[str],
    intents: List[Union[ImplicitGoalIntent, ExplicitGoalIntent, Intent]],
    workspace_id: Optional[str],
) -> Dict[str, Any]:
    """Reply for the signifier-only fast-path (no LLM call)."""
    return {
        "plan_type": PLAN_TYPE_BT,
        "tree": tree,
        "explanation": "Plan recovered from signifiers (no LLM call needed).",
        "intents": [
            i.to_wire_dict() if hasattr(i, 'to_wire_dict') else i.to_dict()
            for i in intents
        ],
        "workspace_id": workspace_id,
        "signifier_reuse": True,
        "signifier_ids": signifier_ids,
    }


def envelope_llm_plans(
    plans_by_intent: Dict[str, Dict[str, Any]],
    intents: List[Union[ImplicitGoalIntent, ExplicitGoalIntent, Intent]],
    workspace_id: Optional[str],
) -> Dict[str, Any]:
    """Reply for multiple LLM-generated plans (one per atomic intent).

    Returns a list of independent plans, one per intent. UserAssistant
    will execute these asynchronously (in parallel).
    """
    plans = []
    has_impossible = False

    for intent_obj in intents:
        intent_str = intent_obj.to_query_string()
        result = plans_by_intent.get(intent_str, {})

        plan_entry: Dict[str, Any] = {
            "tree": result.get("tree") or None,
            "explanation": result.get("explanation", ""),
            "intent": (
                intent_obj.to_wire_dict()
                if hasattr(intent_obj, 'to_wire_dict')
                else intent_obj.to_dict()
            ),
        }
        if result.get("impossible"):
            plan_entry["impossible"] = True
            has_impossible = True

        plans.append(plan_entry)

    out: Dict[str, Any] = {
        "plan_type": PLAN_TYPE_BT,
        "plans": plans,  # List of independent plans
        "workspace_id": workspace_id,
    }
    if has_impossible:
        out["impossible"] = True
    return out
