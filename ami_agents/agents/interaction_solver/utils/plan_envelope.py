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


def envelope_plan_entries(
    entries: List[Dict[str, Any]],
    workspace_id: Optional[str],
) -> Dict[str, Any]:
    """Reply assembled from per-intent plan entries, in the request's order.

    The same multi-plan shape as `envelope_llm_plans`, for replies some of
    whose entries were planned deterministically and some by the workflow.
    An entry without a tree is not executed; its explanation says why.
    """
    out: Dict[str, Any] = {
        "plan_type": PLAN_TYPE_BT,
        "plans": entries,
        "workspace_id": workspace_id,
    }
    if any(e.get("impossible") for e in entries):
        out["impossible"] = True
    # Goals the solver could not choose devices for: the UA asks the user.
    if any(e.get("requires_clarification") for e in entries):
        out["requires_clarification"] = True
    return out


def plan_entries_from_envelope(
    envelope: Dict[str, Any],
    intents: List[Any],
) -> List[Dict[str, Any]]:
    """The workflow's reply for `intents`, as plan entries.

    A multi-plan reply already has one entry per intent. A single-tree reply
    covers all its intents with one tree, so it becomes one entry listing
    them. An error reply becomes one tree-less entry per intent, carrying the
    error so the summary can say what went wrong.
    """
    wire = [i.to_wire_dict() for i in intents]
    plans = envelope.get("plans")
    if isinstance(plans, list):
        return [p for p in plans if isinstance(p, dict)]
    if envelope.get("error"):
        return [{"tree": None, "intent": w, "error": envelope["error"],
                 "explanation": envelope.get("detail", "")} for w in wire]
    entry: Dict[str, Any] = {
        "tree": envelope.get("tree") or None,
        "explanation": envelope.get("explanation", ""),
        "intent": wire[0] if wire else None,
        "intents": wire,
    }
    for key in ("impossible", "signifier_reuse", "signifier_ids"):
        if envelope.get(key):
            entry[key] = envelope[key]
    return [entry]
