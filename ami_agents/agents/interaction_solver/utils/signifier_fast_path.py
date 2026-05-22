"""Pure helpers for the signifier-only BT recovery fast path.

If every intent has at least one signifier match, we can build a BT directly
from those signifiers without ever calling the LLM or asking EnvExplorer for
capabilities/state. The orchestration (querying signifiers and replying) lives
in the behaviours; this module is the reusable transformation.
"""

from typing import Any, Dict, List, Optional

from ....bt_planning.signifier_bridge import build_bt_from_signifiers
from ...user_assistant.models import Intent


def collect_signifier_ids(
    signifier_matches: Dict[str, Any], intents: List[Intent]
) -> List[str]:
    """Flatten the per-intent ``exact_matches`` into one ordered list of signifier IDs."""
    ids: List[str] = []
    for intent in intents:
        match = signifier_matches.get(intent.to_query_string(), {})
        if isinstance(match, dict):
            for m in match.get("exact_matches", []) or []:
                if isinstance(m, dict):
                    ids.append(str(m.get("signifier_id", "")))
    return ids


def try_build_signifier_only_tree(
    signifier_matches: Dict[str, Any], intents: List[Intent]
) -> Optional[Dict[str, Any]]:
    """Return a BT JSON IR built purely from signifier matches, or None.

    Returns ``None`` if the matches dict is empty or any intent fails to
    bind to a stored signifier in ``build_bt_from_signifiers``.
    """
    if not isinstance(signifier_matches, dict) or not signifier_matches:
        return None
    return build_bt_from_signifiers(signifier_matches, intents)
