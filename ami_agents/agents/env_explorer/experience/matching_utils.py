from typing import Any, Dict, List, Optional
from ..utils.text_processing import artifact_tokens, tokens_from_identifier


def rank_signifier_matches(matches: List[Dict[str, Any]], intent_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Rank and sort signifier matches by intent_type and similarity.

    Args:
        matches: List of signifier match dictionaries
        intent_type: Optional intent type to prioritize ("EXPLICIT" or "IMPLICIT")

    Returns:
        Sorted list of matches with matching intent_type first, then by similarity
    """
    if not intent_type or not matches:
        return matches

    def _rank_key(m: Dict[str, Any]) -> tuple:
        has_matching_intent_type = (m.get("intent_type") == intent_type)
        similarity = m.get("intent_similarity", 0.0)
        # Return tuple: (match_type_priority, similarity)
        # Higher priority = comes first (so negate for reverse sort)
        return (not has_matching_intent_type, -similarity)

    sorted_matches = sorted(matches, key=_rank_key)
    return sorted_matches


def intent_compatible(
    intent_query: Optional[str] = None,
    signifier_intent: Optional[str] = None,
    affordance_uri: Optional[str] = None,
    payload_hint: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Check intent compatibility with affordance semantics using heuristic guardrails.

    This function implements several heuristics to prevent reusing signifiers in incompatible contexts:
    1. Artifact token compatibility (e.g., light308 vs blinds308)
    2. Polarity guardrails for on/off operations
    3. Property-setting intent alignment

    Args:
        intent_query: The query intent to match
        signifier_intent: The stored signifier intent
        affordance_uri: The affordance URI
        payload_hint: Optional payload structure hint

    Returns:
        True if intents are compatible, False otherwise
    """
    qi = str(intent_query or "").strip().lower()
    si = str(signifier_intent or "").strip().lower()
    au = str(affordance_uri or "").strip().lower()

    # 1) Artifact token compatibility (best-effort).
    # If both mention artifact-like tokens (e.g. light308, blinds308), require overlap.
    q_art = artifact_tokens(qi)
    s_art = artifact_tokens(f"{si} {au}")
    if q_art and s_art and not (q_art & s_art):
        return False

    # 2) Polarity guardrails for on/off.
    if "turn on" in qi and "turn off" in si:
        return False
    if "turn off" in qi and "turn on" in si:
        return False

    # 3) Property-setting intents: require payload/affordance alignment.
    payload_keys: set[str] = set()
    if isinstance(payload_hint, dict):
        # Keep original casing so we can token-split camelCase keys (e.g., lightIntensity).
        payload_keys = {str(k).strip() for k in payload_hint.keys()}

    is_set_intent = qi.startswith("set ") and " to " in qi
    if is_set_intent:
        # If we are "setting" something but the signifier has no payload keys AND
        # the affordance URI doesn't hint at a setter, treat it as incompatible.
        if not payload_keys and not any(x in au for x in ("set", "update")):
            return False

        # Generic property extraction: parse "set <...> <property> to <...>" and
        # require that the inferred property appears in payload keys (preferred) or
        # at least in the affordance URI.
        try:
            mid = qi.split(" to ", 1)[0].removeprefix("set ").strip()
        except Exception:
            mid = ""

        # Remove artifact-like tokens (e.g. light308) from the middle segment.
        prop_phrase = " ".join([t for t in mid.split() if t and t not in q_art]).strip()

        prop_tokens = tokens_from_identifier(prop_phrase)
        if prop_tokens:
            # Compare against payload keys (best signal) and affordance URI as fallback.
            aff_tokens = tokens_from_identifier(au.rsplit("/", 1)[-1])
            key_token_sets = [tokens_from_identifier(k) for k in payload_keys] if payload_keys else []

            if payload_keys:
                if not any(prop_tokens & ks for ks in key_token_sets):
                    return False
            else:
                if not (prop_tokens & aff_tokens):
                    return False

    return True