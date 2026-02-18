"""
Pure utility functions for the User Assistant.

Extracted from the former tools.py so they can be reused by behaviours
without depending on spade_llm.
"""

import hashlib
import json
from typing import Any, Dict


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences from a string."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped.replace("```json", "").replace("```", "").strip()
    return stripped


def loose_json_loads(text: str) -> Any | None:
    """Best-effort JSON loader.

    Handles:
    - clean JSON
    - JSON inside Markdown code fences
    - leading/trailing extra text (tries to raw-decode from first token)
    """
    cleaned = strip_code_fences(text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for token in ("{", "[", '"'):
        idx = cleaned.find(token)
        if idx < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            return obj
        except Exception:
            continue

    return None


def coerce_plan_dict(value: Any) -> Dict[str, Any] | None:
    """Coerce various plan-shaped inputs into a JSON dict.

    Supports:
    - raw plan JSON string
    - wrapper JSON: ``{"ok": true, "plan_json": "...", ...}``
    - nested string-in-string
    """
    candidate: Any = value
    for _ in range(5):
        if isinstance(candidate, dict):
            maybe_plan_json = candidate.get("plan_json")
            if isinstance(maybe_plan_json, str) and maybe_plan_json.strip():
                candidate = maybe_plan_json
                continue
            return candidate

        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                return None
            loaded = loose_json_loads(text)
            if loaded is None:
                return None
            candidate = loaded
            continue

        try:
            candidate = json.loads(json.dumps(candidate))
            continue
        except Exception:
            return None

    return candidate if isinstance(candidate, dict) else None


def canonicalize_plan_for_hash(plan: Dict[str, Any]) -> tuple[str, str]:
    """Return ``(canonical_json, sha256_hex)`` for a plan dict."""
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_bt_nodes(node: dict) -> int:
    """Count nodes in a BT JSON IR tree."""
    if not isinstance(node, dict):
        return 0
    count = 1
    for child in node.get("children", []):
        count += count_bt_nodes(child)
    return count


def bt_preview(node: dict, depth: int = 0) -> str:
    """Generate a compact preview of a BT JSON IR tree."""
    if not isinstance(node, dict):
        return ""
    name = node.get("name", "?")
    ntype = node.get("type", "?")
    parts = [f"{name}({ntype})"]
    if ntype == "action":
        url = node.get("action_url", "")
        action_name = url.rstrip("/").rsplit("/", 1)[-1] if url else "?"
        params = node.get("parameters", {})
        parts = [f"{name}:action({action_name})"]
        if params:
            parts[0] += f" params={params}"
    elif ntype == "condition":
        prop = node.get("property_url", "")
        prop_name = prop.rstrip("/").rsplit("/", 1)[-1] if prop else "?"
        expected = node.get("expected_value", "?")
        parts = [f"{name}:cond({prop_name}=={expected})"]
    children = node.get("children", [])
    if children and depth < 2:
        child_previews = [bt_preview(c, depth + 1) for c in children[:4]]
        child_str = ", ".join(child_previews)
        if len(children) > 4:
            child_str += ", ..."
        parts[0] += f" [{child_str}]"
    return parts[0]
