"""JSON I/O helpers tolerant to LLM output quirks."""

import json
from typing import Any


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
