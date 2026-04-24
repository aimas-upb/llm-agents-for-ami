"""Helpers for gathering planning context (affordances + state snapshots)."""

import json
from typing import Any


def parse_or_empty(raw: str) -> Any:
    """Parse a JSON string, returning ``{}`` on failure.

    Used to tolerate EnvExplorer responses that are blank or malformed.
    """
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def match_workspace(value: Any, workspace_id: str) -> bool:
    """Return True if ``value`` refers to ``workspace_id``.

    Accepts either an exact workspace URI or a short workspace name
    (e.g. ``lab308``). For short names, matches common URI shapes:
    ``.../workspaces/lab308#workspace``, ``.../workspaces/lab308``,
    ``.../workspaces/lab308/...``. Falls back to substring match for
    user-friendly inputs.
    """
    if not value:
        return False
    ws = (workspace_id or "").strip()
    if not ws:
        return False
    s = str(value)

    if ws.startswith("http://") or ws.startswith("https://"):
        return s == ws or ws in s

    if s == ws:
        return True
    if f"/{ws}#" in s:
        return True
    if f"/{ws}/" in s:
        return True
    if s.endswith("/" + ws):
        return True
    return ws in s
