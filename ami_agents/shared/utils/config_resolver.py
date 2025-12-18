"""
Configuration lookup helpers.

The repo currently mixes multiple config shapes:
- flat test configs (e.g., {"yggdrasil_url": "http://..."})
- legacy nested shapes (e.g., {"environment": {"yggdrasil": {"url": "http://..."}}})
- environment.yaml shape (e.g., {"environment": {"integration_engine": {"yggdrasil": {"url": "http://..."}}}})

These helpers centralize lookup so agents behave consistently.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional


def _get(mapping: Mapping[str, Any], *path: str) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _first_non_empty_str(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        s = str(value).strip()
        if s:
            return s
    return None


def resolve_yggdrasil_url(
    config: Optional[Mapping[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    default: str = "http://localhost:8080/",
) -> str:
    """
    Resolve the Yggdrasil HMAS endpoint URL from config/env.

    Supports (in priority order):
    - config["yggdrasil_url"]
    - config["yggdrasil"]["url"]
    - config["environment"]["yggdrasil"]["url"] (legacy)
    - config["environment"]["integration_engine"]["yggdrasil"]["url"] (environment.yaml)
    - config["integration_engine"]["yggdrasil"]["url"] (if integration_engine passed directly)
    - env["YGGDRASIL_URL"]
    - default
    """
    cfg: Mapping[str, Any] = config or {}
    env_map: Mapping[str, str] = env or os.environ

    return _first_non_empty_str(
        _get(cfg, "yggdrasil_url"),
        _get(cfg, "yggdrasil", "url"),
        _get(cfg, "environment", "yggdrasil", "url"),
        _get(cfg, "environment", "integration_engine", "yggdrasil", "url"),
        _get(cfg, "integration_engine", "yggdrasil", "url"),
        env_map.get("YGGDRASIL_URL"),
        default,
    ) or default

