"""Embedded RD4 Signifier Engine (vendored).

The engine code is structured as a standalone package rooted at
`ami_agents/shared/memory/`, with a top-level import namespace of `src`.

To use it embedded in this repo, add this directory to `sys.path` so that
imports like `from src.storage.registry import SignifierRegistry` work.
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_engine_root() -> Path:
    """Return the filesystem root of the vendored signifier engine."""

    return Path(__file__).resolve().parent


def get_default_storage_dir() -> Path:
    """Return the default on-disk storage directory for signifiers."""

    return get_engine_root() / "storage"


def ensure_engine_on_path() -> str:
    """Ensure the vendored signifier engine root is importable as `src.*`."""

    engine_root = get_engine_root()
    engine_root_str = str(engine_root)
    if engine_root_str not in sys.path:
        sys.path.insert(0, engine_root_str)
    return engine_root_str
