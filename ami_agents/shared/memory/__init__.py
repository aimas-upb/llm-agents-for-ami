"""The memory engine (vendored): remembered goal executions, matched by exact key.

A separate component: the EnvExplorer manages one embedded instance, and the
same code is served on its own as the community memory (`src/api`). The code is
a standalone package rooted at `ami_agents/shared/memory/`, with a top-level
import namespace of `src`:

    src/models/signifier.py        GoalSignifier, the record
    src/storage/memory_store.py    MemoryStore: record / match / list
    src/validation/                SHACL context validation (kept, not used for
                                   matching yet)

To use it embedded in this repo, add this directory to `sys.path` (see
`ensure_engine_on_path`) so that imports like
`from src.storage.memory_store import MemoryStore` work.
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
