#!/usr/bin/env python3
"""
Single point of access to the vendored Matter registry.

`tests/simuhome` is not importable as a package from the repo root (the sibling
scripts are all run directly), so anchor sys.path on it once here rather than
repeating the dance in every module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The vendored Matter registry lives with the SHTD integration package, which is
# its primary consumer; the Phase A tables are a second one.
SHTD_DIR = (Path(__file__).resolve().parents[3]
            / "ami_agents" / "environment" / "integration" / "SimuHome")
if str(SHTD_DIR) not in sys.path:
    sys.path.insert(0, str(SHTD_DIR))

from matter_model.registry import load_registry  # noqa: E402


def get_registry() -> Any:
    """The vendored registry (lru_cached upstream, so this is cheap to call)."""
    return load_registry()
