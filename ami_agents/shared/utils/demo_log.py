"""
Small helper for emitting visually distinct demo logs.

We intentionally keep this dependency-free (no colorama). If ANSI colors are
undesirable, set NO_COLOR=1 (or AMI_NO_COLOR=1).
"""

from __future__ import annotations

import os

_NO_COLOR = bool(os.getenv("AMI_NO_COLOR")) or bool(os.getenv("NO_COLOR"))

_ANSI_RESET = "\x1b[0m"
_ANSI_BOLD_CYAN = "\x1b[1;36m"
_ANSI_BOLD_GREEN = "\x1b[1;32m"


def demo_prefix() -> str:
    if _NO_COLOR:
        return "[DEMO]"
    return f"{_ANSI_BOLD_CYAN}[DEMO]{_ANSI_RESET}"


def demo(message: str) -> str:
    """Format a message with a demo-highlighted prefix."""
    return f"{demo_prefix()} {message}"

