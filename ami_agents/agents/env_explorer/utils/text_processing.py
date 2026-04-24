"""
Text processing utilities extracted from EnvExplorer agent.
"""

import re
from typing import Set


def artifact_tokens(s: str) -> Set[str]:
    """
    Extract artifact-like tokens using regex pattern.

    Finds tokens matching pattern: word followed by 1-4 digits (e.g., light308, blinds42).

    Args:
        s: Input string to extract tokens from

    Returns:
        Set of artifact tokens found in the string
    """
    return set(re.findall(r"\b[a-z_]+[0-9]{1,4}\b", s.lower()))


def tokens_from_identifier(s: str) -> Set[str]:
    """
    Split camelCase identifiers and extract meaningful tokens.

    Splits camelCase boundaries, underscores, and extracts tokens that are 3+ characters.
    Useful for parsing property names like "lightIntensity" -> {"light", "intensity"}.

    Args:
        s: Input identifier string

    Returns:
        Set of tokens extracted from the identifier (3+ chars only)
    """
    s = str(s or "").strip()
    if not s:
        return set()

    # Split camelCase boundaries, underscores, and non-alphanumerics
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ")
    parts = re.findall(r"\b[a-z0-9]+\b", s.lower())

    # Drop very short tokens to reduce noise
    return {p for p in parts if len(p) >= 3}