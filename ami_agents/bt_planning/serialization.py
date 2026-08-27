"""
Persist/load a BehaviorTree JSON IR for cross-agent transport (issue #22).

The BT IR is pure JSON — including custom ``compute`` nodes, whose logic is
referenced by registered name rather than embedded as code (see
``nodes/compute_node.py``). That means a tree can be sent between the
InteractionSolver and the UserAssistant as a UTF-8 JSON string (the existing
SPADE message-body format) and reconstructed on the far side, with **no
``pickle`` and no arbitrary-code-execution risk**.

This module gives that contract an explicit, testable home:

- :func:`serialize_tree`   IR dict   -> JSON ``str``
- :func:`to_bytes`         IR dict   -> UTF-8 ``bytes`` (for byte-message transport)
- :func:`deserialize_tree` ``str``/``bytes`` -> IR dict
- :func:`validate_tree`    IR dict   -> list of error strings (registry-driven)

The round-trip ``deserialize_tree(serialize_tree(spec)) == spec`` holds for any
tree built from registered node types.
"""

import json
from typing import Any, List, Union

from .nodes.registry import validate_node


def serialize_tree(spec: dict, *, indent: Union[int, None] = None) -> str:
    """Serialise a BT IR dict to a JSON string."""
    if not isinstance(spec, dict):
        raise TypeError(f"BT IR must be a dict, got {type(spec).__name__}")
    return json.dumps(spec, indent=indent, ensure_ascii=False)


def to_bytes(spec: dict) -> bytes:
    """Serialise a BT IR dict to UTF-8 bytes for byte-message transport."""
    return serialize_tree(spec).encode("utf-8")


def deserialize_tree(payload: Union[str, bytes, bytearray]) -> dict:
    """Load a BT IR dict from a JSON string or UTF-8 bytes."""
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    spec = json.loads(payload)
    if not isinstance(spec, dict):
        raise ValueError("Behavior tree payload must decode to a JSON object")
    return spec


def validate_tree(spec: Any) -> List[str]:
    """Validate a BT IR dict against the node registry. Empty list == valid."""
    return validate_node(spec, "tree")
