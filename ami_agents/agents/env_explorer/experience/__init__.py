"""The EnvExplorer's handle on the memory engine (`ami_agents/shared/memory`).

`engine_utils` opens the engine's MemoryStore; `shacl_utils` generates SHACL
shapes from context conditions (kept for context validation, not used for
matching).
"""

from .engine_utils import (
    ensure_experience_engine_ready,
    get_experience_engine_config,
    list_experience_engine_signifiers,
)
from .shacl_utils import generate_shacl_shapes_from_conditions

__all__ = [
    "ensure_experience_engine_ready",
    "get_experience_engine_config",
    "list_experience_engine_signifiers",
    "generate_shacl_shapes_from_conditions",
]
