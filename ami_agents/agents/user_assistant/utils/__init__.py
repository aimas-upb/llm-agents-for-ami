"""Utility modules for the UserAssistant agent.

Re-exports the flat surface area that callers (including tests) relied on
when ``utils`` was a single module.
"""

from .json_io import strip_code_fences, loose_json_loads
from .plan import (
    coerce_plan_dict,
    canonicalize_plan_for_hash,
    count_bt_nodes,
    bt_preview,
)
from .llm_client import LLMClientConfig, build_llm_client, build_llm_call_kwargs, build_behaviour_llm_client
from .ontology_context import (
    build_capabilities_context,
    get_capabilities_context,
    get_capabilities_context_json,
)

__all__ = [
    "build_capabilities_context",
    "get_capabilities_context",
    "get_capabilities_context_json",
    "strip_code_fences",
    "loose_json_loads",
    "coerce_plan_dict",
    "canonicalize_plan_for_hash",
    "count_bt_nodes",
    "bt_preview",
    "LLMClientConfig",
    "build_llm_client",
    "build_llm_call_kwargs",
    "build_behaviour_llm_client",
]
