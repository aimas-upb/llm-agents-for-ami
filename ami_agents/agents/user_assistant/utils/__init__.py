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
    build_ontology_context,
    get_capability_ontology_context,
    get_capability_ontology_context_json,
    get_ontology_context,
    get_ontology_context_json,
    iter_classes,
)

__all__ = [
    "build_ontology_context",
    "get_capability_ontology_context",
    "get_capability_ontology_context_json",
    "get_ontology_context",
    "get_ontology_context_json",
    "iter_classes",
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
