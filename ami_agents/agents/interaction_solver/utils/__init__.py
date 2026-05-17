"""Utility modules for the InteractionSolver agent."""

from .llm_client import LLMClientConfig, build_llm_client
from .plan_envelope import (
    envelope_error,
    envelope_llm_plan,
    envelope_missing_intents,
    envelope_signifier_reuse,
)
from .planning_context import match_workspace, parse_or_empty
from .signifier_fast_path import (
    collect_signifier_ids,
    try_build_signifier_only_tree,
)
from .signifier_matching import (
    merge_signifier_matches,
    query_community_signifier_match,
    query_local_signifier_match,
)

__all__ = [
    "LLMClientConfig",
    "build_llm_client",
    "collect_signifier_ids",
    "envelope_error",
    "envelope_llm_plan",
    "envelope_missing_intents",
    "envelope_signifier_reuse",
    "match_workspace",
    "merge_signifier_matches",
    "parse_or_empty",
    "query_community_signifier_match",
    "query_local_signifier_match",
    "try_build_signifier_only_tree",
]
