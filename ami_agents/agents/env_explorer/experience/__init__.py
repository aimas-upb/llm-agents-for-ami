"""
Experience Engine Module for EnvExplorer Agent.

This module contains utilities and components for experience management,
including signifier matching, SHACL generation, and natural language processing.

Extracted from the main agent to improve modular organization and separation of concerns.
"""

# Re-export main experience management utilities
from .matching_utils import intent_compatible, rank_signifier_matches
from .shacl_utils import generate_shacl_shapes_from_conditions
from .nl_generation import generate_nl_description
from .engine_utils import ensure_experience_engine_ready, get_experience_engine_config, build_experience_engine_context_snapshot, list_experience_engine_signifiers

__all__ = [
    'intent_compatible',
    'rank_signifier_matches',
    'generate_shacl_shapes_from_conditions',
    'generate_nl_description',
    'ensure_experience_engine_ready',
    'get_experience_engine_config',
    'build_experience_engine_context_snapshot',
    'list_experience_engine_signifiers',
]