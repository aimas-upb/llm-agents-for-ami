"""
Utils package for EnvExplorer agent.
"""

from .text_processing import artifact_tokens, tokens_from_identifier
from .matching_utils import (
    rank_signifier_matches, workspace_match, intent_compatible
)
from .rd4_engine_utils import (
    ensure_rd4_ready, get_signifier_config, build_rd4_context_snapshot, list_rd4_signifiers
)
from .data_formatting import (
    format_capabilities_summary, format_capabilities_payload,
    generate_shacl_shapes_from_conditions, generate_nl_description
)

__all__ = [
    # Text processing
    'artifact_tokens', 'tokens_from_identifier',
    # Matching utilities
    'rank_signifier_matches', 'workspace_match', 'intent_compatible',
    # RD4 engine utilities
    'ensure_rd4_ready', 'get_signifier_config', 'build_rd4_context_snapshot', 'list_rd4_signifiers',
    # Data formatting utilities
    'format_capabilities_summary', 'format_capabilities_payload',
    'generate_shacl_shapes_from_conditions', 'generate_nl_description',
]