"""
Focused prompts for the User Assistant agent, split by pipeline stage.

The LLM is called only for:
- NLU: understanding the user's request and extracting structured intents.
- NLG: summarising plans and query results in user-friendly language.

Three stages, one module each:

- ``intent_segmentation_prompts`` — split an utterance into atomic intents
  and type each one (no device/capability knowledge).
- ``intent_parsing_prompts`` — parse ONE atomic query (state or capability)
  into ontology classes.
- ``goal_structuring_prompts`` — structure ONE atomic goal, by the shape its
  qualifiers give it (simple, or with dependencies).
- ``intent_response_prompts`` — render plans and query results as prose.

Every prompt is re-exported here, so ``from ...prompts import X`` keeps
working regardless of which module X lives in.
"""

from .goal_structuring_prompts import (
    GOAL_STRUCTURING_DEPENDENCY_PROMPT,
    GOAL_STRUCTURING_SIMPLE_PROMPT,
)
from .intent_parsing_prompts import (
    ENV_CAPABILITIES_REQUEST_PARSER_PROMPT,
    ENV_STATE_REQUEST_PARSER_PROMPT,
)
from .intent_response_prompts import (
    STATE_ANSWER_PROMPT,
    CAPABILITY_ANSWER_PROMPT,
    PLAN_SUMMARY_SYSTEM_PROMPT,
    QUERY_RESPONSE_SYSTEM_PROMPT,
)
from .intent_segmentation_prompts import ATOMIC_SEGMENTATION_SYSTEM_PROMPT

__all__ = [
    # stage 1 — segmentation
    "ATOMIC_SEGMENTATION_SYSTEM_PROMPT",
    # stage 2 — parsing
    "ENV_CAPABILITIES_REQUEST_PARSER_PROMPT",
    "ENV_STATE_REQUEST_PARSER_PROMPT",
    "STATE_ANSWER_PROMPT",
    "GOAL_STRUCTURING_SIMPLE_PROMPT",
    "GOAL_STRUCTURING_DEPENDENCY_PROMPT",
    # stage 3 — response
    "CAPABILITY_ANSWER_PROMPT",
    "PLAN_SUMMARY_SYSTEM_PROMPT",
    "QUERY_RESPONSE_SYSTEM_PROMPT",
]
