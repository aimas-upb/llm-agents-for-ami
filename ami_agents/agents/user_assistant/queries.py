"""
Answering a query — capabilities and state.

Both are terminal: the user asked something, this replies, nothing is planned
and nothing executes. They live outside the behaviours so the request FSM and
the old receiver can share one implementation rather than each keeping a copy,
which is how the two drifted apart before.

`behaviour` is whatever SPADE behaviour is asking; it supplies `agent`,
`logger`, and `reply`.
"""

from __future__ import annotations

import json

from ...shared.models.messages import MessageType
from ...shared.utils.demo_log import demo
from ...shared.utils.spade_rpc import rpc_call, RpcTimeoutError
from . import pipeline
from .models import ConversationPhase


async def answer_capabilities_query(behaviour, msg, thread: str, conv,
                                    capabilities_ctx: str, extraction: dict) -> None:
    """Handle a capabilities query (structured + LLM formatting)."""
    if not capabilities_ctx:
        capabilities_ctx = await pipeline.fetch_capabilities(behaviour.agent, behaviour.logger)

    if not capabilities_ctx:
        await behaviour.reply("Unable to fetch environment capabilities right now.")
        conv.phase = ConversationPhase.IDLE
        return

    # Parse and filter capabilities JSON to reduce size
    try:
        caps_dict = json.loads(capabilities_ctx) if isinstance(capabilities_ctx, str) else capabilities_ctx
        ## log caps_dict with indentation for debugging
        # behaviour.logger.info(demo(f"Raw capabilities JSON: {json.dumps(caps_dict, indent=2)}"))

        filtered_caps = pipeline.filter_capabilities_json(caps_dict)
        filtered_json = json.dumps(filtered_caps, indent=2)

        # behaviour.logger.info(demo(f"Filtered capabilities JSON: {filtered_json}"))
    except Exception as e:
        behaviour.logger.warning(f"Failed to filter capabilities JSON: {e}, using unfiltered")
        filtered_json = capabilities_ctx

    formatted = await pipeline.format_capabilities_response(behaviour.agent, behaviour.logger, filtered_json, extraction)
    await behaviour.reply(formatted)
    conv.phase = ConversationPhase.IDLE

async def answer_state_query(behaviour, msg, thread: str, conv,
                             extraction: dict) -> None:
    """Handle a state query (deterministic fetch with cache + LLM formatting).

    The extraction dict contains LLM-parsed fields: artifact_type, workspace_type,
    artifact_name, property_name, parameter_name. We send these to EnvExplorer,
    which resolves names to artifact_id and property_uri, fetches state, and returns data.
    """
    explorer_jid = behaviour.agent.target_jids.get("explorer")
    if not explorer_jid:
        await behaviour.reply("Error: EnvExplorer is not configured.")
        conv.phase = ConversationPhase.IDLE
        return

    # Send entire extraction to EnvExplorer for name→ID resolution
    payload = extraction

    behaviour.logger.info(
        demo(f"UA -> EnvExplorer ENV_STATE_REQUEST: extraction={json.dumps(extraction)}")
    )
    try:
        result = await rpc_call(
            behaviour.agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_STATE_REQUEST.value,
            body=payload,
            expect_type=MessageType.ENV_STATE_RESPONSE.value,
            timeout=behaviour.agent.rpc_call_timeout,
        )

        # Log the raw state response before formatting
        behaviour.logger.info(demo(f"ENV_STATE_RESPONSE received: {result.body}"))

        # Format the raw state response for user-friendly output
        # Use the extracted text_intent from the parsed intent, not the full user message
        text_intent = extraction.get("text_intent", conv.user_message)
        formatted = await pipeline.format_query_response(behaviour.agent, behaviour.logger, result.body, text_intent)
        await behaviour.reply(formatted)
    except RpcTimeoutError:
        await behaviour.reply("Timeout querying environment state. Please try again.")
    except Exception as exc:
        await behaviour.reply(f"Error querying state: {exc}")

    conv.phase = ConversationPhase.IDLE

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
