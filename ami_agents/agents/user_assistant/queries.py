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
from .behaviours.state_answer import StateAnswerBehaviour
from .models import ConversationPhase
from .utils import state_answers


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
    """Answer one ENV_STATE_REQUEST, per the outcome EnvExplorer reports.

    `extraction` holds the ontology classes the structuring stage produced --
    a location, a device kind, and a device property or environment variable.
    EnvExplorer resolves those to the affordances that can answer, reads them,
    and reports one of four outcomes.

    Three of the four are phrased here, from the response alone. Only a
    mismatch, and a reading whose value is a dictionary, need the user's
    phrasing read; those go through `StateAnswerBehaviour`.
    """
    explorer_jid = behaviour.agent.target_jids.get("explorer")
    if not explorer_jid:
        await behaviour.reply("Error: EnvExplorer is not configured.")
        conv.phase = ConversationPhase.IDLE
        return

    behaviour.logger.info(
        demo(f"UA -> EnvExplorer ENV_STATE_REQUEST: extraction={json.dumps(extraction)}")
    )
    try:
        result = await rpc_call(
            behaviour.agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_STATE_REQUEST.value,
            body=extraction,
            expect_type=MessageType.ENV_STATE_RESPONSE.value,
            timeout=behaviour.agent.rpc_call_timeout,
        )
        behaviour.logger.info(demo(f"ENV_STATE_RESPONSE received: {result.body}"))

        response = json.loads(result.body or "{}")
        text_intent = extraction.get("text_intent") or conv.user_message
        await behaviour.reply(await _phrase(behaviour, response, text_intent))
    except RpcTimeoutError:
        await behaviour.reply("Timeout querying environment state. Please try again.")
    except Exception as exc:
        behaviour.logger.error("State query failed: %s", exc, exc_info=True)
        await behaviour.reply(f"Error querying state: {exc}")

    conv.phase = ConversationPhase.IDLE


async def _phrase(behaviour, response: dict, text_intent: str) -> str:
    """The sentence for one state response."""
    if "error" in response:
        return f"I could not look that up: {response.get('detail') or response['error']}"

    outcome = response.get("outcome")

    if outcome == "no_affordance":
        return state_answers.no_affordance(response)
    if outcome == "indeterminate_affordance":
        return state_answers.indeterminate_affordance(response)

    if not state_answers.needs_interpretation(response):
        return state_answers.resolved_basic(response)

    # A mismatch, or a structured reading: the answer depends on what the user
    # asked, so a model reads the question against the readings.
    answering = StateAnswerBehaviour(
        text_intent, outcome, response.get("affordances") or [],
        logger=behaviour.logger)
    behaviour.agent.add_behaviour(answering)
    await answering.join()
    return answering.result or state_answers.fallback(response)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
