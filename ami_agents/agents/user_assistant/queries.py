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
from .behaviours.capability_answer import CapabilityAnswerBehaviour
from .behaviours.state_answer import StateAnswerBehaviour
from .models import ConversationPhase
from .utils import capability_answers, state_answers


async def answer_capabilities_query(behaviour, msg, thread: str, conv,
                                    extraction: dict) -> None:
    """Answer one ENV_CAPABILITIES_REQUEST, per the outcome EnvExplorer reports.

    `extraction` holds the ontology classes the structuring stage produced and
    the performative (`query` / `query_if`). EnvExplorer resolves the classes
    against the TD graph -- reading nothing -- and reports what provides them.

    A yes/no, a "cannot" and a "none" are phrased here from the response alone;
    a found list goes through `CapabilityAnswerBehaviour`.
    """
    explorer_jid = behaviour.agent.target_jids.get("explorer")
    if not explorer_jid:
        await behaviour.reply("Error: EnvExplorer is not configured.")
        conv.phase = ConversationPhase.IDLE
        return

    behaviour.logger.info(demo(
        f"UA -> EnvExplorer ENV_CAPABILITY_QUERY_REQUEST: "
        f"extraction={json.dumps(extraction)}"))
    try:
        result = await rpc_call(
            behaviour.agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_CAPABILITY_QUERY_REQUEST.value,
            body=extraction,
            expect_type=MessageType.ENV_CAPABILITY_QUERY_RESPONSE.value,
            timeout=behaviour.agent.rpc_call_timeout,
        )
        behaviour.logger.info(demo(
            f"ENV_CAPABILITY_QUERY_RESPONSE received: {(result.body or '')[:1000]}"))

        response = json.loads(result.body or "{}")
        text_intent = extraction.get("text_intent") or conv.user_message
        await behaviour.reply(
            await _phrase_capabilities(behaviour, response, text_intent))
    except RpcTimeoutError:
        await behaviour.reply("Timeout querying environment capabilities. Please try again.")
    except Exception as exc:
        behaviour.logger.error("Capability query failed: %s", exc, exc_info=True)
        await behaviour.reply(f"Error querying capabilities: {exc}")

    conv.phase = ConversationPhase.IDLE


async def _phrase_capabilities(behaviour, response: dict, text_intent: str) -> str:
    """The sentence for one capability response."""
    if not capability_answers.needs_interpretation(response):
        return capability_answers.phrase(response)

    answering = CapabilityAnswerBehaviour(
        text_intent, response.get("entries") or [], logger=behaviour.logger)
    behaviour.agent.add_behaviour(answering)
    await answering.join()
    return answering.result or capability_answers.fallback(response)


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

    if outcome == "resolved_artifact" and not response.get("affordances"):
        # Should not happen -- the outcome exists because artifacts were found.
        return state_answers.no_affordance(response)

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
