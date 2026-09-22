"""Behaviour: resolve ontology classes to the affordances that can answer them.

Answers *what can provide this value, and where* -- without reading anything.
A planner asking "what senses temperature in the kitchen?" wants exactly this
and no HTTP traffic; a state query wants it as its first half.

The answer is a SPARQL query over the discovered Thing Descriptions plus the
ontologies that define the vocabulary, never string matching on names: a
property affordance is typed with its homeont class and carries its own read
target, so a class-level match lands directly on the thing to read.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..utils.state_resolution import (
    StateOutcome,
    StateResolution,
    discovered_graph,
    resolve_state_request,
)


class PropertyResolutionBehaviour(CyclicBehaviour):
    """Handle ENV_RESOLVE_REQUEST: classes in, affordances out, no reads."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.ENV_RESOLVE_REQUEST.value:
            return

        self.agent.logger.info(demo(
            f"Received ENV_RESOLVE_REQUEST from {msg.sender}"))

        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}
            self.agent.logger.warning(
                "Failed to parse resolve request payload, using empty payload")

        resolution = resolve_payload(self.agent, payload)
        response = resolution_response(resolution, payload)

        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.ENV_RESOLVE_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, correlation_id)
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)


def resolve_payload(agent, payload: Dict[str, Any]) -> StateResolution:
    """Resolve one class payload against the discovered environment.

    Returns the `StateResolution` itself, not a response body, so the state
    behaviour can hand its affordances to a retrieval behaviour and serialise
    only once the values are in. A plain function rather than a method so both
    behaviours share it without one inheriting from the other.
    """
    agent.logger.info(demo(
        f"Resolving by class: location={payload.get('location_class')} "
        f"device={payload.get('device_class')} "
        f"property={payload.get('device_property')} "
        f"env_var={payload.get('environment_variable')}"))

    try:
        graph = discovered_graph(
            agent.artifacts.values(),
            agent.environment_map.values(),
            logger=agent.logger,
        )
        resolution = resolve_state_request(
            graph,
            location_class=payload.get("location_class"),
            device_class=payload.get("device_class"),
            device_property=payload.get("device_property"),
            environment_variable=payload.get("environment_variable"),
        )
    except Exception as exc:
        agent.logger.error("State resolution failed: %s", exc, exc_info=True)
        return StateResolution(
            outcome=StateOutcome.INDETERMINATE,
            query={"error": "state_resolution_failed"},
            detail=f"resolution failed: {exc}",
        )

    agent.logger.info(demo(
        f"Resolution: {resolution.outcome.value} -- {resolution.detail}"))
    return resolution


def resolution_response(resolution: StateResolution,
                        payload: Dict[str, Any]) -> Dict[str, Any]:
    """The wire body for a resolution, with the intent it answers echoed back.

    One affordance list, each entry carrying its own value once read -- not a
    second parallel list of readings repeating the same fields.
    """
    response = resolution.as_dict()
    response["text_intent"] = payload.get("text_intent")
    return response
