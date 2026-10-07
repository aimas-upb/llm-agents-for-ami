"""Behaviour: answer ENV_CAPABILITY_QUERY_REQUEST -- what can do or report X.

The request arrives as ontology class identifiers from the UA's capability
structuring stage: a location, a kind of device, a device property or an
environment variable, and a command. Answering it is resolution only -- a
capability is a fact about the Thing Descriptions, so no device is read.

Not to be confused with ENV_CAPABILITIES_REQUEST, the ISA's dump of the whole
environment; see `environment_capabilities_behaviour`.
"""

import json
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..utils.capability_resolution import (
    CapabilityOutcome,
    CapabilityResolution,
    resolve_capability_request,
)
from ..utils.state_resolution import discovered_graph


class CapabilityQueryBehaviour(CyclicBehaviour):
    """Handle ENV_CAPABILITY_QUERY_REQUEST: classes in, affordances out."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.ENV_CAPABILITY_QUERY_REQUEST.value:
            return

        self.agent.logger.info(demo(
            f"Received ENV_CAPABILITY_QUERY_REQUEST from {msg.sender}"))

        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}
            self.agent.logger.warning(
                "Failed to parse capability query payload, using empty payload")

        if isinstance(payload, dict):
            response = capability_response(resolve_capability_payload(
                self.agent, payload), payload)
        else:
            response = {
                "error": "malformed_payload",
                "detail": f"expected a JSON object, got {type(payload).__name__}",
            }

        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.ENV_CAPABILITY_QUERY_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, correlation_id)
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)

        self.agent.logger.info(demo(
            f"Sent capability response: {response.get('outcome') or response.get('error')}"))


def resolve_capability_payload(agent, payload: Dict[str, Any]) -> CapabilityResolution:
    """Resolve one class payload against the discovered environment."""
    agent.logger.info(demo(
        f"Resolving capability: location={payload.get('location_class')} "
        f"device={payload.get('device_class')} "
        f"property={payload.get('device_property')} "
        f"env_var={payload.get('environment_variable')} "
        f"command={payload.get('command')}"))

    try:
        graph = discovered_graph(
            agent.artifacts.values(),
            agent.environment_map.values(),
            logger=agent.logger,
        )
        resolution = resolve_capability_request(
            graph,
            location_class=payload.get("location_class"),
            device_class=payload.get("device_class"),
            device_property=payload.get("device_property"),
            environment_variable=payload.get("environment_variable"),
            command=payload.get("command"),
            artifact_name=payload.get("artifact_name"),
        )
    except Exception as exc:
        agent.logger.error("Capability resolution failed: %s", exc, exc_info=True)
        return CapabilityResolution(
            outcome=CapabilityOutcome.INDETERMINATE,
            query={"error": "capability_resolution_failed"},
            detail=f"resolution failed: {exc}",
        )

    agent.logger.info(demo(
        f"Capability resolution: {resolution.outcome.value} -- {resolution.detail}"))
    return resolution


def capability_response(resolution: CapabilityResolution,
                        payload: Dict[str, Any]) -> Dict[str, Any]:
    """The wire body, with the intent and its performative echoed back.

    The UA phrases a `query_if` as yes/no and a `query` as a list; carrying the
    performative back keeps the response self-describing.
    """
    response = resolution.as_dict()
    response["text_intent"] = payload.get("text_intent")
    response["request_performative"] = payload.get("request_performative") or "query"
    return response
