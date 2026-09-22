"""Behaviour: answer ENV_STATE_REQUEST -- the value of a named property.

The request arrives as ontology class identifiers from the UA's structuring
stage: a location, a kind of device, and the property or environment variable
asked about. Answering it is two steps, and they are separate behaviours
because they are separately useful -- resolution says *what can answer this and
where*, retrieval says *what the answer is*. This composes them.

A request for the whole environment's state is a different question and has its
own behaviour; see `environment_snapshot_behaviour`.
"""

import json
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from .property_resolution_behaviour import resolve_payload, resolution_response
from .value_retrieval_behaviour import ValueRetrievalBehaviour

# The slots the UA's ENV_STATE structuring stage fills. A request carrying none
# of them names nothing to look for.
CLASS_KEYS = ("location_class", "device_class", "device_property",
              "environment_variable")


class EnvironmentStateBehaviour(CyclicBehaviour):
    """Handle ENV_STATE_REQUEST: resolve the classes, read what they resolve to."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.ENV_STATE_REQUEST.value:
            return

        self.agent.logger.info(demo(
            f"Received ENV_STATE_REQUEST from {msg.sender}"))

        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}
            self.agent.logger.warning(
                "Failed to parse state request payload, using empty payload")

        if not isinstance(payload, dict):
            response = {
                "error": "malformed_payload",
                "detail": f"expected a JSON object, got {type(payload).__name__}",
            }
        elif any(payload.get(key) for key in CLASS_KEYS):
            response = await self._answer_by_class(payload)
        else:
            # Nothing was named. Say so rather than quietly answering a
            # different question -- the whole environment's state is
            # ENV_SNAPSHOT_REQUEST.
            response = {
                "error": "no_property_named",
                "detail": ("expected ontology class identifiers "
                           f"{list(CLASS_KEYS)}; got keys {sorted(payload)}. "
                           "For the whole environment's state, send "
                           f"{MessageType.ENV_SNAPSHOT_REQUEST.value}."),
            }

        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.ENV_STATE_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, correlation_id)
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)

        if "error" in response:
            self.agent.logger.info(demo(
                f"Sent error response: {response.get('error')}"))
        else:
            self.agent.logger.info(demo("Sent state response"))

    async def _answer_by_class(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the classes, then read whatever resolved.

        Everything that resolved is read, including a `mismatched_affordance`.
        "Is anything still on in the kitchen" matches several different on/off
        properties and is *only* answerable with their values; returning them
        unread would push a disambiguation question at the user that the values
        themselves answer. The outcome describes what was found; it does not
        decide whether to look.
        """
        resolution = resolve_payload(self.agent, payload)

        if resolution.affordances:
            retrieval = ValueRetrievalBehaviour(
                resolution.affordances, logger=self.agent.logger)
            self.agent.add_behaviour(retrieval)
            await retrieval.join()
            if retrieval.error:
                self.agent.logger.warning(
                    "Values unavailable for %s affordance(s): %s",
                    len(resolution.affordances), retrieval.error)

        return resolution_response(resolution, payload)
