"""Behaviour: every artifact's last known state, unfiltered.

A different question from ENV_STATE. That one asks about a particular property
of a particular kind of device in a particular place, and resolves ontology
classes to answer it. This one names nothing and resolves nothing -- it is the
bulk read a planner wants as background before planning, and the signifier
matcher wants as the context a past experience was recorded in.

The state comes from what discovery and change events have already put on the
agent; nothing is dereferenced here.
"""

import json
from typing import Any, Dict, List

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.models.environment import AffordanceType
from ....shared.utils.demo_log import demo


class EnvironmentSnapshotBehaviour(CyclicBehaviour):
    """Handle ENV_SNAPSHOT_REQUEST: all artifacts, all their state."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.ENV_SNAPSHOT_REQUEST.value:
            return

        self.agent.logger.info(demo(
            f"Received ENV_SNAPSHOT_REQUEST from {msg.sender}"))

        response = environment_snapshot(self.agent)

        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.ENV_SNAPSHOT_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, correlation_id)
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)


def environment_snapshot(agent) -> Dict[str, Any]:
    """Every artifact's last known state, keyed by artifact id.

    The shape consumers index and filter: they look artifacts up by id and
    narrow them by `workspace_id`, so `artifacts` must be a mapping rather than
    a list. State is copied, so a caller editing the response cannot reach back
    into the agent's own record.
    """
    artifacts: Dict[str, Any] = {}
    total_properties = 0

    for artifact_id, artifact in agent.artifacts.items():
        state = dict(artifact.current_state)
        artifacts[artifact_id] = {
            "name": artifact.name,
            "workspace_id": getattr(artifact, "workspace_id", None),
            "state": state,
            "state_property_urls": state_property_urls(agent, artifact_id),
        }
        total_properties += len(state)

    agent.logger.info(demo(
        f"Environment snapshot: {len(artifacts)} artifacts, "
        f"{total_properties} properties"))

    return {
        "artifacts": artifacts,
        "summary": {
            "total_artifacts": len(artifacts),
            "total_properties": total_properties,
            "workspaces": sorted({
                entry["workspace_id"] for entry in artifacts.values()
                if entry.get("workspace_id")
            }),
        },
    }


def state_property_urls(agent, artifact_id: str) -> List[str]:
    """Where an artifact's state is read from, per its own TD.

    HASP names one state property per entity (`lightState`, `climateState`),
    so a consumer cannot build the URL as `{artifact}/properties/state`; it
    takes these. Empty when the TD advertises none (a SimuHome device names
    each attribute instead) or the engine is not available.
    """
    engine = getattr(agent, "integration_engine", None)
    if engine is None:
        return []
    urls: List[str] = []
    for affordance in engine.get_affordances_for_artifact(artifact_id):
        if getattr(affordance, "affordance_type", None) != AffordanceType.PROPERTY:
            continue
        name = str(getattr(affordance, "name", "") or "")
        href = getattr(getattr(affordance, "form", None), "href", None)
        if href and (name == "state" or name.endswith("State")):
            urls.append(str(href))
    return sorted(urls)
