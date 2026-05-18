import json

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo


class EnvironmentSemanticQueryBehaviour(CyclicBehaviour):
    """Handle semantic environment queries backed by the integration engine."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        if msg.get_metadata("type") != MessageType.ENV_SEMANTIC_QUERY_REQUEST.value:
            return

        self.agent.logger.info(demo("Received ENV_SEMANTIC_QUERY_REQUEST from %s"), str(msg.sender))

        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}

        workspace_id = payload.get("workspace_id")
        observable_properties = payload.get("observable_properties")
        if isinstance(observable_properties, str):
            observable_properties = [observable_properties]
        if not isinstance(observable_properties, list):
            observable_properties = []
        self.agent.logger.info(
            demo("Semantic query request parsed: workspace_id=%r observable_properties=%s"),
            workspace_id,
            observable_properties,
        )

        results = []
        for prop in observable_properties:
            if not isinstance(prop, str) or not prop.strip():
                continue
            self.agent.logger.info(
                demo("Semantic query -> HASP: workspace_id=%r observable_property=%r"),
                workspace_id,
                prop.strip(),
            )
            result = await self.agent.integration_engine.query_actions_affecting_observable_property(
                str(workspace_id or "").strip(),
                prop.strip(),
            )
            action_count = len(result.get("actions") or []) if isinstance(result, dict) else 0
            self.agent.logger.info(
                demo("Semantic query <- HASP: observable_property=%r actions=%d"),
                prop.strip(),
                action_count,
            )
            results.append(result)

        response_payload = {
            "workspace_id": workspace_id,
            "observable_properties": [p for p in observable_properties if isinstance(p, str) and p.strip()],
            "results": results,
        }
        self.agent.logger.info(
            demo("Semantic query response prepared: workspace_id=%r result_sets=%d"),
            workspace_id,
            len(results),
        )

        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", MessageType.ENV_SEMANTIC_QUERY_RESPONSE.value)

        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, correlation_id)
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)
