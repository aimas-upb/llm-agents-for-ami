import json

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo


class EnvironmentSemanticQueryBehaviour(CyclicBehaviour):
    """Handle semantic environment queries backed by the integration engine."""

    @staticmethod
    def _candidate_property_urls_for_observable_property(
        state_payload: dict,
        observable_property: str,
    ) -> list[str]:
        if not isinstance(state_payload, dict):
            return []
        artifacts = state_payload.get("artifacts")
        if not isinstance(artifacts, dict):
            return []

        prop = str(observable_property or "").strip().lower()
        keyword_map = {
            "glare": ("glare",),
            "luminosity": ("internal_light", "desk_light", "external_light", "illumin", "luminos"),
            "desk_luminosity": ("desk_light", "illumin", "luminos"),
            "thermal_comfort": ("temperature",),
            "humidity": ("humidity",),
            "air_quality": ("co2", "air_quality", "air quality"),
            "occupancy_presence": ("presence", "person_counter", "occup", "motion"),
            "activity_state": ("clock", "activity", "projector", "display"),
        }
        keywords = keyword_map.get(prop, (prop,))

        candidates: list[str] = []
        seen: set[str] = set()
        for artifact_id, artifact_info in artifacts.items():
            if not isinstance(artifact_info, dict):
                continue
            text = " ".join([str(artifact_id or ""), str(artifact_info.get("name") or "")]).lower()
            if not any(keyword in text for keyword in keywords):
                continue
            artifact_url = str(artifact_id or "")
            if artifact_url.endswith("#artifact"):
                artifact_url = artifact_url[: -len("#artifact")]
            if not artifact_url:
                continue
            property_url = f"{artifact_url}/properties/state"
            if property_url not in seen:
                seen.add(property_url)
                candidates.append(property_url)
        return candidates

    @staticmethod
    def _full_state_snapshot(agent) -> dict:
        artifacts_snapshot = {}
        for aid, artifact in getattr(agent, "artifacts", {}).items():
            artifacts_snapshot[aid] = {
                "name": getattr(artifact, "name", aid),
                "workspace_id": getattr(artifact, "workspace_id", None),
                "state": dict(getattr(artifact, "current_state", {}) or {}),
            }
        return {"artifacts": artifacts_snapshot}

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
        action_urls = payload.get("action_urls")
        if isinstance(observable_properties, str):
            observable_properties = [observable_properties]
        if not isinstance(observable_properties, list):
            observable_properties = []
        if isinstance(action_urls, str):
            action_urls = [action_urls]
        if not isinstance(action_urls, list):
            action_urls = []
        self.agent.logger.info(
            demo("Semantic query request parsed: workspace_id=%r observable_properties=%s action_urls=%s"),
            workspace_id,
            observable_properties,
            action_urls,
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

        action_effect_results = []
        state_payload = self._full_state_snapshot(self.agent)
        for action_url in action_urls:
            if not isinstance(action_url, str) or not action_url.strip():
                continue
            action_url = action_url.strip()
            self.agent.logger.info(
                demo("Semantic action-effects query -> HASP: workspace_id=%r action_url=%r"),
                workspace_id,
                action_url,
            )
            query_action_effects = getattr(self.agent.integration_engine, "query_action_effects", None)
            if not callable(query_action_effects):
                result = {
                    "workspace_id": workspace_id,
                    "action_url": action_url,
                    "effects": [],
                    "error": "semantic_action_effects_not_supported",
                }
            else:
                result = await query_action_effects(str(workspace_id or "").strip(), action_url)
            if isinstance(result, dict):
                for effect in result.get("effects") or []:
                    if not isinstance(effect, dict):
                        continue
                    prop_name = str(effect.get("property_name") or "").strip()
                    if not prop_name:
                        prop_uri = str(effect.get("property_uri") or "")
                        prop_name = prop_uri.rstrip("/").rsplit("/", 1)[-1] if prop_uri else ""
                    effect["readable_property_urls"] = self._candidate_property_urls_for_observable_property(
                        state_payload,
                        prop_name,
                    )
            effect_count = len(result.get("effects") or []) if isinstance(result, dict) else 0
            self.agent.logger.info(
                demo("Semantic action-effects query <- HASP: action_url=%r effects=%d"),
                action_url,
                effect_count,
            )
            action_effect_results.append(result)

        response_payload = {
            "workspace_id": workspace_id,
            "observable_properties": [p for p in observable_properties if isinstance(p, str) and p.strip()],
            "results": results,
            "action_urls": [a for a in action_urls if isinstance(a, str) and a.strip()],
            "action_effects": action_effect_results,
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
