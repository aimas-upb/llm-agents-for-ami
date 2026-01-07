"""
EnvExplorer Agent - Environment discovery and monitoring.

Classical SPADE agent that crawls, monitors, and manages environment knowledge.
"""

import asyncio
import logging
import json
import uuid
from typing import Any, Dict, List, Optional
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour, OneShotBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template

from ...shared.protocols.agent_protocol import IAgent
from ...shared.models.messages import Message, MessageType, AffordanceMatchRequest, AffordanceMatchResponse
from ...shared.models.messages import META_CORRELATION_ID, META_CONVERSATION_ID, ensure_correlation_id
from ...shared.models.environment import (
    Workspace, Artifact, Affordance, Signifier, ChangeEvent, ChangeEventType, AffordanceType
)
from ...shared.models.plan import BehaviorTreePlan
from ...shared.utils.config_resolver import resolve_yggdrasil_url
from ...shared.utils.demo_log import demo
from ...environment.connection.hmas_client import IHMASClient
from ...environment.integration.integration_engine import YggdrasilIntegration


class EnvExplorerAgent(Agent, IAgent):
    """
    EnvExplorer agent for environment discovery and monitoring.

    Responsibilities:
    - Discover and map the HMAS environment
    - Monitor for changes (artifacts, capabilities, state)
    - Store and retrieve signifiers (usage experiences)
    - Match affordances to goals
    - Notify other agents of changes
    """

    def __init__(self, jid: str, password: str, config: Dict[str, Any],
                 hmas_client: IHMASClient):
        """
        Initialize EnvExplorer agent.

        Args:
            jid: SPADE JID for the agent.
            password: SPADE password.
            config: Agent configuration.
            hmas_client: HMAS client for environment interaction.
        """
        super().__init__(jid, password)
        self.config = config or {}
        self.hmas_client = hmas_client
        self.environment_map = {}
        self.artifacts = {}
        self.affordances = {}
        self.signifiers_store = None
        self.yggdrasil_url = resolve_yggdrasil_url(self.config)
        self.integration_engine = YggdrasilIntegration(self.yggdrasil_url)
        self.discovery_complete = False
        self.logger = logging.getLogger(__name__)

        # --- Embedded RD4 signifier engine (ami_agents/shared/memory) ---
        self._rd4_engine_ready: bool = False
        self._rd4_engine_lock: asyncio.Lock = asyncio.Lock()
        self._rd4_storage_dir: Optional[str] = None
        self._rd4_registry: Any = None
        self._rd4_matcher_registry: Any = None
        self._rd4_context_builder: Any = None
        self._rd4_shacl_validator: Any = None
        self._rd4_default_matcher_version: str = "v0"
        self._rd4_default_min_similarity: float = 0.0

    async def setup(self):
        """
        Setup the agent (SPADE lifecycle method).

        Connects to the Yggdrasil HMAS instance and performs an initial
        environment crawl to populate workspace/artifact/affordance maps.
        """
        self.logger.info(demo(f"EnvExplorer booting (yggdrasil_url={self.yggdrasil_url})"))
        self.logger.info("EnvExplorerAgent starting...")
        self.add_behaviour(InitialDiscoveryBehaviour())

        # Route environment capability/state requests to the handler using templates
        cap_template = Template()
        cap_template.set_metadata("type", MessageType.ENV_CAPABILITIES_REQUEST.value)
        state_template = Template()
        state_template.set_metadata("type", MessageType.ENV_STATE_REQUEST.value)

        self.add_behaviour(EnvironmentRequestHandler(), template=cap_template)
        self.add_behaviour(EnvironmentRequestHandler(), template=state_template)

        # Signifier engine requests (embedded memory + matcher)
        sign_match_template = Template()
        sign_match_template.set_metadata("type", MessageType.SIGNIFIER_MATCH_REQUEST.value)
        sign_record_template = Template()
        sign_record_template.set_metadata("type", MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value)
        sign_list_template = Template()
        sign_list_template.set_metadata("type", MessageType.SIGNIFIER_LIST_REQUEST.value)

        self.add_behaviour(SignifierRequestHandler(), template=sign_match_template)
        self.add_behaviour(SignifierRequestHandler(), template=sign_record_template)
        self.add_behaviour(SignifierRequestHandler(), template=sign_list_template)

        self.add_behaviour(EventProcessingBehaviour(self.integration_engine))

    def _generate_capabilities_summary(self) -> str:
        """
        Formats the internal artifact map into a detailed string for the LLM.
        Includes Forms and Input Schemas so the LLM understands parameters.
        """
        # 1. Check readiness
        if not self.discovery_complete:
            return "Environment discovery is still in progress. Please try again later."

        # 2. Read from Agent Memory (populated by InitialDiscoveryBehaviour)
        artifacts = self.artifacts.values()
        
        if not artifacts:
            return "No artifacts found in the environment."

        summary = "Available Environment Capabilities:\n"
        
        for artifact in artifacts:
            # 3. Retrieve actions
            # We use the engine's helper to filter affordances for this artifact ID
            actions = self.integration_engine.get_affordances_for_artifact(artifact.artifact_id)
            
            # Filter for ACTION types (we only care about what we can DO)
            action_affordances = [a for a in actions if a.affordance_type.value == "action"]
            
            if action_affordances:
                summary += f"Artifact: {artifact.name}\n"
                summary += f"  ID: {artifact.artifact_id}\n"
                summary += f"  Capabilities:\n"
                
                for action in action_affordances:
                    summary += f"    - Action: {action.name}\n"
                    
                    # Include Form Details (Method + URL)
                    # This helps the LLM distinguish between GET (read) and POST (write)
                    if action.form:
                        summary += f"      Target: [{action.form.method}] {action.form.href}\n"
                    
                    # Include Input Schema (Parameters)
                    # This tells the LLM what arguments (e.g. brightness level) are required
                    if action.input_schema:
                        summary += f"      Schema: {json.dumps(action.input_schema)}\n"
                
                summary += "\n"
        
        return summary

    def _generate_capabilities_payload(self) -> Dict[str, Any]:
        """
        Machine-readable capabilities payload for other agents (planning, etc.).
        Includes a human-friendly 'summary' field for convenience.
        """
        if not self.discovery_complete:
            return {
                "discovery_complete": False,
                "error": "discovery_in_progress",
                "summary": "Environment discovery is still in progress. Please try again later.",
                "workspaces": [],
                "artifacts": [],
                "affordances": [],
            }

        artifacts = list(self.artifacts.values())
        if not artifacts:
            return {
                "discovery_complete": True,
                "summary": "No artifacts found in the environment.",
                "workspaces": [],
                "artifacts": [],
                "affordances": [],
            }

        # Workspaces (for multi-workspace UX and scoping)
        workspaces_out: List[Dict[str, Any]] = []
        try:
            for ws in (self.environment_map or {}).values():
                workspaces_out.append(
                    {
                        "workspace_id": ws.workspace_id,
                        "name": ws.name,
                        "workspace_type": getattr(ws.workspace_type, "value", str(ws.workspace_type)),
                        "parent_workspace_id": getattr(ws, "parent_workspace_id", None),
                        "sub_workspaces": list(getattr(ws, "sub_workspaces", []) or []),
                        "artifacts": list(getattr(ws, "artifacts", []) or []),
                    }
                )
        except Exception:
            workspaces_out = []

        affordances_out: List[Dict[str, Any]] = []
        artifacts_out: List[Dict[str, Any]] = []

        for artifact in artifacts:
            affs = self.integration_engine.get_affordances_for_artifact(artifact.artifact_id)
            action_affordances = [a for a in affs if a.affordance_type == AffordanceType.ACTION]

            actions_out: List[Dict[str, Any]] = []
            for action in action_affordances:
                form = action.form
                actions_out.append(
                    {
                        "affordance_id": action.affordance_id,
                        "name": action.name,
                        "description": action.description,
                        "artifact_id": action.artifact_id,
                        "affordance_type": action.affordance_type.value,
                        "semantic_types": list(action.semantic_types or []),
                        "form": {
                            "href": getattr(form, "href", None),
                            "method": getattr(form, "method", None),
                            "content_type": getattr(form, "content_type", None),
                            "operation_type": getattr(form, "operation_type", None),
                            "additional_fields": getattr(form, "additional_fields", None) or {},
                        }
                        if form
                        else None,
                        "input_schema": action.input_schema,
                        "output_schema": action.output_schema,
                    }
                )

                affordances_out.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "artifact_name": artifact.name,
                        "workspace_id": getattr(artifact, "workspace_id", None),
                        "affordance_id": action.affordance_id,
                        "affordance_type": action.affordance_type.value,
                        "action_name": action.name,
                        "method": getattr(action.form, "method", None) if action.form else None,
                        "target": getattr(action.form, "href", None) if action.form else None,
                        "content_type": getattr(action.form, "content_type", None) if action.form else None,
                        "input_schema": action.input_schema,
                    }
                )

            artifacts_out.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "workspace_id": getattr(artifact, "workspace_id", None),
                    "actions": actions_out,
                }
            )

        return {
            "discovery_complete": True,
            "summary": self._generate_capabilities_summary(),
            "workspaces": workspaces_out,
            "artifacts": artifacts_out,
            "affordances": affordances_out,
        }

    def _get_signifier_config(self) -> Dict[str, Any]:
        return (self.config or {}).get("signifiers", {}) or {}

    async def _ensure_rd4_engine_ready(self) -> None:
        if self._rd4_engine_ready:
            return

        async with self._rd4_engine_lock:
            if self._rd4_engine_ready:
                return

            from ...shared import memory as rd4_memory

            rd4_memory.ensure_engine_on_path()

            from src.matching.registry import IntentMatcherRegistry  # type: ignore[import-not-found]
            from src.storage.registry import SignifierRegistry  # type: ignore[import-not-found]
            from src.validation.context_builder import ContextGraphBuilder  # type: ignore[import-not-found]
            from src.validation.shacl_validator import SHACLValidator  # type: ignore[import-not-found]

            cfg = self._get_signifier_config()

            storage_dir = cfg.get("storage_dir")
            if not storage_dir:
                storage_dir = str(rd4_memory.get_default_storage_dir())

            enable_authoring_validation = bool(cfg.get("enable_authoring_validation", False))

            self._rd4_storage_dir = str(storage_dir)
            self._rd4_registry = SignifierRegistry(
                storage_dir=self._rd4_storage_dir,
                enable_authoring_validation=enable_authoring_validation,
            )

            # Default to v0 to avoid downloading embedding models unexpectedly.
            matcher_registry = IntentMatcherRegistry(default_version="v0")
            preferred_version = str(cfg.get("matcher_version") or "v0")
            if preferred_version in matcher_registry.list_versions():
                matcher_registry.set_default_version(preferred_version)
            self._rd4_matcher_registry = matcher_registry
            self._rd4_default_matcher_version = matcher_registry.get_default_version()

            try:
                self._rd4_default_min_similarity = float(cfg.get("min_similarity", 0.0))
            except Exception:
                self._rd4_default_min_similarity = 0.0

            self._rd4_context_builder = ContextGraphBuilder()
            self._rd4_shacl_validator = SHACLValidator(enable_caching=bool(cfg.get("enable_shacl_cache", False)))

            self._rd4_engine_ready = True
            self.logger.info(
                demo("RD4 signifier engine ready (storage_dir=%s, matcher_default=%s)"),
                self._rd4_storage_dir,
                self._rd4_default_matcher_version,
            )

    def _ws_match(self, value: Any, workspace_id: str) -> bool:
        if not workspace_id:
            return True
        if not value:
            return False

        ws = str(workspace_id).strip()
        s = str(value)

        if ws.startswith("http://") or ws.startswith("https://"):
            return s == ws or ws in s

        if s == ws:
            return True
        if f"/{ws}#" in s:
            return True
        if f"/{ws}/" in s:
            return True
        if s.endswith("/" + ws):
            return True
        return ws in s

    def _build_rd4_context_snapshot(self, workspace_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        context: Dict[str, Dict[str, Any]] = {}

        for artifact in (self.artifacts or {}).values():
            if workspace_id and not self._ws_match(getattr(artifact, "workspace_id", None), workspace_id):
                continue

            artifact_uri = str(getattr(artifact, "artifact_id", "") or "")
            if not artifact_uri:
                continue

            state = getattr(artifact, "current_state", {}) or {}
            if not isinstance(state, dict):
                continue

            # ContextGraphBuilder expects: {artifact_uri: {property_uri: value}}
            context[artifact_uri] = {str(k): v for k, v in state.items()}

        return context

    async def _rd4_list_signifiers(self) -> Dict[str, Any]:
        await self._ensure_rd4_engine_ready()

        signifiers = self._rd4_registry.list_signifiers(limit=10000) if self._rd4_registry else []
        out = []
        for s in signifiers:
            out.append(
                {
                    "signifier_id": s.signifier_id,
                    "version": s.version,
                    "status": getattr(s.status, "value", str(s.status)),
                    "intent": getattr(s.intent, "nl_text", ""),
                    "affordance_uri": s.affordance_uri,
                }
            )

        return {"total": len(out), "signifiers": out}

    async def _rd4_match_signifiers(
        self,
        *,
        intent: str,
        workspace_id: Optional[str] = None,
        context: Optional[Dict[str, Dict[str, Any]]] = None,
        k: int = 10,
        matcher_version: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> Dict[str, Any]:
        await self._ensure_rd4_engine_ready()

        intent = str(intent or "").strip()
        if not intent:
            return {"error": "missing_intent", "matches": [], "final_matches": [], "total_signifiers": 0}

        if context is None:
            context = self._build_rd4_context_snapshot(workspace_id=workspace_id)

        all_signifiers = self._rd4_registry.list_signifiers(limit=10000) if self._rd4_registry else []
        signifier_dicts = [s.model_dump() for s in all_signifiers]

        if not signifier_dicts:
            self.logger.info(demo("Signifier match: 0 stored signifiers (storage empty)."))
            return {"matches": [], "final_matches": [], "total_signifiers": 0}

        version_to_use = str(matcher_version or self._rd4_default_matcher_version or "v0")
        if min_similarity is None:
            min_similarity = self._rd4_default_min_similarity

        self.logger.info(
            demo("Signifier match: intent=%r matcher=%s min_similarity=%s k=%s"),
            intent,
            version_to_use,
            min_similarity,
            k,
        )

        try:
            match_results = self._rd4_matcher_registry.match(
                intent_query=intent,
                signifiers=signifier_dicts,
                k=int(k),
                version=version_to_use,
                min_similarity=float(min_similarity),
            )
        except Exception:
            version_to_use = "v0"
            match_results = self._rd4_matcher_registry.match(
                intent_query=intent,
                signifiers=signifier_dicts,
                k=int(k),
                version=version_to_use,
            )

        context_graph, _ = self._rd4_context_builder.normalize_context(context)

        matches: List[Dict[str, Any]] = []
        for match in match_results:
            s = self._rd4_registry.get(match.signifier_id)
            if not s:
                continue

            shacl_conforms = True
            shacl_violations: List[str] = []

            if getattr(getattr(s, "context", None), "shacl_shapes", None):
                validation = self._rd4_shacl_validator.validate_signifier_context(
                    context_graph,
                    s.context.shacl_shapes,
                    format="turtle",
                )
                shacl_conforms = bool(validation.conforms)
                shacl_violations = [v.message for v in (validation.violations or [])]

            structured = getattr(getattr(s, "intent", None), "structured", None)
            payload_hint = structured.get("payload") if isinstance(structured, dict) else None

            matches.append(
                {
                    "signifier_id": s.signifier_id,
                    "affordance_uri": s.affordance_uri,
                    "intent": getattr(s.intent, "nl_text", ""),
                    "intent_similarity": round(float(match.similarity), 4),
                    "matcher_version": version_to_use,
                    "shacl_conforms": shacl_conforms,
                    "shacl_violations": shacl_violations,
                    "payload_hint": payload_hint,
                }
            )

        final_matches = [m["signifier_id"] for m in matches if m.get("shacl_conforms")]

        self.logger.info(
            demo("Signifier match result: matches=%d final_matches=%d"),
            len(matches),
            len(final_matches),
        )

        return {
            "intent": intent,
            "matches": matches,
            "final_matches": final_matches,
            "total_signifiers": len(signifier_dicts),
        }

    async def _rd4_record_execution(
        self,
        *,
        plan: Any,
        execution_report: Any,
        sender: Optional[str] = None,
        thread: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_rd4_engine_ready()

        plan_obj = plan
        if isinstance(plan_obj, str):
            try:
                plan_obj = json.loads(plan_obj)
            except Exception:
                plan_obj = None

        if not isinstance(plan_obj, dict):
            return {"ok": False, "error": "invalid_plan"}

        steps = plan_obj.get("steps")
        if not isinstance(steps, list) or not steps:
            return {"ok": False, "error": "missing_steps"}

        ok_step_ids: set[str] = set()
        if isinstance(execution_report, dict):
            results = execution_report.get("results")
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict) or not r.get("ok"):
                        continue
                    sid = r.get("step_id")
                    if sid is not None:
                        ok_step_ids.add(str(sid))

        from src.models.signifier import (  # type: ignore[import-not-found]
            IntentContext,
            IntentionDescription,
            Provenance,
            Signifier as RD4Signifier,
            SignifierStatus,
        )

        created: List[str] = []
        skipped: List[Dict[str, Any]] = []

        for step in steps:
            if not isinstance(step, dict):
                skipped.append({"error": "invalid_step"})
                continue

            step_id = step.get("step_id")
            if ok_step_ids and step_id is not None and str(step_id) not in ok_step_ids:
                continue

            intent = str(step.get("intent") or "").strip()
            affordance_uri = str(step.get("affordance_uri") or step.get("affordance_id") or "").strip()
            if not intent or not affordance_uri:
                skipped.append({"step_id": step_id, "error": "missing_intent_or_affordance"})
                continue

            payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
            action_name = step.get("action_name")
            evidence = []
            for reason in step.get("reasons") or []:
                if not isinstance(reason, dict):
                    continue
                ev_list = reason.get("evidence") or []
                if isinstance(ev_list, list):
                    evidence.extend([ev for ev in ev_list if isinstance(ev, dict)])

            signifier_id = f"exec-{uuid.uuid4().hex}"

            # Best-effort SHACL context shape:
            # Require that the artifact+property URIs referenced in evidence exist in the current context graph.
            # This demonstrates "context match" while remaining stable across simple state resets (values may change,
            # but the relevant artifacts/properties remain present).
            shacl_shapes = None
            try:
                props_by_artifact: Dict[str, set[str]] = {}
                for ev in evidence:
                    art = ev.get("artifact")
                    prop = ev.get("property")
                    if not (isinstance(art, str) and art.startswith("http")):
                        continue
                    if not (isinstance(prop, str) and prop.startswith("http")):
                        continue
                    props_by_artifact.setdefault(art, set()).add(prop)

                if props_by_artifact:
                    ttl_lines = [
                        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
                        "",
                    ]
                    shape_idx = 0
                    for art_uri, prop_uris in props_by_artifact.items():
                        shape_idx += 1
                        shape_iri = f"<urn:ami:signifier:{signifier_id}:shape:{shape_idx}>"
                        ttl_lines.append(f"{shape_iri} a sh:NodeShape ;")
                        ttl_lines.append(f"  sh:targetNode <{art_uri}> ;")
                        for p_uri in sorted(prop_uris):
                            ttl_lines.append("  sh:property [")
                            ttl_lines.append(f"    sh:path <{p_uri}> ;")
                            ttl_lines.append("    sh:minCount 1 ;")
                            ttl_lines.append("  ] ;")
                        # Replace final ';' with '.' to end the shape.
                        if ttl_lines and ttl_lines[-1].endswith(";"):
                            ttl_lines[-1] = ttl_lines[-1].rstrip(";").rstrip() + "."
                        ttl_lines.append("")

                    shacl_shapes = "\n".join(ttl_lines).strip() or None
            except Exception:
                shacl_shapes = None

            intent_structured = {"intent": intent, "payload": payload}
            if action_name:
                intent_structured["action_name"] = str(action_name)

            ctx_meta = {
                "workspace_id": workspace_id,
                "thread": thread,
                "step_id": step_id,
                "evidence": evidence,
            }

            signifier = RD4Signifier(
                signifier_id=signifier_id,
                version=1,
                status=SignifierStatus.ACTIVE,
                intent=IntentionDescription(nl_text=intent, structured=intent_structured),
                context=IntentContext(nl_description=json.dumps(ctx_meta), structured_conditions=[], shacl_shapes=shacl_shapes),
                affordance_uri=affordance_uri,
                provenance=Provenance(created_by=str(sender or self.jid), source="execution"),
            )

            try:
                self._rd4_registry.create(signifier)
                created.append(signifier_id)
                self.logger.info(
                    demo("Stored signifier (no SHACL context) %s: intent=%r -> affordance=%s"),
                    signifier_id,
                    intent,
                    affordance_uri,
                )
            except Exception as e:
                skipped.append({"step_id": step_id, "error": "create_failed", "detail": str(e)})

        return {
            "ok": True,
            "created_count": len(created),
            "created_ids": created,
            "skipped": skipped,
        }

    async def start(self, *args, **kwargs) -> None:
        """
        Start the EnvExplorer agent.

        Uses SPADE's start to trigger setup/discovery.
        """
        return await super().start(*args, **kwargs)

    async def stop(self) -> None:
        """
        Stop the EnvExplorer agent.

        TODO: Implementation steps:
        1. Stop all behaviors
        2. Unsubscribe from environment events
        3. Disconnect HMAS client
        4. Close signifier storage
        5. Cleanup resources
        """
        if self.integration_engine:
            await self.integration_engine.stop_notification_listener()
        await super().stop()

    async def send_message(self, message: Message) -> bool:
        """
        Send a message to another agent.

        TODO: Implementation steps:
        1. Validate message
        2. Serialize to SPADE message format
        3. Send via SPADE
        4. Log sent message
        5. Return success status
        """
        pass

    async def receive_message(self, message: Message) -> None:
        """
        Receive and process a message.

        TODO: Implementation steps:
        1. Deserialize SPADE message
        2. Route based on message type
        3. Handle appropriately
        """
        pass


class InitialDiscoveryBehaviour(OneShotBehaviour):
    """Behavior for initial environment discovery."""

    async def run(self):
        """
        Perform initial discovery of the environment.

        TODO: Implementation steps:
        1. Wait for agent to be ready
        2. Crawl entire HMAS environment
        3. Build environment map (workspaces)
        4. Collect all artifacts and Thing Descriptions
        5. Extract affordances from Thing Descriptions
        6. Store in agent's data structures
        7. Subscribe to all workspace change notifications
        8. Mark discovery as complete
        9. Notify UserAssistant and InteractionSolver
        10. Stop this behavior (one-time only)
        """
        self.agent.logger.info(demo("Starting EnvExplorer discovery..."))
        
        success = await self.agent.integration_engine.initialize({})
        if not success:
            self.agent.logger.error("Failed to initialize Integration Engine.")
            return

        # Start webhook listener for event notifications before subscribing
        callback_url = await self.agent.integration_engine.start_notification_listener()
        
        try:
            await self.agent.integration_engine.explore_hmas_environment()
            
            self.agent.environment_map = self.agent.integration_engine.workspace_map
            self.agent.artifacts = self.agent.integration_engine.artifact_map
            self.agent.affordances = self.agent.integration_engine.affordance_map

            self.agent.logger.info("Subscribing to artifact events...")
            for artifact_id, artifact in self.agent.artifacts.items():
                # The engine uses the internal listener automatically (callback_url=None)
                success = await self.agent.integration_engine.subscribe_to_artifact(
                    artifact_id, callback_url=callback_url
                )
                if success:
                    self.agent.logger.debug(f"Subscribed to {artifact.name}")
                else:
                    self.agent.logger.warning(f"Could not subscribe to {artifact.name} (might be static)")
            
            self.agent.discovery_complete = True
            self.agent.logger.info(f"Discovery Complete. Found {len(self.agent.artifacts)} artifacts.")
            self.agent.logger.info(
                demo("Discovery summary: workspaces=%d artifacts=%d affordances=%d"),
                len(self.agent.environment_map or {}),
                len(self.agent.artifacts or {}),
                len(self.agent.affordances or {}),
            )
            
            await self.notify_discovery_complete()
            
        except Exception as e:
            self.agent.logger.error(f"Error during discovery: {e}", exc_info=True)

        pass

    async def subscribe_to_changes(self) -> None:
        """
        Subscribe to environment change notifications.

        TODO: Implementation steps:
        1. For each workspace in environment_map:
           a. Subscribe to workspace change events
           b. Register callback for change handling
        """
        pass

    async def notify_discovery_complete(self) -> None:
        """
        Notify other agents that discovery is complete.

        TODO: Implementation steps:
        1. Create ENV_DISCOVERY_COMPLETE message
        2. Send to UserAssistant
        3. Send to InteractionSolver
        4. Log notification sent
        """
        discovery_cfg = (self.agent.config or {}).get("discovery", {}) or {}
        if not discovery_cfg.get("notify_on_discovery_complete", True):
            return

        notify_agents = discovery_cfg.get("notify_agents") or []
        if not notify_agents:
            self.agent.logger.info("Discovery complete: no notify_agents configured.")
            return

        payload = {
            "discovery_complete": True,
            "artifacts_count": len(self.agent.artifacts or {}),
            "affordances_count": len(self.agent.affordances or {}),
            "yggdrasil_url": getattr(self.agent, "yggdrasil_url", None),
        }

        sent = 0
        for jid in notify_agents:
            try:
                msg = SpadeMessage(to=str(jid))
                msg.set_metadata("type", MessageType.ENV_DISCOVERY_COMPLETE.value)
                msg.set_metadata(META_CORRELATION_ID, ensure_correlation_id({}))
                msg.body = json.dumps(payload)
                await self.send(msg)
                sent += 1
            except Exception as e:
                self.agent.logger.warning(f"Failed to notify {jid} of discovery complete: {e}")

        self.agent.logger.info(f"Discovery complete notification sent to {sent}/{len(notify_agents)} agents.")

class EventProcessingBehaviour(CyclicBehaviour):
    """
    Behavior for processing asynchronous environment events from the mailbox.
    """
    def __init__(self, integration_engine):
        super().__init__()
        self.integration = integration_engine

    async def run(self):
        # 1. Block until an event arrives (efficient)
        try:
            # Check if listener is running
            if not self.integration.notification_listener:
                await asyncio.sleep(1) # Wait for setup
                return

            event_data = await self.integration.event_queue.get()
            
            # 2. Extract Identity
            # Yggdrasil sends "artifactUri" in the payload
            artifact_uri = event_data.get("artifactUri")
            if not artifact_uri:
                return

            # 3. Find Local Artifact
            # Handle potential suffix mismatch (http://.../light vs http://.../light#artifact)
            artifact = self.integration.artifact_map.get(artifact_uri)
            if not artifact:
                # Try fuzzy match if exact match fails
                artifact = next((a for a in self.integration.artifact_map.values() 
                                    if artifact_uri in a.artifact_id or a.artifact_id in artifact_uri), None)
            
            if not artifact:
                self.agent.logger.warning(f"Received event for unknown artifact: {artifact_uri}")
                return

            # 4. Update State (Digital Twin)
            property_uri = event_data.get("propertyUri")
            value = event_data.get("value")
            
            if property_uri and value is not None:
                # Update Internal State
                artifact.current_state[property_uri] = value
                self.agent.logger.info(f"STATE UPDATE: {artifact.name} -> {property_uri} = {value}")
                
        except Exception as e:
            self.agent.logger.error(f"Error processing event: {e}")

class ChangeMonitoringBehaviour(PeriodicBehaviour):
    """Behavior for monitoring environment changes."""

    async def run(self):
        """
        Periodically check for environment changes.

        TODO: Implementation steps:
        1. Check for new change events
        2. For each change event:
           a. Update internal data structures
           b. Analyze impact on existing plans
           c. Notify UserAssistant if plans affected
        """
        pass

    async def handle_change_event(self, event: ChangeEvent) -> None:
        """
        Handle a specific change event.

        TODO: Implementation steps:
        1. Based on event type:
           - ARTIFACT_ADDED: Add to artifacts map
           - ARTIFACT_REMOVED: Remove from artifacts map
           - CAPABILITY_CHANGED: Update affordances
           - STATE_CHANGED: Update artifact state
        2. Analyze plan impact
        3. Send notifications if needed
        """
        pass

    async def analyze_plan_impact(self, event: ChangeEvent) -> List[str]:
        """
        Analyze which plans are affected by the change.

        TODO: Implementation steps:
        1. Query UserAssistant for maintenance plans
        2. For each plan:
           a. Check if change affects triggering conditions
           b. Check if change affects plan affordances
        3. Collect affected plan_ids
        4. Return list
        """
        pass


class EnvironmentRequestHandler(CyclicBehaviour):
    """Generic handler for incoming environment requests."""

    async def run(self):
        """
        Handle affordance match requests.

        TODO: Implementation steps:
        1. Wait for AffordanceMatchRequest message
        2. Process request
        3. Send AffordanceMatchResponse back
        """
        msg = await self.receive(timeout=1)
        if msg:
            msg_type = msg.get_metadata("type")
            
            # Check for capabilities request
            if msg_type == MessageType.ENV_CAPABILITIES_REQUEST.value:
                self.agent.logger.info(demo("Received ENV_CAPABILITIES_REQUEST from %s"), str(msg.sender))
                
                # Generate Response (machine-readable JSON payload + summary)
                response_payload = self.agent._generate_capabilities_payload()
                
                # Send Reply
                reply = msg.make_reply()
                reply.body = json.dumps(response_payload)
                reply.set_metadata("type", MessageType.ENV_CAPABILITIES_RESPONSE.value)
                
                # Preserve Correlation ID
                correlation_id = msg.get_metadata(META_CORRELATION_ID)
                if correlation_id:
                    reply.set_metadata(META_CORRELATION_ID, correlation_id)
                # Preserve thread as conversation id carrier (if set)
                if msg.thread:
                    reply.thread = msg.thread
                    
                await self.send(reply)
            
            # Check for state request (full snapshot or filtered)
            elif msg_type == MessageType.ENV_STATE_REQUEST.value:
                self.agent.logger.info(demo("Received ENV_STATE_REQUEST from %s"), str(msg.sender))

                try:
                    payload = json.loads(msg.body or "{}")
                except json.JSONDecodeError:
                    payload = {}

                artifact_id = (
                    payload.get("artifact_id")
                    or payload.get("artifact")
                    or payload.get("artifact_uri")
                )
                property_uri = payload.get("property_uri") or payload.get("property")

                response_payload = {}

                if artifact_id:
                    artifact = self.agent.artifacts.get(artifact_id)
                    if not artifact:
                        response_payload = {
                            "error": "artifact_not_found",
                            "artifact_id": artifact_id,
                        }
                    else:
                        state = dict(artifact.current_state)
                        if property_uri:
                            if property_uri in state:
                                response_payload = {
                                    "artifact_id": artifact_id,
                                    "property_uri": property_uri,
                                    "value": state.get(property_uri),
                                }
                            else:
                                response_payload = {
                                    "error": "property_not_found",
                                    "artifact_id": artifact_id,
                                    "property_uri": property_uri,
                                }
                        else:
                            response_payload = {
                                "artifact_id": artifact_id,
                                "name": artifact.name,
                                "workspace_id": getattr(artifact, "workspace_id", None),
                                "state": state,
                            }
                else:
                    artifacts_snapshot = {}
                    for aid, artifact in self.agent.artifacts.items():
                        artifacts_snapshot[aid] = {
                            "name": artifact.name,
                            "workspace_id": getattr(artifact, "workspace_id", None),
                            "state": dict(artifact.current_state),
                        }
                    response_payload = {"artifacts": artifacts_snapshot}

                reply = msg.make_reply()
                reply.body = json.dumps(response_payload)
                reply.set_metadata("type", MessageType.ENV_STATE_RESPONSE.value)

                correlation_id = msg.get_metadata("correlation_id")
                if correlation_id:
                    reply.set_metadata("correlation_id", correlation_id)
                if msg.thread:
                    reply.thread = msg.thread

                await self.send(reply)
        pass

    async def match_affordances(self, request: AffordanceMatchRequest) -> AffordanceMatchResponse:
        """
        Match affordances to a goal.

        TODO: Implementation steps:
        1. Extract goal intent
        2. Search signifiers for similar intents
        3. Rank signifiers by:
           a. Intent similarity
           b. Context similarity
        4. If signifiers found, prioritize them
        5. If no signifiers or need more options:
           a. Use hybrid reasoning (rule-based + LLM)
           b. Apply physics-informed modeling
           c. Find suitable affordances
        6. Compile list of matched affordances
        7. Create and return AffordanceMatchResponse
        """
        pass

    async def search_signifiers(self, intent: str, context: Dict[str, Any],
                               threshold: float = 0.8) -> List[Signifier]:
        """
        Search for signifiers matching an intent.

        TODO: Implementation steps:
        1. Query signifier storage
        2. Compute intent similarity (using embeddings)
        3. Filter by threshold
        4. If context matching enabled:
           a. Compute context similarity
           b. Filter by context threshold
        5. Sort by similarity scores
        6. Return matched signifiers
        """
        pass

    async def hybrid_affordance_matching(self, intent: str,
                                        context: Dict[str, Any]) -> List[Affordance]:
        """
        Use hybrid reasoning to match affordances.

        TODO: Implementation steps:
        1. Apply rule-based matching:
           a. Extract keywords from intent
           b. Match against affordance descriptions
        2. Apply LLM reasoning:
           a. Prepare prompt with intent and affordances
           b. Ask LLM to select suitable affordances
           c. Parse LLM response
        3. If physics-informed modeling enabled:
           a. Consider physical constraints
           b. Consider device capabilities
        4. Combine results
        5. Return ranked affordances
        """
        pass


class SignifierRequestHandler(CyclicBehaviour):
    """Handler for RD4 signifier engine requests (embedded in EnvExplorer)."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        msg_type = msg.get_metadata("type")

        if msg_type == MessageType.SIGNIFIER_LIST_REQUEST.value:
            self.agent.logger.info(demo("Received SIGNIFIER_LIST_REQUEST from %s"), str(msg.sender))
            response_payload = await self.agent._rd4_list_signifiers()
            reply_type = MessageType.SIGNIFIER_LIST_RESPONSE.value

        elif msg_type == MessageType.SIGNIFIER_MATCH_REQUEST.value:
            self.agent.logger.info(demo("Received SIGNIFIER_MATCH_REQUEST from %s"), str(msg.sender))
            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                payload = {}

            intent = payload.get("intent") or payload.get("query") or payload.get("goal") or ""
            workspace_id = payload.get("workspace_id")
            k = payload.get("k", 10)
            matcher_version = payload.get("matcher_version")
            min_similarity = payload.get("min_similarity")

            response_payload = await self.agent._rd4_match_signifiers(
                intent=str(intent),
                workspace_id=str(workspace_id) if workspace_id else None,
                k=int(k) if str(k).isdigit() else 10,
                matcher_version=str(matcher_version) if matcher_version else None,
                min_similarity=float(min_similarity) if min_similarity is not None else None,
            )
            reply_type = MessageType.SIGNIFIER_MATCH_RESPONSE.value

        elif msg_type == MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value:
            self.agent.logger.info(demo("Received SIGNIFIER_RECORD_EXECUTION_REQUEST from %s"), str(msg.sender))
            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                payload = {}

            response_payload = await self.agent._rd4_record_execution(
                plan=payload.get("plan"),
                execution_report=payload.get("execution_report") or payload.get("execution") or {},
                sender=str(msg.sender) if getattr(msg, "sender", None) else None,
                thread=str(msg.thread) if getattr(msg, "thread", None) else None,
                workspace_id=payload.get("workspace_id"),
            )
            reply_type = MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value

        else:
            return

        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", reply_type)

        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)


class SignifierManager:
    """Manages storage and retrieval of signifiers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SignifierManager.

        Args:
            config: Signifier configuration.
        """
        self.config = config
        self.storage = None  # Database connection

    async def initialize_storage(self) -> None:
        """
        Initialize signifier storage backend.

        TODO: Implementation steps:
        1. Connect to database (SQLite, etc.)
        2. Create tables if not exist
        3. Load indices for fast retrieval
        """
        pass

    async def store_signifier(self, signifier: Signifier) -> None:
        """
        Store a signifier.

        TODO: Implementation steps:
        1. Check if similar signifier exists
        2. If exists, increment usage_count
        3. Otherwise, insert new signifier
        4. Update indices
        """
        pass

    async def retrieve_signifiers_by_intent(self, intent: str,
                                           threshold: float = 0.8) -> List[Signifier]:
        """
        Retrieve signifiers matching an intent.

        TODO: Implementation steps:
        1. Generate intent embedding
        2. Query vector store or compute similarities
        3. Filter by threshold
        4. Return matched signifiers
        """
        pass

    async def retrieve_signifiers_by_affordance(self, affordance_id: str) -> List[Signifier]:
        """
        Retrieve all signifiers for a specific affordance.

        TODO: Implementation steps:
        1. Query database for affordance_id
        2. Return all matching signifiers
        """
        pass

    async def extract_signifiers_from_plan(self, plan: BehaviorTreePlan) -> List[Signifier]:
        """
        Extract signifiers from a behavior tree plan.

        TODO: Implementation steps:
        1. Get all leaf action nodes from plan
        2. For each action node:
           a. Create signifier with affordance_id
           b. Record intent (from plan goal)
           c. Record context (from plan metadata)
           d. Record parameters used
        3. Store signifiers
        4. Return list of created signifiers
        """
        pass

    async def get_context_for_signifier(self, signifier_id: str) -> Dict[str, Any]:
        """
        Get the context of use for a signifier.

        TODO: Implementation steps:
        1. Retrieve signifier from database
        2. Return context dictionary
        """
        pass
