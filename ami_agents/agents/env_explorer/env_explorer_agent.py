"""
EnvExplorer Agent - Environment discovery and monitoring.

Refactored SPADE agent with proper package structure for behaviors and utilities.
"""

import asyncio
import logging
import json
import re
import uuid
from typing import Any, Dict, List, Optional
from spade.agent import Agent
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

# Import extracted behaviors
from .behaviors import (
    InitialDiscoveryBehaviour,
    EventProcessingBehaviour,
    EnvironmentRequestHandler,
    SignifierRequestHandler
)

# Import extracted utilities
from .utils import (
    artifact_tokens, tokens_from_identifier,
    intent_compatible, workspace_match, rank_signifier_matches,
    format_capabilities_summary, format_capabilities_payload,
    ensure_rd4_ready, get_signifier_config, build_rd4_context_snapshot,
    list_rd4_signifiers, generate_shacl_shapes_from_conditions, generate_nl_description
)

# Configuration constants as fallbacks
_RD4_DEFAULTS = {
    "default_matcher_version": "v2",
    "default_min_similarity": 0.5,
    "signifier_limit": 10000,
}
_TIMEOUTS = {
    "message_reception": 5.0,
}


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
            hmas_client: HMAS client instance.
        """
        super().__init__(jid, password)

        self.config = config or {}
        self.hmas_client = hmas_client

        # Environment state
        self.environment_map: Dict[str, Workspace] = {}
        self.artifacts: Dict[str, Artifact] = {}
        self.affordances: Dict[str, Affordance] = {}
        self.signifiers_store = None

        # Integration engine
        self.yggdrasil_url = resolve_yggdrasil_url(config)
        self.integration_engine = YggdrasilIntegration(
            self.yggdrasil_url, hmas_client
        )

        # Agent state
        self.discovery_complete = False
        self.logger = logging.getLogger(f"EnvExplorerAgent[{jid}]")

        # RD4 engine state
        self._rd4_engine_ready = False
        self._rd4_engine_lock = asyncio.Lock()
        self._rd4_storage_dir: Optional[str] = None
        self._rd4_registry = None
        self._rd4_matcher_registry = None
        self._rd4_context_builder = None
        self._rd4_shacl_validator = None
        self._rd4_default_matcher_version = self.config.get("rd4_engine", {}).get("default_matcher_version", _RD4_DEFAULTS["default_matcher_version"])
        self._rd4_default_min_similarity = self.config.get("rd4_engine", {}).get("default_min_similarity", _RD4_DEFAULTS["default_min_similarity"])
        self._rd4_shacl_validation_enabled = self.config.get("rd4_engine", {}).get("shacl_validation_enabled", "false").lower() == "true"

    async def setup(self):
        """Set up the agent behaviors and templates."""
        self.logger.info(demo("Setting up EnvExplorerAgent..."))

        # Create message templates
        env_cap_template = Template()
        env_cap_template.set_metadata("type", MessageType.ENV_CAPABILITIES_REQUEST.value)

        env_state_template = Template()
        env_state_template.set_metadata("type", MessageType.ENV_STATE_REQUEST.value)

        # Signifier engine requests (need separate templates for each type)
        sign_match_template = Template()
        sign_match_template.set_metadata("type", MessageType.SIGNIFIER_MATCH_REQUEST.value)

        sign_record_template = Template()
        sign_record_template.set_metadata("type", MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value)

        sign_list_template = Template()
        sign_list_template.set_metadata("type", MessageType.SIGNIFIER_LIST_REQUEST.value)

        # Add behaviors using extracted classes (FIXED: original order restored)
        self.add_behaviour(InitialDiscoveryBehaviour())
        self.add_behaviour(EnvironmentRequestHandler(), template=env_cap_template)
        self.add_behaviour(EnvironmentRequestHandler(), template=env_state_template)
        self.add_behaviour(SignifierRequestHandler(), template=sign_match_template)
        self.add_behaviour(SignifierRequestHandler(), template=sign_record_template)
        self.add_behaviour(SignifierRequestHandler(), template=sign_list_template)
        self.add_behaviour(EventProcessingBehaviour(self.integration_engine))  # MOVED TO END

    # Utility methods that delegate to extracted utilities
    def _generate_capabilities_summary(self) -> str:
        """Generate capabilities summary for human consumption."""
        return format_capabilities_summary(self)

    def _generate_capabilities_payload(self) -> Dict[str, Any]:
        """Generate capabilities payload for machine consumption."""
        return format_capabilities_payload(self)

    def _get_signifier_config(self) -> Dict[str, Any]:
        """Get signifier configuration with defaults."""
        return get_signifier_config(self.config)

    def _ws_match(self, value: Any, workspace_id: str) -> bool:
        """Check if workspace matches pattern."""
        return workspace_match(value, workspace_id)

    def _build_rd4_context_snapshot(self, workspace_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Build RD4 context snapshot from current state."""
        return build_rd4_context_snapshot(self, workspace_id)

    async def _ensure_rd4_engine_ready(self) -> None:
        """Ensure RD4 engine is ready."""
        await ensure_rd4_ready(self)

    def _rd4_intent_compatible(
        self,
        intent_query: Optional[str] = None,
        signifier_intent: Optional[str] = None,
        affordance_uri: Optional[str] = None,
        payload_hint: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check intent compatibility with affordance semantics."""
        return intent_compatible(intent_query, signifier_intent, affordance_uri, payload_hint)

    async def _rd4_list_signifiers(self) -> Dict[str, Any]:
        """List all stored signifiers."""
        return await list_rd4_signifiers(self, _RD4_DEFAULTS)

    async def _rd4_match_signifiers(
        self,
        intent: str,
        workspace_id: Optional[str] = None,
        k: int = 10,
        matcher_version: Optional[str] = None,
        min_similarity: Optional[float] = None,
        intent_type: Optional[str] = None,
        query_structured_intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Match signifiers using RD4 engine."""
        await self._ensure_rd4_engine_ready()
        if not self._rd4_matcher_registry or not self._rd4_registry:
            return {"ok": False, "error": "rd4_engine_not_ready"}

        if not intent:
            return {"error": "missing_intent", "matches": [], "final_matches": [], "total_signifiers": 0}

        context = self._build_rd4_context_snapshot(workspace_id=workspace_id)

        try:
            all_signifiers = self._rd4_registry.list_signifiers(limit=10000)
            signifier_dicts = [s.model_dump() for s in all_signifiers]

            if not signifier_dicts:
                self.logger.info(demo("Signifier match: 0 stored signifiers (storage empty)."))
                return {"matches": [], "final_matches": [], "total_signifiers": 0}


            version_to_use = str(matcher_version or self._rd4_default_matcher_version or "v0")
            if min_similarity is None:
                min_similarity = self._rd4_default_min_similarity

            self.logger.info(
                demo("[INTENT_TYPE] Signifier match: intent=%r matcher=%s min_similarity=%s k=%s intent_type=%r"),
                intent,
                version_to_use,
                min_similarity,
                k,
                intent_type,
            )
            print(f"[MATCHER_VERSION_DEBUG] Using matcher version: {version_to_use}")

            # Log context validation strategy
            if intent_type == "EXPLICIT":
                self.logger.info(
                    demo("[INTENT_TYPE] Context validation DISABLED for EXPLICIT intent (exact target specified)")
                )
            elif intent_type == "IMPLICIT":
                self.logger.info(
                    demo("[INTENT_TYPE] Context validation ENABLED for IMPLICIT intent (inferring from environment)")
                )
            else:
                self.logger.info(
                    demo("[INTENT_TYPE] Context validation ENABLED (intent_type not specified, defaulting to validation)")
                )

            # Use RD4 matcher registry for matching
            print(f"[V2_DEBUG] About to call matcher with {len(signifier_dicts)} signifiers")
            print(f"[V2_DEBUG] Query: intent='{intent}', version='{version_to_use}', query_structured_intent={query_structured_intent}")
            try:
                match_results = self._rd4_matcher_registry.match(
                    intent_query=intent,
                    signifiers=signifier_dicts,
                    k=int(k),
                    version=version_to_use,
                    min_similarity=float(min_similarity),
                    query_structured_intent=query_structured_intent,
                )
                print(f"[V2_DEBUG] Matcher returned {len(match_results) if match_results else 0} results")
                for i, result in enumerate(match_results[:3]):  # Show first 3
                    print(f"[V2_DEBUG] Result {i}: similarity={getattr(result, 'similarity', 'N/A')}, signifier_id={getattr(result, 'signifier_id', 'N/A')}")

            except Exception as e:
                self.logger.error(
                    demo("!!! V2 MATCHER FAILED, falling back to v0: %s - %s"),
                    type(e).__name__,
                    str(e),
                    exc_info=True,
                )
                print(f"[V2_MATCHER_ERROR] Exception: {type(e).__name__}: {str(e)}")
                version_to_use = "v0"
                match_results = self._rd4_matcher_registry.match(
                    intent_query=intent,
                    signifiers=signifier_dicts,
                    k=int(k),
                    version=version_to_use,
                    query_structured_intent=query_structured_intent,
                )

            context_graph, _ = self._rd4_context_builder.normalize_context(context)

            matches: List[Dict[str, Any]] = []
            print(f"[REGISTRY_DEBUG] Processing {len(match_results)} match results")
            for i, match in enumerate(match_results):
                print(f"[REGISTRY_DEBUG] Match {i}: signifier_id={match.signifier_id}, similarity={getattr(match, 'similarity', 'N/A')}")
                s = self._rd4_registry.get(match.signifier_id)
                if not s:
                    print(f"[REGISTRY_DEBUG] Signifier {match.signifier_id} NOT FOUND in registry!")
                    continue
                print(f"[REGISTRY_DEBUG] Signifier {match.signifier_id} found in registry")

                shacl_conforms = True
                shacl_violations: List[str] = []

                # EXPLICIT intents: skip context validation (user specified exact target)
                # IMPLICIT intents: validate context (need to infer from environment state)
                should_validate_context = (intent_type != "EXPLICIT")

                print(f"[SHACL_DEBUG] should_validate_context={should_validate_context}, intent_type='{intent_type}'")
                print(f"[SHACL_DEBUG] shacl_validation_enabled={self._rd4_shacl_validation_enabled}")
                print(f"[SHACL_DEBUG] signifier has shacl_shapes: {bool(getattr(getattr(s, 'context', None), 'shacl_shapes', None))}")

                if should_validate_context and self._rd4_shacl_validation_enabled and getattr(getattr(s, "context", None), "shacl_shapes", None):
                    print(f"[SHACL_DEBUG] Running SHACL validation for signifier {s.signifier_id}")
                    validation = self._rd4_shacl_validator.validate_signifier_context(
                        context_graph,
                        s.context.shacl_shapes,
                        format="turtle",
                    )
                    # ValidationResult object has .conforms and .violations attributes
                    shacl_conforms = bool(getattr(validation, 'conforms', False))
                    violations = getattr(validation, 'violations', []) or []
                    shacl_violations = [getattr(v, 'message', str(v)) for v in violations]
                    print(f"[SHACL_DEBUG] SHACL validation result: conforms={shacl_conforms}, violations={len(violations)}")
                else:
                    skip_reason = []
                    if not should_validate_context:
                        skip_reason.append("explicit intent")
                    if not self._rd4_shacl_validation_enabled:
                        skip_reason.append("validation disabled in config")
                    if not getattr(getattr(s, "context", None), "shacl_shapes", None):
                        skip_reason.append("no SHACL shapes")
                    print(f"[SHACL_DEBUG] Skipping SHACL validation ({', '.join(skip_reason)})")

                print(f"[SHACL_DEBUG] Final shacl_conforms value: {shacl_conforms}")

                # RESTORED: Check intent compatibility (was missing in refactoring!)
                structured = getattr(getattr(s, "intent", None), "structured", None)
                payload_hint = structured.get("payload") if isinstance(structured, dict) else None
                signifier_intent = getattr(s.intent, "nl_text", "")

                print(f"[COMPATIBILITY_DEBUG] Checking compatibility: query='{intent}' vs signifier='{signifier_intent}'")
                is_compatible = self._rd4_intent_compatible(
                    intent_query=intent,
                    signifier_intent=signifier_intent,
                    affordance_uri=s.affordance_uri,
                    payload_hint=payload_hint,
                )
                print(f"[COMPATIBILITY_DEBUG] Result: {is_compatible}")

                if not is_compatible:
                    print(f"[COMPATIBILITY_DEBUG] FILTERED OUT: signifier '{signifier_intent}' not compatible with query '{intent}'")
                    continue

                print(f"[COMPATIBILITY_DEBUG] PASSED: signifier '{signifier_intent}' is compatible with query '{intent}'")

                match_data = {
                    "signifier_id": s.signifier_id,
                    "intent": signifier_intent,
                    "affordance_uri": s.affordance_uri,
                    "intent_similarity": match.similarity,
                    "intent_type": s.intent_type,
                    "workspace_id": workspace_id,
                    "shacl_conforms": shacl_conforms,
                    "shacl_violations": shacl_violations,
                    "payload_hint": payload_hint,  # RESTORED: missing field
                }
                matches.append(match_data)

            # Filter by SHACL conformance if validating context
            print(f"[SHACL_FILTER_DEBUG] Before SHACL filter: {len(matches)} matches")
            for i, match in enumerate(matches):
                print(f"[SHACL_FILTER_DEBUG] Match {i}: signifier_id={match.get('signifier_id')}, shacl_conforms={match.get('shacl_conforms')}")

            if intent_type != "EXPLICIT":
                original_count = len(matches)
                matches = [m for m in matches if m.get("shacl_conforms", True)]
                filtered_count = len(matches)
                print(f"[SHACL_FILTER_DEBUG] SHACL filter applied: {original_count} -> {filtered_count} matches")
                if original_count != filtered_count:
                    print(f"[SHACL_FILTER_DEBUG] FILTERED OUT {original_count - filtered_count} matches due to SHACL validation")

            # Rank matches by intent_type and similarity
            if intent_type:
                matches = rank_signifier_matches(matches, intent_type)
            else:
                matches.sort(key=lambda m: m.get("intent_similarity", 0.0), reverse=True)

            print(f"[FINAL_RESULT_DEBUG] Returning {len(matches)} matches to InteractionSolver")
            for i, match in enumerate(matches[:3]):  # Show first 3
                print(f"[FINAL_RESULT_DEBUG] Match {i}: signifier_id={match.get('signifier_id', 'N/A')}, similarity={match.get('intent_similarity', 'N/A')}")

            return {"matches": matches, "final_matches": matches, "total_signifiers": len(signifier_dicts)}

        except Exception as e:
            self.logger.error(f"Error matching signifiers: {e}", exc_info=True)
            return {"ok": False, "error": "exception", "detail": str(e)}

    def _generate_shacl_shapes_from_conditions(
        self, structured_conditions: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate SHACL shapes from structured conditions."""
        return generate_shacl_shapes_from_conditions(structured_conditions)

    def _generate_nl_description(
        self, structured_conditions: List[Dict[str, Any]], ctx_meta: Dict[str, Any]
    ) -> str:
        """Generate natural language description from structured conditions."""
        return generate_nl_description(structured_conditions, ctx_meta)

    async def _rd4_record_execution(
        self,
        *,
        plan: Any = None,
        execution_report: Any = None,
        sender: Optional[str] = None,
        thread: Optional[str] = None,
        workspace_id: Optional[str] = None,
        signifiers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Record execution signifiers."""
        self.logger.info(demo(">>> _rd4_record_execution CALLED: plan_type=%s, execution_report_type=%s, signifiers_provided=%s"), type(plan).__name__, type(execution_report).__name__, signifiers is not None)

        try:
            await self._ensure_rd4_engine_ready()
            if not self._rd4_registry:
                return {"ok": False, "error": "rd4_engine_not_ready"}
        except Exception as e:
            self.logger.error(demo("!!! RD4 engine setup failed: %s"), str(e), exc_info=True)
            return {"ok": False, "error": "rd4_engine_setup_failed", "detail": str(e)}

        # NEW PATH: If signifiers are already provided (from BT extraction), skip plan parsing
        if signifiers and isinstance(signifiers, list) and len(signifiers) > 0:
            self.logger.info(demo(">>> Using pre-extracted signifiers (count=%d), skipping plan parsing"), len(signifiers))
            # Jump directly to storing signifiers (reuse code below)
            try:
                from src.models.signifier import (  # type: ignore[import-not-found]
                    IntentContext,
                    IntentionDescription,
                    Provenance,
                    Signifier as RD4Signifier,
                    SignifierStatus,
                )
            except ImportError as e:
                self.logger.error(demo("!!! EXCEPTION in _rd4_record_execution: ImportError - %s"), str(e), exc_info=True)
                return {"ok": False, "error": "import_failed", "detail": str(e)}

            created: List[str] = []
            skipped: List[Dict[str, Any]] = []

            for sig_dict in signifiers:
                if not isinstance(sig_dict, dict):
                    skipped.append({"error": "invalid_signifier"})
                    continue

                try:
                    # Convert pre-extracted signifier dict to RD4Signifier
                    signifier_id = sig_dict.get("signifier_id") or f"sig-{uuid.uuid4().hex}"
                    intent_text = sig_dict.get("intent", "")
                    affordance_uri = sig_dict.get("affordance_uri", "")
                    action_name = sig_dict.get("action_name")
                    payload_hint = sig_dict.get("payload_hint", {})

                    if not affordance_uri:
                        skipped.append({"signifier_id": signifier_id, "error": "missing_affordance_uri"})
                        continue

                    # Build structured intent
                    intent_structured = {"intent": intent_text, "payload": payload_hint}
                    if action_name:
                        intent_structured["action_name"] = str(action_name)

                    # Include original structured_intent (action, artifact, parameter, value) if available
                    structured_intent_orig = sig_dict.get("structured_intent")
                    if structured_intent_orig and isinstance(structured_intent_orig, dict):
                        intent_structured["structured_intent"] = structured_intent_orig
                        self.logger.info(
                            demo("[STRUCTURED_INTENT] Preserving original structured_intent: %s"),
                            structured_intent_orig
                        )

                    # Build context metadata
                    ctx_meta = {
                        "workspace_id": sig_dict.get("workspace_id"),
                        "thread": thread,
                        "was_successful": sig_dict.get("was_successful", True),
                        "source": sig_dict.get("source", "bt_execution"),
                        "node_name": sig_dict.get("node_name"),
                    }

                    # Extract structured_conditions from signifier dict (built from state_snapshot)
                    structured_conditions = sig_dict.get("structured_conditions", [])
                    # Ensure it's a list (defensive)
                    if not isinstance(structured_conditions, list):
                        structured_conditions = []

                    # Extract intent_type (EXPLICIT or IMPLICIT classification)
                    intent_type = sig_dict.get("intent_type")
                    self.logger.info(
                        demo("[INTENT_TYPE] Extracted from signifier dict: intent_type=%r for signifier %s"),
                        intent_type,
                        signifier_id,
                    )
                    # Validate it's one of the expected values
                    if intent_type and str(intent_type).upper() not in ("EXPLICIT", "IMPLICIT"):
                        self.logger.warning(
                            demo("[INTENT_TYPE] Invalid intent_type %r for signifier %s, setting to None"),
                            intent_type,
                            signifier_id,
                        )
                        intent_type = None

                    # Generate SHACL shapes from structured_conditions for context validation
                    shacl_shapes = self._generate_shacl_shapes_from_conditions(structured_conditions)
                    if shacl_shapes:
                        self.logger.info(
                            demo("Generated SHACL shapes for signifier %s (%d conditions)"),
                            signifier_id,
                            len(structured_conditions),
                        )
                    else:
                        self.logger.debug(
                            demo("No SHACL shapes generated for signifier %s (no valid conditions)"),
                            signifier_id,
                        )

                    # Generate natural language description from structured conditions
                    nl_description = self._generate_nl_description(structured_conditions, ctx_meta)

                    signifier = RD4Signifier(
                        signifier_id=signifier_id,
                        version=1,
                        status=SignifierStatus.ACTIVE,
                        intent=IntentionDescription(
                            nl_text=str(intent_text),
                            structured=intent_structured
                        ),
                        context=IntentContext(
                            nl_description=nl_description,
                            structured_conditions=structured_conditions,
                            shacl_shapes=shacl_shapes,  # Generated from structured_conditions
                        ),
                        affordance_uri=affordance_uri,
                        intent_type=intent_type,  # Pass intent_type for memory engine filtering
                        provenance=Provenance(created_by=str(sender or self.jid), source="bt_execution"),
                    )

                    self._rd4_registry.create(signifier)
                    created.append(signifier_id)
                    self.logger.info(
                        demo("[INTENT_TYPE] Stored signifier %s: intent=%r -> affordance=%s, intent_type=%r"),
                        signifier_id,
                        intent_text,
                        affordance_uri,
                        intent_type,
                    )

                except Exception as e:
                    self.logger.error(
                        demo("Failed to create signifier from pre-extracted: %s - %s"),
                        type(e).__name__,
                        str(e),
                        exc_info=True,
                    )
                    skipped.append({"signifier_id": signifier_id if 'signifier_id' in locals() else "unknown", "error": "create_failed", "detail": str(e)})

            result = {
                "ok": True,
                "created_count": len(created),
                "created_ids": created,
                "skipped": skipped,
            }
            self.logger.info(
                demo("Signifier recording result: created=%d, skipped=%d"),
                len(created),
                len(skipped),
            )
            return result

        # OLD PATH: Extract signifiers from plan steps
        plan_obj = plan
        if isinstance(plan_obj, str):
            try:
                plan_obj = json.loads(plan_obj)
            except Exception:
                plan_obj = None

        if not isinstance(plan_obj, dict):
            self.logger.warning(demo(">>> EARLY RETURN: invalid_plan (plan_obj type=%s)"), type(plan_obj).__name__)
            return {"ok": False, "error": "invalid_plan", "hint": "no plan or signifiers provided"}

        steps = plan_obj.get("steps")
        if not isinstance(steps, list) or not steps:
            self.logger.warning(demo(">>> EARLY RETURN: missing_steps (steps type=%s, empty=%s)"), type(steps).__name__, not steps if isinstance(steps, list) else "N/A")
            return {"ok": False, "error": "missing_steps", "hint": "plan must have 'steps' array or provide 'signifiers' directly"}

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

        try:
            from src.models.signifier import (  # type: ignore[import-not-found]
                IntentContext,
                IntentionDescription,
                Provenance,
                Signifier as RD4Signifier,
                SignifierStatus,
            )
        except ImportError as e:
            self.logger.error(demo("!!! OLD PATH import failed: ImportError - %s"), str(e), exc_info=True)
            return {"ok": False, "error": "import_failed", "detail": str(e)}

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
                "was_successful": True,  # OLD PATH assumes success
            }

            # Generate natural language description
            nl_description = self._generate_nl_description([], ctx_meta)

            signifier = RD4Signifier(
                signifier_id=signifier_id,
                version=1,
                status=SignifierStatus.ACTIVE,
                intent=IntentionDescription(nl_text=intent, structured=intent_structured),
                context=IntentContext(nl_description=nl_description, structured_conditions=[], shacl_shapes=shacl_shapes),
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
                self.logger.error(
                    demo("Failed to create signifier %s: %s - %s"),
                    signifier_id,
                    type(e).__name__,
                    str(e),
                    exc_info=True,
                )
                skipped.append({"step_id": step_id, "error": "create_failed", "detail": str(e)})

        result = {
            "ok": True,
            "created_count": len(created),
            "created_ids": created,
            "skipped": skipped,
        }
        self.logger.info(
            demo("Signifier recording result: created=%d, skipped=%d"),
            len(created),
            len(skipped),
        )
        return result

    # Agent lifecycle methods
    async def start(self, *args, **kwargs):
        """Start the EnvExplorer agent."""
        self.logger.info(demo("Starting EnvExplorer agent"))
        await super().start(*args, **kwargs)

    async def stop(self):
        """Stop the EnvExplorer agent and clean up resources."""
        self.logger.info(demo("Stopping EnvExplorer agent"))
        if self.integration_engine:
            try:
                await self.integration_engine.stop_notification_listener()
            except Exception as e:
                self.logger.error(f"Error stopping integration engine: {e}")
        await super().stop()

    # Implement the previously stubbed message methods
    async def send_message(self, message: Message, recipient_jid: str) -> bool:
        """
        Send message to another agent.

        Args:
            message: Message to send
            recipient_jid: JID of recipient agent

        Returns:
            True if message was sent successfully
        """
        try:
            spade_msg = SpadeMessage(to=recipient_jid)
            spade_msg.set_metadata("type", message.message_type.value)
            spade_msg.set_metadata(META_CORRELATION_ID, message.correlation_id)
            spade_msg.body = message.model_dump_json()
            await self.send(spade_msg)
            self.logger.debug(f"Sent message to {recipient_jid}: {message.message_type}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending message to {recipient_jid}: {e}")
            return False

    async def receive_message(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive message with timeout.

        Args:
            timeout: Timeout in seconds (uses default if None)

        Returns:
            Parsed message or None if no message received
        """
        timeout = timeout or self.config.get("timeouts", {}).get("message_reception", _TIMEOUTS["message_reception"])
        try:
            spade_msg = await self.receive(timeout=timeout)
            if spade_msg:
                return Message.model_validate_json(spade_msg.body)
            return None
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            return None