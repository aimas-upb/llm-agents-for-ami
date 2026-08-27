import json
import logging
from typing import Any, Dict, List, Optional
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..experience import ensure_experience_engine_ready, build_experience_engine_context_snapshot


class SignifierMatchBehaviour(CyclicBehaviour):
    """
    Contains all the business logic for matching user intents against stored signifiers
    using the v3 experience engine matcher (environment variable exact matching) and SHACL validation.

    For v3 matching: exact environment variable matches are returned if SHACL context validation passes.
    Non-conforming matches are provided as affordance hints for BT planning.
    """

    def _match_explicit_by_fields(
        self, query_structured_intent: Optional[Dict[str, Any]], signifier_dicts: List[Dict]
    ) -> List:
        """Filter signifiers using exact-field matching for EXPLICIT intents.

        Matches all populated (non-null, non-"NA") fields from the structured intent:
        - action.affordance_type
        - action.parameter
        - target.artifact_type
        - target.artifact_name
        - target.workspace_type

        Does NOT match on action.value (injected during BT construction).

        Compares against signifier's stored structured_intent field.
        """
        if not query_structured_intent:
            return []

        query_action = query_structured_intent.get("action", {}) or {}
        query_target = query_structured_intent.get("target", {}) or {}

        # Build constraint fields (exclude "NA" and None)
        constraints = {}
        if query_action.get("affordance_type") not in (None, "NA"):
            constraints["affordance_type"] = query_action.get("affordance_type")
        if query_action.get("parameter") not in (None, "NA"):
            constraints["parameter"] = query_action.get("parameter")
        if query_target.get("artifact_type") not in (None, "NA"):
            constraints["artifact_type"] = query_target.get("artifact_type")
        if query_target.get("artifact_name") not in (None, "NA"):
            constraints["artifact_name"] = query_target.get("artifact_name")
        if query_target.get("workspace_type") not in (None, "NA"):
            constraints["workspace_type"] = query_target.get("workspace_type")

        self.agent.logger.debug(
            f"EXPLICIT matching with constraints: {constraints}"
        )

        # Filter signifiers by exact match on all constraint fields
        matching_signifiers = []
        for sig_dict in signifier_dicts:
            # Get structured_intent from signifier (stored during signifier recording)
            sig_structured = sig_dict.get("structured_intent") or {}
            sig_action = sig_structured.get("action", {}) or {}
            sig_target = sig_structured.get("target", {}) or {}

            # Check each constraint
            match = True
            for constraint_field, constraint_value in constraints.items():
                if constraint_field == "affordance_type":
                    sig_value = sig_action.get("affordance_type")
                elif constraint_field == "parameter":
                    sig_value = sig_action.get("parameter")
                elif constraint_field == "artifact_type":
                    sig_value = sig_target.get("artifact_type")
                elif constraint_field == "artifact_name":
                    sig_value = sig_target.get("artifact_name")
                elif constraint_field == "workspace_type":
                    sig_value = sig_target.get("workspace_type")
                else:
                    sig_value = None

                if sig_value != constraint_value:
                    match = False
                    break

            if match:
                # Create a match result object (mimics matcher registry output)
                matching_signifiers.append(
                    type('MatchResult', (), {
                        'signifier_id': sig_dict.get('signifier_id'),
                        'similarity': 1.0,  # Exact match
                    })()
                )

        self.agent.logger.info(
            f"EXPLICIT exact-field matching: {len(matching_signifiers)} matches out of {len(signifier_dicts)} signifiers"
        )
        return matching_signifiers

    async def run(self):
        """Main behavior loop - handles signifier match requests."""
        msg = await self.receive(timeout=1)
        if not msg:
            return

        # Only process SIGNIFIER_MATCH_REQUEST messages
        msg_type = msg.get_metadata("type")
        if msg_type != MessageType.SIGNIFIER_MATCH_REQUEST.value:
            return

        self.agent.logger.debug(f"SignifierMatchBehaviour received request from {msg.sender}")

        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}
            self.agent.logger.warning("Failed to parse request payload, using empty payload")

        # Extract request parameters
        intent = payload.get("intent") or payload.get("query") or payload.get("goal") or ""
        workspace_id = payload.get("workspace_id")
        k = payload.get("k", 10)
        matcher_version = payload.get("matcher_version")
        min_similarity = payload.get("min_similarity")
        intent_type = payload.get("intent_type")
        query_structured_intent = payload.get("query_structured_intent")
        affected_env_vars = payload.get("affected_env_vars")

        # Log request details
        self.agent.logger.debug(f"Processing signifier match: intent='{intent}', workspace_id={workspace_id}")
        self.agent.logger.debug(f"Parameters: k={k}, matcher_version={matcher_version}, min_similarity={min_similarity}")
        self.agent.logger.debug(f"Intent type: {intent_type}, structured_intent={query_structured_intent}, has_env_vars={bool(affected_env_vars)}")

        # Perform signifier matching
        response_payload = await self._match_signifiers(
            intent=str(intent),
            workspace_id=str(workspace_id) if workspace_id else None,
            k=int(k) if str(k).isdigit() else 10,
            matcher_version=str(matcher_version) if matcher_version else None,
            min_similarity=float(min_similarity) if min_similarity is not None else None,
            intent_type=str(intent_type).upper() if intent_type and str(intent_type).upper() in ("EXPLICIT", "IMPLICIT") else None,
            query_structured_intent=query_structured_intent if isinstance(query_structured_intent, dict) else None,
            affected_env_vars=affected_env_vars if isinstance(affected_env_vars, list) else None,
        )

        # Send response
        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", MessageType.SIGNIFIER_MATCH_RESPONSE.value)

        # Preserve correlation ID and thread
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)

    async def _match_signifiers(
        self,
        intent: str,
        workspace_id: Optional[str] = None,
        k: int = 10,
        matcher_version: Optional[str] = None,
        min_similarity: Optional[float] = None,
        intent_type: Optional[str] = None,
        query_structured_intent: Optional[Dict[str, Any]] = None,
        affected_env_vars: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Match signifiers using Experience engine engine - complete business logic.

        This contains all the matching logic that was previously in the agent,
        now properly encapsulated in a behavior.

        For implicit intents with affected_env_vars, use v3 environment-variable matching.
        """
        # Ensure Experience engine is ready
        await ensure_experience_engine_ready(self.agent)
        if not self.agent._experience_engine_matcher_registry or not self.agent._experience_engine_registry:
            return {"ok": False, "error": "experience_engine_not_ready"}

        if not intent:
            return {"ok": False, "error": "missing_intent", "exact_matches": [], "affordance_hints": [], "total_signifiers": 0}

        # Build context snapshot
        context = build_experience_engine_context_snapshot(self.agent, workspace_id=workspace_id)

        try:
            # Get all signifiers from registry
            all_signifiers = self.agent._experience_engine_registry.list_signifiers(limit=10000)
            signifier_dicts = [s.model_dump() for s in all_signifiers]

            if not signifier_dicts:
                self.agent.logger.info(demo("Signifier match: 0 stored signifiers (storage empty)."))
                return {"ok": True, "exact_matches": [], "affordance_hints": [], "total_signifiers": 0}

            # Determine matcher version and similarity threshold
            version_to_use = str(matcher_version or self.agent._experience_engine_default_matcher_version or "v0")
            if min_similarity is None:
                min_similarity = self.agent._experience_engine_default_min_similarity

            self.agent.logger.info(
                demo(f"[INTENT_TYPE] Signifier match: intent={intent!r} matcher={version_to_use} min_similarity={min_similarity} k={k} intent_type={intent_type!r}")
            )
            self.agent.logger.debug(f"Using matcher version: {version_to_use}")

            # Log context validation strategy
            if intent_type == "EXPLICIT":
                self.agent.logger.info(
                    demo("[INTENT_TYPE] Context validation DISABLED for EXPLICIT intent (exact target specified)")
                )
            elif intent_type == "IMPLICIT":
                self.agent.logger.info(
                    demo("[INTENT_TYPE] Context validation ENABLED for IMPLICIT intent (inferring from environment)")
                )
            else:
                self.agent.logger.info(
                    demo("[INTENT_TYPE] Context validation ENABLED (intent_type not specified, defaulting to validation)")
                )

            # Perform Experience engine matching
            self.agent.logger.debug(f"Calling matcher with {len(signifier_dicts)} signifiers")
            self.agent.logger.debug(f"Query: intent='{intent}', version='{version_to_use}', structured_intent={query_structured_intent}")

            # Determine matcher version and routing
            if intent_type == "EXPLICIT":
                # EXPLICIT: Use exact-field matching from structured intent
                match_results = self._match_explicit_by_fields(
                    query_structured_intent, signifier_dicts
                )
            else:
                # IMPLICIT: v3 if affected_env_vars provided, otherwise use configured version
                if affected_env_vars is not None:
                    self.agent.logger.info(
                        demo("[V3 MATCHER] Using environment variable matching (v3) for implicit intent")
                    )
                    version_to_use = "v3"

                try:
                    match_results = self.agent._experience_engine_matcher_registry.match(
                        intent_query=intent,
                        signifiers=signifier_dicts,
                        k=int(k),
                        version=version_to_use,
                        min_similarity=float(min_similarity),
                        query_structured_intent=query_structured_intent,
                        affected_env_vars=affected_env_vars,
                    )
                    self.agent.logger.debug(f"Matcher returned {len(match_results) if match_results else 0} results")
                    for i, result in enumerate(match_results[:3]):  # Show first 3
                        self.agent.logger.debug(f"Result {i}: similarity={getattr(result, 'similarity', 'N/A')}, signifier_id={getattr(result, 'signifier_id', 'N/A')}")

                except Exception as e:
                    self.agent.logger.error(
                        demo(f"!!! MATCHER FAILED: {type(e).__name__} - {e}"),
                        exc_info=True,
                    )
                    return {"ok": False, "error": "matcher_failed", "detail": str(e), "exact_matches": [], "affordance_hints": [], "total_signifiers": len(signifier_dicts)}

            # Build context graph for SHACL validation
            context_graph, _ = self.agent._experience_engine_context_builder.normalize_context(context)

            # Process match results
            matches: List[Dict[str, Any]] = []
            td_sosa_supported = bool(
                (getattr(self.agent, "semantic_capabilities", {}) or {}).get("td_sosa_supported", False)
            )
            query_td_sosa_props = (
                self._infer_td_sosa_properties(intent, query_structured_intent)
                if td_sosa_supported
                else set()
            )
            if query_td_sosa_props:
                self.agent.logger.info(
                    demo("TD-SOSA signifier matching enabled: inferred_properties=%s"),
                    sorted(query_td_sosa_props),
                )
            self.agent.logger.debug(f"Processing {len(match_results)} match results")

            for i, match in enumerate(match_results):
                self.agent.logger.debug(f"Processing match {i}: signifier_id={match.signifier_id}, similarity={getattr(match, 'similarity', 'N/A')}")

                # Get signifier from registry
                s = self.agent._experience_engine_registry.get(match.signifier_id)
                if not s:
                    self.agent.logger.debug(f"Signifier {match.signifier_id} NOT FOUND in registry!")
                    continue
                self.agent.logger.debug(f"Signifier {match.signifier_id} found in registry")

                # SHACL validation
                shacl_conforms = True
                shacl_violations: List[str] = []

                # EXPLICIT intents: skip context validation (user specified exact target)
                # IMPLICIT intents: validate context (need to infer from environment state)
                should_validate_context = (intent_type != "EXPLICIT")

                self.agent.logger.debug(f"SHACL validation: should_validate={should_validate_context}, intent_type='{intent_type}'")
                self.agent.logger.debug(f"SHACL validation enabled: {self.agent._experience_engine_shacl_validation_enabled}")
                self.agent.logger.debug(f"Signifier has SHACL shapes: {bool(getattr(getattr(s, 'context', None), 'shacl_shapes', None))}")

                if should_validate_context and self.agent._experience_engine_shacl_validation_enabled and getattr(getattr(s, "context", None), "shacl_shapes", None):
                    self.agent.logger.debug(f"Running SHACL validation for signifier {s.signifier_id}")
                    validation = self.agent._experience_engine_shacl_validator.validate_signifier_context(
                        context_graph,
                        s.context.shacl_shapes,
                        format="turtle",
                    )
                    # ValidationResult object has .conforms and .violations attributes
                    shacl_conforms = bool(getattr(validation, 'conforms', False))
                    violations = getattr(validation, 'violations', []) or []
                    shacl_violations = [getattr(v, 'message', str(v)) for v in violations]
                    self.agent.logger.debug(f"SHACL validation result: conforms={shacl_conforms}, violations={len(violations)}")
                else:
                    skip_reason = []
                    if not should_validate_context:
                        skip_reason.append("explicit intent")
                    if not self.agent._experience_engine_shacl_validation_enabled:
                        skip_reason.append("validation disabled in config")
                    if not getattr(getattr(s, "context", None), "shacl_shapes", None):
                        skip_reason.append("no SHACL shapes")
                    self.agent.logger.debug(f"Skipping SHACL validation ({', '.join(skip_reason)})")

                self.agent.logger.debug(f"Final SHACL conforms value: {shacl_conforms}")

                # v3 matching: exact environment variable match means intent is already validated.
                # SHACL validation is the only remaining check. No additional compatibility filtering needed.
                structured = getattr(getattr(s, "intent", None), "structured", None)
                payload_hint = structured.get("payload") if isinstance(structured, dict) else None
                signifier_intent = getattr(s.intent, "nl_text", "")

                # Build match data
                match_data = {
                    "signifier_id": s.signifier_id,
                    "intent": signifier_intent,
                    "affordance_uri": s.affordance_uri,
                    "intent_similarity": match.similarity,
                    "intent_type": s.intent_type,
                    "workspace_id": workspace_id,
                    "shacl_conforms": shacl_conforms,
                    "shacl_violations": shacl_violations,
                    "payload_hint": payload_hint,
                    "td_sosa_properties": sorted(signifier_td_sosa_props),
                    "td_sosa_semantic_match": semantic_match,
                }
                matches.append(match_data)

            # For EXPLICIT intents: return all matches (no context validation was done)
            # For IMPLICIT intents: return all matches, but keep SHACL info to separate:
            #   - exact_matches (SHACL conforming) for fast-path
            #   - affordance_hints (SHACL non-conforming) for BT planner
            self.agent.logger.debug(f"Total matches collected: {len(matches)}")
            for i, match in enumerate(matches):
                self.agent.logger.debug(f"Match {i}: signifier_id={match.get('signifier_id')}, shacl_conforms={match.get('shacl_conforms')}, intent_type={match.get('intent_type')}")

            # Separate matches by intent type semantics
            if intent_type == "EXPLICIT":
                # EXPLICIT: user specified exact target, no context validation needed
                # All matches are exact_matches; affordance_hints unused
                exact_matches = matches
                affordance_hints = []
                self.agent.logger.info(f"EXPLICIT intent signifier match: {len(exact_matches)} matches returned (no context validation)")
            else:
                # IMPLICIT: system inferred target, context validation required via SHACL
                # Split by conformance: exact_matches (SHACL valid), affordance_hints (SHACL invalid)
                exact_matches = [m for m in matches if m.get("shacl_conforms") is True]
                affordance_hints = [m for m in matches if m.get("shacl_conforms") is not True]
                self.agent.logger.info(f"IMPLICIT intent signifier match: {len(exact_matches)} exact (context valid), {len(affordance_hints)} hints (context invalid)")
            for i, match in enumerate(exact_matches[:3]):  # Show first 3 exact matches
                self.agent.logger.debug(f"Exact match {i}: signifier_id={match.get('signifier_id', 'N/A')}")
            for i, match in enumerate(affordance_hints[:3]):  # Show first 3 hints
                self.agent.logger.debug(f"Affordance hint {i}: signifier_id={match.get('signifier_id', 'N/A')}, violations={match.get('shacl_violations', [])}")

            return {
                "ok": True,
                "exact_matches": exact_matches,
                "affordance_hints": affordance_hints,
                "total_signifiers": len(signifier_dicts)
            }

        except Exception as e:
            self.agent.logger.error(f"Error matching signifiers: {e}", exc_info=True)
            return {"ok": False, "error": "exception", "detail": str(e)}
