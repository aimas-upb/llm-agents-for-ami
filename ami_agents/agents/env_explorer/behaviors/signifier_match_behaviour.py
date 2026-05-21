import json
import logging
from typing import Any, Dict, List, Optional
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..experience import intent_compatible, rank_signifier_matches
from ..experience import ensure_experience_engine_ready, build_experience_engine_context_snapshot


class SignifierMatchBehaviour(CyclicBehaviour):
    """
    Contains all the business logic for matching user intents against stored signifiers
    using Experience engine engine, SHACL validation, and intent compatibility checking.
    """

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
            return {"error": "missing_intent", "matches": [], "final_matches": [], "total_signifiers": 0}

        # Build context snapshot
        context = build_experience_engine_context_snapshot(self.agent, workspace_id=workspace_id)

        try:
            # Get all signifiers from registry
            all_signifiers = self.agent._experience_engine_registry.list_signifiers(limit=10000)
            signifier_dicts = [s.model_dump() for s in all_signifiers]

            if not signifier_dicts:
                self.agent.logger.info(demo("Signifier match: 0 stored signifiers (storage empty)."))
                return {"matches": [], "final_matches": [], "total_signifiers": 0}

            # Determine matcher version and similarity threshold
            version_to_use = str(matcher_version or self.agent._experience_engine_default_matcher_version or "v0")
            if min_similarity is None:
                min_similarity = self.agent._experience_engine_default_min_similarity

            self.agent.logger.info(
                demo("[INTENT_TYPE] Signifier match: intent=%r matcher=%s min_similarity=%s k=%s intent_type=%r"),
                intent,
                version_to_use,
                min_similarity,
                k,
                intent_type,
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

            # For v3 matching (implicit intents with affected_env_vars), use direct env-var matching
            if affected_env_vars is not None:
                self.agent.logger.info(
                    demo("[V3 MATCHER] Using environment variable matching (v3) for implicit intent")
                )
                # Convert affected_env_vars to comparable format
                query_vars = {(v["variable"], v["direction"]) for v in affected_env_vars}
                match_results = []

                for signifier_dict in signifier_dicts:
                    # Extract signifier's affected_env_vars
                    signifier_vars_raw = signifier_dict.get("affected_env_vars", [])
                    if not isinstance(signifier_vars_raw, list):
                        continue

                    signifier_vars = {(v.get("variable"), v.get("direction")) for v in signifier_vars_raw}

                    # v3 match: exact match on affected_env_vars (both must have identical elements)
                    if query_vars == signifier_vars:
                        self.agent.logger.debug(
                            f"[V3 MATCHER] Exact env-var match found: signifier_id={signifier_dict.get('signifier_id')}"
                        )
                        # Create a mock match result object
                        class MockMatch:
                            def __init__(self, sig_id):
                                self.signifier_id = sig_id
                                self.similarity = 1.0  # Exact match
                        match_results.append(MockMatch(signifier_dict.get("signifier_id")))

                self.agent.logger.info(
                    demo("[V3 MATCHER] Found %d env-var matches"),
                    len(match_results),
                )
                if not match_results:
                    self.agent.logger.info(
                        demo("[V3 MATCHER] No exact matches on environment variables, returning empty")
                    )
                    return {"matches": [], "final_matches": [], "total_signifiers": len(signifier_dicts)}
            else:
                try:
                    match_results = self.agent._experience_engine_matcher_registry.match(
                        intent_query=intent,
                        signifiers=signifier_dicts,
                        k=int(k),
                        version=version_to_use,
                        min_similarity=float(min_similarity),
                        query_structured_intent=query_structured_intent,
                    )
                    self.agent.logger.debug(f"Matcher returned {len(match_results) if match_results else 0} results")
                    for i, result in enumerate(match_results[:3]):  # Show first 3
                        self.agent.logger.debug(f"Result {i}: similarity={getattr(result, 'similarity', 'N/A')}, signifier_id={getattr(result, 'signifier_id', 'N/A')}")

                except Exception as e:
                    self.agent.logger.error(
                        demo("!!! V2 MATCHER FAILED, falling back to v0: %s - %s"),
                        type(e).__name__,
                        str(e),
                        exc_info=True,
                    )
                    self.agent.logger.error(f"V2 Matcher exception: {type(e).__name__}: {str(e)}")
                    version_to_use = "v0"
                    match_results = self.agent._experience_engine_matcher_registry.match(
                        intent_query=intent,
                        signifiers=signifier_dicts,
                        k=int(k),
                        version=version_to_use,
                        query_structured_intent=query_structured_intent,
                    )

            # Build context graph for SHACL validation
            context_graph, _ = self.agent._experience_engine_context_builder.normalize_context(context)

            # Process match results
            matches: List[Dict[str, Any]] = []
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

                # Check intent compatibility
                structured = getattr(getattr(s, "intent", None), "structured", None)
                payload_hint = structured.get("payload") if isinstance(structured, dict) else None
                signifier_intent = getattr(s.intent, "nl_text", "")

                self.agent.logger.debug(f"Checking compatibility: query='{intent}' vs signifier='{signifier_intent}'")
                is_compatible = intent_compatible(
                    intent_query=intent,
                    signifier_intent=signifier_intent,
                    affordance_uri=s.affordance_uri,
                    payload_hint=payload_hint,
                )
                self.agent.logger.debug(f"Compatibility result: {is_compatible}")

                if not is_compatible:
                    self.agent.logger.debug(f"FILTERED OUT: signifier '{signifier_intent}' not compatible with query '{intent}'")
                    continue

                self.agent.logger.debug(f"PASSED: signifier '{signifier_intent}' is compatible with query '{intent}'")

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
                }
                matches.append(match_data)

            # Filter by SHACL conformance if validating context
            self.agent.logger.debug(f"Before SHACL filter: {len(matches)} matches")
            for i, match in enumerate(matches):
                self.agent.logger.debug(f"Match {i}: signifier_id={match.get('signifier_id')}, shacl_conforms={match.get('shacl_conforms')}")

            if intent_type != "EXPLICIT":
                original_count = len(matches)
                matches = [m for m in matches if m.get("shacl_conforms", True)]
                filtered_count = len(matches)
                self.agent.logger.debug(f"SHACL filter applied: {original_count} -> {filtered_count} matches")
                if original_count != filtered_count:
                    self.agent.logger.debug(f"FILTERED OUT {original_count - filtered_count} matches due to SHACL validation")

            # Rank matches by intent_type and similarity
            if intent_type:
                matches = rank_signifier_matches(matches, intent_type)
            else:
                matches.sort(key=lambda m: m.get("intent_similarity", 0.0), reverse=True)

            self.agent.logger.info(f"Returning {len(matches)} matches to requester")
            for i, match in enumerate(matches[:3]):  # Show first 3
                self.agent.logger.debug(f"Final match {i}: signifier_id={match.get('signifier_id', 'N/A')}, similarity={match.get('intent_similarity', 'N/A')}")

            return {"matches": matches, "final_matches": matches, "total_signifiers": len(signifier_dicts)}

        except Exception as e:
            self.agent.logger.error(f"Error matching signifiers: {e}", exc_info=True)
            return {"ok": False, "error": "exception", "detail": str(e)}