import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..experience import ensure_experience_engine_ready
from ..experience import generate_shacl_shapes_from_conditions, generate_nl_description


class SignifierRecordBehaviour(CyclicBehaviour):
    """
    Contains all the business logic for recording execution signifiers from
    behavior tree executions and plan steps, including SHACL shape generation,
    intent processing, and registry operations.
    """

    async def run(self):
        """Main behavior loop - handles signifier recording requests."""
        msg = await self.receive(timeout=1)
        if not msg:
            return

        # Only process SIGNIFIER_RECORD_EXECUTION_REQUEST messages
        msg_type = msg.get_metadata("type")
        if msg_type != MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value:
            return

        self.agent.logger.debug(f"SignifierRecordBehaviour received request from {msg.sender}")

        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}
            self.agent.logger.warning("Failed to parse record request payload, using empty payload")

        try:
            # Process signifier recording
            response_payload = await self._record_execution(
                plan=payload.get("plan"),
                execution_report=payload.get("execution_report") or payload.get("execution") or {},
                sender=str(msg.sender) if getattr(msg, "sender", None) else None,
                thread=str(msg.thread) if getattr(msg, "thread", None) else None,
                workspace_id=payload.get("workspace_id"),
                signifiers=payload.get("signifiers"),  # Pre-extracted signifiers from BT
            )
        except Exception as e:
            self.agent.logger.error(
                demo(f"!!! EXCEPTION in signifier recording: {type(e).__name__} - {e}"),
                exc_info=True,
            )
            response_payload = {"ok": False, "error": "exception", "detail": str(e)}

        # Send response
        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value)

        # Preserve correlation ID and thread
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)

    async def _record_execution(
        self,
        *,
        plan: Any = None,
        execution_report: Any = None,
        sender: Optional[str] = None,
        thread: Optional[str] = None,
        workspace_id: Optional[str] = None,
        signifiers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Record execution signifiers - complete business logic.

        This contains all the recording logic that was previously in the agent,
        now properly encapsulated in a behavior.
        """
        self.agent.logger.info(
            demo(f">>> SignifierRecordBehaviour processing: plan_type={type(plan).__name__}, execution_report_type={type(execution_report).__name__}, signifiers_provided={signifiers is not None}")
        )

        # Ensure Experience engine engine is ready
        try:
            await ensure_experience_engine_ready(self.agent)
            if not self.agent._experience_engine_registry:
                return {"ok": False, "error": "experience_engine_not_ready"}
        except Exception as e:
            self.agent.logger.error(demo(f"!!! Experience engine engine setup failed: {e}"), exc_info=True)
            return {"ok": False, "error": "experience_engine_setup_failed", "detail": str(e)}

        # NEW PATH: If signifiers are already provided (from BT extraction), skip plan parsing
        if signifiers and isinstance(signifiers, list) and len(signifiers) > 0:
            self.agent.logger.info(demo(f">>> Using pre-extracted signifiers (count={len(signifiers)}), skipping plan parsing"))
            return await self._process_pre_extracted_signifiers(signifiers, sender, thread)

        # OLD PATH: Extract signifiers from plan steps
        return await self._process_plan_steps(plan, execution_report, sender, thread, workspace_id)

    async def _process_pre_extracted_signifiers(
        self,
        signifiers: List[Dict[str, Any]],
        sender: Optional[str],
        thread: Optional[str]
    ) -> Dict[str, Any]:
        """Process pre-extracted signifiers from BT execution."""
        try:
            from src.models.signifier import (  # type: ignore[import-not-found]
                IntentContext,
                IntentionDescription,
                Provenance,
                Signifier as ExperienceEngineSignifier,
                SignifierStatus,
            )
        except ImportError as e:
            self.agent.logger.error(demo(f"!!! EXCEPTION in signifier recording: ImportError - {e}"), exc_info=True)
            return {"ok": False, "error": "import_failed", "detail": str(e)}

        created: List[str] = []
        skipped: List[Dict[str, Any]] = []

        for sig_dict in signifiers:
            if not isinstance(sig_dict, dict):
                skipped.append({"error": "invalid_signifier"})
                continue

            try:
                # Convert pre-extracted signifier dict to ExperienceEngineSignifier
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
                td_sosa_metadata = sig_dict.get("td_sosa")
                if isinstance(td_sosa_metadata, dict) and td_sosa_metadata:
                    intent_structured["td_sosa"] = td_sosa_metadata

                # Include original structured_intent (action, artifact, parameter, value) if available
                structured_intent_orig = sig_dict.get("structured_intent")
                if structured_intent_orig and isinstance(structured_intent_orig, dict):
                    intent_structured["structured_intent"] = structured_intent_orig
                    self.agent.logger.info(
                        demo(f"[STRUCTURED_INTENT] Preserving original structured_intent: {structured_intent_orig}")
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
                semantic_structured_conditions = structured_conditions
                structured_conditions = [
                    {
                        "artifact": condition.get("artifact"),
                        "property_affordance": condition.get("property_affordance"),
                        "value_conditions": condition.get("value_conditions") or [],
                    }
                    for condition in structured_conditions
                    if isinstance(condition, dict)
                ]
                if isinstance(td_sosa_metadata, dict) and td_sosa_metadata:
                    intent_structured["td_sosa_context_conditions"] = semantic_structured_conditions

                # Extract intent_type (EXPLICIT or IMPLICIT classification)
                intent_type = sig_dict.get("intent_type")
                self.agent.logger.info(
                    demo(f"[INTENT_TYPE] Extracted from signifier dict: intent_type={intent_type!r} for signifier {signifier_id}")
                )
                # Validate and normalize to uppercase
                if intent_type:
                    intent_type_upper = str(intent_type).upper()
                    if intent_type_upper not in ("EXPLICIT", "IMPLICIT"):
                        self.agent.logger.warning(
                            demo(f"[INTENT_TYPE] Invalid intent_type {intent_type!r} for signifier {signifier_id}, setting to None")
                        )
                        intent_type = None
                    else:
                        intent_type = intent_type_upper  # Normalize to uppercase

                # Generate SHACL shapes from structured_conditions for context validation
                shacl_shapes = generate_shacl_shapes_from_conditions(structured_conditions)
                if shacl_shapes:
                    self.agent.logger.info(
                        demo(f"Generated SHACL shapes for signifier {signifier_id} ({len(structured_conditions)} conditions)")
                    )
                else:
                    self.agent.logger.debug(
                        demo(f"No SHACL shapes generated for signifier {signifier_id} (no valid conditions)")
                    )

                # Generate natural language description from structured conditions
                nl_description = generate_nl_description(structured_conditions, ctx_meta)

                signifier = ExperienceEngineSignifier(
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
                    affected_env_vars=sig_dict.get("affected_env_vars"),  # For v3 env-var matching
                    provenance=Provenance(created_by=str(sender or self.agent.jid), source="bt_execution"),
                )

                self.agent._experience_engine_registry.create(signifier)
                created.append(signifier_id)
                self.agent.logger.info(
                    demo(f"[INTENT_TYPE] Stored signifier {signifier_id}: intent={intent_text!r} -> affordance={affordance_uri}, intent_type={intent_type!r}")
                )

            except Exception as e:
                self.agent.logger.error(
                    demo(f"Failed to create signifier from pre-extracted: {type(e).__name__} - {e}"),
                    exc_info=True,
                )
                skipped.append({
                    "signifier_id": signifier_id if 'signifier_id' in locals() else "unknown",
                    "error": "create_failed",
                    "detail": str(e)
                })

        result = {
            "ok": True,
            "created_count": len(created),
            "created_ids": created,
            "skipped": skipped,
        }
        self.agent.logger.info(
            demo(f"Signifier recording result: created={len(created)}, skipped={len(skipped)}")
        )
        return result

    async def _process_plan_steps(
        self,
        plan: Any,
        execution_report: Any,
        sender: Optional[str],
        thread: Optional[str],
        workspace_id: Optional[str]
    ) -> Dict[str, Any]:
        """Process plan steps to extract signifiers (OLD PATH)."""
        plan_obj = plan
        if isinstance(plan_obj, str):
            try:
                plan_obj = json.loads(plan_obj)
            except Exception:
                plan_obj = None

        if not isinstance(plan_obj, dict):
            self.agent.logger.warning(demo(f">>> EARLY RETURN: invalid_plan (plan_obj type={type(plan_obj).__name__})"))
            return {"ok": False, "error": "invalid_plan", "hint": "no plan or signifiers provided"}

        steps = plan_obj.get("steps")
        if not isinstance(steps, list) or not steps:
            self.agent.logger.warning(
                demo(f">>> EARLY RETURN: missing_steps (steps type={type(steps).__name__}, empty={not steps if isinstance(steps, list) else 'N/A'})")
            )
            return {"ok": False, "error": "missing_steps", "hint": "plan must have 'steps' array or provide 'signifiers' directly"}

        # Determine successful steps
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
                Signifier as ExperienceEngineSignifier,
                SignifierStatus,
            )
        except ImportError as e:
            self.agent.logger.error(demo(f"!!! OLD PATH import failed: ImportError - {e}"), exc_info=True)
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

            # Generate SHACL shapes from evidence
            shacl_shapes = self._generate_shacl_from_evidence(evidence, signifier_id)

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
            nl_description = generate_nl_description([], ctx_meta)

            signifier = ExperienceEngineSignifier(
                signifier_id=signifier_id,
                version=1,
                status=SignifierStatus.ACTIVE,
                intent=IntentionDescription(nl_text=intent, structured=intent_structured),
                context=IntentContext(
                    nl_description=nl_description,
                    structured_conditions=[],
                    shacl_shapes=shacl_shapes
                ),
                affordance_uri=affordance_uri,
                provenance=Provenance(created_by=str(sender or self.agent.jid), source="execution"),
            )

            try:
                self.agent._experience_engine_registry.create(signifier)
                created.append(signifier_id)
                self.agent.logger.info(
                    demo(f"Stored signifier (no SHACL context) {signifier_id}: intent={intent!r} -> affordance={affordance_uri}")
                )
            except Exception as e:
                self.agent.logger.error(
                    demo(f"Failed to create signifier {signifier_id}: {type(e).__name__} - {e}"),
                    exc_info=True,
                )
                skipped.append({"step_id": step_id, "error": "create_failed", "detail": str(e)})

        result = {
            "ok": True,
            "created_count": len(created),
            "created_ids": created,
            "skipped": skipped,
        }
        self.agent.logger.info(
            demo(f"Signifier recording result: created={len(created)}, skipped={len(skipped)}")
        )
        return result

    def _generate_shacl_from_evidence(self, evidence: List[Dict[str, Any]], signifier_id: str) -> Optional[str]:
        """Generate SHACL shapes from evidence for context validation."""
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

                return "\n".join(ttl_lines).strip() or None
        except Exception:
            return None

        return None
