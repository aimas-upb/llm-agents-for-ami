"""
Explicit Tool definitions for the User Assistant.
"""

import asyncio
import json
import logging
import hashlib
from typing import Any, Dict, List

from spade_llm import LLMTool

from ...shared.models.messages import MessageType
from ...shared.utils.demo_log import demo
from ...shared.utils.spade_rpc import rpc_call, RpcTimeoutError

logger = logging.getLogger("UserAssistant")


def _strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    # Best-effort removal of Markdown fences like ```json ... ```
    stripped = stripped.replace("```json", "").replace("```", "").strip()
    return stripped


def _loose_json_loads(text: str) -> Any | None:
    """
    Best-effort JSON loader.

    Handles:
    - clean JSON
    - JSON inside Markdown code fences
    - leading/trailing extra text (tries to raw-decode from first token)
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    # Prefer objects, then arrays, then strings.
    for token in ("{", "[", "\""):
        idx = cleaned.find(token)
        if idx < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            return obj
        except Exception:
            continue

    return None


def _coerce_plan_dict(value: Any) -> Dict[str, Any] | None:
    """
    Coerce various plan-shaped inputs into a JSON object (dict).

    Supports:
    - raw plan JSON string
    - wrapper JSON from retrieve_and_clear_latest_plan: {"ok": true, "plan_json": "...", ...}
    - JSON string literal containing a plan JSON string
    """
    candidate: Any = value
    for _ in range(5):
        if isinstance(candidate, dict):
            maybe_plan_json = candidate.get("plan_json")
            if isinstance(maybe_plan_json, str) and maybe_plan_json.strip():
                candidate = maybe_plan_json
                continue
            return candidate

        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                return None
            loaded = _loose_json_loads(text)
            if loaded is None:
                return None
            candidate = loaded
            continue

        # Try to round-trip via JSON for any other types.
        try:
            candidate = json.loads(json.dumps(candidate))
            continue
        except Exception:
            return None

    return candidate if isinstance(candidate, dict) else None


def _canonicalize_plan_for_hash(plan: Dict[str, Any]) -> tuple[str, str]:
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QueryCapabilitiesTool(LLMTool):
    """
    Tool to query the EnvExplorer via XMPP.
    """

    def __init__(self, target_jid: str):
        super().__init__(
            name="query_environment_capabilities",
            description="Asks the Environment Explorer agent about available actions or capabilities.",
            parameters={
                "type": "object",
                "properties": {
                    "explorer_query": {
                        "type": "string",
                        "description": "The query string (e.g. 'List capabilities')."
                    }
                },
                "required": ["explorer_query"]
            },
            func=self.run_impl
        )
        self.agent = None
        self.target_jid = target_jid

    def set_agent(self, agent):
        """Binds the tool to the agent instance."""
        self.agent = agent

    async def run_impl(self, explorer_query: str):
        if not self.agent:
            return "Error: Agent not initialized in tool."

        logger.info(demo(f"UA -> EnvExplorer ENV_CAPABILITIES_REQUEST: to={self.target_jid} query={explorer_query!r}"))
        try:
            result = await rpc_call(
                self.agent,
                to_jid=self.target_jid,
                request_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
                body=explorer_query,
                expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
                timeout=15.0,
            )
            return result.body
        except RpcTimeoutError:
            return "Error: Timeout waiting for reply."
        except Exception as e:
            return f"Error querying capabilities: {e}"


class QueryEnvironmentStateTool(LLMTool):
    """
    Tool to request environment state (snapshot or filtered) from EnvExplorer via XMPP.
    """

    def __init__(self, target_jid: str):
        super().__init__(
            name="query_environment_state",
            description="Fetches the current environment state from EnvExplorer. By default returns all artifacts and their properties; can be scoped by artifact_id/property_uri.",
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "Optional artifact identifier to scope the state query.",
                    },
                    "property_uri": {
                        "type": "string",
                        "description": "Optional property URI to fetch a single property value.",
                    },
                },
            },
            func=self.run_impl,
        )
        self.agent = None
        self.target_jid = target_jid

    def set_agent(self, agent):
        """Binds the tool to the agent instance."""
        self.agent = agent

    async def run_impl(self, artifact_id: str | None = None, property_uri: str | None = None):
        if not self.agent:
            return "Error: Agent not initialized in tool."

        logger.info(
            demo(
                "UA -> EnvExplorer ENV_STATE_REQUEST: to=%s artifact_id=%r property_uri=%r"
            ),
            self.target_jid,
            artifact_id,
            property_uri,
        )
        payload = {
            **({"artifact_id": artifact_id} if artifact_id else {}),
            **({"property_uri": property_uri} if property_uri else {}),
        }

        try:
            result = await rpc_call(
                self.agent,
                to_jid=self.target_jid,
                request_type=MessageType.ENV_STATE_REQUEST.value,
                body=payload,
                expect_type=MessageType.ENV_STATE_RESPONSE.value,
                timeout=15.0,
            )
            return result.body
        except RpcTimeoutError:
            return "Error: Timeout waiting for reply."
        except Exception as e:
            return f"Error querying state: {e}"


class RequestInteractionPlanTool(LLMTool):
    """
    Request a JSON-Plan 1.2 from the InteractionSolver using a list of intents.

    Under the hood this sends a GOAL_REQUEST message to the solver and waits for PLAN_CREATED.
    """

    def __init__(self):
        super().__init__(
            name="request_interaction_plan",
            description=(
                "Send a list of user intents to the Interaction-Solver and return the JSON-Plan 1.2 response. "
                "Use this after deriving intents and scanning environment capabilities."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "intent_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of concise environment-level intents to satisfy (multiple allowed).",
                    },
                    "workspace_id": {
                        "type": "string",
                        "description": "Optional workspace identifier (or name) to scope planning to a single workspace.",
                    }
                },
                "required": ["intent_list"],
            },
            func=self.run_impl,
        )
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    async def run_impl(self, intent_list: List[str], workspace_id: str | None = None):
        if not self.agent:
            return "Error: Agent not initialized in tool."

        solver_jid = (getattr(self.agent, "target_jids", {}) or {}).get("solver")
        if not solver_jid:
            # Best-effort fallback: assume default local name on same XMPP domain.
            try:
                domain = str(getattr(self.agent, "jid", "")).split("@", 1)[1].split("/", 1)[0]
                solver_jid = f"interaction_solver@{domain}"
            except Exception:
                solver_jid = None
        if not solver_jid:
            return json.dumps({"error": "solver_jid_not_configured"}, indent=2)

        # Normalize intents list
        intents = [str(i).strip() for i in (intent_list or []) if str(i).strip()]
        if not intents:
            return json.dumps({"error": "missing_intents"}, indent=2)

        thread = getattr(self.agent, "active_conversation_id", None)
        timeout = float((getattr(self.agent, "config", {}) or {}).get("planning", {}).get("timeout", 60))

        logger.info(
            demo("UA -> InteractionSolver GOAL_REQUEST: intents=%s workspace_id=%r"),
            intents,
            workspace_id,
        )
        try:
            result = await rpc_call(
                self.agent,
                to_jid=str(solver_jid),
                request_type=MessageType.GOAL_REQUEST.value,
                body={**({"workspace_id": str(workspace_id)} if workspace_id else {}), "intents": intents},
                expect_type=MessageType.PLAN_CREATED.value,
                timeout=timeout,
                thread=thread,
            )
            return result.body
        except RpcTimeoutError:
            return json.dumps({"error": "timeout"}, indent=2)
        except Exception as e:
            return json.dumps({"error": "rpc_failed", "detail": str(e)}, indent=2)


class StoreLatestPlanTool(LLMTool):
    """Store the latest plan in UserAssistant's per-conversation memory."""

    def __init__(self):
        super().__init__(
            name="store_latest_plan",
            description=(
                "Store the exact JSON-Plan 1.2 string as the latest proposed plan for this conversation. "
                "Call this immediately after receiving a valid plan, before summarizing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_json": {"type": "string", "description": "Exact JSON-Plan 1.2 string to store."}
                },
                "required": ["plan_json"],
            },
            func=self.run_impl,
        )
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    async def run_impl(self, plan_json: str):
        if not self.agent:
            return "Error: Agent not initialized in tool."

        thread = getattr(self.agent, "active_conversation_id", None) or "__default__"
        raw_plan = (plan_json or "").strip()
        if not raw_plan:
            return json.dumps({"ok": False, "error": "empty_plan"}, indent=2)

        plan_obj = _coerce_plan_dict(raw_plan)
        if not isinstance(plan_obj, dict):
            return json.dumps({"ok": False, "error": "invalid_plan_json"}, indent=2)

        canonical_plan_str, plan_hash = _canonicalize_plan_for_hash(plan_obj)
        self.agent.store_latest_plan(thread, canonical_plan_str, plan_hash)

        step_count = None
        step_preview = None
        try:
            if isinstance(plan_obj.get("steps"), list):
                steps = plan_obj.get("steps") or []
                step_count = len(steps)

                def _short_artifact(uri: str) -> str:
                    s = str(uri or "")
                    if not s:
                        return "?"
                    if "/artifacts/" in s:
                        s = s.split("/artifacts/", 1)[1]
                    if "#" in s:
                        s = s.split("#", 1)[0]
                    if "/" in s:
                        s = s.rsplit("/", 1)[-1]
                    return s or "?"

                previews: List[str] = []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    sid = step.get("step_id")
                    action = step.get("action_name") or "?"
                    artifact = _short_artifact(step.get("artifact_uri") or step.get("artifact_id") or "")
                    payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
                    previews.append(f"{sid}:{artifact}.{action} payload={payload}")
                if previews:
                    step_preview = "; ".join(previews[:4]) + ("; ..." if len(previews) > 4 else "")
        except Exception:
            step_count = None

        logger.info(
            demo("Plan stored for approval: thread=%s plan_hash=%s steps=%s"),
            thread,
            plan_hash,
            step_count if step_count is not None else "?",
        )
        if step_preview:
            logger.info(demo("Plan step preview: %s"), step_preview)
        return json.dumps({"ok": True, "plan_hash": plan_hash}, indent=2)


class RetrieveAndClearLatestPlanTool(LLMTool):
    """
    Retrieve and clear the stored plan for this conversation.

    - If discard=true: clears without approving, returns discarded=true.
    - If discard=false: returns plan_json and marks it as approved for execution.
    """

    def __init__(self):
        super().__init__(
            name="retrieve_and_clear_latest_plan",
            description=(
                "Retrieve the stored plan for this conversation and clear it. "
                "Use after user confirms (discard=false) or rejects (discard=true)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "discard": {
                        "type": "boolean",
                        "description": "If true, discard the stored plan without approving it.",
                        "default": False,
                    }
                },
            },
            func=self.run_impl,
        )
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    async def run_impl(self, discard: bool = False):
        if not self.agent:
            return "Error: Agent not initialized in tool."

        thread = getattr(self.agent, "active_conversation_id", None) or "__default__"
        record = self.agent.retrieve_and_clear_latest_plan(thread, approve=not discard)
        if not record:
            return json.dumps({"ok": False, "error": "no_plan_found"}, indent=2)

        if discard:
            logger.info(demo("Plan discarded by user: thread=%s"), thread)
            return json.dumps({"ok": True, "discarded": True}, indent=2)

        logger.info(
            demo("Plan approved for execution: thread=%s plan_hash=%s"),
            thread,
            record.get("plan_hash"),
        )
        return json.dumps({"ok": True, "plan_json": record["plan_json"], "plan_hash": record["plan_hash"]}, indent=2)


class ExecutePlanTool(LLMTool):
    """
    Execute a JSON-Plan 1.2 plan by invoking each step's affordance.

    Uses `UserAssistantAgent.execution_engine` (YggdrasilIntegration) because it already
    implements affordance execution via HCTL forms.
    """

    def __init__(self):
        super().__init__(
            name="execute_plan",
            description=(
                "Executes a JSON-Plan 1.2 by calling each step's affordance in the live environment. "
                "Use ONLY when the user explicitly asks to execute/apply the plan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_json": {
                        "type": "string",
                        "description": "The JSON-Plan 1.2 object serialized as a JSON string.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, validate and show what would run without executing.",
                        "default": False,
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Optional cap on number of steps to execute (from the start).",
                        "minimum": 1,
                    },
                },
                "required": ["plan_json"],
            },
            func=self.run_impl,
        )
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    async def run_impl(self, plan_json: str, dry_run: bool = False, max_steps: int | None = None):
        if not self.agent:
            return "Error: Agent not initialized in tool."

        thread = getattr(self.agent, "active_conversation_id", None) or "__default__"

        approved_hash = self.agent.peek_approved_plan_hash(thread) if not dry_run else None
        approved_plan_json = self.agent.peek_approved_plan_json(thread) if not dry_run else None

        # Hard gate: only allow execution for a plan that was explicitly retrieved-for-execution.
        if not dry_run and (not approved_hash or not approved_plan_json):
            logger.info(demo("Execution blocked: no approved plan in thread=%s"), thread)
            return json.dumps({"error": "not_approved", "detail": "No approved plan available for execution."}, indent=2)

        plan_obj = _coerce_plan_dict(plan_json)
        if plan_obj is None and not dry_run and approved_plan_json:
            # The caller passed an unparseable plan (common LLM serialization issue).
            # Execute the approved plan instead (safer than executing an unapproved variant).
            logger.info(demo("Executing approved plan (input plan_json not parseable): thread=%s"), thread)
            plan_obj = _coerce_plan_dict(approved_plan_json)

        if not isinstance(plan_obj, dict):
            return json.dumps({"error": "invalid_json", "detail": "Could not parse plan_json as JSON."}, indent=2)

        _, provided_hash = _canonicalize_plan_for_hash(plan_obj)
        if not dry_run and approved_hash and provided_hash != approved_hash and approved_plan_json:
            logger.info(demo("Execution plan mismatch: using approved plan: thread=%s"), thread)
            approved_obj = _coerce_plan_dict(approved_plan_json)
            if not isinstance(approved_obj, dict):
                return json.dumps({"error": "not_approved", "detail": "Approved plan is missing or invalid."}, indent=2)
            _, approved_obj_hash = _canonicalize_plan_for_hash(approved_obj)
            if approved_obj_hash != approved_hash:
                return json.dumps({"error": "not_approved", "detail": "Approved plan hash mismatch (internal)."}, indent=2)
            plan_obj = approved_obj

        plan = plan_obj

        if str(plan.get("plan_version")) != "1.2":
            return json.dumps(
                {"error": "invalid_plan_version", "detail": f"Expected plan_version=1.2, got {plan.get('plan_version')!r}"},
                indent=2,
            )

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return json.dumps({"error": "missing_steps", "detail": "Plan has no steps to execute."}, indent=2)

        if max_steps is not None:
            try:
                max_steps = int(max_steps)
            except Exception:
                max_steps = None
        if max_steps:
            steps = steps[:max_steps]

        logger.info(demo("Executing plan: thread=%s steps=%d dry_run=%s"), thread, len(steps), bool(dry_run))

        # Ensure engine ready (loads affordance_map required by execute_affordance)
        try:
            await self.agent.ensure_execution_engine_ready()
        except Exception as e:
            return json.dumps({"error": "engine_init_failed", "detail": str(e)}, indent=2)

        results: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                results.append({"index": idx, "ok": False, "error": "invalid_step", "detail": "Step is not an object."})
                continue

            step_id = step.get("step_id", idx)
            affordance_id = step.get("affordance_uri") or step.get("target") or step.get("affordance_id")
            payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}

            if not affordance_id:
                results.append({"step_id": step_id, "ok": False, "error": "missing_affordance"})
                continue

            if dry_run:
                results.append(
                    {"step_id": step_id, "ok": True, "dry_run": True, "affordance_id": str(affordance_id), "payload": payload}
                )
                continue

            try:
                logger.info(demo("Step %s: invoking affordance=%s payload=%s"), step_id, str(affordance_id), payload)
                response_text = await self.agent.execution_engine.execute_affordance(str(affordance_id), payload)
                ok = response_text is not None
                results.append(
                    {
                        "step_id": step_id,
                        "ok": ok,
                        "affordance_id": str(affordance_id),
                        "payload": payload,
                        "response": response_text,
                    }
                )
                logger.info(demo("Step %s: ok=%s"), step_id, ok)
            except Exception as e:
                results.append(
                    {
                        "step_id": step_id,
                        "ok": False,
                        "affordance_id": str(affordance_id),
                        "payload": payload,
                        "error": "execution_failed",
                        "detail": str(e),
                    }
                )
                logger.info(demo("Step %s: ok=false error=%s"), step_id, str(e))

            await asyncio.sleep(0)

        if not dry_run:
            self.agent.clear_approved_plan_hash(thread)

            # Best-effort: record successful executions as signifiers in EnvExplorer (embedded RD4 engine).
            explorer_jid = (getattr(self.agent, "target_jids", {}) or {}).get("explorer")
            if explorer_jid:
                try:
                    # Do not re-record signifiers for steps that were explicitly recovered from prior signifiers.
                    # (A reused plan should not create duplicate signifiers.)
                    recordable_steps: List[Dict[str, Any]] = []
                    recordable_step_ids: set[str] = set()
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                        meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
                        if meta.get("used_signifier_id"):
                            continue
                        recordable_steps.append(step)
                        sid = step.get("step_id")
                        if sid is not None:
                            recordable_step_ids.add(str(sid))

                    if not recordable_steps:
                        logger.info(demo("Skipping signifier recording: plan steps were recovered from existing signifiers."))
                    else:
                        ok_count = sum(
                            1
                            for r in results
                            if isinstance(r, dict)
                            and r.get("ok")
                            and (not recordable_step_ids or str(r.get("step_id")) in recordable_step_ids)
                        )
                        logger.info(
                            demo("Recording signifiers in EnvExplorer: ok_steps=%d total_steps=%d"),
                            ok_count,
                            len(recordable_steps),
                        )
                        plan_to_record: Any = plan
                        try:
                            if isinstance(plan, dict):
                                plan_to_record = {**plan, "steps": recordable_steps}
                        except Exception:
                            plan_to_record = plan
                        rec_res = await rpc_call(
                            self.agent,
                            to_jid=str(explorer_jid),
                            request_type=MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value,
                            body={
                                "plan": plan_to_record,
                                "execution_report": {"results": results},
                            },
                            expect_type=MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value,
                            timeout=15.0,
                            thread=(thread if thread and thread != "__default__" else None),
                        )
                        created_count = None
                        try:
                            rec_payload = json.loads(rec_res.body or "{}")
                            if isinstance(rec_payload, dict):
                                created_count = rec_payload.get("created_count")
                        except Exception:
                            created_count = None

                        logger.info(
                            demo("Signifier recording completed: created_count=%s"),
                            created_count if created_count is not None else "?",
                        )
                except Exception as e:
                    logger.info(demo("Failed to record signifiers in EnvExplorer: %s"), e)

        ok_count = sum(1 for r in results if isinstance(r, dict) and r.get("ok"))
        logger.info(demo("Execution finished: ok_steps=%d total_steps=%d dry_run=%s"), ok_count, len(results), bool(dry_run))

        return json.dumps(
            {
                "plan_version": "1.2",
                "executed": not dry_run,
                "dry_run": bool(dry_run),
                "steps_executed": len(results),
                "results": results,
            },
            indent=2,
        )
