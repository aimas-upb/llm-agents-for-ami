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
from ...shared.utils.spade_rpc import rpc_call, RpcTimeoutError

logger = logging.getLogger("UserAssistant")


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

        logger.info(f"[Tool] Sending query to {self.target_jid}: {explorer_query}")
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

        logger.info(f"[Tool] Sending state query to {self.target_jid}: artifact_id={artifact_id}, property_uri={property_uri}")
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
        plan_str = (plan_json or "").strip()
        if not plan_str:
            return json.dumps({"ok": False, "error": "empty_plan"}, indent=2)

        plan_hash = hashlib.sha256(plan_str.encode("utf-8")).hexdigest()
        self.agent.store_latest_plan(thread, plan_str, plan_hash)
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
            return json.dumps({"ok": True, "discarded": True}, indent=2)

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
        plan_str = (plan_json or "").strip()

        # Be tolerant if caller passes the wrapper returned by retrieve_and_clear_latest_plan
        # (i.e., {"ok": true, "plan_json": "..."}).
        if plan_str.startswith("{"):
            try:
                maybe_wrapper = json.loads(plan_str)
                if isinstance(maybe_wrapper, dict) and isinstance(maybe_wrapper.get("plan_json"), str):
                    plan_str = maybe_wrapper["plan_json"].strip()
            except Exception:
                pass
        plan_hash = hashlib.sha256(plan_str.encode("utf-8")).hexdigest() if plan_str else ""

        # Hard gate: only allow execution for a plan that was explicitly retrieved-for-execution.
        approved_hash = self.agent.peek_approved_plan_hash(thread) if not dry_run else None
        if not dry_run:
            if not approved_hash:
                return json.dumps({"error": "not_approved", "detail": "No approved plan available for execution."}, indent=2)
            if approved_hash != plan_hash:
                return json.dumps(
                    {"error": "not_approved", "detail": "Plan does not match the last approved plan."}, indent=2
                )

        # Parse plan JSON
        try:
            plan = json.loads(plan_str or "")
        except Exception as e:
            return json.dumps({"error": "invalid_json", "detail": str(e)}, indent=2)

        if not isinstance(plan, dict):
            return json.dumps({"error": "invalid_plan_type", "detail": "Plan must be a JSON object."}, indent=2)

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

            await asyncio.sleep(0)

        if not dry_run:
            self.agent.clear_approved_plan_hash(thread)

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
