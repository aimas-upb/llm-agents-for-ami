"""
Explicit Tool definitions for the User Assistant.
"""

import asyncio
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional

from spade_llm import LLMTool

from ...shared.models.messages import MessageType
from ...shared.utils.demo_log import demo
from ...shared.utils.spade_rpc import rpc_call, RpcTimeoutError
from ...bt_planning.execution.ir_executor import IRExecutor
from ...bt_planning.execution.base import ExecutionResult
from ...bt_planning.signifier_bridge import extract_signifiers_from_bt
from ...shared.community.community_client import CommunitySignifierClient

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


def _count_bt_nodes(node: dict) -> int:
    """Count nodes in a BT JSON IR tree."""
    if not isinstance(node, dict):
        return 0
    count = 1
    for child in node.get("children", []):
        count += _count_bt_nodes(child)
    return count


def _bt_preview(node: dict, depth: int = 0) -> str:
    """Generate a compact preview of a BT JSON IR tree."""
    if not isinstance(node, dict):
        return ""
    name = node.get("name", "?")
    ntype = node.get("type", "?")
    parts = [f"{name}({ntype})"]
    if ntype == "action":
        url = node.get("action_url", "")
        action_name = url.rstrip("/").rsplit("/", 1)[-1] if url else "?"
        params = node.get("parameters", {})
        parts = [f"{name}:action({action_name})"]
        if params:
            parts[0] += f" params={params}"
    elif ntype == "condition":
        prop = node.get("property_url", "")
        prop_name = prop.rstrip("/").rsplit("/", 1)[-1] if prop else "?"
        expected = node.get("expected_value", "?")
        parts = [f"{name}:cond({prop_name}=={expected})"]
    children = node.get("children", [])
    if children and depth < 2:
        child_previews = [_bt_preview(c, depth + 1) for c in children[:4]]
        child_str = ", ".join(child_previews)
        if len(children) > 4:
            child_str += ", ..."
        parts[0] += f" [{child_str}]"
    return parts[0]


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

        # Check state memory cache for single-property lookups
        state_memory = getattr(self.agent, "state_memory", None)
        if property_uri and state_memory and state_memory.has(property_uri):
            cached_value = state_memory.get(property_uri)
            logger.info(
                demo("State cache HIT: property_uri=%r value=%r"),
                property_uri,
                cached_value,
            )
            return json.dumps({"property_uri": property_uri, "value": cached_value, "source": "cache"})

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

            # Cache results in state memory
            if state_memory:
                try:
                    body = result.body
                    state_data = json.loads(body) if isinstance(body, str) else body
                    if isinstance(state_data, dict):
                        if property_uri and "value" in state_data:
                            state_memory.store(property_uri, state_data["value"])
                        elif "artifacts" in state_data and isinstance(state_data["artifacts"], dict):
                            state_memory.store_bulk(state_data["artifacts"])
                        else:
                            state_memory.store_bulk(state_data)
                except Exception:
                    pass  # Best-effort caching

            return result.body
        except RpcTimeoutError:
            return "Error: Timeout waiting for reply."
        except Exception as e:
            return f"Error querying state: {e}"


class RequestInteractionPlanTool(LLMTool):
    """
    Request a behavior tree plan from the InteractionSolver using a list of intents.

    Under the hood this sends a GOAL_REQUEST message to the solver and waits for PLAN_CREATED.
    """

    def __init__(self):
        super().__init__(
            name="request_interaction_plan",
            description=(
                "Send a list of user intents to the Interaction-Solver and return a behavior tree plan. "
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
                    },
                    "intent_type": {
                        "type": "string",
                        "enum": ["implicit", "explicit"],
                        "description": (
                            "Intent type classification:\n"
                            "- 'implicit': User is vague, does NOT specify artifact ID (e.g., 'turn on a light', 'turn on the light'). "
                            "System must infer which artifact from context. Matches on both intent AND context (SHACL validation).\n"
                            "- 'explicit': User specifies exact artifact ID (e.g., 'turn on light308', 'set brightness for light308 to 50%'). "
                            "System does NOT need to infer. Matches only on intent structure, skips context validation.\n"
                            "Default: 'implicit' if not specified."
                        ),
                        "default": "implicit"
                    }
                },
                "required": ["intent_list"],
            },
            func=self.run_impl,
        )
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    async def run_impl(self, intent_list: List[str], workspace_id: str | None = None, intent_type: str = "implicit"):
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

        # Normalize intent_type (default to implicit for backward compatibility)
        intent_type = str(intent_type).lower().strip() if intent_type else "implicit"
        if intent_type not in ("implicit", "explicit"):
            logger.warning(demo("Invalid intent_type=%r, defaulting to 'implicit'"), intent_type)
            intent_type = "implicit"

        thread = getattr(self.agent, "active_conversation_id", None)
        timeout = float((getattr(self.agent, "config", {}) or {}).get("planning", {}).get("timeout", 60))

        # Log the classified intent type
        logger.info(
            demo("UA -> InteractionSolver GOAL_REQUEST: intents=%s workspace_id=%r intent_type=%s"),
            intents,
            workspace_id,
            intent_type.upper(),
        )
        logger.info(
            demo("Intent Type Classified: %s (%s)"),
            intent_type.upper(),
            "vague request, infer from context" if intent_type == "implicit" else "exact artifact ID specified",
        )

        try:
            result = await rpc_call(
                self.agent,
                to_jid=str(solver_jid),
                request_type=MessageType.GOAL_REQUEST.value,
                body={
                    **({"workspace_id": str(workspace_id)} if workspace_id else {}),
                    "intents": intents,
                    "intent_type": intent_type,  # Propagate intent_type to InteractionSolver
                },
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
                "Store the exact behavior tree plan JSON string as the latest proposed plan for this conversation. "
                "Call this immediately after receiving a valid plan, before summarizing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_json": {"type": "string", "description": "Exact behavior tree plan JSON string to store."}
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

        node_count = None
        plan_preview = None
        try:
            tree = plan_obj.get("tree")
            if isinstance(tree, dict) and tree:
                node_count = _count_bt_nodes(tree)
                plan_preview = _bt_preview(tree)
            elif isinstance(plan_obj.get("steps"), list):
                # Backward compat: JSON-Plan 1.2 format
                node_count = len(plan_obj["steps"])
        except Exception:
            node_count = None

        logger.info(
            demo("Plan stored for approval: thread=%s plan_hash=%s nodes=%s"),
            thread,
            plan_hash,
            node_count if node_count is not None else "?",
        )
        if plan_preview:
            logger.info(demo("BT plan preview: %s"), plan_preview)
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


class ExecuteBTTool(LLMTool):
    """
    Execute a behavior tree plan by compiling JSON IR to py_trees and running the tick loop.

    Uses IRExecutor to compile the BT JSON IR to a py_trees tree, then executes via tick loop.
    After execution, extracts signifiers from leaf action nodes for recording.
    """

    def __init__(self):
        super().__init__(
            name="execute_plan",
            description=(
                "Executes a behavior tree plan by compiling it and running the tick loop in the live environment. "
                "Use ONLY when the user explicitly asks to execute/apply the plan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_json": {
                        "type": "string",
                        "description": "The behavior tree plan JSON string (containing 'tree' field with JSON IR).",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, validate the tree structure without executing.",
                        "default": False,
                    },
                },
                "required": ["plan_json"],
            },
            func=self.run_impl,
        )
        self.agent = None
        self._executor = IRExecutor(max_ticks=50)

    def set_agent(self, agent):
        self.agent = agent

    async def run_impl(self, plan_json: str, dry_run: bool = False):
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

        # Extract BT tree spec and execution context
        tree_spec = plan_obj.get("tree")
        intents = plan_obj.get("intents", [])
        is_signifier_reuse = plan_obj.get("signifier_reuse", False)
        workspace_id = plan_obj.get("workspace_id")
        intent_type = plan_obj.get("intent_type")  # Extract intent_type for signifier recording
        logger.info(
            demo("[INTENT_TYPE] Extracted from plan: intent_type=%r"),
            intent_type,
        )
        execution_context = plan_obj.get("execution_context", {})

        if not tree_spec or not isinstance(tree_spec, dict):
            # Check for error in plan
            error = plan_obj.get("error")
            if error:
                return json.dumps({
                    "error": error,
                    "detail": plan_obj.get("detail") or plan_obj.get("explanation", "Plan has no executable tree."),
                }, indent=2)
            return json.dumps({"error": "missing_tree", "detail": "Plan has no behavior tree to execute."}, indent=2)

        # Dry-run: validate tree structure
        if dry_run:
            validation_errors = self._executor.validate_tree(tree_spec)
            node_count = _count_bt_nodes(tree_spec)
            preview = _bt_preview(tree_spec)
            return json.dumps({
                "dry_run": True,
                "valid": len(validation_errors) == 0,
                "node_count": node_count,
                "preview": preview,
                "validation_errors": validation_errors,
            }, indent=2)

        logger.info(
            demo("Executing BT plan: thread=%s nodes=%d signifier_reuse=%s"),
            thread, _count_bt_nodes(tree_spec), is_signifier_reuse,
        )

        # Execute BT in a thread executor (py_trees tick loop is synchronous)
        loop = asyncio.get_event_loop()
        try:
            exec_result: ExecutionResult = await loop.run_in_executor(
                None,
                self._executor.execute_from_spec,
                tree_spec,
            )
        except Exception as e:
            logger.warning(demo("BT execution failed: %s"), e)
            return json.dumps({
                "error": "execution_failed",
                "detail": str(e),
            }, indent=2)

        logger.info(
            demo("BT execution complete: success=%s ticks=%d status=%s"),
            exec_result.success, exec_result.ticks, exec_result.final_status,
        )

        # Clear approved plan hash after execution
        self.agent.clear_approved_plan_hash(thread)

        # Invalidate state cache (plan execution changes environment state)
        state_memory = getattr(self.agent, "state_memory", None)
        if state_memory:
            state_memory.clear()
            logger.info(demo("State memory cache cleared after BT execution"))

        # Extract signifiers from executed BT and record in EnvExplorer
        if exec_result.success and not is_signifier_reuse:
            await self._record_signifiers(
                tree_spec, intents, exec_result, thread, workspace_id, execution_context, intent_type
            )

        return json.dumps({
            "plan_type": "behavior_tree",
            "executed": True,
            "success": exec_result.success,
            "ticks": exec_result.ticks,
            "final_status": exec_result.final_status,
            "tick_history": exec_result.tick_history,
            "error": exec_result.error,
        }, indent=2)

    async def _record_signifiers(
        self,
        tree_spec: dict,
        intents: list,
        exec_result: ExecutionResult,
        thread: str,
        workspace_id: Optional[str] = None,
        execution_context: Optional[dict] = None,
        intent_type: Optional[str] = None,
    ) -> None:
        """Extract signifiers from executed BT and record locally + publish to community."""
        # Query EnvExplorer for fresh state snapshot (for building structured_conditions)
        # NOTE: We don't use execution_context from plan anymore (it made plan JSON too large).
        # Instead, query for state at execution time to get actual, current state.
        state_snapshot = None
        explorer_jid = (getattr(self.agent, "target_jids", {}) or {}).get("explorer")

        if explorer_jid and workspace_id:
            try:
                logger.info(demo("Querying EnvExplorer for state snapshot (workspace_id=%r)"), workspace_id)
                state_response = await rpc_call(
                    self.agent,
                    to_jid=str(explorer_jid),
                    request_type=MessageType.ENV_STATE_REQUEST.value,
                    body={},  # Empty body = request all state
                    expect_type=MessageType.ENV_STATE_RESPONSE.value,
                    timeout=10.0,
                )

                if state_response and state_response.body:
                    import json
                    raw_state = json.loads(state_response.body) if isinstance(state_response.body, str) else state_response.body

                    # Build state_snapshot in expected format: {"artifacts": {...}}
                    if isinstance(raw_state, dict):
                        # Unwrap if EnvExplorer returned {"artifacts": {...}} instead of direct dict
                        if "artifacts" in raw_state and isinstance(raw_state["artifacts"], dict):
                            artifacts_dict = raw_state["artifacts"]
                            logger.info(
                                demo("Unwrapped artifacts from state response: count=%d"),
                                len(artifacts_dict)
                            )
                        else:
                            # Already in direct format
                            artifacts_dict = raw_state

                        # DEBUG: Log artifact details
                        logger.info(
                            demo("Artifacts to filter: total=%d, artifact_ids=%s"),
                            len(artifacts_dict),
                            list(artifacts_dict.keys())[:3]
                        )
                        # Sample first artifact to see workspace_id format
                        if artifacts_dict:
                            sample_id = list(artifacts_dict.keys())[0]
                            sample_info = artifacts_dict[sample_id]
                            sample_ws = sample_info.get("workspace_id") if isinstance(sample_info, dict) else None
                            logger.info(
                                demo("Sample artifact: id=%r, workspace_id=%r"),
                                sample_id, sample_ws
                            )

                        # Filter by workspace_id if needed
                        filtered_artifacts = {}
                        for artifact_id, artifact_info in artifacts_dict.items():
                            if isinstance(artifact_info, dict):
                                artifact_ws = artifact_info.get("workspace_id")
                                # Match workspace_id (exact or substring)
                                if artifact_ws and (str(artifact_ws) == str(workspace_id) or workspace_id in str(artifact_ws)):
                                    filtered_artifacts[artifact_id] = artifact_info

                        state_snapshot = {"artifacts": filtered_artifacts}
                        logger.info(
                            demo("State snapshot retrieved: %d artifacts for workspace_id=%r"),
                            len(filtered_artifacts), workspace_id
                        )
            except Exception as e:
                logger.warning(demo("Failed to query state snapshot: %s (continuing without state)"), e)

        # DEBUG: Log state_snapshot structure before extraction
        if state_snapshot:
            artifacts_in_snapshot = state_snapshot.get("artifacts", {})
            logger.info(
                demo("State snapshot before extraction: has_artifacts=%s, artifact_count=%d"),
                "artifacts" in state_snapshot,
                len(artifacts_in_snapshot) if isinstance(artifacts_in_snapshot, dict) else 0
            )
            # Log first artifact details
            if artifacts_in_snapshot and isinstance(artifacts_in_snapshot, dict):
                first_key = list(artifacts_in_snapshot.keys())[0]
                first_artifact = artifacts_in_snapshot[first_key]
                logger.info(
                    demo("First artifact in snapshot: key=%r, has_properties=%s, properties=%s"),
                    first_key,
                    "properties" in first_artifact if isinstance(first_artifact, dict) else False,
                    list(first_artifact.get("properties", {}).keys())[:3] if isinstance(first_artifact, dict) else []
                )

        logger.info(
            demo("[INTENT_TYPE] Calling extract_signifiers_from_bt with intent_type=%r"),
            intent_type,
        )
        signifiers = extract_signifiers_from_bt(
            tree_spec=tree_spec,
            intents=intents,
            was_successful=exec_result.success,
            workspace_id=workspace_id,
            state_snapshot=state_snapshot,
            intent_type=intent_type,
        )
        logger.info(
            demo("[INTENT_TYPE] Extracted %d signifiers from BT"),
            len(signifiers),
        )

        if not signifiers:
            logger.info(demo("No signifiers extracted from BT"))
            return

        # Debug: log structured_conditions from first signifier
        if signifiers and isinstance(signifiers[0], dict):
            conditions = signifiers[0].get("structured_conditions", [])
            logger.info(
                demo("Signifier extracted with %d structured_conditions"),
                len(conditions) if isinstance(conditions, list) else 0
            )

        logger.info(demo("Recording %d signifiers from BT execution"), len(signifiers))

        # 1. Record signifiers locally via EnvExplorer (embedded RD4 engine)
        explorer_jid = (getattr(self.agent, "target_jids", {}) or {}).get("explorer")
        if explorer_jid:
            try:
                rec_res = await rpc_call(
                    self.agent,
                    to_jid=str(explorer_jid),
                    request_type=MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value,
                    body={
                        "plan_type": "behavior_tree",
                        "tree": tree_spec,
                        "signifiers": signifiers,
                        "execution_result": exec_result.to_dict(),
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
                    demo("Local signifier recording: created_count=%s"),
                    created_count if created_count is not None else "?",
                )
            except Exception as e:
                logger.info(demo("Failed to record signifiers locally: %s"), e)

        # 2. Publish signifiers to community (cross-environment sharing)
        community_client = getattr(self.agent, "community_client", None)
        if isinstance(community_client, CommunitySignifierClient):
            published = 0
            for sig in signifiers:
                try:
                    ok = await community_client.publish_signifier(sig)
                    if ok:
                        published += 1
                except Exception:
                    pass
            logger.info(demo("Community signifier publishing: %d/%d published"), published, len(signifiers))


# Keep backward-compatible alias
ExecutePlanTool = ExecuteBTTool
