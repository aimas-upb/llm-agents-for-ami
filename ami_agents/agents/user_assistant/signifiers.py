"""What a successful plan entry is remembered as -- and the context it ran in.

Memory keeps goals under an exact key built from their structure
(`shared/utils/goal_signifiers.py`), and only what is worth remembering:

    explicit / incomplete   only plans an LLM had to make (`llm: true` on the
                            entry): the whole tree, replayed on the next exact
                            match without a model call
    ambiguous               every successful plan: the context it was made in
                            (the affordances acting on / reading each variable,
                            with the properties' values) and what it used --
                            hints for the planner on the next exact match

    never                   a plan replayed from memory (`source: "memory"`),
                            a plan that did not succeed

The context is the environment **before the plan is applied**: a plan that turns
on a dehumidifier changes the humidity that made it appropriate. So
`capture_execution_context` reads the entry's context properties in
`PlanExecutionBehaviour.on_start`, after the tree is compiled and before its
first tick -- no action has run yet -- and the result is carried untouched to
the end.

Plan execution's whole involvement is two calls:

    context = await capture_execution_context(agent, plan_id, tree, entry=...)  # before
    agent.add_behaviour(SignifierRecordingBehaviour(context, outcome))          # after
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ...bt_planning.nodes.http_client import HTTPClient, HTTPClientConfig
from ...shared.community.community_client import CommunitySignifierClient
from ...shared.models.goal_structure import GoalSpec, GoalStructure
from ...shared.models.messages import MessageType
from ...shared.utils.demo_log import demo
from ...shared.utils.goal_signifiers import signifier_key
from ...shared.utils.logger import LoggerFactory
from ...shared.utils.spade_rpc import rpc_call

# A plan entry replayed from memory: running it again teaches nothing.
MEMORY_SOURCE = "memory"


# --------------------------------------------------------------------------
# The context, captured before the plan is applied
# --------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    """One plan entry, and the world as it was before the plan ran.

    Immutable once captured: plan execution only holds it and hands it back.
    """

    plan_id: str
    request_id: Optional[str]
    thread: str
    tree_spec: Dict[str, Any]
    entry: Dict[str, Any] = field(default_factory=dict)   # the plan entry, as planned
    intent: Optional[Dict[str, Any]] = None               # its goal structure (wire)
    workspace_id: Optional[str] = None
    # Ambiguous goals: property URL -> value read before the first tick (None:
    # unreadable). None when nothing was to be read, or the snapshot failed.
    snapshot: Optional[Dict[str, Any]] = None
    snapshot_failed: bool = False
    captured_at: datetime = field(default_factory=datetime.now)

    def goal(self) -> Optional[GoalSpec]:
        """The entry's one goal, when its intent is a simple goal structure."""
        if not isinstance(self.intent, dict) or not isinstance(self.intent.get("goals"), dict):
            return None
        structure = GoalStructure.from_dict(self.intent)
        if structure.structure != "simple" or len(structure.goals) != 1:
            return None
        (goal,) = structure.goals.values()
        return goal

    @property
    def should_record(self) -> bool:
        """Whether this entry, if it succeeds, is worth remembering."""
        if self.entry.get("source") == MEMORY_SOURCE:
            return False
        goal = self.goal()
        if goal is None or signifier_key(goal) is None:
            return False
        if goal.goal_specificity == "ambiguous":
            return bool(self.entry.get("goal_context")) and not self.snapshot_failed
        return bool(self.entry.get("llm"))


def context_property_urls(goal_context: Optional[Dict[str, Any]]) -> List[str]:
    """Every property URL of an ambiguous entry's context, once each."""
    urls: List[str] = []
    for variable in (goal_context or {}).get("variables") or []:
        for prop in variable.get("properties") or []:
            url = prop.get("url")
            if url and url not in urls:
                urls.append(url)
    return urls


async def read_properties(urls: List[str], client: Optional[HTTPClient] = None) -> Dict[str, Any]:
    """The current value of each property, read concurrently; None where a
    read fails. The HTTP client is the one behaviour-tree nodes use."""
    client = client or HTTPClient(config=HTTPClientConfig())
    loop = asyncio.get_running_loop()

    async def read(url: str) -> Any:
        try:
            response = await loop.run_in_executor(None, client.get, url)
        except Exception:  # noqa: BLE001 - one unreadable property is stored as None
            return None
        return response.body if getattr(response, "is_success", False) else None

    values = await asyncio.gather(*(read(url) for url in urls))
    return dict(zip(urls, values))


async def capture_execution_context(
    agent,
    plan_id: str,
    tree_spec: Dict[str, Any],
    *,
    request_id: Optional[str] = None,
    thread: str = "__default__",
    entry: Optional[Dict[str, Any]] = None,
    intent: Optional[Dict[str, Any]] = None,
    workspace_id: Optional[str] = None,
    logger=None,
    http_client: Optional[HTTPClient] = None,
) -> ExecutionContext:
    """The context for one plan entry. Awaited before the plan's first tick:
    an ambiguous entry's context properties are read here, before any action."""
    logger = logger or LoggerFactory.get_logger("UserAssistant")
    context = ExecutionContext(
        plan_id=plan_id, request_id=request_id, thread=thread, tree_spec=tree_spec,
        entry=dict(entry or {}), intent=intent if isinstance(intent, dict) else None,
        workspace_id=workspace_id)

    goal = context.goal()
    if goal is None or goal.goal_specificity != "ambiguous":
        return context
    urls = context_property_urls(context.entry.get("goal_context"))
    if not urls:
        return context
    try:
        context.snapshot = await read_properties(urls, http_client)
        logger.info(demo(f"[MEMORY] Snapshot before plan={plan_id}: {len(urls)} properties"))
    except Exception as exc:  # noqa: BLE001 - the plan runs; it is just not remembered
        context.snapshot_failed = True
        logger.warning(f"Snapshot before plan={plan_id} failed: {exc}")
    return context


# --------------------------------------------------------------------------
# The outcome, and the record built from both
# --------------------------------------------------------------------------

@dataclass
class ExecutionOutcome:
    """What the run amounted to."""

    success: bool
    final_status: str
    ticks: int = 0
    executed_actions: List[tuple] = field(default_factory=list)
    error: Optional[str] = None


def collect_action_urls(tree_spec: dict) -> List[str]:
    """Every distinct action URL in a tree, in order."""
    return [a["url"] for a in _used_in_tree(tree_spec)["actions"]]


def _used_in_tree(tree_spec: Any) -> Dict[str, List[Dict[str, Any]]]:
    """The affordances a tree uses: its actions (with what they send) and the
    properties it reads (condition, wait_condition and read nodes)."""
    actions: Dict[str, Dict[str, Any]] = {}
    properties: Dict[str, Dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        if kind == "action" and node.get("action_url"):
            sent = node.get("parameters") or {}
            if node.get("parameter_keys"):
                sent = {**sent, **{k: f"<computed: {v}>"
                                   for k, v in node["parameter_keys"].items()}}
            actions.setdefault(str(node["action_url"]), {"url": str(node["action_url"]),
                                                         "parameters": sent})
        elif kind in ("condition", "wait_condition", "read") and node.get("property_url"):
            properties.setdefault(str(node["property_url"]), {"url": str(node["property_url"])})
        for child in node.get("children") or []:
            walk(child)

    walk(tree_spec)
    return {"actions": list(actions.values()), "properties": list(properties.values())}


def build_signifier(context: ExecutionContext) -> Optional[Dict[str, Any]]:
    """The record for a successful entry, or None when there is nothing to keep."""
    goal = context.goal()
    sk = signifier_key(goal) if goal is not None else None
    if sk is None:
        return None
    record: Dict[str, Any] = {
        "kind": sk.kind, "key": sk.key, "intent_text": goal.intent_text,
        "goal": goal.to_wire_dict(), "workspace_id": context.workspace_id,
    }

    # Explicit / incomplete: the whole plan, to replay.
    if sk.kind != "ambiguous":
        record["tree"] = context.tree_spec
        return record

    # Ambiguous: the context with its values before the plan, and what the plan
    # used -- named after the context's own affordances.
    goal_context = json.loads(json.dumps(context.entry.get("goal_context") or {}))
    snapshot = context.snapshot or {}
    names: Dict[str, Dict[str, Any]] = {}
    for variable in goal_context.get("variables") or []:
        for action in variable.get("actions") or []:
            names.setdefault(action.get("url"), {"artifact": action.get("artifact"),
                                                 "action": action.get("action")})
        for prop in variable.get("properties") or []:
            prop["value"] = snapshot.get(prop.get("url"))
            names.setdefault(prop.get("url"), {"artifact": prop.get("artifact"),
                                               "property": prop.get("property")})
    used = _used_in_tree(context.tree_spec)
    for item in used["actions"] + used["properties"]:
        item.update(names.get(item["url"], {}))
    record["context"] = goal_context
    record["used"] = used
    return record


# --------------------------------------------------------------------------
# The behaviour
# --------------------------------------------------------------------------

class SignifierRecordingBehaviour(OneShotBehaviour):
    """Remembers one plan entry, after it has finished.

    A behaviour of its own so recording never delays a tick and its failures
    stay its own: an unreachable EnvExplorer must not turn a plan that ran fine
    into a plan that reports failure.
    """

    def __init__(self, context: ExecutionContext, outcome: ExecutionOutcome, logger=None):
        super().__init__()
        self.context = context
        self.outcome = outcome
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        self.signifier: Optional[Dict[str, Any]] = None
        self.recorded_id: Optional[str] = None
        self.published = False

    async def run(self) -> None:
        context = self.context
        if not self.outcome.success:
            self.logger.debug(f"Not remembering failed plan={context.plan_id}")
            return
        if not context.should_record:
            self.logger.debug(f"Not remembering plan={context.plan_id} "
                              f"(source={context.entry.get('source')})")
            return

        self.signifier = build_signifier(context)
        if self.signifier is None:
            return
        self.logger.info(demo(
            f"[MEMORY] Remembering {self.signifier['kind']} plan={context.plan_id} "
            f"for {self.signifier.get('intent_text')!r}"))
        await self._record_locally()
        await self._publish_to_community()

    async def _record_locally(self) -> None:
        explorer_jid = self.agent.target_jids.get("explorer")
        if not explorer_jid:
            return
        try:
            response = await rpc_call(
                self.agent,
                to_jid=str(explorer_jid),
                request_type=MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value,
                body={"signifier": self.signifier},
                expect_type=MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value,
                timeout=self.agent.rpc_call_timeout,
                thread=(self.context.thread
                        if self.context.thread != "__default__" else None),
            )
            payload = json.loads(response.body or "{}")
            if isinstance(payload, dict):
                self.recorded_id = payload.get("signifier_id")
        except Exception as exc:  # noqa: BLE001 - recording is best-effort
            self.logger.info(demo(f"Failed to record the signifier locally: {exc}"))

    async def _publish_to_community(self) -> None:
        client = getattr(self.agent, "community_client", None)
        if not isinstance(client, CommunitySignifierClient):
            return
        try:
            self.published = await client.publish_signifier(self.signifier)
        except Exception:  # noqa: BLE001 - publishing is best-effort
            self.published = False
