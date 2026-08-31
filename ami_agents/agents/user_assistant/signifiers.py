"""
What a plan's execution should be remembered as — and the context it happened in.

Recording a signifier is not part of running a plan. It is a separate question
asked afterwards: *given what the environment looked like when this was
launched, and what the tree actually did, what is worth remembering?*

Two things follow from that, and shape this module:

**The context is the environment as it was when the request began, not when it
ended.** A plan that turns on a heater changes the very state that made the plan
appropriate; sampling afterwards records the consequence and calls it the
condition. So `ExecutionContext` is captured before the first tick and carried
untouched to the end.

**Extraction is going to change.** The current procedure — state snapshot,
TD-SOSA effect lookup, `extract_signifiers_from_bt` — is being rebuilt against
TD-SOSA proper. Everything specific to it lives behind `SignifierExtractor`, so
that rebuild replaces one class and touches nothing in plan execution. What
crosses the boundary is a context and a result, never a tree walked in place.

Plan execution's whole involvement is two calls:

    context = await capture_execution_context(agent, plan_id, ...)   # before
    agent.add_behaviour(SignifierRecordingBehaviour(context, outcome))  # after
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ...bt_planning.signifier_bridge import extract_signifiers_from_bt
from ...shared.community.community_client import CommunitySignifierClient
from ...shared.models.intents import (
    ExplicitGoalIntent,
    ImplicitGoalIntent,
    Intent,
)
from ...shared.models.messages import MessageType
from ...shared.utils.demo_log import demo
from ...shared.utils.logger import LoggerFactory
from ...shared.utils.spade_rpc import rpc_call


@dataclass
class ExecutionContext:
    """The world as it was when a plan was launched.

    Immutable once captured. Plan execution neither reads nor updates it -- it
    holds the object and hands it back when the plan ends.
    """

    plan_id: str
    request_id: Optional[str]
    thread: str
    tree_spec: Dict[str, Any]
    intents: List[Any] = field(default_factory=list)
    intent_type: Optional[str] = None
    workspace_id: Optional[str] = None
    td_sosa_supported: bool = False
    is_signifier_reuse: bool = False

    # The environment snapshot, taken before the first tick.
    state_snapshot: Optional[Dict[str, Any]] = None
    semantic_effects: Dict[str, List[dict]] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=datetime.now)

    @property
    def should_record(self) -> bool:
        """Signifier reuse teaches nothing: it is a replay of what was learned."""
        if self.is_signifier_reuse:
            return False
        return str(self.intent_type or "").upper() in ("IMPLICIT", "EXPLICIT")

    def merged_snapshot(self) -> Optional[Dict[str, Any]]:
        """The snapshot with TD-SOSA effects folded in, as extraction wants it."""
        snapshot = self.state_snapshot
        if snapshot is None and self.semantic_effects:
            snapshot = {"artifacts": {}}
        if snapshot is not None and self.semantic_effects:
            snapshot = {**snapshot,
                        "td_sosa_action_effects": self.semantic_effects}
        return snapshot


@dataclass
class ExecutionOutcome:
    """What the run amounted to. Deliberately not an `ExecutionResult`.

    Plan execution now ticks a tree itself rather than calling a blocking
    executor, so there is no `ExecutionResult` to pass on. This is the small
    shape recording actually needs, which also keeps the two sides independent.
    """

    success: bool
    final_status: str
    ticks: int = 0
    executed_actions: List[tuple] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_status": self.final_status,
            "ticks": self.ticks,
            "executed_actions": [list(a) for a in self.executed_actions],
            "error": self.error,
        }


def collect_action_urls(tree_spec: dict) -> List[str]:
    """Every distinct action URL in a tree, in order."""
    urls: List[str] = []
    seen: set = set()

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "action":
            url = str(node.get("action_url") or "").strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        for child in node.get("children") or []:
            walk(child)

    walk(tree_spec)
    return urls


async def capture_execution_context(
    agent,
    plan_id: str,
    tree_spec: Dict[str, Any],
    *,
    request_id: Optional[str] = None,
    thread: str = "__default__",
    intents: Optional[List[Any]] = None,
    intent_type: Optional[str] = None,
    workspace_id: Optional[str] = None,
    td_sosa_supported: bool = False,
    is_signifier_reuse: bool = False,
    logger=None,
) -> ExecutionContext:
    """Snapshot the environment *before* a plan runs.

    Called once, ahead of the first tick. Failing to reach EnvExplorer is not
    fatal: a context with no snapshot still records the plan and its outcome,
    just without the state that justified it.
    """
    logger = logger or LoggerFactory.get_logger("UserAssistant")
    context = ExecutionContext(
        plan_id=plan_id,
        request_id=request_id,
        thread=thread,
        tree_spec=tree_spec,
        intents=list(intents or []),
        intent_type=intent_type,
        workspace_id=workspace_id,
        td_sosa_supported=td_sosa_supported,
        is_signifier_reuse=is_signifier_reuse,
    )
    if not context.should_record:
        return context

    explorer_jid = agent.target_jids.get("explorer")
    if not explorer_jid:
        return context

    # EXPLICIT intents name their target, so the surrounding state is not what
    # made them appropriate and is deliberately not recorded.
    if "IMPLICIT" in str(intent_type or "").upper() and workspace_id:
        context.state_snapshot = await _query_state_snapshot(
            agent, explorer_jid, workspace_id, logger)
    elif "EXPLICIT" in str(intent_type or "").upper():
        logger.info(demo(
            "Skipping state context for EXPLICIT intent "
            "(user specified exact target)"))

    if td_sosa_supported and workspace_id:
        context.semantic_effects = await _query_semantic_effects(
            agent, explorer_jid, workspace_id, tree_spec, logger)

    return context


async def _query_state_snapshot(agent, explorer_jid, workspace_id,
                                logger) -> Optional[Dict[str, Any]]:
    try:
        logger.info(demo(
            f"Querying environment state for signifier context "
            f"(workspace_id={workspace_id!r})"))
        response = await rpc_call(
            agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_STATE_REQUEST.value,
            body={},
            expect_type=MessageType.ENV_STATE_RESPONSE.value,
            timeout=agent.signifier_match_timeout,
        )
        if not (response and response.body):
            return None

        raw = (json.loads(response.body)
               if isinstance(response.body, str) else response.body)
        if not isinstance(raw, dict):
            return None

        artifacts = raw.get("artifacts") if isinstance(
            raw.get("artifacts"), dict) else raw
        filtered = {}
        for artifact_id, info in artifacts.items():
            if not isinstance(info, dict):
                continue
            artifact_ws = info.get("workspace_id")
            if artifact_ws and (str(artifact_ws) == str(workspace_id)
                                or workspace_id in str(artifact_ws)):
                filtered[artifact_id] = info

        logger.info(demo(
            f"State snapshot retrieved: {len(filtered)} artifacts for "
            f"workspace_id={workspace_id!r}"))
        return {"artifacts": filtered}
    except Exception as exc:  # noqa: BLE001 - recording proceeds without it
        logger.warning(demo(
            f"Failed to query state snapshot: {exc} (continuing without state)"))
        return None


async def _query_semantic_effects(agent, explorer_jid, workspace_id, tree_spec,
                                  logger) -> Dict[str, List[dict]]:
    action_urls = collect_action_urls(tree_spec)
    if not action_urls:
        return {}
    try:
        logger.info(demo(
            "Querying TD-SOSA action effects for signifier recording: actions=%d"),
            len(action_urls))
        response = await rpc_call(
            agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_SEMANTIC_QUERY_REQUEST.value,
            body={"workspace_id": workspace_id, "action_urls": action_urls},
            expect_type=MessageType.ENV_SEMANTIC_QUERY_RESPONSE.value,
            timeout=agent.signifier_match_timeout,
        )
        raw = (json.loads(response.body)
               if response and isinstance(response.body, str) else {})
        effects: Dict[str, List[dict]] = {}
        if isinstance(raw, dict):
            for item in raw.get("action_effects") or []:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("action_url") or "").strip()
                item_effects = item.get("effects")
                if url and isinstance(item_effects, list):
                    effects[url] = [e for e in item_effects if isinstance(e, dict)]
        logger.info(demo(
            "TD-SOSA action effects retrieved for signifier recording: "
            "actions=%d effects=%d"),
            len(effects), sum(len(v) for v in effects.values()))
        return effects
    except Exception as exc:  # noqa: BLE001 - recording proceeds without them
        logger.warning(demo("Failed to query TD-SOSA action effects: %s"), exc)
        return {}


class SignifierExtractor:
    """Turns a context plus an outcome into signifiers.

    The seam. Everything below is the *current* procedure and is expected to be
    replaced wholesale by the TD-SOSA rebuild; swapping this class for another
    with the same `extract()` is the entire change, and nothing in plan
    execution or the recording behaviour needs to know.
    """

    def extract(self, context: ExecutionContext,
                outcome: ExecutionOutcome) -> List[dict]:
        intent_strings = [self._as_query_string(i) for i in context.intents]
        structured = [
            i.to_wire_dict()
            if isinstance(i, (ImplicitGoalIntent, ExplicitGoalIntent, Intent))
            else (i if isinstance(i, dict) else None)
            for i in context.intents
        ]
        return extract_signifiers_from_bt(
            tree_spec=context.tree_spec,
            intents=[s for s in intent_strings if s],
            was_successful=outcome.success,
            workspace_id=context.workspace_id,
            state_snapshot=context.merged_snapshot(),
            intent_type=context.intent_type,
            structured_intents=[s for s in structured if s is not None],
            td_sosa_supported=context.td_sosa_supported,
            executed_actions=outcome.executed_actions or None,
        )

    @staticmethod
    def _as_query_string(intent: Any) -> str:
        if hasattr(intent, "to_query_string"):
            return intent.to_query_string()
        if isinstance(intent, dict):
            return str(intent.get("text_intent") or "")
        return str(intent or "")


class SignifierRecordingBehaviour(OneShotBehaviour):
    """Records one plan's signifiers, after that plan has finished.

    A behaviour of its own so recording never delays a tick and its failures
    stay its own: an unreachable EnvExplorer must not turn a plan that ran fine
    into a plan that reports failure.
    """

    def __init__(self, context: ExecutionContext, outcome: ExecutionOutcome,
                 extractor: Optional[SignifierExtractor] = None, logger=None):
        super().__init__()
        self.context = context
        self.outcome = outcome
        self.extractor = extractor or SignifierExtractor()
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        self.signifiers: List[dict] = []
        self.created_count: Optional[int] = None
        self.published_count: int = 0

    async def run(self) -> None:
        context, outcome = self.context, self.outcome

        if not context.should_record:
            self.logger.debug(
                f"Not recording signifiers for plan={context.plan_id} "
                f"(reuse={context.is_signifier_reuse} "
                f"intent_type={context.intent_type})")
            return
        if not outcome.success:
            self.logger.debug(
                f"Not recording signifiers for failed plan={context.plan_id}")
            return

        try:
            self.signifiers = self.extractor.extract(context, outcome)
        except Exception as exc:  # noqa: BLE001 - never fail the plan
            self.logger.warning(f"Signifier extraction failed: {exc}")
            return

        if not self.signifiers:
            self.logger.info(demo("No signifiers extracted from BT"))
            return

        self.logger.info(demo(
            f"Recording {len(self.signifiers)} signifiers from BT execution "
            f"(intent_type={context.intent_type})"))

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
                body={
                    "plan_type": "behavior_tree",
                    "tree": self.context.tree_spec,
                    "signifiers": self.signifiers,
                    "execution_result": self.outcome.to_dict(),
                },
                expect_type=MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value,
                timeout=self.agent.rpc_call_timeout,
                thread=(self.context.thread
                        if self.context.thread != "__default__" else None),
            )
            try:
                payload = json.loads(response.body or "{}")
                if isinstance(payload, dict):
                    self.created_count = payload.get("created_count")
            except (json.JSONDecodeError, TypeError):
                pass
            self.logger.info(demo(
                f"Local signifier recording: "
                f"created_count={self.created_count or '?'}"))
        except Exception as exc:  # noqa: BLE001 - recording is best-effort
            self.logger.info(demo(f"Failed to record signifiers locally: {exc}"))

    async def _publish_to_community(self) -> None:
        client = getattr(self.agent, "community_client", None)
        if not isinstance(client, CommunitySignifierClient):
            return
        for signifier in self.signifiers:
            try:
                if await client.publish_signifier(signifier):
                    self.published_count += 1
            except Exception:  # noqa: BLE001 - one failure must not stop the rest
                pass
        self.logger.info(demo(
            f"Community signifier publishing: "
            f"{self.published_count}/{len(self.signifiers)} published"))
