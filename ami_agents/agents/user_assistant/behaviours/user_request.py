"""
One user utterance, handled end to end — and stopping at confirmation.

`UserMessageBehaviour` used to do all of this inline, which meant the UA could
handle exactly one request at a time: a second utterance waited for the first to
finish planning, summarising, and executing. Here each utterance gets its own
FSM instance, and SPADE's own scheduling interleaves them.

The machine deliberately ends at confirmation. Executing a confirmed plan is
`PlanManagementBehaviour`'s job, reached by sending a message rather than by
calling into it, so a plan outlives the request that asked for it and stays
addressable — for status, for cancellation, for explanation — long after this
FSM is gone.

States:

    SEGMENTING ──► EXTRACTING ──► AWAITING_PLAN ──► SUMMARIZING
                        │                                │
                        └──► DONE (queries answer         ▼
                              inline and finish)   AWAITING_CONFIRMATION ──┐
                                                          │ ▲             │
                                                          │ └─ self-loop ─┘
                                                          ▼
                                                      SUBMITTING ──► DONE

Two SPADE details this relies on:

- A `State` reads the FSM instance's own mailbox: `FSMBehaviour._run` assigns
  `behaviour.receive = self.receive` (spade/behaviour.py:726). That is how
  AWAITING_CONFIRMATION hears the user's yes/no.
- A state that sets no next state kills the FSM (spade/behaviour.py:763), which
  is how DONE tears the machine down. The corollary is that a waiting state MUST
  register a self-transition, or the first idle poll ends the request.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from spade.behaviour import FSMBehaviour, State

from ....shared.models.intents import goal_intent_from_dict
from ....shared.models.messages import (
    META_REQUEST_ID,
    MessageType,
    new_plan_id,
)
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ....shared.utils.spade_rpc import rpc_call, RpcTimeoutError
from .. import pipeline, queries
from ..models import (
    CONFIRM_TOKENS,
    ConversationPhase,
    ConversationState,
    REJECT_TOKENS,
)
from ..utils import canonicalize_plan_for_hash, coerce_plan_dict, count_bt_nodes

logger = LoggerFactory.get_logger("UserAssistant")

SEGMENTING = "SEGMENTING"
EXTRACTING = "EXTRACTING"
AWAITING_PLAN = "AWAITING_PLAN"
SUMMARIZING = "SUMMARIZING"
AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
SUBMITTING = "SUBMITTING"
DONE = "DONE"


def _label(category: str) -> str:
    """The demo log's yellow category tag, unless colour is switched off."""
    if os.getenv("AMI_NO_COLOR") or os.getenv("NO_COLOR"):
        return f"[{category}]"
    return f"\x1b[1;33m[{category}]\x1b[0m"


class _RequestState(State):
    """Shared access to the request this FSM is handling.

    SPADE gives a State only `agent` and `receive` (see `FSMBehaviour._run`),
    never a handle on the machine it belongs to -- so the FSM passes itself in.
    """

    def __init__(self, request: "UserRequestBehaviour"):
        super().__init__()
        self.request = request

    @property
    def logger(self):
        return self.request.logger

    @property
    def conv(self) -> ConversationState:
        return self.request.conv

    async def reply(self, text: str) -> None:
        await self.request.reply(text)


class SegmentingState(_RequestState):
    """Split the utterance into atomic intents and classify each one."""

    async def run(self) -> None:
        request = self.request
        self.conv.phase = ConversationPhase.SEGMENTING

        capabilities_ctx = await pipeline.fetch_capabilities(
            request.agent, self.logger, detail_level="summary")
        try:
            caps_summary = json.loads(capabilities_ctx) if capabilities_ctx else {}
        except (json.JSONDecodeError, AttributeError):
            caps_summary = {}

        request.capabilities_ctx = capabilities_ctx
        request.caps_summary = caps_summary

        filtered = pipeline.filter_capabilities_json(caps_summary)
        atomic_intents = await pipeline.segment_into_atomic_intents(
            request.agent, self.logger, request.user_text, json.dumps(filtered))

        if not atomic_intents:
            await self.reply(
                "I couldn't understand your request. Could you rephrase?")
            request.finish("failed")
            self.set_next_state(DONE)
            return

        summary: Dict[str, int] = {}
        for intent in atomic_intents:
            summary[intent.category] = summary.get(intent.category, 0) + 1
        self.logger.info(demo(
            f"[SEGMENTATION] request={request.request_id} "
            f"{len(atomic_intents)} atomic intent(s): {summary}"))
        for intent in atomic_intents:
            self.logger.info(demo(
                f"[SEGMENTATION] - {intent.category}: span={intent.span!r} "
                f"reason={intent.reason}"))

        request.agent.requests.note_intents(request.request_id, [
            {"span": i.span, "category": i.category, "reason": i.reason}
            for i in atomic_intents
        ])
        request.atomic_intents = atomic_intents
        self.set_next_state(EXTRACTING)


class ExtractingState(_RequestState):
    """Parse each span. Queries answer here; goals go on to planning."""

    async def run(self) -> None:
        request = self.request
        self.conv.phase = ConversationPhase.EXTRACTING_INTENTS

        caps = [a for a in request.atomic_intents
                if a.category == "ENV_CAPABILITIES_REQUEST"]
        state = [a for a in request.atomic_intents
                 if a.category == "ENV_STATE_REQUEST"]
        goals = [a for a in request.atomic_intents
                 if a.category == "GOAL_REQUEST"]

        hierarchical = pipeline.build_capabilities_hierarchical_text(
            request.caps_summary)
        hierarchical_state = ""
        if state:
            hierarchical_state = json.dumps(
                pipeline.filter_capabilities_json_for_state(request.caps_summary),
                indent=2)

        for intent in caps:
            extraction = await pipeline.parse_atomic_intent(
                request.agent, self.logger, intent.span, intent.category,
                hierarchical)
            self.logger.info(demo(
                f"{_label('ENV_CAPABILITIES_REQUEST')}:\n"
                f"{json.dumps(extraction, indent=2)}"))
            await request.answer_capabilities_query(extraction)

        for intent in state:
            extraction = await pipeline.parse_atomic_intent(
                request.agent, self.logger, intent.span, intent.category,
                hierarchical_state)
            self.logger.info(demo(
                f"{_label('ENV_STATE_REQUEST')}:\n"
                f"{json.dumps(extraction, indent=2)}"))
            await request.answer_state_query(extraction)

        if not goals:
            # A pure query: answered above, nothing to plan.
            request.finish("answered")
            self.set_next_state(DONE)
            return

        extractions = await asyncio.gather(
            *[pipeline.parse_atomic_intent(request.agent, self.logger,
                                           intent.span, intent.category,
                                           hierarchical)
              for intent in goals],
            return_exceptions=True,
        )

        parsed: List[Any] = []
        for extraction in extractions:
            if isinstance(extraction, Exception):
                self.logger.error("Failed to parse atomic intent: %s", extraction)
                continue
            if isinstance(extraction, dict):
                self.logger.info(demo(
                    f"{_label('GOAL_REQUEST')}:\n"
                    f"{json.dumps(extraction, indent=2)}"))
                parsed.append(goal_intent_from_dict(extraction))

        if not parsed:
            await self.reply("Could not parse your request. Could you rephrase?")
            request.finish("failed")
            self.set_next_state(DONE)
            return

        self.conv.intents = parsed
        request.parsed_intents = parsed
        self.set_next_state(AWAITING_PLAN)


class AwaitingPlanState(_RequestState):
    """Ask the InteractionSolver for a plan."""

    async def run(self) -> None:
        request = self.request
        self.conv.phase = ConversationPhase.AWAITING_PLAN

        solver_jid = request.agent.target_jids.get("solver")
        if not solver_jid:
            self.logger.info(demo(
                "Goal handling failed: missing solver JID for thread=%s"),
                request.thread)
            await self.reply("Error: InteractionSolver is not configured.")
            request.finish("failed")
            self.set_next_state(DONE)
            return

        for index, intent in enumerate(request.parsed_intents):
            self.logger.info(demo(
                f"Parsed goal intent #{index + 1}: "
                f"{json.dumps(intent.to_wire_dict(), indent=2)}"))

        body: Dict[str, Any] = {
            "intents": [i.to_wire_dict() for i in request.parsed_intents],
        }
        if self.conv.workspace_id:
            body["workspace_id"] = str(self.conv.workspace_id)

        categories = [i.to_wire_dict().get("category", "unknown")
                      for i in request.parsed_intents]
        self.logger.info(demo(
            f"UA -> InteractionSolver GOAL_REQUEST: "
            f"{len(request.parsed_intents)} intents, categories: {categories}"))

        timeout = request.agent.goal_request_timeout
        try:
            result = await rpc_call(
                request.agent,
                to_jid=str(solver_jid),
                request_type=MessageType.GOAL_REQUEST.value,
                body=body,
                expect_type=MessageType.PLAN_CREATED.value,
                timeout=timeout,
                thread=request.thread if request.thread != "__default__" else None,
            )
        except RpcTimeoutError:
            self.logger.info(demo("Goal request timed out: thread=%s timeout=%s"),
                             request.thread, timeout)
            await self.reply("Planning timed out. Could you try again?")
            request.finish("failed")
            self.set_next_state(DONE)
            return
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.logger.error("Goal request failed: %s", exc)
            await self.reply(f"Planning failed: {exc}")
            request.finish("failed")
            self.set_next_state(DONE)
            return

        request.plan_body = result.body
        self.set_next_state(SUMMARIZING)


class SummarizingState(_RequestState):
    """Store the plan, hash it, and tell the user what it will do."""

    async def run(self) -> None:
        request = self.request
        self.conv.phase = ConversationPhase.SUMMARIZING_PLAN

        plan_obj = coerce_plan_dict(request.plan_body)
        if not plan_obj:
            await self.reply("I received an unusable plan. Could you try again?")
            request.finish("failed")
            self.set_next_state(DONE)
            return

        canonical, plan_hash = canonicalize_plan_for_hash(plan_obj)
        self.conv.plan_json = canonical
        self.conv.plan_hash = plan_hash
        request.plan_obj = plan_obj
        request.plan_hash = plan_hash

        plans_list = plan_obj.get("plans")
        if isinstance(plans_list, list) and plans_list:
            self.conv.plan_count = len(plans_list)
            total = sum(count_bt_nodes(p.get("tree") or {}) for p in plans_list)
            self.logger.info(demo(
                f"Multi-plan stored: hash={plan_hash} plans={len(plans_list)} "
                f"total_nodes={total}"))
        else:
            self.conv.plan_count = 1
            self.logger.info(demo(
                f"Plan stored: hash={plan_hash} "
                f"nodes={count_bt_nodes(plan_obj.get('tree') or {})}"))

        summary = await pipeline.summarize_plan(
            request.agent, self.logger,
            canonical if isinstance(canonical, str) else json.dumps(plan_obj))
        self.conv.plan_summary = summary
        await self.reply(summary)

        if request.auto_confirms(plan_obj):
            self.logger.info(demo(
                f"Read-only plan auto-confirmed: request={request.request_id}"))
            self.set_next_state(SUBMITTING)
            return

        self.conv.phase = ConversationPhase.AWAITING_CONFIRMATION
        request.start_confirmation_window()
        self.set_next_state(AWAITING_CONFIRMATION)


class AwaitingConfirmationState(_RequestState):
    """Wait for yes/no — and self-loop, or the FSM dies on the first idle poll."""

    async def run(self) -> None:
        request = self.request

        msg = await self.receive(timeout=1)
        if msg is not None:
            text = (msg.body or "").strip().lower()
            request.last_message = msg
            if text in REJECT_TOKENS:
                self.logger.info(demo(
                    f"Plan rejected by user: request={request.request_id}"))
                await self.reply("Okay, I won't do that.")
                self.conv.clear_plan()
                self.conv.phase = ConversationPhase.IDLE
                request.finish("rejected")
                self.set_next_state(DONE)
                return
            if text in CONFIRM_TOKENS:
                self.set_next_state(SUBMITTING)
                return
            # Anything else on this thread is a new request, not an answer.
            await self.reply(
                "I still need a yes or no on the plan I proposed.")
            self.set_next_state(AWAITING_CONFIRMATION)
            return

        if request.confirmation_expired():
            self.logger.info(demo(
                f"Confirmation window elapsed, assuming yes: "
                f"request={request.request_id}"))
            self.set_next_state(SUBMITTING)
            return

        self.set_next_state(AWAITING_CONFIRMATION)


class SubmittingState(_RequestState):
    """Hand the plan to plan management and let this request end."""

    async def run(self) -> None:
        request = self.request
        self.conv.phase = ConversationPhase.EXECUTING

        plan_id = new_plan_id()
        await request.submit_plan(plan_id)
        request.agent.requests.note_plan(request.request_id, plan_id)

        self.logger.info(demo(
            f"Plan submitted for execution: plan={plan_id} "
            f"request={request.request_id}"))
        request.finish("planned")
        self.set_next_state(DONE)


class DoneState(_RequestState):
    """Terminal. Sets no next state, so SPADE tears the FSM down."""

    async def run(self) -> None:
        self.conv.phase = ConversationPhase.IDLE
        self.logger.debug(
            f"Request {self.request.request_id} finished; FSM ending")


class UserRequestBehaviour(FSMBehaviour):
    """The state machine for one utterance.

    Owns everything about that utterance: the text, the ids, the parsed
    intents, the plan awaiting confirmation. Nothing is shared with other
    requests, so two utterances on two threads never see each other's state.
    """

    def __init__(self, request_id: str, thread: str, user_text: str,
                 original_msg, logger=None):
        super().__init__()
        self.request_id = request_id
        self.thread = thread
        self.user_text = user_text
        self.original_msg = original_msg
        self.last_message = original_msg
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")

        # Filled in as the machine advances.
        self.capabilities_ctx: str = ""
        self.caps_summary: Dict[str, Any] = {}
        self.atomic_intents: List[Any] = []
        self.parsed_intents: List[Any] = []
        self.plan_body: Any = None
        self.plan_obj: Optional[Dict[str, Any]] = None
        self.plan_hash: Optional[str] = None
        self._confirmation_deadline: Optional[float] = None

        self._build()

    def _build(self) -> None:
        self.add_state(name=SEGMENTING, state=SegmentingState(self), initial=True)
        self.add_state(name=EXTRACTING, state=ExtractingState(self))
        self.add_state(name=AWAITING_PLAN, state=AwaitingPlanState(self))
        self.add_state(name=SUMMARIZING, state=SummarizingState(self))
        self.add_state(name=AWAITING_CONFIRMATION,
                       state=AwaitingConfirmationState(self))
        self.add_state(name=SUBMITTING, state=SubmittingState(self))
        self.add_state(name=DONE, state=DoneState(self))

        self.add_transition(SEGMENTING, EXTRACTING)
        self.add_transition(SEGMENTING, DONE)
        self.add_transition(EXTRACTING, AWAITING_PLAN)
        self.add_transition(EXTRACTING, DONE)
        self.add_transition(AWAITING_PLAN, SUMMARIZING)
        self.add_transition(AWAITING_PLAN, DONE)
        self.add_transition(SUMMARIZING, AWAITING_CONFIRMATION)
        self.add_transition(SUMMARIZING, SUBMITTING)
        self.add_transition(SUMMARIZING, DONE)
        # Mandatory: without this self-transition, `is_valid_transition` raises
        # and the FSM dies the first time the user does not answer within 1s.
        self.add_transition(AWAITING_CONFIRMATION, AWAITING_CONFIRMATION)
        self.add_transition(AWAITING_CONFIRMATION, SUBMITTING)
        self.add_transition(AWAITING_CONFIRMATION, DONE)
        self.add_transition(SUBMITTING, DONE)

    # -- state shared with the states -----------------------------------

    @property
    def conv(self) -> ConversationState:
        return self.agent.get_conversation(self.thread)

    async def reply(self, text: str) -> None:
        reply = self.last_message.make_reply()
        reply.body = text
        await self.send(reply)

    def finish(self, outcome: str) -> None:
        self.agent.requests.close(self.request_id, outcome)

    # -- confirmation policy --------------------------------------------

    def start_confirmation_window(self) -> None:
        loop = asyncio.get_event_loop()
        self._confirmation_deadline = (
            loop.time() + self.agent.confirmation_timeout)

    def confirmation_expired(self) -> bool:
        if self._confirmation_deadline is None:
            return False
        return asyncio.get_event_loop().time() >= self._confirmation_deadline

    @staticmethod
    def auto_confirms(plan_obj: Dict[str, Any]) -> bool:
        """A plan that only reads needs no permission.

        Confirmation exists so nothing actuates the home unasked. A tree with
        no action nodes changes nothing, so waiting on a yes just makes the
        assistant feel slow.
        """
        trees = []
        plans = plan_obj.get("plans")
        if isinstance(plans, list):
            trees = [p.get("tree") for p in plans if isinstance(p, dict)]
        elif plan_obj.get("tree") is not None:
            trees = [plan_obj.get("tree")]

        def has_action(node: Any) -> bool:
            if not isinstance(node, dict):
                return False
            if node.get("type") == "action":
                return True
            return any(has_action(c) for c in node.get("children") or [])

        return not any(has_action(t) for t in trees)

    # -- handing off ----------------------------------------------------

    async def submit_plan(self, plan_id: str) -> None:
        """Send the confirmed plan to plan management, as a message.

        Deliberately a message and not a call: this FSM is about to end, and
        the plan must outlive it.
        """
        from spade.message import Message

        msg = Message(to=str(self.agent.jid))
        msg.thread = self.thread if self.thread != "__default__" else None
        msg.set_metadata("type", MessageType.PLAN_EXECUTE_REQUEST.value)
        msg.set_metadata(META_REQUEST_ID, self.request_id)
        msg.body = json.dumps({
            "plan_id": plan_id,
            "request_id": self.request_id,
            "thread": self.thread,
            "plan_hash": self.plan_hash,
            # The user's own words travel with the plan: an explanation should
            # answer in the language of the request, not of the tree.
            "goal_description": self.user_text,
            # Context signifier recording will need. Each plan's own intent
            # (and so its own category) travels on the plan entry itself; this
            # is only what is common to the whole request.
            "workspace_id": self.conv.workspace_id,
            "plan": self.plan_obj,
        })
        await self.send(msg)

    # -- inline query answers (delegated back to the receiver's helpers) --

    async def answer_capabilities_query(self, extraction: dict) -> None:
        await queries.answer_capabilities_query(
            self, self.last_message, self.thread, self.conv,
            self.capabilities_ctx, extraction)

    async def answer_state_query(self, extraction: dict) -> None:
        await queries.answer_state_query(
            self, self.last_message, self.thread, self.conv, extraction)
