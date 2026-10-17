"""Behaviour: handle GOAL_REQUEST messages from the UserAssistant.

Thin entry point. Parses the request, waits for environment readiness, and
delegates the actual planning to ``PlanningWorkflowBehaviour`` (which in
turn spawns the per-stage sub-behaviours: env context query, signifier
matching, community lookup, BT plan generation).
"""

import json
from typing import List, Optional, Union

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ....shared.models.goal_structure import GoalStructure
from ....shared.models.intents import Intent
from ..utils.ambiguous_goal_plan import ambiguous_goal
from ..utils.dependency_context import dependency_structure
from ..utils.explicit_goal_plan import deterministic_goal
from ..utils.incomplete_goal_plan import incomplete_goal
from ..utils.plan_envelope import (
    envelope_error,
    envelope_plan_entries,
    plan_entries_from_envelope,
)
from .ambiguous_goal_planning import AmbiguousGoalPlanningBehaviour
from .dependency_goal_planning import DependencyGoalPlanningBehaviour
from .explicit_goal_planning import ExplicitGoalPlanningBehaviour
from .incomplete_goal_planning import IncompleteGoalPlanningBehaviour
from .planning_workflow import PlanningWorkflowBehaviour
from .power_on_guard import PowerOnGuardBehaviour


# Each planning path: its name, the selector that accepts a structure for it,
# and the name of the behaviour that plans the selected goals (looked up when
# planning). Order is the planning order.
ROUTES = (
    ("explicit", deterministic_goal, "ExplicitGoalPlanningBehaviour"),
    ("incomplete", incomplete_goal, "IncompleteGoalPlanningBehaviour"),
    ("ambiguous", ambiguous_goal, "AmbiguousGoalPlanningBehaviour"),
    ("dependency", dependency_structure, "DependencyGoalPlanningBehaviour"),
)


def route(items) -> dict:
    """{route name: [(index, selected goal or structure), ...]} for the
    structured items; an item no selector accepts is in no list (it goes to
    the planning workflow, from its words)."""
    return {name: GoalRequestBehaviour._select(items, selector)
            for name, selector, _ in ROUTES}


def route_behaviour(name: str):
    """The behaviour class that plans the goals of route ``name``."""
    return globals()[next(cls for route_name, _, cls in ROUTES if route_name == name)]


class GoalRequestBehaviour(CyclicBehaviour):
    """Receive GOAL_REQUEST, validate, dispatch the planning workflow."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.GOAL_REQUEST.value:
            return

        intent_objects, workspace_id = self._parse_request(msg)

        intent_objects = [i for i in intent_objects if i.to_query_string().strip()]

        intent_strings = [i.to_query_string() for i in intent_objects]
        self.logger.info(
            demo(f"Received GOAL_REQUEST: intents={intent_strings} workspace_id={workspace_id!r} from={msg.sender}")
        )

        ready = await self.agent.await_environment_ready(
            timeout=self.agent.env_ready_timeout,
        )
        if not ready:
            envelope = envelope_error(
                "env_not_ready",
                "Environment discovery not completed (timeout).",
                intent_objects,
            )
            await self._reply(msg, MessageType.PLAN_CREATED.value, envelope)
            return

        # Structured goals are planned on their own paths: explicit, incomplete
        # and ambiguous achievement goals from the TD graph or by an LLM against
        # a context cut to the goal; dependency structures (conditions, timing,
        # maintenance) by templates. Only plain-text intents reach the workflow.
        routes = route(intent_objects)
        if any(routes.values()):
            envelope = await self._plan_mixed(
                intent_objects, routes["explicit"], routes["incomplete"], workspace_id,
                routes["ambiguous"], routes["dependency"])
        else:
            envelope = await self._run_workflow(
                [self._planner_intent(i) for i in intent_objects], workspace_id)
        self._log_plan_summary(envelope)
        await self._reply(msg, MessageType.PLAN_CREATED.value, envelope)

    async def _run_workflow(self, intents, workspace_id):
        workflow = PlanningWorkflowBehaviour(
            intents=intents,
            workspace_id=workspace_id,
            logger=self.logger,
        )
        self.agent.add_behaviour(workflow)
        await workflow.join()
        return workflow.reply_envelope or envelope_error(
            "plan_generation_failed",
            "Planning workflow returned no envelope.",
            intents,
        )

    @staticmethod
    def _select(items, selector):
        """(index, goal) for every structure the selector accepts."""
        return [(index, goal) for index, item in enumerate(items)
                if isinstance(item, GoalStructure)
                and (goal := selector(item)) is not None]

    async def _plan_mixed(self, items, explicit, incomplete, workspace_id,
                          ambiguous=(), dependency=()) -> dict:
        """Explicit goals (``ExplicitGoalPlanningBehaviour``), incomplete ones
        (``IncompleteGoalPlanningBehaviour``), ambiguous ones
        (``AmbiguousGoalPlanningBehaviour``) and dependency structures
        (``DependencyGoalPlanningBehaviour``) on their own paths, the workflow
        for the rest, merged into one multi-plan reply in the request's order.
        Entries outside dependency structures then get the power-on guard."""
        entries = {}

        # 1. Every routed goal or structure gets an entry here: a tree,
        #    impossible, a clarification request, or an error. One behaviour
        #    per route, in ROUTES order.
        selections = {"explicit": explicit, "incomplete": incomplete,
                      "ambiguous": ambiguous, "dependency": dependency}
        for name, _, _ in ROUTES:
            selected = selections[name]
            if not selected:
                continue
            cls = route_behaviour(name)
            behaviour = cls([goal for _, goal in selected], logger=self.logger)
            self.agent.add_behaviour(behaviour)
            await behaviour.join()
            for (index, _), result in zip(selected, behaviour.results):
                entries[index] = [{**result, "intent": items[index].to_wire_dict()}]

        # 2. Whatever no route took: the planning workflow, from its words.
        rest = [index for index in range(len(items)) if index not in entries]
        if rest:
            rest_intents = [self._planner_intent(items[i]) for i in rest]
            reply = await self._run_workflow(rest_intents, workspace_id)
            rest_entries = plan_entries_from_envelope(reply, rest_intents)
            if len(rest_entries) == len(rest):
                for index, entry in zip(rest, rest_entries):
                    # The planner saw only the words; keep the structure.
                    entries[index] = [{**entry, "intent": items[index].to_wire_dict()}]
            else:
                # One tree for several intents: it stands where the first was.
                entries[rest[0]] = rest_entries

        # 3. Every entry outside a dependency structure: switch the device on
        #    before setting it, where the guidelines say so (dependency
        #    structures, and the steps planned inside them, are left alone).
        dependent = {index for index, _ in selections["dependency"]}
        simple = [entry for index, group in entries.items() if index not in dependent
                  for entry in group]
        if simple:
            guard = PowerOnGuardBehaviour(simple, logger=self.logger)
            self.agent.add_behaviour(guard)
            await guard.join()

        # 4. One reply, entries in the order the user asked for them.
        ordered = [entry for index in sorted(entries) for entry in entries[index]]
        return envelope_plan_entries(ordered, workspace_id)

    @staticmethod
    def _planner_intent(item):
        """What the workflow plans from: a structure's words, until it reads
        structures; legacy intents as they are."""
        if isinstance(item, GoalStructure):
            return Intent(intent_text=item.to_query_string())
        return item

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_request(msg):
        """Extract intents and workspace from the GOAL_REQUEST body.

        A structured goal (`{"goals": ...}`) becomes a GoalStructure; anything
        else is kept as its words.
        """
        intent_objects: List[Union[GoalStructure, Intent]] = []
        workspace_id: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                ws = payload.get("workspace_id") or payload.get("workspace")
                if ws:
                    workspace_id = str(ws)
            raw_intents = payload.get("intents") if isinstance(payload, dict) else None
            if isinstance(raw_intents, list):
                for item in raw_intents:
                    if isinstance(item, dict):
                        if isinstance(item.get("goals"), dict):
                            # A structured goal from the UA's goal structuring.
                            intent_objects.append(GoalStructure.from_dict(item))
                        else:
                            # Fallback for other intent types
                            intent_objects.append(Intent(intent_text=item.get("text_intent", str(item))))
                    elif isinstance(item, str) and item.strip():
                        intent_objects.append(Intent(intent_text=item.strip()))
            else:
                goal_text = (
                    (payload.get("intent") or payload.get("goal"))
                    if isinstance(payload, dict)
                    else None
                )
                if goal_text:
                    intent_objects.append(Intent(intent_text=str(goal_text).strip()))
        except json.JSONDecodeError:
            if msg.body:
                intent_objects.append(Intent(intent_text=str(msg.body).strip()))
        return intent_objects, workspace_id

    async def _reply(self, src_msg, reply_type: str, envelope: dict) -> None:
        reply = src_msg.make_reply()
        reply.set_metadata("type", reply_type)
        reply.body = json.dumps(envelope, indent=2)
        corr = src_msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply.set_metadata(META_CORRELATION_ID, corr)
        if src_msg.thread:
            reply.thread = src_msg.thread
        await self.send(reply)

    def _log_plan_summary(self, envelope: dict) -> None:
        if not isinstance(envelope, dict):
            self.logger.info(demo("Plan created (non-dict body)."))
            return

        # Handle both single-tree format (top-level "tree") and multi-plan format (plans list)
        tree = envelope.get("tree")
        has_tree = tree is not None and isinstance(tree, dict) and bool(tree)

        # Check for multi-plan format
        plans = envelope.get("plans")
        if isinstance(plans, list) and not has_tree:
            # Multi-plan format - check if any plan has a tree
            has_tree = any(p.get("tree") for p in plans if isinstance(p, dict))

        self.logger.info(
            demo(f"Plan created: has_tree={has_tree} from_memory={sum(1 for p in (plans or []) if isinstance(p, dict) and p.get('source') == 'memory')} impossible={envelope.get('impossible', False)} error={envelope.get('error')}")
        )
