"""Behaviour: plan dependency structures -- goals with conditions, timing, maintenance.

Spawned by ``GoalRequestBehaviour`` for the structures ``dependency_structure``
accepts. Fetches the TD graph once, then per structure:

    1. achievement steps   each goal planned on its own by the achievement
                           planners (explicit / incomplete / ambiguous), as if
                           it were a simple goal: its achieve(G) subtree, and
                           the devices it acts on. A goal with a filter is
                           planned per candidate device. Any step impossible,
                           needing clarification or failing -> the whole
                           structure carries that outcome; nothing else runs.
    2. predicates          resolved to the properties they read
                           (`utils/dependency_context.py`); one that cannot be
                           read or has no value -> an error entry.
    3. profiling           the checklist (`utils/goal_profile.py`); when it has
                           questions, one small-model call with GOAL_PROFILE_PROMPT
                           and each id's devices; answers resolved, unanswered
                           ones on their fallbacks. An inconsistent structure ->
                           an impossible entry.
    3b. stated times       a goal whose start / end the user also stated as a
                           clock time: the two must name the same moment
                           (`utils/time_consistency.py`, the clock and each
                           anchor's remaining time read once); a disagreement
                           -> a clarification entry, no plan.
    3c. condition guards   each goal's steps on devices with a condition
                           (`config/planning_guidelines.yaml`): memory, then
                           one small-model call with the whole structure; each
                           guarded step wrapped in its subtree
                           (`utils/condition_check.py`, `condition_guard.py`).
    4. compile             the templates (`utils/dependency_compiler.py`) -> one
                           tree plus the runtime that keeps its keys current.
    4b. guards in order    the compiled tree walked in execution order, where
                           "before" is real across goals: first the condition
                           guards the plan already satisfies are removed
                           (`condition_guard.drop_satisfied`); then a mode,
                           level or operational-state command on a device the
                           `power_on_first` condition lists gets the power-on
                           guard, unless the plan switched that device on
                           before it (`utils/power_on_guard.py`).

The entry: {tree, runtime, explanation, source: "dependency", execution_mode:
"dependency", long_running, profile, profile_call?, guards?, condition_call?, llm?}.
    profile        per question id: question, target, the answer used, its source
    profile_call   the profiling call: model, questions sent, raw answer text,
                   finish reason -- to trace a fallback to what came back
    guards         per conditioned step, its decision (`condition_guard.guards_wire`)
    condition_call the condition-guard call, as profile_call
``timings`` holds, per structure, how long each step took (seconds), for
measurement; the entry does not carry it. The entry runs on the UserAssistant
as a dependency plan: no tick budget, the runtime refreshed before every tick,
until it concludes or is cancelled.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.models.goal_structure import GoalSpec, GoalStructure
from ....bt_planning.nodes.http_client import HTTPClient, HTTPClientConfig
from ....shared.utils.clock import make_clock, parse_moment, to_seconds
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.ambiguous_goal_plan import ambiguous_goal
from ..utils.dependency_compiler import (
    DEFAULT_POLL_SECONDS,
    DEFAULT_TOLERANCE,
    CompileError,
    compile_dependency,
)
from ..utils.dependency_context import (
    PredicateTarget,
    artifact_of_url,
    known_properties,
    profile_context,
    property_info,
    resolve_predicate,
)
from ..utils.environment_graph import fetch_environment_graph
from ..utils.explicit_goal_plan import deterministic_goal
from ..utils.goal_profile import InconsistentStructure, build_checklist
from ..utils.condition_check import guard_trees
from ..utils.condition_guard import drop_satisfied, guards_wire
from ..utils.planning_conditions import load_conditions
from ..utils.power_on_guard import guard_tree, guidelines_path, load_applies_to
from ..utils.profiling import ask_profile
from ..utils.incomplete_goal_plan import as_explicit, candidates, incomplete_goal
from ..utils.time_consistency import DEFAULT_TOLERANCE_SECONDS as DEFAULT_STATED_TIME_TOLERANCE
from ..utils.time_consistency import check as check_times
from .ambiguous_goal_planning import AmbiguousGoalPlanningBehaviour
from .explicit_goal_planning import ExplicitGoalPlanningBehaviour
from .incomplete_goal_planning import IncompleteGoalPlanningBehaviour

SOURCE = "dependency"
EXECUTION_MODE = "dependency"


class _Stop(Exception):
    """A structure that cannot be planned: carries its tree-less entry."""

    def __init__(self, entry: Dict[str, Any]) -> None:
        super().__init__(entry.get("explanation"))
        self.entry = entry


def _actions(tree: Any) -> List[str]:
    """Every action URL in a tree, in tree order -- used to find the devices a
    goal's planned subtree acts on (their TDs go into the profiling context)."""
    out: List[str] = []

    # Depth-first over the IR's nested "children".
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "action" and node.get("action_url"):
                out.append(node["action_url"])
            for child in node.get("children") or []:
                walk(child)

    walk(tree)
    return out


class DependencyGoalPlanningBehaviour(OneShotBehaviour):
    """Plan each structure; ``results`` holds one plan entry per structure."""

    def __init__(self, structures: List[GoalStructure], logger=None) -> None:
        """``structures`` are dependency structures, planned in order; one
        entry per structure ends up in ``results``, at the same index."""
        super().__init__()
        self.structures = structures
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.results: List[Dict[str, Any]] = []
        # Per structure: {"achievement", "predicates", "profiling", "guarding",
        # "compile"} in seconds, each present once its step has run. "guarding"
        # is 3c and 4b together; "compile" is 3b and 4 (without the guards).
        self.timings: List[Dict[str, float]] = []

    async def run(self) -> None:
        """Plan every structure. A structure that cannot be planned yields a
        tree-less entry (impossible / clarification / error), never an
        exception: the other structures are still planned."""
        # The TD graph once, shared by every structure and every step.
        graph = await fetch_environment_graph(self.agent, self.logger)
        for structure in self.structures:
            self.timings.append({})
            try:
                self.results.append(await self._plan(graph, structure))
            except _Stop as stop:
                self.logger.info(demo(f"Dependency plan for {structure.intent_text!r}: "
                                      f"{stop.entry.get('explanation')}"))
                self.results.append({**stop.entry, "source": SOURCE})

    # -- the four steps ----------------------------------------------------

    def _timed(self, step: str, started: float) -> float:
        """Record the seconds since ``started`` under ``step``; the new start."""
        now = time.perf_counter()
        self.timings[-1][step] = now - started
        return now

    async def _plan(self, graph: Graph, structure: GoalStructure) -> Dict[str, Any]:
        """One structure through the four steps (see the module docstring);
        returns its plan entry. Raises _Stop with a tree-less entry when a step
        decides the structure cannot be planned."""
        # 1. Each goal's action, planned as a simple achievement goal.
        started = time.perf_counter()
        subtrees, per_device, devices, llm = await self._steps(graph, structure)
        started = self._timed("achievement", started)

        # 2. What each predicate reads.
        targets: Dict[str, PredicateTarget] = {}
        for pid, pred in structure.predicates.items():
            target = resolve_predicate(graph, pred)
            if isinstance(target, str):
                raise _Stop({"tree": None, "error": "unresolvable_predicate",
                             "explanation": f"{pid}: {target}"})
            targets[pid] = target

        # 3. The profile: the checklist, the small model, the fallbacks.
        # 3.1 The questions (timing, maintenance, anchors); the checklist
        #     builder also catches a structure that can never run.
        try:
            checklist = build_checklist(structure)
        except InconsistentStructure as exc:
            raise _Stop(self._inconsistent(exc))
        started = self._timed("predicates", started)
        # 3.2 One call for every question; `trace` keeps what was sent and the
        #     raw reply for the entry.
        trace: Dict[str, Any] = {}
        profile, fallbacks, asked = await self._profile(graph, structure, checklist,
                                                        devices, targets, trace)
        llm = llm or asked
        started = self._timed("profiling", started)

        # 3b. Stated clock times against their relative timing: a disagreement
        # is a question for the user, not a plan.
        planning = (self.agent.config.get("planning", {}) or {}).get("dependency", {}) or {}
        checked = await self._check_stated_times(graph, structure, profile, planning)

        # 3c. The condition guards of each goal's steps, decided for the
        #     whole structure at once and wrapped in the goals' subtrees.
        condition_trace: Dict[str, Any] = {}
        guarding_started = time.perf_counter()
        decisions, condition_notes = await self._condition_guards(
            graph, structure, subtrees, per_device, condition_trace)
        guarding = time.perf_counter() - guarding_started

        # 4. The tree.
        try:
            compiled = compile_dependency(
                graph, structure, subtrees, targets, profile, per_device=per_device,
                poll_seconds=float(planning.get("poll_seconds", DEFAULT_POLL_SECONDS)),
                tolerance=float(planning.get("default_tolerance", DEFAULT_TOLERANCE)))
        except CompileError as exc:
            raise _Stop({"tree": None, "error": "dependency_compile_failed",
                         "explanation": str(exc)})
        compiled_at = time.perf_counter()
        # 4b. On the compiled tree, so "before" is the plan's real order: drop
        #     the condition guards the plan already satisfies, then switch
        #     devices on before setting them, where the conditions say so.
        compiled.tree, dropped = drop_satisfied(graph, compiled.tree) if decisions \
            else (compiled.tree, [])
        guard_notes: List[str] = []
        applies_to = load_applies_to(guidelines_path(self.agent.config))
        if applies_to is not None:
            compiled.tree, guard_notes = guard_tree(graph, compiled.tree, applies_to, goal=None)
        # The two guard steps are timed apart from the compile they surround.
        self.timings[-1]["guarding"] = guarding + (time.perf_counter() - compiled_at)
        self.timings[-1]["compile"] = compiled_at - started - guarding

        # 5. The entry: the tree, its runtime, and everything needed to explain
        #    it -- the profile, the guards, the calls behind them, the notes.
        notes = fallbacks + checked + condition_notes + compiled.notes + dropped + guard_notes
        self.logger.info(demo(
            f"Dependency plan for {structure.intent_text!r}: {len(structure.goals)} goal(s), "
            f"{len(structure.predicates)} predicate(s), {len(checklist)} question(s), "
            f"{len(compiled.runtime['reads'])} reading(s)"
            + (f"; {len(notes)} note(s)" if notes else "")))
        return {"tree": compiled.tree, "runtime": compiled.runtime, "source": SOURCE,
                "execution_mode": EXECUTION_MODE, "long_running": compiled.long_running,
                "profile": {qid: {k: item[k] for k in ("question", "target", "answer", "source")}
                            for qid, item in profile.items()},
                **({"profile_call": trace} if trace else {}),
                **({"guards": guards_wire(decisions, compiled.tree)} if decisions else {}),
                **({"condition_call": condition_trace} if condition_trace else {}),
                "explanation": "; ".join(notes) or "Planned from the goals' dependencies.",
                **({"llm": True} if llm else {})}

    async def _condition_guards(self, graph: Graph, structure: GoalStructure,
                                subtrees: Dict[str, Dict[str, Any]],
                                per_device: Dict[str, List[Tuple[str, Dict[str, Any]]]],
                                trace: Dict[str, Any]):
        """(decisions, notes): every goal's steps checked against the device
        conditions in one call, the guarded steps wrapped in place --
        ``subtrees`` and ``per_device`` are updated. No conditions (or no
        file): nothing to check. A failure leaves the steps unguarded: the
        check never stops a plan."""
        # 1. The conditions; none -> no check.
        conditions = load_conditions(guidelines_path(self.agent.config))
        if not conditions:
            return {}, []
        # 2. Every tree the compiler will use, with its goal: one per goal,
        #    and one per candidate device of a filtered goal.
        trees = [(gid, tree) for gid, tree in subtrees.items()]
        trees += [(gid, tree) for gid, pairs in per_device.items() for _, tree in pairs]
        try:
            guarded, decisions, notes = await guard_trees(
                self.agent, self.logger, graph, trees, structure.to_wire_dict(),
                conditions, trace)
        except Exception as exc:  # noqa: BLE001 - the plan stands unguarded
            self.logger.warning(f"Condition check skipped for {structure.intent_text!r}: {exc}")
            return {}, []
        # 3. The wrapped trees back where they came from, in the same order.
        out = iter(guarded)
        for gid in subtrees:
            subtrees[gid] = next(out)
        for gid, pairs in per_device.items():
            per_device[gid] = [(title, next(out)) for title, _ in pairs]
        return decisions, notes

    async def _check_stated_times(self, graph: Graph, structure: GoalStructure,
                                  profile: Dict[str, Any], planning: Dict[str, Any]) -> List[str]:
        """Check each stated start / end clock time against the goal's relative
        timing (`utils/time_consistency.py`). Raises _Stop with a clarification
        on a disagreement; returns the notes otherwise. Reads the clock, and
        each anchor's remaining time, once -- only for structures that state a
        clock time beside a relative one."""
        if not any(g.timing and (g.timing.stated_start_time_text or g.timing.stated_end_time_text)
                   for g in structure.goals.values()):
            return []
        # 1. The current time, from the configured clock (the simulator's in
        #    SimuHome). Without one the times are not checked, only noted.
        try:
            now = make_clock((self.agent.config or {}).get("environment") or {}).now()
        except Exception as exc:  # noqa: BLE001 - unverifiable, not wrong
            return [f"could not verify the stated times: no clock ({exc})"]
        # 2. How far away each predicate's event is, where a property tells.
        remaining = {pid: await self._remaining(graph, profile, pid, now)
                     for pid in structure.predicates}
        # 3. Compare; any disagreement stops the plan with a question.
        result = check_times(structure, now, remaining, float(planning.get(
            "stated_time_tolerance_seconds", DEFAULT_STATED_TIME_TOLERANCE)))
        if result.conflicts:
            raise _Stop({"tree": None, "requires_clarification": True,
                         "error": "stated_time_conflict",
                         "explanation": " ".join(result.conflicts)})
        return result.notes

    async def _remaining(self, graph: Graph, profile: Dict[str, Any], pid: str,
                         now) -> Optional[float]:
        """Seconds until predicate ``pid``'s event, read now from the property
        the profile names for it (`time_to_event`); None when nothing does,
        or when the read fails -- the time is then unverifiable, not wrong."""
        # 1. The property the profiling model named for this predicate.
        answer = next((item["answer"] for item in profile.values()
                       if item.get("question") == "time_to_event" and item.get("target") == pid),
                      None)
        source = (answer or {}).get("time_to_event") if isinstance(answer, dict) else None
        if not isinstance(source, dict) or not isinstance(source.get("ref"), dict):
            return None
        ref = source["ref"]
        info = property_info(graph, ref.get("artifact_name"), ref.get("property_name"))
        if info is None:
            return None
        # 2. Read it once, now (blocking HTTP, so off the event loop).
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None, HTTPClient(config=HTTPClientConfig()).get, info.url)
        except Exception:  # noqa: BLE001 - unverifiable, not wrong
            return None
        value = response.body if getattr(response, "is_success", False) else None
        field_name = ref.get("property_field")
        if isinstance(value, dict) and field_name and field_name != "NA":
            value = value.get(field_name)
        # 3. To seconds from now: an end time minus now, or a countdown in the
        #    property's own unit.
        if source.get("reads_as") == "timestamp":
            moment = parse_moment(value)
            return None if moment is None else moment - to_seconds(now)
        try:
            return float(value) * info.seconds_per_unit
        except (TypeError, ValueError):
            return None

    async def _steps(self, graph: Graph, structure: GoalStructure):
        """Plan each goal's own action; returns (subtrees, per_device, devices,
        llm).

        subtrees    goal id -> its achieve(G) tree
        per_device  goal id -> [(device title, tree)] for a filtered incomplete
                    goal ("every light that is on"), one tree per candidate
        devices     goal id -> the artifacts its trees act on
        llm         whether any of these plans was made by an LLM

        Each goal is planned with its timing removed and as an achievement:
        the planners only build the action; when and how long is the
        compiler's job."""
        subtrees: Dict[str, Dict[str, Any]] = {}
        per_device: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        devices: Dict[str, List[Any]] = {}
        llm = False
        filtered = {gid for p in structure.predicates.values() if p.role == "filter"
                    for gid in p.applies_to}

        for gid, goal in structure.goals.items():
            step = replace(goal, goal_kind="achievement", timing=None)
            # A filtered incomplete goal: one tree per candidate device; the
            # filter picks among them when the plan runs.
            if gid in filtered and step.goal_specificity == "incomplete":
                entries = await self._per_device(graph, gid, step)
                per_device[gid] = [(title, entry["tree"]) for title, entry in entries]
                trees = [entry["tree"] for _, entry in entries]
                llm = llm or any(entry.get("llm") for _, entry in entries)
            # Any other goal: one tree, from the planner its specificity needs.
            else:
                entry = await self._step(graph, gid, step)
                subtrees[gid] = entry["tree"]
                trees = [entry["tree"]]
                llm = llm or bool(entry.get("llm"))
            # The devices acted on: their TDs are the profiling context.
            devices[gid] = [artifact_of_url(graph, url) for tree in trees
                            for url in _actions(tree)]
        return subtrees, per_device, devices, llm

    async def _step(self, graph: Graph, gid: str, goal: GoalSpec) -> Dict[str, Any]:
        """One goal through the achievement planner its specificity calls for;
        returns that planner's entry. Raises _Stop when the goal names no
        action, or its planner did not produce a tree."""
        # 1. Pick the planner, by what the goal names (a single goal is
        #    presented as a simple structure, as the planners expect).
        simple = GoalStructure(intent_text=goal.intent_text or gid, structure="simple",
                               goals={"G1": goal})
        if deterministic_goal(simple) is not None:
            behaviour = ExplicitGoalPlanningBehaviour([goal], logger=self.logger, graph=graph)
        elif incomplete_goal(simple) is not None:
            behaviour = IncompleteGoalPlanningBehaviour([goal], logger=self.logger, graph=graph)
        elif ambiguous_goal(simple) is not None:
            behaviour = AmbiguousGoalPlanningBehaviour([goal], logger=self.logger, graph=graph)
        else:
            raise _Stop({"tree": None, "error": "unplannable_goal",
                         "explanation": f"{gid} ({goal.intent_text!r}) names no action to plan"})
        # 2. Run it as its own behaviour and wait for its single entry.
        self.agent.add_behaviour(behaviour)
        await behaviour.join()
        (entry,) = behaviour.results
        self._stop_unless_planned(gid, entry)
        return entry

    async def _per_device(self, graph: Graph, gid: str, goal: GoalSpec):
        """A filtered goal on every candidate device, each planned on its own:
        the filter picks among them when the plan runs. Returns [(device
        title, entry)]; raises _Stop when no device fits or one is not
        planned."""
        # 1. Every device that fits the goal's description.
        found = candidates(graph, goal)
        if not found:
            raise _Stop({"tree": None, "impossible": True,
                         "explanation": f"{gid}: no device fits {goal.intent_text!r}"})
        # 2. The goal made explicit for each device, all planned in one
        #    behaviour run.
        goals = [as_explicit(goal, c) for c in found]
        behaviour = ExplicitGoalPlanningBehaviour(goals, logger=self.logger, graph=graph)
        self.agent.add_behaviour(behaviour)
        await behaviour.join()
        for entry in behaviour.results:
            self._stop_unless_planned(gid, entry)
        return [(c.title, entry) for c, entry in zip(found, behaviour.results)]

    @staticmethod
    def _stop_unless_planned(gid: str, entry: Dict[str, Any]) -> None:
        """Raise _Stop when a goal's step was not planned: the whole structure
        then carries that step's outcome (impossible, clarification, error),
        with the goal id in the explanation."""
        if entry.get("tree"):
            return
        stopped = {k: v for k, v in entry.items()
                   if k in ("impossible", "error", "requires_clarification", "candidates")}
        raise _Stop({"tree": None, **stopped,
                     "explanation": f"{gid}: {entry.get('explanation') or 'not planned'}"})

    @staticmethod
    def _inconsistent(exc: Exception) -> Dict[str, Any]:
        """The entry for a structure that can never be carried out (reported
        by the checklist builder or the profiling model)."""
        return {"tree": None, "impossible": True, "error": "inconsistent_structure",
                "explanation": f"This cannot be carried out as asked: {exc}"}

    async def _profile(self, graph: Graph, structure: GoalStructure,
                       checklist: List[Dict[str, Any]], devices: Dict[str, List[Any]],
                       targets: Dict[str, PredicateTarget], trace: Dict[str, Any]):
        """The resolved profile, its fallback log, and whether a model was asked.
        `trace` receives the questions sent and the model's raw answer. Raises
        _Stop when the model reports the structure inconsistent."""
        # No questions: no call.
        if not checklist:
            return {}, [], False
        # The context: the TDs of every device a goal acts on or a predicate
        # reads, under that goal's / predicate's id.
        context = profile_context(graph, {**devices, **{pid: t.artifacts
                                                        for pid, t in targets.items()}})
        try:
            profile, log = await ask_profile(self.agent, self.logger, structure,
                                             checklist, context, trace)
        except InconsistentStructure as exc:
            raise _Stop(self._inconsistent(exc))
        return profile, log, True
