"""One episode (SimuHome's, or HomeBench's) through our agents' reasoning steps, in-process.

    0. load        reset the simulator to the episode's home; discovery builds the
                   TDs (offline in a deployment: logged, never in the statistics)
    1. segmentation    `pipeline.segment_into_atomic_intents`
    2. structuring     per intent, concurrently as the UA does:
                         GOAL_REQUEST            GoalStructuringBehaviour
                         ENV_STATE_REQUEST       EnvStateStructuringBehaviour
                         ENV_CAPABILITIES_REQUEST EnvCapabilityStructuringBehaviour
                       then the UA's clarification check: a goal only the user can
                       resolve stops here
       state_read      a state / capability question answered by the EnvExplorer
    3. planning        the ISA's routing (`goal_request.route`), one behaviour per
                       route; dependency structures timed as achievement steps and
                       dependency planning (the planner's own `timings`)
    4. execution       the simulator is reset to the same home first (planning
                       never actuates, so the plans start at tick 0, SimuHome's
                       "now"); lockstep for dependency plans and every timed
                       (qt4) episode, plans or not; otherwise the UA's loop
                       (`execution.py`)
    5. response        the text the UA would send: clarification questions, state
                       / capability answers, the plan summary
    6. snapshot        the final home state, 0.5 s after execution (as SimuHome's
                       runner)

The suite (`suites/`) decides what differs: the TD builder discovery uses, and
whether an episode is timed (SimuHome's qt4; none of HomeBench's, whose plans are
ticked or run immediately). The record names it (`record["suite"]`) so scoring
reads the episode the same way.

Each step is timed and its LLM calls booked to it (`meter.py`). The record holds
what every step produced, what the EnvExplorer was asked (as SimuHome tool
calls), the actions, the state snapshots, and the steps in SimuHome's trace
format for its judge.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ami_agents.agents.interaction_solver.behaviours.goal_request import (
    route,
    route_behaviour,
)
from ami_agents.agents.interaction_solver.behaviours.condition_guard import ConditionGuardBehaviour
from ami_agents.agents.interaction_solver.behaviours.power_on_guard import PowerOnGuardBehaviour
from ami_agents.agents.interaction_solver.utils.environment_graph import fetch_environment_graph
from ami_agents.agents.user_assistant.guards import guards_to_record
from ami_agents.agents.interaction_solver.utils.plan_envelope import envelope_plan_entries
from ami_agents.agents.user_assistant import pipeline as ua_pipeline
from ami_agents.agents.user_assistant import queries
from ami_agents.agents.user_assistant.behaviours.env_capability_structuring import (
    EnvCapabilityStructuringBehaviour,
)
from ami_agents.agents.user_assistant.behaviours.env_state_structuring import (
    EnvStateStructuringBehaviour,
)
from ami_agents.agents.user_assistant.behaviours.goal_structuring import (
    GoalStructuringBehaviour,
)
from ami_agents.agents.user_assistant.behaviours.user_request import (
    split_clarifications,
    split_unplanned,
)
from ami_agents.agents.user_assistant.signifiers import (
    build_signifier,
    capture_execution_context,
)
from ami_agents.agents.user_assistant.utils.goal_resolution import structure_clarification
from ami_agents.shared.models.goal_structure import GoalStructure
from ami_agents.agents.user_assistant.utils.plan import (
    canonicalize_plan_for_hash,
    coerce_plan_dict,
)

from . import execution as bench_execution
from .meter import Meter
from .models import Variant, close_clients
from .stubs import (
    Explorer,
    Home,
    isa_agent,
    patched_explorer_rpcs,
    patched_ua_clients,
    response_agent,
    ua_agent,
)
from .suites import DEFAULT_SUITE, Suite, get_suite
from .toollog import ToolLog, parse_url

STOP_AFTER = ("segmentation", "structuring", "planning")


@dataclass
class Options:
    context: str = "none"                 # none | prepend
    step_seconds: float = 10.0            # lockstep step (virtual seconds)
    exec_timeout: float = 60.0            # immediate plans (real seconds)
    stop_after: Optional[str] = None      # segmentation | structuring | planning
    memory: Any = None                    # memory.Memory, or None
    cache: Any = None                     # StepCache, or None (sweeps)
    debug: bool = False
    execution: str = "ticks"              # ticks (immediate / lockstep) | scheduled
    dry_step_seconds: float = 1.0         # scheduled: the dry run's step (virtual seconds)
    real_read_seconds: float = 10.0       # scheduled: how long a simulator reading is reused
    environment: Optional[str] = None     # on | off (HomeBench's TDs); None: not an option


@dataclass
class Target:
    """Where the episode runs: the simulator client and SHTD's address."""

    sim: Any                              # SimuHomeClient
    shtd_base: str
    home_id: str


def _say(options: Options, title: str, body: Any = None) -> None:
    """Debug output: one step, what it produced."""
    if not options.debug:
        return
    print(f"\n== {title}", flush=True)
    if body is not None:
        text = body if isinstance(body, str) else json.dumps(body, indent=1, default=str)
        print(text, flush=True)


def _input_text(episode: Dict[str, Any], options: Options) -> str:
    query = episode["query"]
    if options.context != "prepend":
        return query
    home = episode.get("initial_home_config") or {}
    parts = []
    if home.get("base_time"):
        parts.append(f"It is {home['base_time']}.")
    if episode.get("user_location"):
        parts.append(f"I am in the {str(episode['user_location']).replace('_', ' ')}.")
    return f"({' '.join(parts)}) {query}" if parts else query


def _devices(tree: Any) -> List[str]:
    """The devices a tree acts on, from its action URLs."""
    out: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "action" and node.get("action_url"):
                parts = parse_url(node["action_url"])
                if parts and parts.get("device") and parts["device"] not in out:
                    out.append(parts["device"])
            for child in node.get("children") or []:
                walk(child)

    walk(tree)
    return out


def _entry_outcome(entry: Dict[str, Any]) -> str:
    if entry.get("requires_clarification"):
        return "clarify"
    if entry.get("impossible"):
        return "impossible"
    if entry.get("error"):
        return "error"
    return "plan" if entry.get("tree") else "no_plan"


# --------------------------------------------------------------------------
# The episode
# --------------------------------------------------------------------------

async def run_episode(name: str, episode: Dict[str, Any], variant: Variant,
                      config: Dict[str, Any], target: Target, options: Options,
                      logger, suite: Optional[Suite] = None) -> Dict[str, Any]:
    """Run one episode of ``suite`` (SimuHome when None), already loaded by the
    suite's `load_episode`; the record of everything it did. The LLM clients
    the episode made are closed before it returns, in its own event loop."""
    # 1. The record, filled as the steps run.
    suite = suite or get_suite(DEFAULT_SUITE)
    record: Dict[str, Any] = {
        "suite": suite.name,
        "episode": name, "query_type": episode["meta"]["query_type"],
        "case": episode["meta"]["case"], "meta": episode["meta"],
        "query": episode["query"], "variant": variant.as_dict(),
        "options": {"context": options.context, "step_seconds": options.step_seconds,
                    "stop_after": options.stop_after, "memory": options.memory is not None},
        "intents": [], "structures": [], "queries": [], "plans": [], "goals": [],
        "outcome": None, "error": None, "error_step": None}
    meter = Meter()
    tools = ToolLog()
    current = {"step": "load"}

    # 2. The steps; an error is recorded with the step it happened in.
    try:
        with meter.installed():
            await _run(name, episode, variant, config, target, options, logger,
                       record, meter, tools, current, suite)
    except Exception as exc:  # noqa: BLE001 - recorded, scored as an error
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_step"] = current["step"]
        record["traceback"] = traceback.format_exc()
        record["outcome"] = "error"
        _say(options, f"ERROR in {current['step']}", record["traceback"])
    finally:
        # The episode's LLM clients, closed while their event loop still runs.
        await close_clients()
    # 3. What was measured along the way.
    record["tools_invoked"] = tools.tools_invoked()
    record["tool_log"] = tools.entries
    record["meter"] = meter.as_dict()
    return record


async def _run(name, episode, variant, config, target, options, logger, record, meter,
               tools, current, suite) -> None:
    """The steps of one episode (module docstring), filling ``record`` as it goes.

    ``suite`` supplies the TD builder discovery uses and whether the episode is timed;
    ``current`` names the step running, so `run_episode` can say where an error was.
    """
    sim = target.sim
    home_config = episode["initial_home_config"]

    # 0. Load and discover (not timed as a step: offline in a deployment).
    current["step"] = "load"
    started = time.perf_counter()
    sim.reset(home_config)
    # The suite's TD builder, with its environment mode when the suite has one.
    builder_kwargs = ({"environment": options.environment == "on"}
                      if options.environment else None)
    home = Home(target.shtd_base, target.home_id, sim.home_state(), config, logger,
                builder_cls=suite.td_cls, builder_kwargs=builder_kwargs)
    record["discovery_seconds"] = time.perf_counter() - started
    record["ambient_sensors"] = home.sensors
    tools.sensors = home.sensors
    explorer = Explorer(home, tools, memory=options.memory, logger=logger)
    ua = ua_agent(config, variant, logger)
    isa = isa_agent(config, variant, explorer, logger)
    # The ISA reads the clock to check stated times: the simulator's, through
    # SHTD (production takes it from environment.yaml).
    isa.config["environment"] = {**(isa.config.get("environment") or {}),
                                 "clock": {"source": "http", "http": {
                                     "url": f"{target.shtd_base}/_shtd/time"}}}
    text = _input_text(episode, options)
    record["input"] = text
    _say(options, f"{name} [{record['query_type']} {record['case']}]", text)

    with patched_ua_clients(variant), patched_explorer_rpcs(explorer):
        # 1-2. Segmentation and structuring -- or their cached results, when a
        # sweep variant shares these steps' models with one already run.
        cache_key = _cache_key(name, text, variant)
        cached = options.cache.load(cache_key) if options.cache is not None else None
        if cached is not None:
            front = _restore(cached, meter, tools, explorer)
            _say(options, "segmentation + structuring (cached)", front)
        else:
            marks = (len(meter.steps), len(meter.calls), len(tools.entries),
                     len(explorer.observations))
            front = await _understand(text, ua, logger, meter, options, current)
            if options.cache is not None and options.stop_after != "segmentation":
                options.cache.save(cache_key, {
                    **front, "meter_steps": meter.steps[marks[0]:],
                    "meter_calls": meter.calls[marks[1]:],
                    "tools": tools.entries[marks[2]:],
                    "observations": explorer.observations[marks[3]:]})
        record["intents"] = front["intents"]
        if not front["intents"]:
            record["outcome"] = "nothing"
            record["response"] = "I couldn't understand your request. Could you rephrase?"
            record["steps"] = _trace(record, explorer)
            return
        if options.stop_after == "segmentation":
            record["outcome"] = "stopped"
            return

        structures, ua_questions = [], []
        for item in front["goals"]:
            if item.get("structure"):
                structure = GoalStructure.from_dict(item["structure"])
                # The question asked of the UA's own structure; a step cache
                # written before it was kept falls back to the rebuilt one.
                question = (item["question"] if "question" in item
                            else structure_clarification(structure))
                if question:
                    item["clarification"] = question
                    ua_questions.append(f"About \"{item['text']}\": {question}")
                else:
                    structures.append(structure)
            record["structures"].append(item)
        _say(options, "structuring", record["structures"])

        # State and capability questions: the EnvExplorer reads / looks up.
        answers: List[str] = []
        responder = SimpleNamespace(agent=response_agent(config, variant, logger),
                                    logger=logger)
        pending_answers = []
        current["step"] = "state_read"
        for item in front["queries"]:
            extraction = dict(item.get("result") or {}, text_intent=item["text"])
            async with meter.step("state_read", kind=item["kind"]):
                if item["kind"] == "state":
                    # As the UA asks: a named device the home lacks is listed,
                    # not read (`device_names.state_lookup`).
                    response = await explorer.state(queries.state_lookup(extraction))
                else:
                    response = await explorer.resolve_candidates(None, logger, extraction)
            record["queries"].append({"kind": item["kind"], "text": item["text"],
                                      "extraction": item.get("result"),
                                      "error": item.get("error"), "response": response})
            pending_answers.append((item["kind"], response, extraction, item["text"]))
        if record["queries"]:
            _say(options, "state / capability queries", record["queries"])
        if options.stop_after == "structuring":
            record["outcome"] = "stopped"
            return

        # 3. Planning: the ISA's routes, one behaviour per route.
        current["step"] = "planning"
        entries: Dict[int, Dict[str, Any]] = {}
        routes = route(structures)
        for route_name, selected in routes.items():
            if not selected:
                continue
            behaviour = route_behaviour(route_name)([goal for _, goal in selected],
                                                    logger=logger)
            behaviour.agent = isa
            async with meter.step("planning", route=route_name, goals=len(selected)):
                if route_name == "dependency":
                    await _run_dependency(behaviour, meter)
                else:
                    await behaviour.run()
            for (index, _), result in zip(selected, behaviour.results):
                entries[index] = {**result, "route": route_name,
                                  "intent": structures[index].to_wire_dict()}
        # The condition-guard check, then the power-on guard, on every entry
        # outside a dependency structure, as the ISA's GoalRequestBehaviour
        # does after its routes (one TD graph for both).
        simple = [e for e in entries.values() if e["route"] != "dependency"]
        if simple:
            graph = await fetch_environment_graph(isa, logger)
            for name, cls in (("condition_guard", ConditionGuardBehaviour),
                              ("power_on_guard", PowerOnGuardBehaviour)):
                guard = cls(simple, logger=logger, graph=graph)
                guard.agent = isa
                async with meter.step("guarding", kind=name, goals=len(simple)):
                    await guard.run()
        for index, structure in enumerate(structures):
            if index not in entries:
                # The agents would send it to the planning workflow, from its words.
                entries[index] = {"tree": None, "error": "unrouted", "route": "workflow",
                                  "explanation": "No structured planning path accepts this "
                                                 "structure (the workflow is not run here).",
                                  "intent": structure.to_wire_dict()}
        ordered = [entries[i] for i in sorted(entries)]
        record["plans"] = [{**{k: v for k, v in e.items() if k != "goal_context"},
                            "outcome": _entry_outcome(e), "devices": _devices(e.get("tree"))}
                           for e in ordered]
        # The properties the planner read to check a request's timing (an
        # anchor's countdown, say) are reads of the home like any other.
        for entry in ordered:
            for read in entry.get("reads") or []:
                tools.read(read.get("url"), None, "planning")
        _say(options, "planning", record["plans"])
        if options.stop_after == "planning":
            record["outcome"] = "stopped"
            return

        # 4. Execution, from a fresh home at tick 0.
        current["step"] = "execution"
        runnable = [e for e in ordered if isinstance(e.get("tree"), dict) and e["tree"]]
        # Timed episodes are always run in lockstep, plans or not: their goals
        # are checked at set times, so the home is followed through them.
        timed = suite.timed(record["query_type"])
        contexts = []
        if runnable or timed:
            sim.reset(home_config)
            if options.memory is not None:
                for index, entry in enumerate(runnable):
                    contexts.append(await capture_execution_context(
                        None, f"{name}-{index}", entry["tree"], entry=entry,
                        intent=entry.get("intent"), logger=logger))
            async with meter.step("execution") as step:
                result = await asyncio.to_thread(
                    _execute, runnable, tools, sim, config, options,
                    home_config, episode, timed, home.turtle)
                step.update(mode=result.mode, virtual_seconds=result.virtual_seconds,
                            lockstep_steps=result.steps)
            record["execution"] = result.as_dict()
            record["snapshots"] = result.snapshots
            record["timeline"] = result.timeline
            _say(options, f"execution ({result.mode})",
                 {"actions": result.actions, "plans": result.plans,
                  "virtual_seconds": result.virtual_seconds,
                  "real_seconds": round(result.real_seconds, 2)})
            if options.memory is not None:
                record["memory_recorded"] = _remember(options.memory, contexts,
                                                      result.plans)
        else:
            record["execution"] = {"mode": "none", "actions": [], "plans": []}

        # 5. Response: what the UA would say.
        current["step"] = "response"
        async with meter.step("response"):
            # The UA's own replies, a named device the home lacks included.
            for kind, response, extraction, intent_text in pending_answers:
                if kind == "state":
                    answers.append(await queries.state_reply(
                        responder, response, extraction, intent_text))
                else:
                    answers.append(await queries.capabilities_reply(
                        responder, response, extraction, intent_text))
            summary = None
            if ordered:
                envelope = envelope_plan_entries(
                    [{k: v for k, v in e.items() if k != "route"} for e in ordered], None)
                plan_obj = coerce_plan_dict(json.dumps(envelope)) or {}
                plan_obj, questions = split_clarifications(plan_obj)
                ua_questions += questions
                # Entries that are not plans are told as such, not summarised.
                plan_obj, unplanned = split_unplanned(plan_obj)
                answers += unplanned
                if plan_obj.get("plans") or plan_obj.get("tree"):
                    canonical, _ = canonicalize_plan_for_hash(plan_obj)
                    summary = await ua_pipeline.summarize_plan(
                        responder.agent, logger,
                        canonical if isinstance(canonical, str) else json.dumps(plan_obj))
        record["response"] = "\n".join([*answers, *ua_questions, *([summary] if summary
                                                                     else [])])
        _say(options, "response", record["response"])

    # 6. The home as the episode left it.
    current["step"] = "snapshot"
    await asyncio.sleep(0.5)
    record["final_home_state"] = sim.home_state()
    record["outcome"] = _outcome(record, ua_questions)
    record["steps"] = _trace(record, explorer)


async def _understand(text, ua, logger, meter, options, current) -> Dict[str, Any]:
    """Segmentation, then every intent structured concurrently (as the UA
    spawns them). Plain data, so it can be cached."""
    current["step"] = "segmentation"
    async with meter.step("segmentation"):
        intents = await ua_pipeline.segment_into_atomic_intents(ua, logger, text)
    front: Dict[str, Any] = {
        "intents": [{"text": i.text, "type": i.type, "qualifiers": i.qualifiers,
                     "reason": i.reason} for i in intents],
        "goals": [], "queries": []}
    _say(options, "segmentation", front["intents"])
    if not intents or options.stop_after == "segmentation":
        return front

    current["step"] = "structuring"
    goals = [i for i in intents if i.type == "GOAL_REQUEST"]
    states = [i for i in intents if i.type == "ENV_STATE_REQUEST"]
    capabilities = [i for i in intents if i.type == "ENV_CAPABILITIES_REQUEST"]

    async def structure(behaviour, label):
        behaviour.agent = ua
        async with meter.step("structuring.intent", kind=label, text=behaviour.intent_text):
            await behaviour.run()
        return behaviour

    async with meter.step("structuring"):
        done = await asyncio.gather(
            *[structure(GoalStructuringBehaviour(i.text, i.qualifiers, logger=logger), "goal")
              for i in goals],
            *[structure(EnvStateStructuringBehaviour(i.text, logger=logger), "state")
              for i in states],
            *[structure(EnvCapabilityStructuringBehaviour(i.text, logger=logger),
                        "capabilities") for i in capabilities])
    # The UA's question is asked of the structure it built, as UserRequestBehaviour
    # does: the wire dict leaves out what stays in the UA (a question set by
    # resolution, how the request picks out its device), so a structure rebuilt
    # from it would never be held back for them.
    for behaviour in done[:len(goals)]:
        result = behaviour.result
        front["goals"].append({
            "text": behaviour.intent_text, "error": behaviour.error,
            "context_unavailable": behaviour.context_unavailable,
            "structure": result.to_wire_dict() if result else None,
            "question": structure_clarification(result) if result else None,
            "device_references": ({gid: g.device_reference for gid, g in result.goals.items()}
                                  if result else None)})
    for kind, behaviour in [("state", b) for b in done[len(goals):len(goals) + len(states)]] + \
            [("capabilities", b) for b in done[len(goals) + len(states):]]:
        front["queries"].append({"kind": kind, "text": behaviour.intent_text,
                                 "result": behaviour.result, "error": behaviour.error})
    return front


def _cache_key(name: str, text: str, variant: Variant) -> str:
    """What the first two steps' results depend on: the episode, the input,
    and the two steps' models."""
    import hashlib
    return hashlib.sha1(f"{name}|{text}|{variant.key('segmentation', 'structuring')}"
                        .encode()).hexdigest()[:16]


def _restore(cached: Dict[str, Any], meter: Meter, tools: ToolLog,
             explorer: Explorer) -> Dict[str, Any]:
    """Put back what the cached steps measured, asked and observed."""
    meter.steps += [{**step, "cached": True} for step in cached.get("meter_steps", [])]
    meter.calls += [{**call, "cached": True} for call in cached.get("meter_calls", [])]
    for entry in cached.get("tools", []):
        tools.add(entry["tool"], entry["params"], entry.get("source", "cached"),
                  entry.get("outcome", {}).get("ok", True))
    explorer.observations += cached.get("observations", [])
    return {k: cached[k] for k in ("intents", "goals", "queries")}


class StepCache:
    """Segmentation + structuring results on disk, one file per key."""

    def __init__(self, directory) -> None:
        from pathlib import Path
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.directory / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(self, key: str, data: Dict[str, Any]) -> None:
        if data.get("intents") is None:
            return
        (self.directory / f"{key}.json").write_text(json.dumps(data, default=str),
                                                     encoding="utf-8")


async def _run_dependency(behaviour, meter: Meter) -> None:
    """A dependency behaviour, with its LLM calls booked to the achievement
    steps, to profiling, to guarding or to the rest of the dependency
    planning; its own timings become the steps. Profiling and guarding are
    added as stages of their own, timed inside the "planning" step that runs
    this behaviour (`within="planning"`: the stage summary takes them out of
    planning)."""
    steps, profile, guards = behaviour._steps, behaviour._profile, behaviour._condition_guards

    # 1. Each part's LLM calls booked to its stage.
    async def booked_steps(*args, **kwargs):
        with meter.tag("planning.achievement"):
            return await steps(*args, **kwargs)

    async def booked_profile(*args, **kwargs):
        with meter.tag("profiling"):
            return await profile(*args, **kwargs)

    async def booked_guards(*args, **kwargs):
        with meter.tag("guarding"):
            return await guards(*args, **kwargs)

    behaviour._steps, behaviour._profile = booked_steps, booked_profile
    behaviour._condition_guards = booked_guards
    await behaviour.run()

    # 2. The behaviour's own timings, per structure, as steps.
    for timing in behaviour.timings:
        if "achievement" in timing:
            meter.add("planning.achievement", timing["achievement"])
        for stage in ("profiling", "guarding"):
            if stage in timing:
                meter.add(stage, timing[stage], within="planning", route="dependency")
        dependency = {k: timing[k] for k in ("predicates", "compile") if k in timing}
        if dependency:
            meter.add("planning.dependency", sum(dependency.values()), parts=dependency)


def _execute(runnable, tools, sim, config, options, home_config, episode, timed,
             turtle: str = ""):
    """Execution in a thread (it blocks): scheduled workflows from a dry run
    (`--execution scheduled`), else lockstep or the UA's loop."""
    bt = config.get("bt_execution", {}) or {}
    # 1. Scheduled: every episode, timed or not, through the dry run. Imported
    #    here: only SimuHome's scheduled runs use it.
    if options.execution == "scheduled":
        from . import scheduling as bench_scheduling
        return bench_scheduling.run_scheduled(
            runnable, tools, sim, turtle=turtle,
            tick_interval=float(home_config.get("tick_interval", 0.1)),
            eval_goals=(episode.get("eval") or {}).get("goals") or [],
            dry_step_seconds=options.dry_step_seconds,
            real_read_seconds=options.real_read_seconds)
    # 2. Ticks: lockstep for timed or waiting plans, the UA's loop otherwise.
    if timed or bench_execution.needs_lockstep(runnable):
        return bench_execution.run_lockstep(
            runnable, tools, sim, step_seconds=options.step_seconds,
            tick_interval=float(home_config.get("tick_interval", 0.1)),
            eval_goals=(episode.get("eval") or {}).get("goals") or [])
    return bench_execution.run_immediate(
        runnable, tools, tick_period=float(bt.get("min_tick_yield", 0.01)),
        tick_budget=int((bt.get("max_ticks", {}) or {}).get("user_assistant", 100)),
        timeout=options.exec_timeout)


def _remember(memory, contexts, plans) -> List[str]:
    """Record the signifier and the condition-guard decisions of each
    successful plan the UA would remember; returns the ids recorded."""
    recorded = []
    for context, plan in zip(contexts, plans):
        # 1. Nothing is remembered from a plan that did not succeed.
        if plan.get("status") != "SUCCESS":
            continue
        # 2. The guard decisions, under the UA's rules (`guards_to_record`).
        recorded.extend(memory.record_guard(g) for g in guards_to_record(context.entry))
        # 3. The signifier, under the UA's recording rules.
        if not context.should_record:
            continue
        signifier = build_signifier(context)
        if signifier:
            recorded.append(memory.record(signifier))
    return recorded


def _outcome(record: Dict[str, Any], ua_questions: List[str]) -> str:
    """executed | answered | impossible | clarify | failed. A plan that
    succeeded without acting (its conditions already held) counts as executed."""
    execution = record.get("execution", {})
    if any(a.get("success") for a in execution.get("actions", [])) or \
            any(p.get("status") == "SUCCESS" for p in execution.get("plans", [])):
        return "executed"
    outcomes = [p["outcome"] for p in record["plans"]]
    if ua_questions or "clarify" in outcomes:
        return "clarify"
    if outcomes and all(o == "impossible" for o in outcomes):
        return "impossible"
    if record["queries"] and not record["plans"]:
        return "answered"
    return "failed"


def _trace(record: Dict[str, Any], explorer: Explorer) -> List[Dict[str, Any]]:
    """Our steps in SimuHome's trace format ({step, thought, action,
    action_input, observation}), ending with `finish` and the UA's reply.
    The judge reads the thoughts and checks the answer against the observations."""
    steps: List[Dict[str, Any]] = []

    def add(thought, action, action_input, observation):
        steps.append({"step": len(steps) + 1, "thought": thought, "action": action,
                      "action_input": action_input,
                      "observation": json.dumps(observation, default=str)[:4000]})

    add("Split the request into atomic intents.", "segmentation",
        {"text": record.get("input")}, record["intents"])
    for item in record["structures"]:
        add(f"Structured the goal {item['text']!r}"
            + (f"; asked the user: {item['clarification']}" if item.get("clarification")
               else ""), "goal_structuring", {"text": item["text"]},
            item.get("structure") or {"error": item.get("error")})
    for observation in explorer.observations:
        add("Asked the environment.", f"env_{observation.get('query')}",
            {k: v for k, v in observation.items() if k in ("text", "payload", "extraction")},
            observation)
    for plan in record["plans"]:
        add(plan.get("explanation") or "Planned.", "planning",
            {"intent": (plan.get("intent") or {}).get("text_intent"),
             "route": plan.get("route")},
            {"outcome": plan["outcome"], "devices": plan.get("devices"),
             **({"reads": plan["reads"]} if plan.get("reads") else {})})
    for action in (record.get("execution") or {}).get("actions", []):
        add("Executed an action of the plan.", "execute_action",
            {"url": action["url"], "parameters": action.get("parameters")},
            {"status_code": action.get("status_code"), "success": action.get("success"),
             "time": action.get("time")})
    steps.append({"step": len(steps) + 1, "thought": "Replied to the user.",
                  "action": "finish", "action_input": {"answer": record.get("response") or ""},
                  "observation": ""})
    return steps
