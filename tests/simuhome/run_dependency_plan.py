"""Run one dependency plan against the running SimuHome stack, in its virtual time.

Needs the simulator and SHTD running (started by you), e.g. on 14000 / 8097:

    python ami_agents/environment/integration/SimuHome/run_inspector.py \\
        --episode qt1_feasible_seed_1 --sim-port 14000 --port 8098
    python ami_agents/environment/integration/SimuHome/shtd.py \\
        --sim http://127.0.0.1:14000/api --home qt1_feasible_seed_1 --port 8097

Then:

    CLOCK_SOURCE=http CLOCK_URL=http://localhost:8097/_shtd/time \\
    conda run -n ami-agents python tests/simuhome/run_dependency_plan.py \\
        --episode qt1_feasible_seed_1.json --case delay --fake-profile

Cases (hand-built GoalStructures on two on/off lights of one room, so no
structuring call is needed):

    delay       in 2 minutes turn on light 1
    at_time     at <virtual now + 3 min, next whole minute> turn on light 1
    after_goal  turn on light 1, then 1 minute later turn off light 2
    trigger     when light 1 turns on, turn on light 2 (this script turns
                light 1 on after --trigger-after virtual seconds)

The plan is built with the dependency planner's own pieces, in-process (the
achievement steps deterministically, predicate resolution, the checklist, the
profiling call -- faked with --fake-profile, else one gpt-5-nano call when the
checklist has questions -- and the compiler), printed, then executed: the
runtime refreshed before every tick from the configured clock, the actions sent
to SHTD. It reports when each action fired in virtual time and checks it fired
at or after its due time, within --slack virtual seconds, and that the light
reads back as set. Exit code 0 on pass.

It actuates the simulated home; it does not start or stop any server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

from rdflib import Graph

from ami_agents.agents.env_explorer.utils.data_formatting import format_capabilities_detailed_rdf
from ami_agents.agents.interaction_solver.behaviours.dependency_goal_planning import (
    DependencyGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.utils.dependency_compiler import compile_dependency
from ami_agents.agents.interaction_solver.utils.dependency_context import resolve_predicate
from ami_agents.agents.interaction_solver.utils.explicit_goal_plan import plan_explicit_goal
from ami_agents.agents.interaction_solver.utils.goal_profile import (
    build_checklist,
    fallback_profile,
)
from ami_agents.agents.interaction_solver.utils.llm_client import build_llm_client
from ami_agents.bt_planning.nodes.affordance_nodes import ActionAffordanceNode
from ami_agents.bt_planning.nodes.registry import compile_node
from ami_agents.bt_planning.runtime import PlanRuntime
from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
from ami_agents.shared.models.goal_structure import GoalStructure
from ami_agents.shared.utils.clock import make_clock
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.shared.utils.vocabulary import close_types

DEFAULT_BENCHMARK = PROJECT_ROOT.parent / "SimuHome" / "data" / "benchmark"


# --------------------------------------------------------------------------
# The home and the two lights
# --------------------------------------------------------------------------

def load_home(episode: str):
    path = Path(episode)
    if not path.is_file():
        path = DEFAULT_BENCHMARK / episode
    return path.stem, json.loads(path.read_text())["initial_home_config"]


def build_graph(shtd: str, home_id: str, config) -> Graph:
    builder = SimuHomeTD(shtd, home_id, config)
    rooms = config["rooms"]
    agent = SimpleNamespace(
        discovery_complete=True,
        environment_map={r: SimpleNamespace(rdf=builder.room_workspace(r).serialize(format="turtle"))
                         for r in rooms},
        artifacts={device: SimpleNamespace(thing_description=SimpleNamespace(
            rdf=builder.artifact(r, device).serialize(format="turtle")))
            for r in rooms
            for device in [d["device_id"] for d in rooms[r].get("devices", [])]
            + [s for _t, s, _row in builder.ambient_sensors(r)]})
    graph = Graph().parse(data=format_capabilities_detailed_rdf(agent), format="turtle")
    close_types(graph)
    return graph


def two_lights(config):
    """A room with two on/off lights: (room, light_1, light_2)."""
    for room, data in config["rooms"].items():
        lights = sorted(d["device_id"] for d in data.get("devices", [])
                        if "on_off_light" in d["device_id"])
        if len(lights) >= 2:
            return room, lights[0], lights[1]
    raise SystemExit("no room with two on/off lights in this episode")


def light_goal(light, on, **timing):
    return {"goal_specificity": "explicit", "goal_kind": "achievement",
            "intent_text": f"turn {'on' if on else 'off'} {light}", "artifact_name": light,
            "artifact_class": "homeont:OnOffLight", "affordance_class": "homeont:SetOnOffCommand",
            "affordance_name": "onOff", "parameter_name": "NA", "goal_effect": "set",
            "target_value_text": "on" if on else "off",
            "target_value_determined": "true" if on else "false", "timing": timing}


# --------------------------------------------------------------------------
# The cases
# --------------------------------------------------------------------------

def case(name, light_1, light_2, now: datetime):
    """(structure, {goal: (light, expected value)}, due-time rules)."""
    if name == "delay":
        goals = {"G1": light_goal(light_1, True, starts="after_delay", offset_text="2 minutes")}
        return goals, {}, {"G1": ("start", 120)}
    if name == "at_time":
        due = (now + timedelta(minutes=3)).replace(second=0) + timedelta(minutes=1)
        goals = {"G1": light_goal(light_1, True, starts="at_time",
                                  time_text=due.strftime("%H:%M"))}
        return goals, {}, {"G1": ("at", due)}
    if name == "after_goal":
        goals = {"G1": light_goal(light_1, True, starts="now"),
                 "G2": light_goal(light_2, False, starts="after_goal", ref_goal="G1",
                                  offset_text="1 minute")}
        return goals, {}, {"G1": ("start", 0), "G2": ("after", "G1", 60)}
    if name == "trigger":
        goals = {"G2": light_goal(light_2, True, starts="on_trigger")}
        predicates = {"P1": {"predicate_specificity": "explicit",
                             "predicate_subject": "device_property",
                             "predicate_text": f"{light_1} turns on", "artifact_name": light_1,
                             "property_name": "onOff", "comparison": "==",
                             "target_value_determined": "true", "role": "trigger",
                             "applies_to": ["G2"]}}
        return goals, predicates, {"G2": ("trigger", 0)}
    raise SystemExit(f"unknown case {name!r}")


# --------------------------------------------------------------------------
# Planning, with the planner's own pieces
# --------------------------------------------------------------------------

async def plan(graph, structure, fake_profile: bool):
    subtrees = {}
    for gid, goal in structure.goals.items():
        entry = plan_explicit_goal(graph, goal)
        if not entry or not entry.get("tree"):
            raise SystemExit(f"{gid} could not be planned: {entry}")
        subtrees[gid] = entry["tree"]
    targets = {pid: resolve_predicate(graph, p) for pid, p in structure.predicates.items()}
    for pid, target in targets.items():
        if isinstance(target, str):
            raise SystemExit(f"{pid}: {target}")
    checklist = build_checklist(structure)
    if not checklist:
        profile = {}
    elif fake_profile:
        profile, _ = fallback_profile(checklist, "--fake-profile")
    else:
        config = ConfigLoader.load_with_env_vars(
            str(PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml"))["interaction_solver"]
        behaviour = DependencyGoalPlanningBehaviour([])
        behaviour.agent = SimpleNamespace(explicit_llm=build_llm_client(
            config, section="explicit_planning"), config=config)
        devices = {gid: [] for gid in structure.goals}
        profile, _, _ = await behaviour._profile(graph, structure, checklist, devices, targets)
    return compile_dependency(graph, structure, subtrees, targets, profile)


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------

def post(url, value):
    request = urllib.request.Request(url, data=json.dumps({"value": value}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(request, timeout=10).read()


def read(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episode", required=True)
    parser.add_argument("--case", required=True, choices=["delay", "at_time", "after_goal", "trigger"])
    parser.add_argument("--shtd", default="http://localhost:8097")
    parser.add_argument("--fake-profile", action="store_true",
                        help="no LLM: every checklist question on its fallback")
    parser.add_argument("--timeout", type=float, default=10.0, help="virtual minutes")
    parser.add_argument("--tick", type=float, default=0.5, help="real seconds between ticks")
    parser.add_argument("--slack", type=float, default=30.0,
                        help="virtual seconds an action may fire after it is due")
    parser.add_argument("--trigger-after", type=float, default=60.0,
                        help="trigger case: virtual seconds before light 1 is turned on")
    args = parser.parse_args()

    clock = make_clock({"clock": {"source": os.getenv("CLOCK_SOURCE", "system"),
                                  "http": {"url": os.getenv("CLOCK_URL", "")}}})
    print(f"clock: {type(clock).__name__} {os.getenv('CLOCK_URL', '')}")
    home_id, config = load_home(args.episode)
    graph = build_graph(args.shtd, home_id, config)
    room, light_1, light_2 = two_lights(config)
    on_off = {light: f"{args.shtd}/workspaces/{home_id}/{room}/artifacts/{light}/actions/onOff"
              for light in (light_1, light_2)}
    state = {light: f"{args.shtd}/workspaces/{home_id}/{room}/artifacts/{light}/properties/onOff"
             for light in (light_1, light_2)}

    # A known start: both lights off.
    post(on_off[light_1], False)
    post(on_off[light_2], False)
    start = clock.now()
    goals, predicates, due_rules = case(args.case, light_1, light_2, start)
    structure = GoalStructure.from_dict({"goals": goals, "predicates": predicates},
                                        intent_text=args.case, structure="dependency")
    compiled = asyncio.run(plan(graph, structure, args.fake_profile))
    print("tree:", json.dumps(compiled.tree, indent=1))
    print("runtime:", json.dumps(compiled.runtime, indent=1))
    if compiled.notes:
        print("notes:", compiled.notes)

    # Execute: refresh the runtime, tick, note when each action has run.
    def compile_child(spec):
        return compile_node(spec, compile_child)

    tree = compile_child(compiled.tree)
    tree.setup_with_descendants()
    runtime = PlanRuntime(compiled.runtime, clock=clock, synchronous=True)
    actions = {node.name: node for node in tree.iterate() if isinstance(node, ActionAffordanceNode)}
    fired = {}
    triggered_at = None
    deadline = start + timedelta(minutes=args.timeout)
    print(f"start (virtual): {start}")
    while True:
        now = clock.now()
        if args.case == "trigger" and triggered_at is None and \
                (now - start).total_seconds() >= args.trigger_after:
            post(on_off[light_1], True)
            triggered_at = clock.now()
            print(f"{triggered_at}  trigger: {light_1} turned on by the script")
        runtime.before_tick()
        tree.tick_once()
        for name, node in actions.items():
            if name not in fired and getattr(node, "_executed_successfully", False):
                fired[name] = clock.now()
                print(f"{fired[name]}  fired {name}")
        if tree.status.name != "RUNNING" or now >= deadline:
            break
        time.sleep(args.tick)
    print(f"tree: {tree.status.name}")

    # Check each goal's action against its due time, and read it back.
    ok = tree.status.name == "SUCCESS"
    by_goal = {gid: f"{goals[gid]['artifact_name']}.onOff" for gid in goals}
    for gid, rule in due_rules.items():
        when = fired.get(by_goal[gid])
        if rule[0] == "start":
            due = start + timedelta(seconds=rule[1])
        elif rule[0] == "at":
            due = rule[1]
        elif rule[0] == "after":
            due = (fired.get(by_goal[rule[1]]) or deadline) + timedelta(seconds=rule[2])
        else:
            due = triggered_at or deadline
        late = (when - due).total_seconds() if when else None
        passed = when is not None and 0 <= late <= args.slack
        expected = goals[gid]["target_value_determined"] == "true"
        value = read(state[goals[gid]["artifact_name"]])
        passed = passed and value == expected
        ok = ok and passed
        print(f"{gid}: due {due}, fired {when}, "
              f"{'late by %.0fs' % late if late is not None else 'never'}, "
              f"reads {value!r} (expected {expected!r}) -> {'PASS' if passed else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
