"""Run one phrase through the UA's segmenter and goal structurer, against a
SimuHome home -- no SPADE, no XMPP, no simulator.

What runs is the production code: `pipeline.segment_into_atomic_intents` and
`GoalStructuringBehaviour` (route -> goal context -> LLM -> resolution). Only
the two RPCs to EnvExplorer are replaced, by calls to the same EnvExplorer code
in-process, over the home's Thing Descriptions built by SHTD's TD builder:

    goal context       EnvironmentCapabilitiesBehaviour._generate_goal_context_payload
    candidate lookup   capability_query_behaviour.resolve_capability_payload

Flow:
    1. the home: an episode's `initial_home_config` -> TDs and workspaces
    2. segmentation: one LLM call -> atomic intents with their qualifiers
    3. per GOAL_REQUEST: GoalStructuringBehaviour.run() -> one LLM call, plus
       the lookups resolution makes; the structure is printed with its
       incomplete-goal case and any clarification question

Costs one segmentation call plus one structuring call per goal (OPENAI_API_KEY
from the environment or the repo's .env). `--context-only` makes no LLM call:
it prints the goal context the structurer would get, for the phrase as one goal.

    conda run -n ami-agents python tests/simuhome/structure_goal.py \\
        "Dim the living room lights by 20%" --episode qt1_feasible_seed_1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

from ami_agents.agents.env_explorer.behaviors.capability_query_behaviour import (
    capability_response,
    resolve_capability_payload,
)
from ami_agents.agents.env_explorer.behaviors.environment_capabilities_behaviour import (
    EnvironmentCapabilitiesBehaviour,
)
from ami_agents.agents.env_explorer.utils.actuation_context import (
    DEFAULT_MAX_ACTION_AFFORDANCES,
    DEFAULT_MAX_ARTIFACTS,
)
from ami_agents.agents.user_assistant import pipeline
from ami_agents.agents.user_assistant.behaviours.goal_structuring import (
    GoalStructuringBehaviour,
)
from ami_agents.agents.user_assistant.utils.goal_resolution import clarification_question
from ami_agents.agents.user_assistant.utils.goal_routing import route_goal
from ami_agents.agents.user_assistant.utils.llm_client import (
    build_behaviour_llm_client,
    build_llm_call_kwargs,
)
from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.shared.utils.incomplete_goals import incomplete_case

DEFAULT_BENCHMARK = PROJECT_ROOT.parent / "SimuHome" / "data" / "benchmark"
AGENTS_YAML = PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml"


# --------------------------------------------------------------------------
# The home: TDs and workspaces, shaped as EnvExplorer holds them
# --------------------------------------------------------------------------

def load_home(episode: str) -> Dict[str, Any]:
    """An episode's `initial_home_config`, by file name or path."""
    path = Path(episode)
    if not path.is_file():
        path = DEFAULT_BENCHMARK / episode
    if not path.is_file():
        raise SystemExit(f"Episode not found: {episode} (looked in {DEFAULT_BENCHMARK})")
    return json.loads(path.read_text(encoding="utf-8"))["initial_home_config"]


def explorer_stub(home: Dict[str, Any], config: Dict[str, Any], logger) -> SimpleNamespace:
    """What EnvExplorer's goal-context and lookup code read from their agent:
    the discovered artifacts and workspaces as RDF, and the scoping thresholds."""
    builder = SimuHomeTD("http://localhost:8097", "home", home)
    rooms = home["rooms"]
    actuation = config.get("env_explorer", {}).get("actuation_context", {})
    return SimpleNamespace(
        logger=logger,
        environment_map={r: SimpleNamespace(rdf=builder.room_workspace(r).serialize(format="turtle"))
                         for r in rooms},
        # Every artifact a room workspace lists: the devices and SHTD's
        # virtual ambient sensors.
        artifacts={device: SimpleNamespace(thing_description=SimpleNamespace(
                       rdf=builder.artifact(r, device).serialize(format="turtle")))
                   for r in rooms
                   for device in [d["device_id"] for d in rooms[r].get("devices", [])]
                   + [s for _t, s, _row in builder.ambient_sensors(r)]},
        actuation_max_artifacts=int(actuation.get("max_artifacts", DEFAULT_MAX_ARTIFACTS)),
        actuation_max_actions=int(actuation.get("max_action_affordances",
                                                     DEFAULT_MAX_ACTION_AFFORDANCES)),
    )


# --------------------------------------------------------------------------
# The UA side: an agent stub, and the two RPCs answered in-process
# --------------------------------------------------------------------------

def ua_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The UA's config as `main.py` builds it: its own section plus the shared
    ones (the per-behaviour LLM settings stay under its own `llm`)."""
    merged = dict(config.get("user_assistant", {}))
    for key in ("default_llm", "timeouts", "environment", "bt_execution", "http"):
        merged[key] = config.get(key, {})
    return merged


def ua_stub(config: Dict[str, Any]) -> SimpleNamespace:
    """What the segmenter and GoalStructuringBehaviour read from their agent."""
    cfg = build_behaviour_llm_client(config, "atomic_segmentation")
    return SimpleNamespace(config=config, llm_client=cfg.client, llm_model=cfg.model,
                           build_llm_kwargs=lambda: build_llm_call_kwargs(cfg))


def wire_explorer(explorer: SimpleNamespace, show_context: bool) -> None:
    """Replace the UA's two EnvExplorer RPCs with the same code, in-process."""
    capabilities = EnvironmentCapabilitiesBehaviour()
    capabilities.agent = explorer

    async def fetch_goal_context(agent, logger, goal_text, with_properties=False,
                                 with_environment=False):
        payload = capabilities._generate_goal_context_payload(
            goal_text, with_properties=with_properties, with_environment=with_environment)
        if payload.get("error"):
            print(f"  ! goal context failed: {payload.get('detail')}")
            return ""
        scope = payload.get("scope") or {}
        text = payload.get("text") or ""
        print(f"  goal context: rule={scope.get('rule')} rooms={scope.get('matched_rooms', [])} "
              f"devices={scope.get('matched_devices', [])} chars={len(text)}")
        if show_context:
            print("  " + text.replace("\n", "\n  "))
        return text

    async def resolve_candidates(agent, logger, payload):
        response = capability_response(resolve_capability_payload(explorer, payload), payload)
        names = sorted({e.get("artifact_name") for e in response.get("entries") or []})
        print(f"  lookup {json.dumps({k: v for k, v in payload.items() if k != 'text_intent'})}"
              f" -> {response.get('outcome')} {names}")
        return response

    pipeline.fetch_goal_context = fetch_goal_context
    pipeline.resolve_candidates = resolve_candidates


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------

async def structure(phrase: str, ua: SimpleNamespace, logger) -> None:
    # 1. Segmentation: the atomic intents and their qualifiers.
    intents = await pipeline.segment_into_atomic_intents(ua, logger, phrase)
    print(f"\n== segmentation ({ua.llm_model}): {len(intents)} intent(s)")
    for intent in intents:
        print(f"  {intent.type}: {intent.text!r} qualifiers={intent.qualifiers}")

    # 2. Each goal through the real structuring behaviour.
    for intent in intents:
        if intent.type != "GOAL_REQUEST":
            continue
        print(f"\n== structuring {intent.text!r}")
        behaviour = GoalStructuringBehaviour(intent.text, intent.qualifiers, logger=logger)
        behaviour.agent = ua
        await behaviour.run()
        if behaviour.result is None:
            print(f"  ! not structured: {behaviour.error}")
            continue
        print(json.dumps(behaviour.result.to_wire_dict(), indent=2))
        for gid, goal in behaviour.result.goals.items():
            print(f"  {gid}: case={incomplete_case(goal)} "
                  f"clarification={clarification_question(goal)!r}")


async def context_only(phrase: str, qualifiers, explorer_wired) -> None:
    """The goal context for the phrase as one goal, with no LLM call."""
    route = route_goal(qualifiers)
    print(f"route: {route.structure} properties={route.with_properties} "
          f"environment={route.with_environment}")
    await pipeline.fetch_goal_context(None, None, phrase,
                                      with_properties=route.with_properties,
                                      with_environment=route.with_environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phrase", help="the user's request")
    parser.add_argument("--episode", required=True,
                        help="SimuHome episode JSON (file name in the benchmark dir, or a path) "
                             "whose initial_home_config is the home")
    parser.add_argument("--show-context", action="store_true",
                        help="print the goal context each structuring call gets")
    parser.add_argument("--context-only", action="store_true",
                        help="no LLM: print the goal context for the phrase as one goal")
    parser.add_argument("--qualifiers", default="incomplete,achievement",
                        help="segmenter qualifiers to route with, for --context-only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logger = logging.getLogger("structure_goal")

    config = ConfigLoader.load_with_env_vars(str(AGENTS_YAML))
    home = load_home(args.episode)
    print("home: " + "; ".join(f"{r}: {len(v.get('devices', []))} devices"
                               for r, v in home["rooms"].items()))
    wire_explorer(explorer_stub(home, config, logger), args.show_context or args.context_only)

    if args.context_only:
        asyncio.run(context_only(args.phrase, args.qualifiers.split(","), True))
        return
    asyncio.run(structure(args.phrase, ua_stub(ua_config(config)), logger))


if __name__ == "__main__":
    main()
