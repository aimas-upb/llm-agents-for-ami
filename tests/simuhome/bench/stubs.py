"""In-process stand-ins for the three agents, around the agents' own code.

Nothing here reasons: every decision is made by the code the agents run. The
stand-ins only carry what that code reads from its agent, and answer in-process
what would otherwise be a message to another agent.

    Home                 the discovered home: TDs and workspaces built by SHTD's
                         TD builder from the simulator's state, with SHTD's URLs
                         (discovery -- offline in a deployment, never timed)
    Explorer             the EnvExplorer's per-request answers: goal context,
                         candidate lookup, state queries, the capability graph,
                         memory matches -- each one logged as the SimuHome tool
                         call it stands for (`toollog.py`)
    ua_agent / isa_agent / response_agent
                         SimpleNamespaces with what the UA and ISA behaviours
                         read; `add_behaviour` runs a nested behaviour so its
                         `join()` works without SPADE
    patched_ua_clients   the UA behaviours that build their own client get the
                         variant's model instead (test-only)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

from rdflib import Graph

from ami_agents.agents.env_explorer.behaviors.capability_query_behaviour import (
    capability_response,
    resolve_capability_payload,
)
from ami_agents.agents.env_explorer.behaviors.environment_capabilities_behaviour import (
    EnvironmentCapabilitiesBehaviour,
)
from ami_agents.agents.env_explorer.behaviors.property_resolution_behaviour import (
    resolution_response,
    resolve_payload,
)
from ami_agents.agents.env_explorer.behaviors.value_retrieval_behaviour import (
    ValueRetrievalBehaviour,
)
from ami_agents.agents.env_explorer.utils.actuation_context import (
    DEFAULT_MAX_ACTION_AFFORDANCES,
    DEFAULT_MAX_ARTIFACTS,
)
from ami_agents.agents.env_explorer.utils.data_formatting import (
    format_capabilities_detailed_rdf,
)
from ami_agents.agents.interaction_solver.utils.llm_client import (
    LLMClientConfig as ISAClientConfig,
)
from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
from ami_agents.shared.models.messages import MessageType
from ami_agents.shared.utils.vocabulary import close_types

from .models import Variant
from .toollog import ToolLog

# What the agents read from config, as main.py merges it.
SHARED_KEYS = ("default_llm", "timeouts", "environment", "bt_execution", "http")


def section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """One agent's config: its own section plus the shared ones."""
    merged = dict(config.get(name, {}) or {})
    for key in SHARED_KEYS:
        merged.setdefault(key, config.get(key, {}))
    return merged


# --------------------------------------------------------------------------
# Running nested behaviours without SPADE
# --------------------------------------------------------------------------

def add_behaviour(agent: Any, behaviour: Any) -> None:
    """Start ``behaviour`` as SPADE would, so its ``join()`` waits for it.

    SPADE's ``join`` polls an event that only ``set_agent`` creates; set means
    still running. The behaviour runs once (they are all OneShot here)."""
    behaviour.agent = agent
    behaviour._is_done = asyncio.Event()
    behaviour._is_done.set()

    async def run() -> None:
        try:
            await behaviour.run()
        finally:
            behaviour._is_done.clear()

    behaviour._bench_task = asyncio.ensure_future(run())


# --------------------------------------------------------------------------
# The home, as discovered
# --------------------------------------------------------------------------

class Home:
    """The home's TDs and workspaces, as the EnvExplorer holds them after
    discovery, with SHTD's URLs so plans act through SHTD."""

    def __init__(self, shtd_base: str, home_id: str, state: Dict[str, Any],
                 config: Dict[str, Any], logger) -> None:
        self.shtd_base = shtd_base.rstrip("/")
        self.home_id = home_id
        self.builder = SimuHomeTD(self.shtd_base, home_id, state)
        # What each room workspace lists, as the EnvExplorer's crawl finds it:
        # the simulator's devices and SHTD's virtual ambient sensors (one per
        # room variable -- what a room-state question is answered from).
        self.sensors: Dict[str, str] = {
            sensor: room for room in state["rooms"]
            for _token, sensor, _row in self.builder.ambient_sensors(room)}
        self.rooms: Dict[str, List[str]] = {
            room: [d["device_id"] for d in data.get("devices", [])]
            + [s for s, r in self.sensors.items() if r == room]
            for room, data in state["rooms"].items()}
        self.device_room = {device: room for room, devices in self.rooms.items()
                            for device in devices}
        actuation = (config.get("env_explorer", {}) or {}).get("actuation_context", {}) or {}
        self.agent = SimpleNamespace(
            logger=logger, discovery_complete=True,
            environment_map={room: SimpleNamespace(
                rdf=self.builder.room_workspace(room).serialize(format="turtle"))
                for room in self.rooms},
            artifacts={device: SimpleNamespace(thing_description=SimpleNamespace(
                rdf=self.builder.artifact(room, device).serialize(format="turtle")))
                for room, devices in self.rooms.items() for device in devices},
            actuation_max_artifacts=int(actuation.get("max_artifacts", DEFAULT_MAX_ARTIFACTS)),
            actuation_max_actions=int(actuation.get("max_action_affordances",
                                                    DEFAULT_MAX_ACTION_AFFORDANCES)))
        # What the ISA asks for with ENV_CAPABILITIES (detailed): the whole home.
        self.turtle = format_capabilities_detailed_rdf(self.agent)

    def graph(self) -> Graph:
        graph = Graph().parse(data=self.turtle, format="turtle")
        close_types(graph)
        return graph

    def room_id(self, name: str) -> List[str]:
        """The room ids a scope names: a room id as is, a room class by class."""
        return [name] if name in self.rooms else self.rooms_of_location(name)

    def rooms_of_location(self, location_class: Optional[str]) -> List[str]:
        """The rooms a location class names (by the class's local name)."""
        if not location_class:
            return []
        wanted = str(location_class).split(":")[-1].split("#")[-1].split("/")[-1].lower()
        out = []
        for room in self.rooms:
            cls = self.builder._room_class(room)
            local = str(cls).split("#")[-1].split("/")[-1].lower() if cls is not None else ""
            if local == wanted or room.replace("_", "") == wanted.replace("_", ""):
                out.append(room)
        return out


# --------------------------------------------------------------------------
# The EnvExplorer's per-request answers
# --------------------------------------------------------------------------

class Explorer:
    """Answers what the UA and ISA would ask the EnvExplorer, with its code,
    and logs each answer as the SimuHome tool calls it stands for."""

    def __init__(self, home: Home, tools: ToolLog, memory=None, logger=None) -> None:
        self.home = home
        self.tools = tools
        self.memory = memory
        self.logger = logger
        self.capabilities = EnvironmentCapabilitiesBehaviour()
        self.capabilities.agent = home.agent
        # Observations for the record (and the judge's steps).
        self.observations: List[Dict[str, Any]] = []

    # UA: goal structuring -------------------------------------------------

    async def fetch_goal_context(self, agent, logger, goal_text, with_properties=False,
                                 with_environment=False) -> str:
        payload = self.capabilities._generate_goal_context_payload(
            goal_text, with_properties=with_properties, with_environment=with_environment)
        if payload.get("error"):
            self.observations.append({"query": "goal_context", "text": goal_text,
                                      "error": payload.get("detail")})
            return ""
        scope = payload.get("scope") or {}
        rooms = [self.home.room_id(room) for room in scope.get("matched_rooms") or []]
        rooms = [room for found in rooms for room in found]
        rooms += [self.home.device_room[d] for d in scope.get("matched_devices") or []
                  if d in self.home.device_room]
        for room in dict.fromkeys(rooms):
            self.tools.room_devices(room, "goal_context")
        self.observations.append({"query": "goal_context", "text": goal_text,
                                  "scope": scope})
        return payload.get("text") or ""

    async def resolve_candidates(self, agent, logger, payload) -> Dict[str, Any]:
        response = capability_response(resolve_capability_payload(self.home.agent, payload),
                                        payload)
        names = [e.get("artifact_name") for e in response.get("entries") or []]
        for room in dict.fromkeys(self.home.device_room[n] for n in names
                                  if n in self.home.device_room):
            self.tools.room_devices(room, "candidates")
        self.observations.append({"query": "candidates",
                                  "payload": {k: v for k, v in payload.items()
                                              if k != "text_intent"},
                                  "outcome": response.get("outcome"), "devices": names})
        return response

    # UA: state questions --------------------------------------------------

    async def state(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """ENV_STATE_REQUEST: resolve the classes, read what they resolve to."""
        resolution = resolve_payload(self.home.agent, extraction)
        for room in self.home.rooms_of_location(extraction.get("location_class")):
            self.tools.room_devices(room, "state_query")
        if resolution.affordances:
            retrieval = ValueRetrievalBehaviour(resolution.affordances, logger=self.logger)
            await retrieval.run()
            for affordance in resolution.affordances:
                if affordance.target:
                    self.tools.read(affordance.target, self.home, "state_query")
        response = resolution_response(resolution, extraction)
        self.observations.append({"query": "state", "extraction": extraction,
                                  "response": response})
        return response

    # ISA ------------------------------------------------------------------

    async def query(self, message_type: str, body: Any = None, expect_type: str = None,
                    **_: Any) -> str:
        """The ISA's `_query_env_explorer`."""
        if message_type == MessageType.ENV_CAPABILITIES_REQUEST.value:
            # The discovery result handed over: not a request-time query.
            return json.dumps({"payload": self.home.turtle})
        if message_type == MessageType.SIGNIFIER_MATCH_REQUEST.value:
            record = None
            if self.memory is not None and isinstance(body, dict):
                record = self.memory.match(body.get("kind"), body.get("key"))
            self.observations.append({"query": "memory", "kind": (body or {}).get("kind"),
                                      "hit": record is not None})
            return json.dumps({"ok": True, "match": record})
        return json.dumps({"ok": False, "error": f"not served in the bench: {message_type}"})


# --------------------------------------------------------------------------
# The agents
# --------------------------------------------------------------------------

def ua_agent(config: Dict[str, Any], variant: Variant, logger) -> SimpleNamespace:
    """What segmentation, structuring and state answering read from the UA."""
    seg = variant["segmentation"]
    agent = SimpleNamespace(config=section(config, "user_assistant"), logger=logger,
                            llm_client=seg.client(), llm_model=seg.model,
                            build_llm_kwargs=seg.call_kwargs)
    agent.add_behaviour = lambda behaviour: add_behaviour(agent, behaviour)
    return agent


def response_agent(config: Dict[str, Any], variant: Variant, logger) -> SimpleNamespace:
    """The UA as it phrases its reply: the plan summary's client is the
    variant's response model (the agent uses its segmentation client)."""
    spec = variant["response"]
    agent = SimpleNamespace(config=section(config, "user_assistant"), logger=logger,
                            llm_client=spec.client(), llm_model=spec.model,
                            build_llm_kwargs=spec.call_kwargs)
    agent.add_behaviour = lambda behaviour: add_behaviour(agent, behaviour)
    return agent


def isa_agent(config: Dict[str, Any], variant: Variant, explorer: Explorer,
              logger) -> SimpleNamespace:
    """What the planning behaviours read from the ISA."""
    isa = section(config, "interaction_solver")
    planning = isa.get("planning", {}) or {}
    general: ISAClientConfig = variant["planning.general"].isa_config()
    agent = SimpleNamespace(
        config=isa, logger=logger,
        llm_client=general.client, model=general.model, base_url=general.base_url,
        temperature=general.temperature, reasoning_effort=general.reasoning_effort,
        max_tokens=general.max_tokens, max_completion_tokens=general.max_completion_tokens,
        bt_planner=AsyncBTPlanner(max_attempts=int(
            (planning.get("llm_planning", {}) or {}).get("max_planning_attempts", 2))),
        explicit_llm=variant["planning.explicit"].isa_config(),
        explicit_bt_planner=AsyncBTPlanner(max_attempts=int(
            (planning.get("explicit_planning", {}) or {}).get("max_planning_attempts", 2))),
        community_client=None,
        _query_env_explorer=explorer.query)
    agent.add_behaviour = lambda behaviour: add_behaviour(agent, behaviour)
    return agent


@contextmanager
def patched_ua_clients(variant: Variant) -> Iterator[None]:
    """The UA behaviours that build their own client (structuring, state
    answers) get the variant's model. Test-only: the agents are unchanged."""
    from ami_agents.agents.user_assistant.behaviours import (
        capability_answer,
        env_capability_structuring,
        env_state_structuring,
        goal_structuring,
        state_answer,
    )
    choices = {goal_structuring: "structuring", env_state_structuring: "structuring",
               env_capability_structuring: "structuring",
               state_answer: "response", capability_answer: "response"}
    saved = {module: module.build_behaviour_llm_client for module in choices}

    def factory(step):
        return lambda config, key: variant[step].ua_config()

    try:
        for module, step in choices.items():
            module.build_behaviour_llm_client = factory(step)
        yield
    finally:
        for module, original in saved.items():
            module.build_behaviour_llm_client = original


@contextmanager
def patched_explorer_rpcs(explorer: Explorer) -> Iterator[None]:
    """The UA's two EnvExplorer RPCs, answered by ``explorer``."""
    from ami_agents.agents.user_assistant import pipeline
    saved = (pipeline.fetch_goal_context, pipeline.resolve_candidates)
    pipeline.fetch_goal_context = explorer.fetch_goal_context
    pipeline.resolve_candidates = explorer.resolve_candidates
    try:
        yield
    finally:
        pipeline.fetch_goal_context, pipeline.resolve_candidates = saved
