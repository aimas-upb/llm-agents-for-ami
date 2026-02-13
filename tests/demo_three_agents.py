"""
Demo script for running three complete agent instances.

Each instance contains:
- EnvExplorer: Discovers and monitors the environment
- InteractionSolver: Plans and coordinates goal execution
- UserAssistant: Interfaces with the user

Instances are identified by a short UUID (8 characters).
"""
import argparse
import asyncio
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import spade


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import yaml

from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
from ami_agents.shared.models.community import Community, AffordanceMatch
from ami_agents.shared.utils.demo_log import demo
from ami_agents.environment.connection.hmas_client import IHMASClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# XMPP Configuration
XMPP_SERVER = os.getenv("XMPP_SERVER", "localhost")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "password")

# Number of agent instances to create
NUM_INSTANCES = 3


def generate_short_uuid() -> str:
    """Generate a short UUID (first 8 characters)."""
    return uuid.uuid4().hex[:8]

class _DropHighFrequencyStateUpdateSpamFilter(logging.Filter):
    """Drop ultra-frequent sensor state updates to keep console readable."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (filter name is logging API)
        try:
            msg = record.getMessage()
        except Exception:
            return True
        # Example:
        # "STATE UPDATE: clock308 -> http://.../clock308/props/timeOfDay = 18:03"
        if "STATE UPDATE:" in msg:
            if "clock308" in msg and "/props/timeOfDay" in msg:
                return False
            if "lightSensor308" in msg and "/props/luminosity" in msg:
                return False
            if "peoplePresenceSensor308" in msg and "/props/presence" in msg:
                return False
        return True


class _DemoOnlyLogFilter(logging.Filter):
    """Allow only log entries containing the [DEMO] prefix."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:
            return False
        return "[DEMO]" in msg


class _StripDemoPrefixFilter(logging.Filter):
    """Remove the literal [DEMO] prefix (and ANSI variant) from log messages."""

    _TOKENS = (
        "[DEMO] ",
        "[DEMO]",
        "\x1b[1;36m[DEMO]\x1b[0m ",
        "\x1b[1;36m[DEMO]\x1b[0m",
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = str(record.msg)
        except Exception:
            return True

        for token in self._TOKENS:
            if token in msg:
                record.msg = msg.replace(token, "", 1).lstrip()
                break
        return True


def _configure_console_logging() -> None:
    """
    Make the manual test output readable:
    - Root logger at WARNING (quiet by default)
    - Keep key project loggers at INFO
    - Silence chatty dependencies (slixmpp/aiohttp/httpx/spade_llm)
    - Filter clock308 timeOfDay update spam
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    # Keep our high-signal logs.
    for name in [
        "3AgentsDemo",
        "UserAssistant",
        "InteractionSolver",
        "ami_agents.agents.env_explorer.env_explorer_agent",
        "ami_agents.environment.integration.integration_engine",
    ]:
        logging.getLogger(name).setLevel(logging.INFO)

    # Silence very chatty libs.
    for noisy in [
        "spade",
        "slixmpp",
        "aiohttp.access",
        "httpx",
        "spade_llm",
        "spade_llm.providers",
        "spade_llm.behaviour",
        "ami_agents.shared.ontologies.loader",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Drop only high-frequency sensor spam, keep important state changes (e.g., light/blinds updates).
    filt = _DropHighFrequencyStateUpdateSpamFilter()
    root = logging.getLogger()
    root.addFilter(filt)
    logging.getLogger("ami_agents.agents.env_explorer.env_explorer_agent").addFilter(filt)
    for h in list(getattr(root, "handlers", []) or []):
        h.addFilter(filt)


def _enable_demo_only_logging() -> None:
    """Install a filter that keeps only [DEMO]-tagged log records."""
    demo_filter = _DemoOnlyLogFilter()
    root = logging.getLogger()
    root.addFilter(demo_filter)
    for handler in list(getattr(root, "handlers", []) or []):
        handler.addFilter(demo_filter)


def _strip_demo_prefix_from_logs() -> None:
    """Attach a filter that strips the [DEMO] prefix after we've gated logs."""
    strip_filter = _StripDemoPrefixFilter()
    root = logging.getLogger()
    root.addFilter(strip_filter)
    for handler in list(getattr(root, "handlers", []) or []):
        handler.addFilter(strip_filter)


_configure_console_logging()
logger = logging.getLogger("3AgentsDemo")

_NO_COLOR = os.getenv("AMI_NO_COLOR") or os.getenv("NO_COLOR")
_RED = "" if _NO_COLOR else "\033[31m"
_RESET = "" if _NO_COLOR else "\033[0m"
LAB308_WORKSPACE_URI = "http://localhost:8080/workspaces/lab308#workspace"
LAB308_BASE = LAB308_WORKSPACE_URI.split("#")[0]


def _print_red_line(text: str) -> None:
    if _RED:
        print(f"{_RED}{text}{_RESET}")
    else:
        print(str(text))


async def _reset_lab308_state() -> None:
    """Ensure lights are off and blinds closed before the demo starts."""
    light_off_url = f"{LAB308_BASE}/artifacts/lights_308/ha/light/turn_off"
    blinds_close_url = f"{LAB308_BASE}/artifacts/blinds_308/setClosedPercentage"
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(light_off_url, json={})
            await session.post(blinds_close_url, json={"closedPercentage": 100})
        logger.info("Reset lab308 state: lights_308 off, blinds_308 closed.")
    except Exception as exc:
        logger.warning("Failed to reset lab308 state before demo: %s", exc)


class DummyHMASClient(IHMASClient):
    """Minimal HMAS client stub to satisfy EnvExplorer constructor."""

    async def connect(self, endpoint_url: str) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_workspace(self, workspace_id: str):
        return None

    async def list_workspaces(self, parent_id: str | None = None):
        return []

    async def get_artifact(self, artifact_id: str):
        return None

    async def list_artifacts(self, workspace_id: str):
        return []

    async def get_thing_description(self, artifact_id: str):
        return None

    async def invoke_action(self, artifact_id: str, action_name: str, params: dict):
        return {}

    async def read_property(self, artifact_id: str, property_name: str):
        return None

    async def write_property(self, artifact_id: str, property_name: str, value):
        return False

    async def subscribe_to_event(self, artifact_id: str, event_name: str, callback: callable) -> str:
        return ""

    async def unsubscribe_from_event(self, subscription_id: str) -> bool:
        return True

    async def subscribe_to_workspace_changes(self, workspace_id: str, callback: callable) -> str:
        return ""

    async def crawl_environment(self, root_workspace_id: str, depth: int = -1) -> dict:
        return {}


def _require_env(var: str) -> str:
    val = os.getenv(var, "").strip()
    if not val:
        raise RuntimeError(f"Missing required environment variable: {var}")
    return val


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual full-flow test runner (thesis demo friendly).")
    env_min_similarity = os.getenv("SIGNIFIER_MIN_SIMILARITY", "").strip()
    try:
        env_min_similarity_f = float(env_min_similarity) if env_min_similarity else None
    except Exception:
        env_min_similarity_f = None
    p.add_argument(
        "--sequence",
        default=None,
        help=(
            'Which sequence to run: "2"/"startup", "3"/"demo", "4"/"reuse"/"reuse-demo", "reuse-check", "basic". '
            "If omitted, uses legacy env flags."
        ),
    )
    p.add_argument(
        "--clear-signifiers",
        action="store_true",
        help="Clear embedded RD4 signifier storage before starting (overrides CLEAR_SIGNIFIERS).",
    )
    p.add_argument(
        "--pause-for-sensor",
        action="store_true",
        help="(sequence 3/demo) Pause before IMPLICIT request (useful for live demos).",
    )
    p.add_argument(
        "--hold",
        action="store_true",
        help="Hold at the end of startup-only mode until you press Enter (useful for demos).",
    )
    p.add_argument(
        "--signifier-matcher",
        choices=("v0", "v1"),
        default=os.getenv("SIGNIFIER_MATCHER_VERSION", "").strip() or "v1",
        help=(
            "Intent matcher version for the embedded RD4 engine (default: v1). "
            "Can also be set via SIGNIFIER_MATCHER_VERSION."
        ),
    )
    p.add_argument(
        "--signifier-min-similarity",
        type=float,
        default=env_min_similarity_f if env_min_similarity_f is not None else 0.75,
        help=(
            "Minimum intent similarity threshold for signifier matching/reuse "
            "(default: 0.75). Can also be set via SIGNIFIER_MIN_SIMILARITY."
        ),
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Show only log lines that include the [DEMO] prefix (filters all other logs).",
    )
    return p.parse_args(argv)


def _resolve_sequence(args: argparse.Namespace) -> str:
    """
    Resolve sequence name.

    CLI args take precedence; if --sequence is omitted, fall back to legacy env vars.
    """
    if args.sequence is None:
        if _env_flag("RUN_REUSE_FLOW"):
            return "reuse-check"
        if _env_flag("RUN_DEMO_SEQUENCE"):
            return "demo"
        if _env_flag("RUN_STARTUP_SEQUENCE"):
            return "startup"
        return "basic"

    s = str(args.sequence).strip().lower()
    if s in ("basic", "1"):
        return "basic"
    if s in ("startup", "boot", "2"):
        return "startup"
    if s in ("demo", "interactions", "3"):
        return "demo"
    if s in ("reuse-check", "reuse_only", "reuse-only", "reusecheck"):
        return "reuse-check"
    if s in ("reuse", "reuse-demo", "reuse_demo", "4"):
        return "reuse-demo"
    raise ValueError(f"Unknown --sequence value: {args.sequence!r}")


class AgentInstance:
    """Represents a complete agent instance with all three sub-agents."""

    def __init__(
        self,
        instance_id: str,
        env_explorer: EnvExplorerAgent,
        interaction_solver: InteractionSolverAgent,
        user_assistant: UserAssistantAgent
    ):
        self.instance_id = instance_id
        self.env_explorer = env_explorer
        self.interaction_solver = interaction_solver
        self.user_assistant = user_assistant

    @property
    def name(self) -> str:
        return self.instance_id

    async def start(self):
        logger.info(demo(f"Starting instance '{self.name}'..."))
        await asyncio.gather(
            self.env_explorer.start(),
            self.interaction_solver.start(),
            self.user_assistant.start(),
        )
        logger.info(demo(f"Instance '{self.name}' started successfully!"))

    async def stop(self):
        logger.info(demo(f"Stopping instance '{self.name}'..."))
        await asyncio.gather(
            self.env_explorer.stop(),
            self.interaction_solver.stop(),
            self.user_assistant.stop(),
            return_exceptions=True
        )
        logger.info(demo(f"Instance '{self.name}' stopped."))


async def create_agent_instance(instance_id: str, instance_index: int) -> AgentInstance:
    # Generate unique JIDs using short UUID
    env_jid = f"env_{instance_id}@{XMPP_SERVER}"
    solver_jid = f"solver_{instance_id}@{XMPP_SERVER}"
    assistant_jid = f"assistant_{instance_id}@{XMPP_SERVER}"
    args = _parse_args(sys.argv[1:])
    yggdrasil_url = os.getenv("YGGDRASIL_URL", "http://localhost:8080/").strip()

    if args.clear_signifiers or _env_flag("CLEAR_SIGNIFIERS"):
        storage_dir = PROJECT_ROOT / "ami_agents" / "shared" / "memory" / "storage"
        for subdir in ["rdf", "json", "indexes"]:
            subpath = storage_dir / subdir
            if subpath.exists():
                shutil.rmtree(subpath)
        logger.info("Cleared signifier storage at %s", storage_dir)

    sequence = _resolve_sequence(args)
    run_reuse_flow = sequence == "reuse-check"
    run_demo_sequence = sequence == "demo"
    run_reuse_demo_sequence = sequence == "reuse-demo"
    run_startup_sequence = sequence == "startup"

    logger.info(demo("Running manual test sequence=%s"), sequence)
    logger.info(demo("EnvExplorer entrypoint (Yggdrasil URL)=%s"), yggdrasil_url)
    await _reset_lab308_state()
    # CLI convenience: keep the internal demo sequence env toggles working.
    if args.pause_for_sensor:
        os.environ["PAUSE_FOR_SENSOR"] = "1"

    # EnvExplorer: configure discovery notifications (optional but helpful)
    env_config = {
        "yggdrasil_url": yggdrasil_url,
        "yggdrasil": {"url": yggdrasil_url},
        "discovery": {
            "notify_on_discovery_complete": True,
            "notify_agents": [solver_jid],
        },
        "signifiers": {
            "matcher_version": args.signifier_matcher,
            "min_similarity": float(args.signifier_min_similarity),
        },
    }
    if args.signifier_matcher or args.signifier_min_similarity is not None:
        env_config["signifiers"] = {
            **({"matcher_version": args.signifier_matcher} if args.signifier_matcher else {}),
            **({"min_similarity": float(
                args.signifier_min_similarity)} if args.signifier_min_similarity is not None else {}),
        }

    # UserAssistant and Solver configs (new agents.yaml-compatible structure)
    # Default model requested by user:
    model = os.getenv("OPENAI_MODEL", "o3")
    # This model name is typically served via OpenRouter's OpenAI-compatible API.
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        base_url = "https://openrouter.ai/api/v1" if ":" in model or "/" in model else "https://api.openai.com/v1"

    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip() or (
        "high" if model.startswith("o") and "openai.com" in base_url else ""
    )
    api_timeout = os.getenv("OPENAI_TIMEOUT", "").strip() or os.getenv("OPENAI_HTTP_TIMEOUT", "").strip()
    try:
        api_timeout_s = float(api_timeout) if api_timeout else (
            120.0 if model.startswith("o") and "openai.com" in base_url else 30.0)
    except Exception:
        api_timeout_s = 120.0 if model.startswith("o") and "openai.com" in base_url else 30.0

    try:
        planning_timeout_s = float(
            os.getenv("AMI_PLANNING_TIMEOUT", "").strip() or ("180" if model.startswith("o") else "90"))
    except Exception:
        planning_timeout_s = 180.0 if model.startswith("o") else 90.0

    # Keep orchestrator wait times aligned with planning timeout unless explicitly overridden.
    os.environ.setdefault("AMI_PLAN_TIMEOUT_S", str(planning_timeout_s))
    os.environ.setdefault("AMI_EXEC_TIMEOUT_S", str(max(planning_timeout_s, 180.0)))

    llm_cfg = {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": base_url,
                    "model": model,
                    **({} if model.startswith("o") else {"temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.5"))}),
                    **({} if model.startswith("o") else {"max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1500"))}),
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
                }
            },
            "retry": {"timeout": api_timeout_s},
        },
        "planning": {
            "timeout": planning_timeout_s,
            "llm_planning": {
                "model": model,
                **({} if model.startswith("o") else {"temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.5"))}),
                **({} if model.startswith("o") else {"max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1500"))}),
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            },
            "context_gathering": {"timeout": 60},
        },
    }
    # Target JIDs for inter-agent communication within this instance
    target_jids = {
        "explorer": env_jid,
        "solver": solver_jid,
        "assistant": assistant_jid,
    }

    logger.info(demo(f"Creating instance '{instance_id}':"))
    logger.info(demo(f"  EnvExplorer: {env_jid}, port: {8086 + instance_index}"))
    logger.info(demo(f"  InteractionSolver: {solver_jid}"))
    logger.info(demo(f"  UserAssistant: {assistant_jid}"))

    env_explorer = EnvExplorerAgent(
        jid=env_jid,
        password=XMPP_PASSWORD,
        config=env_config,
        hmas_client=DummyHMASClient(),
        integration_port = 8086 + instance_index
    )

    interaction_solver = InteractionSolverAgent(
        jid=solver_jid,
        password=XMPP_PASSWORD,
        target_jids=target_jids,
        config=llm_cfg,
        plan_generator=object()
    )

    user_assistant = UserAssistantAgent(
        jid=assistant_jid,
        password=XMPP_PASSWORD,
        target_jids=target_jids,
        config=llm_cfg,
    )

    return AgentInstance(
        instance_id=instance_id,
        env_explorer=env_explorer,
        interaction_solver=interaction_solver,
        user_assistant=user_assistant,
    )


async def main():

    instances: List[AgentInstance] = []

    logger.info(demo(f"Creating {NUM_INSTANCES} agent instances..."))
    instance_index = 0
    for _ in range(NUM_INSTANCES):
        instance_id = generate_short_uuid()
        instance = await create_agent_instance(instance_id, instance_index)
        instances.append(instance)
        instance_index += 1

    logger.info(demo(f"Created {len(instances)} instances:"))
    for inst in instances:
        logger.info(demo(f"  - {inst.name}"))

    logger.info(demo("Creating 'lighting' community for lab308 light control..."))

    member_ids = [instances[0].instance_id, instances[1].instance_id]

    lighting_affordance = AffordanceMatch(
        description="Community for controlling lighting in lab308 (turn on/off lights, adjust brightness)"
    )

    lighting_community = Community(
        affordance_match=lighting_affordance,
        community_id="lab308_lighting",
        member_ids=member_ids,
    )

    for i in range(2):
        instances[i].interaction_solver.communities.append(lighting_community)
        logger.info(demo(f"Agent '{instances[i].instance_id}' assigned to 'lighting' community"))

    logger.info(demo(f"Agent '{instances[2].instance_id}' is NOT in any community"))

    logger.info(demo(f"Lighting community created with {lighting_community.member_count} members:"))
    for member_id in lighting_community.member_ids:
        logger.info(demo(f"  - {member_id}"))


    logger.info(demo("Starting all agent instances..."))
    await asyncio.gather(*[inst.start() for inst in instances])

    logger.info(demo("All instances started! Summary:"))
    for inst in instances:
        logger.info(demo(f"  [{inst.name}]"))
        logger.info(demo(f"    EnvExplorer: {inst.env_explorer.jid}"))
        logger.info(demo(f"    InteractionSolver: {inst.interaction_solver.jid}"))
        logger.info(demo(f"    UserAssistant: {inst.user_assistant.jid}"))
        logger.info(demo(f"    Communities: {len(inst.interaction_solver.communities)}"))
    logger.info(demo("Waiting for environment discovery on all instances..."))
    await asyncio.sleep(5)
    logger.info(demo("COMMUNITY STATUS CHECK:"))
    logger.info(demo(f"Community 'lighting' has {lighting_community.member_count} members"))
    for inst in instances:
        solver_communities = len(inst.interaction_solver.communities)
        in_lighting = lighting_community in inst.interaction_solver.communities
        logger.info(demo(f"  Instance '{inst.name}': {solver_communities} community/communities, in lighting={in_lighting}"))

if __name__ == "__main__":
    spade.run(main())

