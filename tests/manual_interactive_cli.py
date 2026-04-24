"""\
Interactive CLI driver for AMI agents.

Usage:
    python tests/manual_interactive_cli.py [options]

This script starts EnvExplorer, InteractionSolver, and UserAssistant agents,
then opens a simple console prompt. Anything you type is sent to the
UserAssistant (as if you were chatting with it via SPADE); responses return to
the console. Type "exit" or "quit" to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file automatically from project root
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)

from ami_agents.shared.utils import spade_compat  # noqa: F401
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
from ami_agents.environment.connection.hmas_client import IHMASClient
from ami_agents.shared.utils.demo_log import demo


class _HighFrequencyFilter(logging.Filter):
    """Drop ultra-frequent sensor state updates to keep console readable."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "STATE UPDATE:" in msg:
            if "clock308" in msg and "/props/timeOfDay" in msg:
                return False
            if "lightSensor308" in msg and "/props/luminosity" in msg:
                return False
            if "peoplePresenceSensor308" in msg and "/props/presence" in msg:
                return False
        return True


class _DemoOnlyFilter(logging.Filter):
    """Allow only log entries containing the [DEMO] prefix."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:
            return False
        return "[DEMO]" in msg


def _configure_logging(*, demo_only: bool, no_logs: bool) -> None:
    if no_logs:
        logging.disable(logging.CRITICAL)
        return

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    for name in [
        "ManualInteractiveCLI",
        "UserAssistant",
        "InteractionSolver",
        "ami_agents.agents.env_explorer.env_explorer_agent",
        "ami_agents.environment.integration.integration_engine",
    ]:
        logging.getLogger(name).setLevel(logging.INFO)

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

    filt = _HighFrequencyFilter()
    root = logging.getLogger()
    root.addFilter(filt)
    logging.getLogger("ami_agents.agents.env_explorer.env_explorer_agent").addFilter(filt)
    for handler in list(getattr(root, "handlers", []) or []):
        handler.addFilter(filt)

    if demo_only:
        demo_filter = _DemoOnlyFilter()
        root.addFilter(demo_filter)
        for handler in list(getattr(root, "handlers", []) or []):
            handler.addFilter(demo_filter)


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


class ConsoleUserAgent(Agent):
    """Tiny SPADE agent that proxies console input/output to UserAssistant."""

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        assistant_jid: str,
        response_timeout: float,
    ):
        super().__init__(jid, password)
        self.assistant_jid = assistant_jid
        self.thread_id = str(uuid.uuid4())
        self.response_timeout = float(response_timeout)
        self.reply_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stop_requested = False

    async def setup(self):
        self.add_behaviour(self.ConsoleLoop())
        t = Template()
        t.set_metadata("message_type", "llm")
        t.thread = self.thread_id
        self.add_behaviour(self.ReceiveLLMReply(), template=t)

    class ReceiveLLMReply(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return
            await self.agent.reply_queue.put(msg.body or "")

    class ConsoleLoop(CyclicBehaviour):
        async def run(self):
            if self.agent._stop_requested:
                await asyncio.sleep(0.1)
                return

            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(None, input, "You> ")
            except (EOFError, KeyboardInterrupt):
                text = "exit"

            if text is None:
                return

            normalized = text.strip()
            if not normalized:
                return

            if normalized.lower() in {"exit", "quit", "q"}:
                print("Exiting interactive session...")
                self.agent._stop_requested = True
                await self.agent.stop()
                return

            msg = SpadeMessage(to=self.agent.assistant_jid)
            msg.set_metadata("message_type", "llm")
            msg.thread = self.agent.thread_id
            msg.body = normalized
            await self.send(msg)

            print("Assistant> (thinking...)")
            try:
                reply = await asyncio.wait_for(
                    self.agent.reply_queue.get(),
                    timeout=self.agent.response_timeout,
                )
            except asyncio.TimeoutError:
                print("Assistant> No reply received within timeout.")
                return

            print("\nAssistant>\n" + reply + "\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI for AMI agents.")
    parser.add_argument("--demo", action="store_true", help="Show only [DEMO] log messages.")
    parser.add_argument(
        "--clear-signifiers",
        action="store_true",
        help="Clear embedded RD4 signifier storage before starting.",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for a UserAssistant reply (default: 120).",
    )
    parser.add_argument(
        "--skip-prewarm",
        action="store_true",
        help="Skip pre-warming the UserAssistant execution engine.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Suppress all logging output; only the console dialogue remains.",
    )
    return parser.parse_args(argv)


async def main():
    args = _parse_args(sys.argv[1:])
    _configure_logging(demo_only=args.demo, no_logs=args.no_logs)
    logger = logging.getLogger("ManualInteractiveCLI")

    xmpp_server = os.getenv("SPADE_SERVER", "localhost")
    password = os.getenv("SPADE_PASSWORD", "password")
    yggdrasil_url = os.getenv("YGGDRASIL_URL", "http://localhost:8080/").strip()

    _require_env("OPENAI_API_KEY")

    explorer_jid = f"env_explorer@{xmpp_server}"
    assistant_jid = f"user_assistant@{xmpp_server}"
    solver_jid = f"interaction_solver@{xmpp_server}"
    console_jid = f"console_cli@{xmpp_server}"

    if args.clear_signifiers:
        storage_dir = PROJECT_ROOT / "ami_agents" / "shared" / "memory" / "storage"
        for subdir in ["rdf", "json", "indexes"]:
            subpath = storage_dir / subdir
            if subpath.exists():
                shutil.rmtree(subpath)
        logger.info("Cleared signifier storage at %s", storage_dir)

    logger.info(demo("Starting interactive CLI (Yggdrasil=%s)"), yggdrasil_url)

    env_config = {
        "yggdrasil_url": yggdrasil_url,
        "yggdrasil": {"url": yggdrasil_url},
        "discovery": {
            "notify_on_discovery_complete": True,
            "notify_agents": [solver_jid],
        },
    }

    model = os.getenv("OPENAI_MODEL", "o3")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        base_url = "https://openrouter.ai/api/v1" if ":" in model or "/" in model else "https://api.openai.com/v1"

    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip() or (
        "high" if model.startswith("o") and "openai.com" in base_url else ""
    )
    api_timeout = os.getenv("OPENAI_TIMEOUT", "").strip() or os.getenv("OPENAI_HTTP_TIMEOUT", "").strip()
    try:
        api_timeout_s = float(api_timeout) if api_timeout else (
            120.0 if model.startswith("o") and "openai.com" in base_url else 30.0
        )
    except Exception:
        api_timeout_s = 120.0 if model.startswith("o") and "openai.com" in base_url else 30.0

    # Load configuration for provider settings
    try:
        config = ConfigLoader.merge_configs(
            ConfigLoader.load_with_env_vars("ami_agents/config/agents.yaml"),
            ConfigLoader.load_with_env_vars("ami_agents/config/environment.yaml")
        )
        llm_config = config.get("llm", {})
        provider_name = llm_config.get("default_provider", "openai")
        provider_cfg = llm_config.get("providers", {}).get(provider_name, {})
    except:
        # Fallback if config loading fails
        provider_cfg = {"temperature": 0.7, "max_tokens": 1500}

    llm_cfg = {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": base_url,
                    "model": model,
                    **({} if model.startswith("o") else {"temperature": float(os.getenv("OPENAI_TEMPERATURE") or str(provider_cfg.get("temperature", 0.7)))}),
                    **({} if model.startswith("o") else {"max_tokens": int(os.getenv("OPENAI_MAX_TOKENS") or str(provider_cfg.get("max_tokens", 1500)))}),
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
                }
            },
            "retry": {"timeout": api_timeout_s},
        },
        "planning": {
            "timeout": float(os.getenv("AMI_PLANNING_TIMEOUT", "180" if model.startswith("o") else "90")),
            "llm_planning": {
                "model": model,
                **({} if model.startswith("o") else {"temperature": float(os.getenv("OPENAI_TEMPERATURE") or str(provider_cfg.get("temperature", 0.7)))}),
                **({} if model.startswith("o") else {"max_tokens": int(os.getenv("OPENAI_MAX_TOKENS") or str(provider_cfg.get("max_tokens", 1500)))}),
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            },
            "context_gathering": {"timeout": 10},
        },
    }

    explorer = EnvExplorerAgent(explorer_jid, password, env_config, hmas_client=DummyHMASClient())
    assistant = UserAssistantAgent(
        assistant_jid,
        password,
        config=llm_cfg,
        target_jids={"explorer": explorer_jid, "solver": solver_jid},
    )
    solver = InteractionSolverAgent(
        solver_jid,
        password,
        config=llm_cfg,
        plan_generator=object(),
        target_jids={"explorer": explorer_jid},
    )
    console = ConsoleUserAgent(
        console_jid,
        password,
        assistant_jid=assistant_jid,
        response_timeout=args.response_timeout,
    )

    try:
        logger.info("Checking Yggdrasil reachable at %s ...", yggdrasil_url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(yggdrasil_url) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"Yggdrasil returned HTTP {resp.status}")
        except Exception as e:
            raise RuntimeError(f"Yggdrasil is not reachable at {yggdrasil_url}: {e}") from e

        # InteractionSolver must start FIRST to register behaviors that can receive
        # the ENV_DISCOVERY_COMPLETE notification sent by EnvExplorer during discovery.
        logger.info("Starting InteractionSolver...")
        await solver.start(auto_register=True)

        logger.info("Starting EnvExplorer...")
        await explorer.start(auto_register=True)

        logger.info("Starting UserAssistant...")
        await assistant.start(auto_register=True)

        for _ in range(600):
            if getattr(explorer, "discovery_complete", False):
                break
            await asyncio.sleep(0.5)
        if not getattr(explorer, "discovery_complete", False):
            raise RuntimeError("EnvExplorer discovery did not complete in time.")

        if not args.skip_prewarm:
            logger.info(demo("Pre-warming UserAssistant execution engine (YggdrasilIntegration)..."))
            integ_logger = logging.getLogger("ami_agents.environment.integration.integration_engine")
            prev_level = integ_logger.level
            try:
                integ_logger.setLevel(logging.WARNING)
                await assistant.ensure_execution_engine_ready()
                logger.info(demo("UserAssistant execution engine ready."))
            finally:
                integ_logger.setLevel(prev_level)

        logger.info("Starting console agent. Type messages and press Enter. Use 'exit' to quit.")
        await console.start(auto_register=True)

        while console.is_alive():
            await asyncio.sleep(0.5)

    finally:
        logger.info("Stopping agents...")
        for agent in [console, assistant, solver, explorer]:
            if not agent:
                continue
            try:
                await agent.stop()
            except Exception:
                pass
        await asyncio.sleep(1)


if __name__ == "__main__":
    spade.run(main())
