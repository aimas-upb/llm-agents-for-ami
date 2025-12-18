"""\
Manual end-to-end flow: UserAssistant -> InteractionSolver -> EnvExplorer.

Flow:
1) Start EnvExplorerAgent (REAL YggdrasilIntegration).
2) Start UserAssistantAgent.
3) Start InteractionSolverAgent.
4) Ask UserAssistant what workspaces are available.
5) Send a workspace-scoped request for lab308 (where light308 is).
6) UserAssistant proposes a plan and asks for confirmation.
7) Send "yes" to confirm and observe execution report.

Run:
    python tests/manual_test_full_flow_plan.py

Prereqs:
- A running XMPP server (SPADE built-in server or external), typically on localhost:5222.
- OPENAI_API_KEY set (both UserAssistant and InteractionSolver call an OpenAI-compatible API).
- A running Yggdrasil instance (default http://localhost:8080/ or set YGGDRASIL_URL).

Notes:
- This test defaults to the OpenRouter model `mistralai/devstral-2512:free`.
- If you are not using OpenRouter, set OPENAI_BASE_URL + OPENAI_MODEL accordingly.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import spade
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message as SpadeMessage

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
from ami_agents.environment.connection.hmas_client import IHMASClient


class _DropClockTimeOfDaySpamFilter(logging.Filter):
    """Drop ultra-frequent clock timeOfDay state updates to keep console readable."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (filter name is logging API)
        try:
            msg = record.getMessage()
        except Exception:
            return True
        # Example:
        # "STATE UPDATE: clock308 -> http://.../clock308/props/timeOfDay = 18:03"
        if "STATE UPDATE:" in msg and "clock308" in msg and "/props/timeOfDay" in msg:
            return False
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
    )

    # Keep our high-signal logs.
    for name in [
        "ManualFullFlowPlan",
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

    # Drop only clock timeOfDay spam, keep other state changes (e.g., light updates).
    filt = _DropClockTimeOfDaySpamFilter()
    root = logging.getLogger()
    for h in list(getattr(root, "handlers", []) or []):
        h.addFilter(filt)


_configure_console_logging()
logger = logging.getLogger("ManualFullFlowPlan")


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


class OrchestratorAgent(Agent):
    """Simulates a user: list workspaces -> scoped request -> approve -> observe execution."""

    def __init__(self, jid: str, password: str, *, assistant_jid: str):
        super().__init__(jid, password)
        self.assistant_jid = assistant_jid
        self.thread_id = str(uuid.uuid4())
        self.first_reply: Optional[str] = None
        self.second_reply: Optional[str] = None

    async def setup(self):
        self.add_behaviour(self.RunFlow())

    class RunFlow(OneShotBehaviour):
        async def run(self):
            list_ws_query = "What workspaces are available?"
            scoped_query = "In lab308, it's too dark in here."

            # 1) Ask for workspaces
            logger.info('Asking UserAssistant for workspaces: "%s"', list_ws_query)
            msg1 = SpadeMessage(to=self.agent.assistant_jid)
            msg1.set_metadata("message_type", "llm")
            msg1.thread = self.agent.thread_id
            msg1.body = list_ws_query
            await self.send(msg1)

            # 2) Wait for workspace list reply
            deadline = asyncio.get_running_loop().time() + 60.0
            reply = None
            while asyncio.get_running_loop().time() < deadline:
                candidate = await self.receive(timeout=1)
                if not candidate:
                    continue
                # match thread and sender (ignore XMPP resource part)
                if getattr(candidate, "thread", None) != self.agent.thread_id:
                    continue
                if not str(candidate.sender).startswith(self.agent.assistant_jid):
                    continue
                reply = candidate
                break

            if not reply:
                print("\nERROR: Timed out waiting for workspace list reply from UserAssistant.\n")
                await self.agent.stop()
                return

            self.agent.first_reply = reply.body or ""
            print("\n" + "=" * 60)
            print("[UserAssistant Workspace Reply]")
            print("-" * 60)
            print(self.agent.first_reply)
            print("=" * 60 + "\n")

            # 3) Send a workspace-scoped request (lab308 contains light308 in your environment)
            logger.info('Sending scoped request to UserAssistant: "%s"', scoped_query)
            msg2 = SpadeMessage(to=self.agent.assistant_jid)
            msg2.set_metadata("message_type", "llm")
            msg2.thread = self.agent.thread_id
            msg2.body = scoped_query
            await self.send(msg2)

            # 4) Wait for plan proposal (assistant asks for confirmation)
            deadline = asyncio.get_running_loop().time() + 90.0
            reply2 = None
            while asyncio.get_running_loop().time() < deadline:
                candidate = await self.receive(timeout=1)
                if not candidate:
                    continue
                if getattr(candidate, "thread", None) != self.agent.thread_id:
                    continue
                if not str(candidate.sender).startswith(self.agent.assistant_jid):
                    continue
                reply2 = candidate
                break

            if not reply2:
                print("\nERROR: Timed out waiting for plan proposal from UserAssistant.\n")
                await self.agent.stop()
                return

            plan_proposal = reply2.body or ""
            print("\n" + "=" * 60)
            print("[UserAssistant Plan Proposal]")
            print("-" * 60)
            print(plan_proposal)
            print("=" * 60 + "\n")

            # 5) Approve the proposed plan
            logger.info('Sending approval ("yes") to UserAssistant...')
            approve = SpadeMessage(to=self.agent.assistant_jid)
            approve.set_metadata("message_type", "llm")
            approve.thread = self.agent.thread_id
            approve.body = "yes"
            await self.send(approve)

            # 6) Wait for execution confirmation/report
            deadline = asyncio.get_running_loop().time() + 120.0
            reply3 = None
            while asyncio.get_running_loop().time() < deadline:
                candidate = await self.receive(timeout=1)
                if not candidate:
                    continue
                if getattr(candidate, "thread", None) != self.agent.thread_id:
                    continue
                if not str(candidate.sender).startswith(self.agent.assistant_jid):
                    continue
                reply3 = candidate
                break

            if not reply3:
                print("\nERROR: Timed out waiting for execution result from UserAssistant.\n")
                await self.agent.stop()
                return

            self.agent.second_reply = reply3.body or ""
            print("\n" + "=" * 60)
            print("[UserAssistant Execution Reply]")
            print("-" * 60)
            print(self.agent.second_reply)
            print("=" * 60 + "\n")

            await self.agent.stop()


async def main():
    xmpp_server = os.getenv("SPADE_SERVER", "localhost")
    password = os.getenv("SPADE_PASSWORD", "password")

    # Require LLM API key (both UA and solver will call it).
    _require_env("OPENAI_API_KEY")

    yggdrasil_url = os.getenv("YGGDRASIL_URL", "http://localhost:8080/").strip()

    explorer_jid = f"env_explorer@{xmpp_server}"
    assistant_jid = f"user_assistant@{xmpp_server}"
    solver_jid = f"interaction_solver@{xmpp_server}"
    orchestrator_jid = f"orchestrator@{xmpp_server}"

    # EnvExplorer: configure discovery notifications (optional but helpful)
    env_config = {
        "yggdrasil_url": yggdrasil_url,
        "yggdrasil": {"url": yggdrasil_url},
        "discovery": {
            "notify_on_discovery_complete": True,
            "notify_agents": [solver_jid],
        }
    }

    # UserAssistant and Solver configs (new agents.yaml-compatible structure)
    # Default model requested by user:
    model = os.getenv("OPENAI_MODEL", "mistralai/devstral-2512:free")
    # This model name is typically served via OpenRouter's OpenAI-compatible API.
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        base_url = "https://openrouter.ai/api/v1" if ":" in model or "/" in model else "https://api.openai.com/v1"

    llm_cfg = {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": base_url,
                    "model": model,
                    "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.5")),
                    "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1500")),
                }
            },
            "retry": {"timeout": 30},
        },
        "planning": {
            "llm_planning": {
                "model": model,
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.5")),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1500")),
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

    orchestrator = OrchestratorAgent(
        orchestrator_jid,
        password,
        assistant_jid=assistant_jid,
    )

    try:
        # Preflight: ensure Yggdrasil reachable (helps diagnose \"no artifacts\" / discovery hang).
        logger.info("Checking Yggdrasil reachable at %s ...", yggdrasil_url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(yggdrasil_url) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"Yggdrasil returned HTTP {resp.status}")
        except Exception as e:
            raise RuntimeError(f"Yggdrasil is not reachable at {yggdrasil_url}: {e}") from e

        logger.info("Starting EnvExplorer...")
        await explorer.start(auto_register=True)

        logger.info("Starting UserAssistant...")
        await assistant.start(auto_register=True)

        logger.info("Starting InteractionSolver...")
        await solver.start(auto_register=True)

        # Wait for discovery to finish before sending the user query (so tool calls see real capabilities).
        for _ in range(600):
            if getattr(explorer, "discovery_complete", False):
                break
            await asyncio.sleep(0.5)
        if not getattr(explorer, "discovery_complete", False):
            raise RuntimeError("EnvExplorer discovery did not complete in time.")

        logger.info("Starting Orchestrator (drives the test)...")
        await orchestrator.start(auto_register=True)

        while orchestrator.is_alive():
            await asyncio.sleep(0.5)

    finally:
        logger.info("Stopping agents...")
        try:
            await orchestrator.stop()
        except Exception:
            pass
        try:
            await solver.stop()
        except Exception:
            pass
        try:
            await assistant.stop()
        except Exception:
            pass
        try:
            await explorer.stop()
        except Exception:
            pass

        await asyncio.sleep(1)


if __name__ == "__main__":
    spade.run(main())
