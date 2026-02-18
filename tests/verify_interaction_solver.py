"""
Verification of InteractionSolverAgent and EnvExplorerAgent Integration using REAL LLM.

This script:
1. Checks if Yggdrasil and XMPP server are reachable.
2. Starts EnvExplorerAgent to discover the environment.
3. Starts InteractionSolverAgent (with REAL LLM).
4. Uses a ProbeAgent to send a GOAL_REQUEST ("turn on the light").
5. Verifies that InteractionSolver returns a valid PLAN_CREATED message with JSON-Plan 1.2.

Run with:
    python tests/verify_interaction_solver.py
"""

import asyncio
import logging
import sys
import json
import os
import uuid
import aiohttp
from pathlib import Path
from typing import Any, Dict

# 1. Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spade
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template

from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.shared.models.messages import MessageType
from ami_agents.environment.connection.hmas_client import IHMASClient

# --- Configuration ---
YGGDRASIL_URL = os.getenv("YGGDRASIL_URL", "http://localhost:8080/")
XMPP_SERVER = os.getenv("SPADE_SERVER", "localhost")
XMPP_PASSWORD = os.getenv("SPADE_PASSWORD", "password")
XMPP_PORT = int(os.getenv("SPADE_PORT", "5222"))

# LLM settings (OpenAI-compatible)
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("OPENAI_MODEL", "o3")
LLM_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip() or (
    "high" if LLM_MODEL.startswith("o") and "openai.com" in LLM_BASE_URL else ""
)
LLM_TIMEOUT = os.getenv("OPENAI_TIMEOUT", "").strip() or os.getenv("OPENAI_HTTP_TIMEOUT", "").strip()
try:
    LLM_TIMEOUT_S = float(LLM_TIMEOUT) if LLM_TIMEOUT else (120.0 if LLM_MODEL.startswith("o") and "openai.com" in LLM_BASE_URL else 30.0)
except Exception:
    LLM_TIMEOUT_S = 120.0 if LLM_MODEL.startswith("o") and "openai.com" in LLM_BASE_URL else 30.0
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")

EXPLORER_JID = f"env_explorer@{XMPP_SERVER}"
SOLVER_JID = f"interaction_solver@{XMPP_SERVER}"
PROBE_JID = f"probe@{XMPP_SERVER}"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("InteractionSolverTest")
# Increase verbosity for solver to see LLM interaction
logging.getLogger("InteractionSolver").setLevel(logging.INFO)

# --- Mocks and Helpers ---

class _DummyPlanGenerator:
    """Placeholder plan generator."""
    pass

class _DummyHMASClient(IHMASClient):
    """Minimal HMAS client to satisfy EnvExplorer constructor."""
    async def connect(self, endpoint_url: str) -> bool: return True
    async def disconnect(self) -> None: return None
    async def get_workspace(self, workspace_id: str): return None
    async def list_workspaces(self, parent_id: str = None): return []
    async def get_artifact(self, artifact_id: str): return None
    async def list_artifacts(self, workspace_id: str): return []
    async def get_thing_description(self, artifact_id: str): return None
    async def invoke_action(self, artifact_id: str, action_name: str, params: Dict[str, Any]): return {}
    async def read_property(self, artifact_id: str, property_name: str): return {}
    async def write_property(self, artifact_id: str, property_name: str, value: Any) -> bool: return True
    async def subscribe_to_event(self, artifact_id: str, event_name: str, callback: callable) -> str: return "sub"
    async def unsubscribe_from_event(self, subscription_id: str) -> bool: return True
    async def subscribe_to_workspace_changes(self, workspace_id: str, callback: callable) -> str: return "sub"
    async def crawl_environment(self, root_workspace_id: str, depth: int = -1): return {}

async def _check_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        conn = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(conn, timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False

def _default_config():
    if not LLM_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    return {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "api_key": LLM_API_KEY,
                    "base_url": LLM_BASE_URL,
                    "model": LLM_MODEL,
                    **({} if LLM_MODEL.startswith("o") else {"temperature": 0.5}),  # more deterministic JSON
                    **({} if LLM_MODEL.startswith("o") else {"max_tokens": 1500}),
                    **({"reasoning_effort": LLM_REASONING_EFFORT} if LLM_REASONING_EFFORT else {}),
                }
            },
            "retry": {"timeout": LLM_TIMEOUT_S},
        },
        "planning": {
            "llm_planning": {
                "model": LLM_MODEL,
                **({} if LLM_MODEL.startswith("o") else {"temperature": 0.5}),
                **({} if LLM_MODEL.startswith("o") else {"max_tokens": 1500}),
                **({"reasoning_effort": LLM_REASONING_EFFORT} if LLM_REASONING_EFFORT else {}),
            },
            "context_gathering": {"timeout": 10},
        },
    }

class PlanProbeAgent(Agent):
    """Helper agent that sends GOAL_REQUEST and awaits PLAN_CREATED."""
    def __init__(self, jid, password, solver_jid, goal_text):
        super().__init__(jid, password)
        self.solver_jid = solver_jid
        self.goal_text = goal_text
        self.plan_reply = None
        self.plan_received = False
        self.request_thread = str(uuid.uuid4())

    async def setup(self):
        self.add_behaviour(self.SendGoalBehaviour())
        t = Template()
        t.set_metadata("type", MessageType.PLAN_CREATED.value)
        # Ensure we only accept the reply to OUR request
        t.thread = self.request_thread
        self.add_behaviour(self.ReceivePlanBehaviour(), t)

    class SendGoalBehaviour(OneShotBehaviour):
        async def run(self):
            logger.info(f"Sending GOAL_REQUEST: {self.agent.goal_text}")
            msg = SpadeMessage(to=self.agent.solver_jid)
            msg.set_metadata("type", MessageType.GOAL_REQUEST.value)
            msg.thread = self.agent.request_thread
            msg.body = json.dumps({"intents": [{"action": "unknown", "artifact": "unknown", "intent_text": self.agent.goal_text}]})
            await self.send(msg)

    class ReceivePlanBehaviour(CyclicBehaviour):
        async def run(self):
            # Increased timeout significantly because real LLM + tools takes time
            msg = await self.receive(timeout=45)
            if msg:
                logger.info(f"Received PLAN_CREATED from {msg.sender}")
                # Keep empty strings too, but mark that a message was received.
                self.agent.plan_reply = msg.body if msg.body is not None else ""
                self.agent.plan_received = True
                await self.agent.stop()

async def main():
    print("\n" + "="*60)
    print("VERIFICATION: Interaction Solver Plan Generation (REAL LLM)")
    print("="*60 + "\n")

    # 1. Pre-flight Yggdrasil Check
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(YGGDRASIL_URL) as resp:
                if resp.status != 200:
                    logger.error(f"Yggdrasil reachable but returned {resp.status}")
                    return
    except Exception as e:
        logger.error(f"Yggdrasil is not reachable at {YGGDRASIL_URL}. Aborting.")
        return

    # 1b. Pre-flight XMPP Check (helps debug 'agent never connects' issues)
    if not await _check_tcp(XMPP_SERVER, XMPP_PORT, timeout=3.0):
        logger.error(f"XMPP server is not reachable at {XMPP_SERVER}:{XMPP_PORT}. Aborting.")
        return

    # 2. Setup Agents
    env_config = {
        "yggdrasil_url": YGGDRASIL_URL,
        "yggdrasil": {"url": YGGDRASIL_URL},
        # Make sure EnvExplorer sends ENV_DISCOVERY_COMPLETE to the solver.
        "discovery": {
            "notify_on_discovery_complete": True,
            "notify_agents": [SOLVER_JID],
        },
    }

    env_explorer = EnvExplorerAgent(
        EXPLORER_JID,
        XMPP_PASSWORD,
        env_config,
        hmas_client=_DummyHMASClient()
    )

    solver = InteractionSolverAgent(
        SOLVER_JID,
        XMPP_PASSWORD,
        config=_default_config(),
        plan_generator=_DummyPlanGenerator(),
        target_jids={"explorer": EXPLORER_JID}
    )

    # 3. Start System
    probe = None
    try:
        logger.info("Starting agents...")
        await env_explorer.start(auto_register=True)
        await solver.start(auto_register=True)

        # Wait for potential discovery
        logger.info("Waiting for agent initialization (Discovery)...")
        # Give Explorer enough time to crawl Yggdrasil (can be slow on first RDF parse)
        for _ in range(60):
            if getattr(env_explorer, "discovery_complete", False):
                break
            await asyncio.sleep(1)
        
        logger.info(f"Discovery Complete? {getattr(env_explorer, 'discovery_complete', False)}")
        if not getattr(env_explorer, "discovery_complete", False):
            logger.error("EnvExplorer discovery did not complete in time; solver tools will likely fail. Aborting.")
            return

        # 4. Probe
        probe = PlanProbeAgent(PROBE_JID, XMPP_PASSWORD, SOLVER_JID, "increase the luminosity in the room")
        await probe.start(auto_register=True)

        # Wait for reply
        logger.info("Waiting for plan response (Timeout: 45s)...")
        while probe.is_alive():
            await asyncio.sleep(0.5)

        # 5. Verify
        if probe.plan_received:
            raw = probe.plan_reply if probe.plan_reply is not None else ""
            print("\n" + "-"*40)
            print("📦 [RECEIVED PLAN RAW]:")
            print(raw)
            print("-"*40 + "\n")

            try:
                plan = json.loads(raw)
                print("\n" + "-"*40)
                print("📦 [RECEIVED JSON PLAN]:")
                print(json.dumps(plan, indent=2))
                print("-"*40 + "\n")

                if plan.get("plan_version") == "1.2" and "steps" in plan:
                    print("\n" + "="*60)
                    print("✅ TEST PASSED: Received valid JSON-Plan 1.2.")
                    print("="*60 + "\n")
                else:
                    print("\n" + "="*60)
                    print(f"❌ TEST FAILED: Invalid plan format.")
                    print("="*60 + "\n")
            except json.JSONDecodeError:
                print("\n" + "="*60)
                print("❌ TEST FAILED: Response was not JSON (see raw above).")
                print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("❌ TEST FAILED: Timed out waiting for PLAN_CREATED.")
            print("="*60 + "\n")

    finally:
        logger.info("Shutting down agents...")
        if probe: await probe.stop()
        await solver.stop()
        await env_explorer.stop()

if __name__ == "__main__":
    spade.run(main())
