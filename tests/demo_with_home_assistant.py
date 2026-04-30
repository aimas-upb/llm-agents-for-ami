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

Sequences (recommended via CLI):
    python tests/manual_test_full_flow_plan.py --sequence 2
        Startup-only (thesis sequence 2): starts UserAssistant + EnvExplorer + InteractionSolver,
        shows the boot/discovery logs, then exits (no user interactions).

    python tests/manual_test_full_flow_plan.py --sequence 3
        Interaction demo (thesis sequence 3):
        1) list all active devices
        2) show state of light308
        3) EXPLICIT request (approve + execute + record signifiers)
        4) IMPLICIT request (approve + execute + record signifier)

    python tests/manual_test_full_flow_plan.py --sequence 4
        Reuse demo (thesis sequence 4):
        - Simulates an external reset (light off, blinds mostly closed)
        - Repeats an IMPLICIT request with a PRIOR signifier present
        - Expects InteractionSolver to reuse signifiers (no ENV_CAPABILITIES/ENV_STATE query in solver)
        - Approves and executes the recovered plan WITHOUT recording a new signifier.

    python tests/manual_test_full_flow_plan.py --sequence 5
        Intent Type Classification Test (sequence 5):
        - Tests IMPLICIT goal: "Turn on a light" (vague, no artifact ID, system infers)
        - Tests IMPLICIT goal: "Turn on the light" (vague, no artifact ID, system infers)
        - Tests EXPLICIT goal: "Toggle light308" (exact artifact ID specified)
        - Verifies logging shows correct intent_type classification
        - Demonstrates difference in matching behavior (context-aware vs context-free)

    python tests/manual_test_full_flow_plan.py --sequence reuse-check
        Reuse-check only: does NOT start UserAssistant; verifies solver reused expected signifier affordance.

    python tests/manual_test_full_flow_plan.py --sequence basic
        Minimal flow kept for quick debugging.

 Extra CLI flags:
     --clear-signifiers
     --pause-for-sensor
     --hold
     --demo
        Filters console logging so only `[DEMO]` messages are shown (useful during live demos).

 Optional env flags (legacy; CLI preferred):
     CLEAR_SIGNIFIERS=1
         Clears embedded Experience Engine signifier storage before starting (reproducible run).

    RUN_STARTUP_SEQUENCE=1
        Startup-only mode (equivalent to `--sequence 2`).

    RUN_DEMO_SEQUENCE=1
        Runs a longer thesis-demo friendly sequence:
        1) list devices
        2) show state of light308
        3) EXPLICIT request (execute + record signifiers)
        4) IMPLICIT request (plan proposal; no execution by default)

     PAUSE_FOR_SENSOR=1
         In demo sequence only: pauses before the IMPLICIT request (useful for live demos; no external UI required).
 
     RUN_REUSE_FLOW=1
         Runs ONLY the reuse check flow (equivalent to `--sequence reuse-check`; does not run the UserAssistant conversation).
         Expects signifiers to already exist from a prior run (or manual seeding).

Timeout tuning (useful for `o*` reasoning models):
    OPENAI_TIMEOUT=240
        Increases the HTTP timeout used for OpenAI calls (default: 120s for `o*` on api.openai.com).

    AMI_PLANNING_TIMEOUT=300
        Increases the RPC timeout waiting for the solver to return a plan (default: 180s for `o*`).

Prereqs:
- A running XMPP server (SPADE built-in server or external), typically on localhost:5222.
- OPENAI_API_KEY set (both UserAssistant and InteractionSolver call an OpenAI-compatible API).
- A running Yggdrasil instance (default http://localhost:8080/ or set YGGDRASIL_URL).

Notes:
- This test defaults to the OpenAI model `o3` (unless overridden by OPENAI_MODEL).
- For reasoning models (`o*`), this test sets `reasoning_effort=high` by default (override via OPENAI_REASONING_EFFORT).
- If you are using a non-OpenAI provider, set OPENAI_BASE_URL + OPENAI_MODEL accordingly.
"""

import asyncio
import argparse
import json
import logging
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import spade
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message as SpadeMessage
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file automatically from project root
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)

from ami_agents.shared.utils import spade_compat  # noqa: F401
from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
from ami_agents.environment.connection.hmas_client import IHMASClient
from ami_agents.shared.models.messages import MessageType
from ami_agents.shared.utils.demo_log import demo
from ami_agents.shared.utils.spade_rpc import RpcTimeoutError, rpc_call
from ami_agents.shared.utils.config_loader import ConfigLoader


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
logger = logging.getLogger("ManualFullFlowPlan")

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
    """Ensure lights are off and blinds at 50% before the demo starts."""
    light_off_url = f"{LAB308_BASE}/artifacts/lights_308/ha/light/turn_off"
    blinds_url = f"{LAB308_BASE}/artifacts/blinds_308/ha/cover/set_cover_position"
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(light_off_url, json={})
            await session.post(blinds_url, json={"position": 50})
        logger.info("Reset lab308 state: lights_308 off, blinds_308 50%.")
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
            'Which sequence to run: "2"/"startup", "3"/"demo", "4"/"reuse"/"reuse-demo", "5"/"intent-type", "reuse-check", "basic". '
            "If omitted, uses legacy env flags."
        ),
    )
    p.add_argument(
        "--clear-signifiers",
        action="store_true",
        help="Clear embedded Experience Engine signifier storage before starting (overrides CLEAR_SIGNIFIERS).",
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
        choices=("v0", "v1", "v2"),
        default=os.getenv("SIGNIFIER_MATCHER_VERSION", "").strip() or "v1",
        help=(
            "Intent matcher version for the embedded Experience Engine engine (default: v1). "
            "v0=string contains, v1=embeddings, v2=structured intent. "
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
    if s in ("intent-type", "intent-test", "classification", "5"):
        return "intent-type-test"
    if s in ("toggle-only", "toggle-test", "6"):
        return "toggle-only-test"
    raise ValueError(f"Unknown --sequence value: {args.sequence!r}")


class OrchestratorAgent(Agent):
    """Simulates a user: list workspaces -> scoped request -> approve -> observe execution."""

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        assistant_jid: str,
        explorer_jid: str,
        solver_jid: str,
        run_reuse_flow: bool,
        run_demo_sequence: bool,
        run_reuse_demo_sequence: bool,
        run_intent_type_test_sequence: bool,
        run_toggle_only_test_sequence: bool,
        yggdrasil_url: str,
    ):
        super().__init__(jid, password)
        self.assistant_jid = assistant_jid
        self.explorer_jid = explorer_jid
        self.solver_jid = solver_jid
        self.run_reuse_flow = bool(run_reuse_flow)
        self.run_demo_sequence = bool(run_demo_sequence)
        self.run_reuse_demo_sequence = bool(run_reuse_demo_sequence)
        self.run_intent_type_test_sequence = bool(run_intent_type_test_sequence)
        self.run_toggle_only_test_sequence = bool(run_toggle_only_test_sequence)
        self.yggdrasil_url = str(yggdrasil_url or "").strip()
        self.thread_id = str(uuid.uuid4())
        self.first_reply: Optional[str] = None
        self.second_reply: Optional[str] = None

    async def setup(self):
        self.add_behaviour(self.RunFlow())

    class RunFlow(OneShotBehaviour):
        async def _get_state_snapshot(self) -> Dict[str, Any]:
            try:
                res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.ENV_STATE_REQUEST.value,
                    body={},
                    expect_type=MessageType.ENV_STATE_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                payload = json.loads(res.body or "{}")
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}

        def _find_artifact_id(self, state_payload: Dict[str, Any], token: str) -> str:
            token = str(token or "").strip().lower()
            if not token:
                return ""

            artifacts = state_payload.get("artifacts")
            if not isinstance(artifacts, dict):
                return ""

            for aid, info in artifacts.items():
                if token in str(aid).lower():
                    return str(aid)
                if isinstance(info, dict) and token in str(info.get("name") or "").lower():
                    return str(aid)
            return ""

        async def _print_selected_states(self, title: str, *, tokens: list[str]) -> None:
            snapshot = await self._get_state_snapshot()
            artifacts = snapshot.get("artifacts") if isinstance(snapshot.get("artifacts"), dict) else {}

            out: Dict[str, Any] = {}
            for t in tokens:
                aid = self._find_artifact_id(snapshot, t)
                if aid and aid in artifacts:
                    out[aid] = artifacts.get(aid)

            print("\n" + "=" * 60)
            print(title)
            print("-" * 60)
            print(json.dumps(out, indent=2))
            print("=" * 60 + "\n")

        async def _wait_user_assistant_reply(self, *, timeout_s: float) -> Optional[str]:
            deadline = asyncio.get_running_loop().time() + float(timeout_s)
            while asyncio.get_running_loop().time() < deadline:
                candidate = await self.receive(timeout=1)
                if not candidate:
                    continue
                if getattr(candidate, "thread", None) != self.agent.thread_id:
                    continue
                if not str(candidate.sender).startswith(self.agent.assistant_jid):
                    continue
                return candidate.body or ""
            return None

        async def _ask_user_assistant(self, text: str, *, timeout_s: float) -> Optional[str]:
            msg = SpadeMessage(to=self.agent.assistant_jid)
            msg.set_metadata("message_type", "llm")
            msg.thread = self.agent.thread_id
            msg.body = text
            _print_red_line(f"[User Query] {text}")
            await self.send(msg)
            return await self._wait_user_assistant_reply(timeout_s=timeout_s)

        async def _send_approval(self, text: str = "yes") -> None:
            approve = SpadeMessage(to=self.agent.assistant_jid)
            approve.set_metadata("message_type", "llm")
            approve.thread = self.agent.thread_id
            approve.body = text
            await self.send(approve)

        async def run(self):
            # Reuse-only mode: assume signifiers already exist and verify the solver reuses them.
            if self.agent.run_reuse_flow:
                await self._run_reuse_only()
                await self.agent.stop()
                return

            if self.agent.run_reuse_demo_sequence:
                await self._run_reuse_demo_sequence()
                await self.agent.stop()
                return

            if self.agent.run_intent_type_test_sequence:
                await self._run_intent_type_test_sequence()
                await self.agent.stop()
                return

            if self.agent.run_toggle_only_test_sequence:
                await self._run_toggle_only_test_sequence()
                await self.agent.stop()
                return

            if self.agent.run_demo_sequence:
                await self._run_demo_sequence()
                await self.agent.stop()
                return

            list_ws_query = "What workspaces are available?"
            scoped_query = "In lab308, it's too dark in here."

            # 1) Ask for workspaces
            logger.info('Asking UserAssistant for workspaces: "%s"', list_ws_query)
            reply_body = await self._ask_user_assistant(list_ws_query, timeout_s=60.0)
            if reply_body is None:
                print("\nERROR: Timed out waiting for workspace list reply from UserAssistant.\n")
                await self.agent.stop()
                return

            self.agent.first_reply = reply_body
            print("\n" + "=" * 60)
            print("[UserAssistant Workspace Reply]")
            print("-" * 60)
            _print_red_line(self.agent.first_reply)
            print("=" * 60 + "\n")

            # 3) Send a workspace-scoped request (lab308 contains light308 in your environment)
            logger.info('Sending scoped request to UserAssistant: "%s"', scoped_query)
            plan_proposal = await self._ask_user_assistant(scoped_query, timeout_s=90.0)
            if plan_proposal is None:
                print("\nERROR: Timed out waiting for plan proposal from UserAssistant.\n")
                await self.agent.stop()
                return

            print("\n" + "=" * 60)
            print("[UserAssistant Plan Proposal]")
            print("-" * 60)
            _print_red_line(plan_proposal)
            print("=" * 60 + "\n")

            # 5) Approve the proposed plan
            logger.info('Sending approval ("yes") to UserAssistant...')
            await self._send_approval("yes")

            # 6) Wait for execution confirmation/report
            exec_reply = await self._wait_user_assistant_reply(timeout_s=120.0)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for execution result from UserAssistant.\n")
                await self.agent.stop()
                return

            self.agent.second_reply = exec_reply
            print("\n" + "=" * 60)
            print("[UserAssistant Execution Reply]")
            print("-" * 60)
            _print_red_line(self.agent.second_reply)
            print("=" * 60 + "\n")

            # 7) Query EnvExplorer for recorded signifiers
            print("\n" + "=" * 60)
            print("[EnvExplorer Signifiers]")
            print("-" * 60)
            try:
                res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.SIGNIFIER_LIST_REQUEST.value,
                    body={},
                    expect_type=MessageType.SIGNIFIER_LIST_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                signifiers_payload = json.loads(res.body or "{}")
            except RpcTimeoutError:
                signifiers_payload = {"error": "timeout"}
            except Exception as e:
                signifiers_payload = {"error": "rpc_failed", "detail": str(e)}

            print(json.dumps(signifiers_payload, indent=2))
            print("=" * 60 + "\n")

            # 8) Run a match query using the first stored signifier's intent (if any)
            match_intent = ""
            expected_signifier_id = ""
            expected_affordance_uri = ""
            try:
                signifiers = signifiers_payload.get("signifiers") or []
                if isinstance(signifiers, list) and signifiers:
                    first = signifiers[0] if isinstance(signifiers[0], dict) else {}
                    match_intent = str(first.get("intent") or "").strip()
                    if match_intent:
                        print("\n" + "=" * 60)
                        print("[EnvExplorer Signifier Match]")
                        print("-" * 60)
                        try:
                            match_res = await rpc_call(
                                self.agent,
                                to_jid=self.agent.explorer_jid,
                                request_type=MessageType.SIGNIFIER_MATCH_REQUEST.value,
                                body={"intent": match_intent, "workspace_id": "lab308", "k": 5},
                                expect_type=MessageType.SIGNIFIER_MATCH_RESPONSE.value,
                                timeout=15.0,
                                thread=self.agent.thread_id,
                            )
                            match_payload = json.loads(match_res.body or "{}")
                        except RpcTimeoutError:
                            match_payload = {"error": "timeout"}
                        except Exception as e:
                            match_payload = {"error": "rpc_failed", "detail": str(e)}

                        print(json.dumps(match_payload, indent=2))
                        print("=" * 60 + "\n")

                        try:
                            matches = match_payload.get("matches") or []
                            finals = match_payload.get("final_matches") or []
                            finals_set = set(str(x) for x in finals) if isinstance(finals, list) else set()

                            if isinstance(matches, list):
                                for m in matches:
                                    if not isinstance(m, dict):
                                        continue
                                    sid = str(m.get("signifier_id") or "")
                                    if sid and sid in finals_set:
                                        expected_signifier_id = sid
                                        expected_affordance_uri = str(m.get("affordance_uri") or "")
                                        break
                        except Exception:
                            pass
            except Exception:
                pass

            await self.agent.stop()

        async def _run_reuse_only(self) -> None:
            print("\n" + "=" * 60)
            print("[Reuse-Only Mode]")
            print("-" * 60)

            # 1) Query EnvExplorer for recorded signifiers
            try:
                res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.SIGNIFIER_LIST_REQUEST.value,
                    body={},
                    expect_type=MessageType.SIGNIFIER_LIST_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                signifiers_payload = json.loads(res.body or "{}")
            except RpcTimeoutError:
                signifiers_payload = {"error": "timeout"}
            except Exception as e:
                signifiers_payload = {"error": "rpc_failed", "detail": str(e)}

            print("[EnvExplorer Signifiers]")
            print(json.dumps(signifiers_payload, indent=2))

            signifiers = signifiers_payload.get("signifiers") or []
            if not isinstance(signifiers, list) or not signifiers:
                print("\nFAIL: No signifiers found. Run the full flow once to generate signifiers.")
                print("=" * 60 + "\n")
                return

            first = signifiers[0] if isinstance(signifiers[0], dict) else {}
            match_intent = str(first.get("intent") or "").strip()
            if not match_intent:
                print("\nFAIL: First stored signifier has no intent field.")
                print("=" * 60 + "\n")
                return

            # 2) Match signifiers for that intent to get expected affordance/signifier id.
            try:
                match_res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.SIGNIFIER_MATCH_REQUEST.value,
                    body={"intent": match_intent, "workspace_id": "lab308", "k": 5},
                    expect_type=MessageType.SIGNIFIER_MATCH_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                match_payload = json.loads(match_res.body or "{}")
            except RpcTimeoutError:
                match_payload = {"error": "timeout"}
            except Exception as e:
                match_payload = {"error": "rpc_failed", "detail": str(e)}

            print("\n[EnvExplorer Signifier Match]")
            print(json.dumps(match_payload, indent=2))

            expected_signifier_id = ""
            expected_affordance_uri = ""
            try:
                matches = match_payload.get("matches") or []
                finals = match_payload.get("final_matches") or []
                finals_set = set(str(x) for x in finals) if isinstance(finals, list) else set()

                if isinstance(matches, list):
                    for m in matches:
                        if not isinstance(m, dict):
                            continue
                        sid = str(m.get("signifier_id") or "")
                        if sid and (not finals_set or sid in finals_set):
                            expected_signifier_id = sid
                            expected_affordance_uri = str(m.get("affordance_uri") or "")
                            break
            except Exception:
                pass

            if not expected_affordance_uri:
                print("\nFAIL: Could not determine expected affordance_uri from match results.")
                print("=" * 60 + "\n")
                return

            # 3) Ask solver to plan for the same intent and verify reuse
            try:
                solver_res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.solver_jid,
                    request_type=MessageType.GOAL_REQUEST.value,
                    body={"intents": [{"action": "unknown", "artifact": "unknown", "intent_text": match_intent}], "workspace_id": "lab308"},
                    expect_type=MessageType.PLAN_CREATED.value,
                    timeout=90.0,
                    thread=self.agent.thread_id,
                )
                plan_obj = json.loads(solver_res.body or "{}")
            except RpcTimeoutError:
                plan_obj = {"error": "timeout"}
            except Exception as e:
                plan_obj = {"error": "rpc_failed", "detail": str(e)}

            print("\n[Solver PLAN_CREATED]")
            print(json.dumps(plan_obj, indent=2))

            reused_affordance = False
            reused_signifier_meta = False
            try:
                steps = plan_obj.get("steps") if isinstance(plan_obj, dict) else None
                if isinstance(steps, list):
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                        if str(step.get("affordance_uri") or "") == expected_affordance_uri:
                            reused_affordance = True

                        meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
                        if expected_signifier_id and meta.get("used_signifier_id") == expected_signifier_id:
                            reused_signifier_meta = True
            except Exception:
                pass

            print("\n[Reuse Check Result]")
            if reused_affordance and (reused_signifier_meta or not expected_signifier_id):
                print("PASS: Solver reused signifier (expected affordance selected).")
            elif reused_affordance and not reused_signifier_meta:
                print("PARTIAL: Solver reused affordance but missing/incorrect used_signifier_id metadata.")
            else:
                print(
                    "FAIL: Solver did not reuse the expected signifier affordance.\n"
                    f"  expected_affordance_uri={expected_affordance_uri}\n"
                    f"  expected_signifier_id={expected_signifier_id or '(unknown)'}"
                )

            print("=" * 60 + "\n")

        async def _run_reuse_demo_sequence(self) -> None:
            """
            Thesis sequence 4 (reuse demo):
            - assumes a prior signifier exists (run sequence 3 beforehand)
            - simulates an external reset (no UserAssistant involvement)
            - repeats an implicit request and expects the solver to recover a plan from signifiers
            - executes the recovered plan without recording a new signifier
            """
            try:
                plan_timeout_s = float(os.getenv("AMI_PLAN_TIMEOUT_S", "180"))
            except Exception:
                plan_timeout_s = 180.0
            try:
                exec_timeout_s = float(os.getenv("AMI_EXEC_TIMEOUT_S", "180"))
            except Exception:
                exec_timeout_s = 180.0

            print("\n" + "=" * 60)
            print("[Reuse Demo Mode]")
            print("-" * 60)

            # 1) Baseline: list stored signifiers (before reuse execution)
            try:
                res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.SIGNIFIER_LIST_REQUEST.value,
                    body={},
                    expect_type=MessageType.SIGNIFIER_LIST_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                before_payload = json.loads(res.body or "{}")
            except RpcTimeoutError:
                before_payload = {"error": "timeout"}
            except Exception as e:
                before_payload = {"error": "rpc_failed", "detail": str(e)}

            before_total = before_payload.get("total") if isinstance(before_payload, dict) else None
            print("[EnvExplorer Signifiers (Before Reuse)]")
            print(json.dumps(before_payload, indent=2))

            signifiers = before_payload.get("signifiers") if isinstance(before_payload, dict) else None
            if not isinstance(signifiers, list) or not signifiers:
                print(
                    "\nFAIL: No signifiers found. Run:\n"
                    "  python tests/manual_test_full_flow_plan.py --sequence 3\n"
                    "to create an IMPLICIT signifier first.\n"
                )
                print("=" * 60 + "\n")
                return

            # 2) External reset (simulated): light off, blinds mostly closed (open ~25%).
            base = (self.agent.yggdrasil_url or "").rstrip("/")
            if not base:
                print("\nFAIL: Missing yggdrasil_url for external reset.\n")
                print("=" * 60 + "\n")
                return

            toggle_url = f"{base}/workspaces/lab308/artifacts/light308/toggle"
            blinds_url = f"{base}/workspaces/lab308/artifacts/blinds_308/ha/cover/set_cover_position"

            snapshot = await self._get_state_snapshot()
            artifacts = snapshot.get("artifacts") if isinstance(snapshot.get("artifacts"), dict) else {}

            def _find_prop_value(state: Dict[str, Any], needle: str) -> Any:
                for k, v in (state or {}).items():
                    if needle in str(k):
                        return v
                return None

            light_aid = self._find_artifact_id(snapshot, "light308")
            light_state = None
            try:
                if light_aid and isinstance(artifacts.get(light_aid), dict):
                    st = artifacts[light_aid].get("state") if isinstance(artifacts[light_aid].get("state"), dict) else {}
                    light_state = _find_prop_value(st, "/props/state")
            except Exception:
                light_state = None

            logger.info(demo("Simulating external reset (no UserAssistant): light=off, blinds_open=25%"))
            try:
                async with aiohttp.ClientSession() as session:
                    # Only toggle if we are confident the light is currently on (avoid accidentally turning it on).
                    if str(light_state).strip().lower() == "on":
                        async with session.post(toggle_url, json={}) as resp:
                            await resp.text()

                    # "lowered to only 25% of the maximum" (i.e., open ~25%) -> position=25
                    async with session.post(blinds_url, json={"position": 25}) as resp:
                        await resp.text()
            except Exception as e:
                print(f"\nWARNING: External reset HTTP calls failed: {e}\n")

            # Allow EnvExplorer to receive webhook updates.
            await asyncio.sleep(1.0)

            await self._print_selected_states(
                "[EnvExplorer State After External Reset]",
                tokens=["light308", "blinds308", "lightSensor308"],
            )

            # 3) Repeat IMPLICIT request with prior signifier present
            implicit_query = "In lab308, I can't see anything on my desk."
            logger.info('DEMO: Sending IMPLICIT (reuse) request: "%s"', implicit_query)
            proposal = await self._ask_user_assistant(implicit_query, timeout_s=plan_timeout_s)
            if proposal is None:
                print("\nERROR: Timed out waiting for IMPLICIT (reuse) plan proposal from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant IMPLICIT Plan Proposal (Reuse)]")
            print("-" * 60)
            _print_red_line(proposal)
            print("=" * 60 + "\n")

            # 4) Approve and execute recovered plan
            logger.info('DEMO: Approving IMPLICIT (reuse) plan ("yes")...')
            await self._send_approval("yes")
            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for IMPLICIT (reuse) execution result from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant IMPLICIT Execution Reply (Reuse)]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")

            # 5) Verify no new signifier was recorded for the reused plan
            try:
                res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.SIGNIFIER_LIST_REQUEST.value,
                    body={},
                    expect_type=MessageType.SIGNIFIER_LIST_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                after_payload = json.loads(res.body or "{}")
            except RpcTimeoutError:
                after_payload = {"error": "timeout"}
            except Exception as e:
                after_payload = {"error": "rpc_failed", "detail": str(e)}

            after_total = after_payload.get("total") if isinstance(after_payload, dict) else None
            print("[EnvExplorer Signifiers (After Reuse)]")
            print(json.dumps(after_payload, indent=2))

            print("\n[Reuse Signifier Recording Check]")
            if before_total is not None and after_total is not None:
                if before_total == after_total:
                    print(f"PASS: No new signifier recorded (total={after_total}).")
                else:
                    print(f"FAIL: Signifier count changed (before={before_total}, after={after_total}).")
            else:
                print(f"NOTE: Could not verify signifier count unchanged (before={before_total}, after={after_total}).")

            print("=" * 60 + "\n")

        async def _run_demo_sequence(self) -> None:
            """
            Longer, thesis-demo friendly sequence:
            - capabilities/workspaces
            - state query (light)
            - explicit request (execute + record signifiers)
            - implicit request (approve + execute + record signifier)
            """
            try:
                simple_timeout_s = float(os.getenv("AMI_SIMPLE_TIMEOUT_S", "90"))
            except Exception:
                simple_timeout_s = 90.0
            try:
                plan_timeout_s = float(os.getenv("AMI_PLAN_TIMEOUT_S", "180"))
            except Exception:
                plan_timeout_s = 180.0
            try:
                exec_timeout_s = float(os.getenv("AMI_EXEC_TIMEOUT_S", "180"))
            except Exception:
                exec_timeout_s = 180.0

            list_devices_query = "List all active (available) devices in the environment."
            state_query = "In lab308, show the state of light308 (is it on and what is its intensity?)."
            turn_on_query = "In lab308, turn on lights_308."
            explicit_query = "In lab308, turn off the light and open the blinds to 80% of the maximum level."
            implicit_query = "In lab308, it's kind of dark in here."

            # 1) Capabilities / workspaces
            logger.info('DEMO: Asking UserAssistant: "%s"', list_devices_query)
            reply = await self._ask_user_assistant(list_devices_query, timeout_s=simple_timeout_s)
            if reply is None:
                print("\nERROR: Timed out waiting for device list reply from UserAssistant.\n")
                return
            print("\n" + "=" * 60)
            print("[UserAssistant Devices Reply]")
            print("-" * 60)
            _print_red_line(reply)
            print("=" * 60 + "\n")

            # 2) State query (forces UA <-> EnvExplorer state conversation)
            logger.info('DEMO: Asking UserAssistant: "%s"', state_query)
            reply = await self._ask_user_assistant(state_query, timeout_s=simple_timeout_s)
            if reply is None:
                print("\nERROR: Timed out waiting for state reply from UserAssistant.\n")
                return
            print("\n" + "=" * 60)
            print("[UserAssistant State Reply]")
            print("-" * 60)
            _print_red_line(reply)
            print("=" * 60 + "\n")

            # 3) Turn on the light
            logger.info('DEMO: Asking UserAssistant: "%s"', turn_on_query)
            reply = await self._ask_user_assistant(turn_on_query, timeout_s=simple_timeout_s)
            if reply is None:
                print("\nERROR: Timed out waiting for turn-on reply from UserAssistant.\n")
                return
            print("\n" + "=" * 60)
            print("[UserAssistant Turn-On Reply]")
            print("-" * 60)
            _print_red_line(reply)
            print("=" * 60 + "\n")
            logger.info('DEMO: Approving turn-on plan ("yes")...')
            await self._send_approval("yes")
            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for turn-on execution result from UserAssistant.\n")
                return
            print("\n" + "=" * 60)
            print("[UserAssistant Turn-On Execution Reply]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")

            # 3) EXPLICIT request (execute to create signifiers)
            logger.info('DEMO: Sending EXPLICIT request: "%s"', explicit_query)
            proposal = await self._ask_user_assistant(explicit_query, timeout_s=plan_timeout_s)
            if proposal is None:
                print("\nERROR: Timed out waiting for EXPLICIT plan proposal from UserAssistant.\n")
                return
            print("\n" + "=" * 60)
            print("[UserAssistant EXPLICIT Plan Proposal]")
            print("-" * 60)
            _print_red_line(proposal)
            print("=" * 60 + "\n")

            logger.info('DEMO: Approving EXPLICIT plan ("yes")...')
            await self._send_approval("yes")

            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for EXPLICIT execution result from UserAssistant.\n")
                return
            print("\n" + "=" * 60)
            print("[UserAssistant EXPLICIT Execution Reply]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")

            # Show state changes via EnvExplorer (no external UI required).
            await self._print_selected_states(
                "[EnvExplorer State After EXPLICIT Execution]",
                tokens=["light308", "blinds308", "lightSensor308"],
            )

            # Query EnvExplorer for recorded signifiers (created by explicit execution)
            print("\n" + "=" * 60)
            print("[EnvExplorer Signifiers After EXPLICIT Execution]")
            print("-" * 60)
            try:
                res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.SIGNIFIER_LIST_REQUEST.value,
                    body={},
                    expect_type=MessageType.SIGNIFIER_LIST_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                signifiers_payload = json.loads(res.body or "{}")
            except RpcTimeoutError:
                signifiers_payload = {"error": "timeout"}
            except Exception as e:
                signifiers_payload = {"error": "rpc_failed", "detail": str(e)}

            print(json.dumps(signifiers_payload, indent=2))
            print("=" * 60 + "\n")

            # Optional manual pause (useful for live demos).
            if os.getenv("PAUSE_FOR_SENSOR", "").strip().lower() in ("1", "true", "yes"):
                prompt = "Press Enter to continue to the IMPLICIT request... "
                await asyncio.get_running_loop().run_in_executor(None, input, prompt)

            # 4) IMPLICIT request (approve + execute + record signifier)
            logger.info('DEMO: Sending IMPLICIT request: "%s"', implicit_query)
            proposal = await self._ask_user_assistant(implicit_query, timeout_s=plan_timeout_s)
            if proposal is None:
                print("\nERROR: Timed out waiting for IMPLICIT reply from UserAssistant.\n")
                return

            def _looks_like_plan_confirmation(text: str) -> bool:
                s = " ".join(str(text or "").strip().lower().split())
                return bool(
                    re.search(
                        r"\bdoes\s+(this|that|the)\s+(plan\s+)?look\s+(good|ok|okay)(\s+to\s+you)?(?=\s*(?:[?.!]|$))",
                        s,
                    )
                )

            # If UA returns an unexpected plan-management message (e.g., discarding a stale pending plan),
            # retry once so the demo can continue deterministically.
            for attempt in (1, 2):
                print("\n" + "=" * 60)
                print("[UserAssistant IMPLICIT Plan Proposal]" if attempt == 1 else "[UserAssistant IMPLICIT Plan Proposal (Retry)]")
                print("-" * 60)
                _print_red_line(proposal)
                print("=" * 60 + "\n")

                if _looks_like_plan_confirmation(proposal):
                    break

                if attempt == 2:
                    print(
                        "\nERROR: IMPLICIT step did not produce a plan confirmation prompt. "
                        "Re-run the sequence (or check UserAssistant prompts/logs).\n"
                    )
                    return

                logger.warning(
                    'DEMO: IMPLICIT reply did not look like a plan proposal (missing confirmation prompt); retrying once. reply="%s"',
                    proposal,
                )
                proposal = await self._ask_user_assistant(implicit_query, timeout_s=plan_timeout_s)
                if proposal is None:
                    print("\nERROR: Timed out waiting for IMPLICIT reply from UserAssistant (retry).\n")
                    return
 
            logger.info('DEMO: Approving IMPLICIT plan ("yes")...')
            await self._send_approval("yes")
            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for IMPLICIT execution result from UserAssistant.\n")
                return
            print("\n" + "=" * 60)
            print("[UserAssistant IMPLICIT Execution Reply]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")
 
            await self._print_selected_states(
                "[EnvExplorer State After IMPLICIT Execution]",
                tokens=["light308", "blinds308", "lightSensor308"],
            )
 
            # Show signifiers after IMPLICIT execution (should include the newly recorded one).
            print("\n" + "=" * 60)
            print("[EnvExplorer Signifiers After IMPLICIT Execution]")
            print("-" * 60)
            try:
                res = await rpc_call(
                    self.agent,
                    to_jid=self.agent.explorer_jid,
                    request_type=MessageType.SIGNIFIER_LIST_REQUEST.value,
                    body={},
                    expect_type=MessageType.SIGNIFIER_LIST_RESPONSE.value,
                    timeout=15.0,
                    thread=self.agent.thread_id,
                )
                signifiers_payload = json.loads(res.body or "{}")
            except RpcTimeoutError:
                signifiers_payload = {"error": "timeout"}
            except Exception as e:
                signifiers_payload = {"error": "rpc_failed", "detail": str(e)}
 
            print(json.dumps(signifiers_payload, indent=2))
            print("=" * 60 + "\n")

        async def _run_intent_type_test_sequence(self) -> None:
            """
            Sequence 5: Intent Type Classification Test
            - Tests EXPLICIT goal classification (indefinite article: "a light")
            - Tests IMPLICIT goal classification (definite article: "the light")
            - Verifies correct intent_type logging and behavior
            """
            try:
                plan_timeout_s = float(os.getenv("AMI_PLAN_TIMEOUT_S", "180"))
            except Exception:
                plan_timeout_s = 180.0
            try:
                exec_timeout_s = float(os.getenv("AMI_EXEC_TIMEOUT_S", "180"))
            except Exception:
                exec_timeout_s = 180.0

            print("\n" + "=" * 60)
            print("[Intent Type Classification Test - Sequence 5]")
            print("=" * 60 + "\n")

            # Test 1: IMPLICIT GOAL (indefinite article, vague)
            implicit_query_1 = "turn on a light."
            logger.info('TEST 1: Sending IMPLICIT request (vague): "%s"', implicit_query_1)
            print("\n" + "=" * 60)
            print("[TEST 1: IMPLICIT Goal - 'turn on A light' (vague)]")
            print("-" * 60)
            print("Expected: LLM should classify as intent_type='implicit'")
            print("Expected logging: 'Intent Type Classified: IMPLICIT (vague request, infer from context)'")
            print("=" * 60 + "\n")

            proposal = await self._ask_user_assistant(implicit_query_1, timeout_s=plan_timeout_s)
            if proposal is None:
                print("\nERROR: Timed out waiting for IMPLICIT plan proposal from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant IMPLICIT Plan Proposal (Test 1)]")
            print("-" * 60)
            _print_red_line(proposal)
            print("=" * 60 + "\n")

            logger.info('Approving IMPLICIT plan ("yes")...')
            await self._send_approval("yes")

            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for IMPLICIT execution result from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant IMPLICIT Execution Reply (Test 1)]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")

            await self._print_selected_states(
                "[EnvExplorer State After IMPLICIT Execution (Test 1)]",
                tokens=["light308", "lights_308"],
            )

            # Small delay between tests
            await asyncio.sleep(2.0)

            # Test 2: IMPLICIT GOAL (definite article, vague)
            implicit_query_2 = "In lab308, turn off the light."
            logger.info('TEST 2: Sending IMPLICIT request (vague): "%s"', implicit_query_2)
            print("\n" + "=" * 60)
            print("[TEST 2: IMPLICIT Goal - 'turn off THE light' (vague)]")
            print("-" * 60)
            print("Expected: LLM should classify as intent_type='implicit'")
            print("Expected logging: 'Intent Type Classified: IMPLICIT (vague request, infer from context)'")
            print("=" * 60 + "\n")

            proposal = await self._ask_user_assistant(implicit_query_2, timeout_s=plan_timeout_s)
            if proposal is None:
                print("\nERROR: Timed out waiting for IMPLICIT plan proposal from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant IMPLICIT Plan Proposal (Test 2)]")
            print("-" * 60)
            _print_red_line(proposal)
            print("=" * 60 + "\n")

            logger.info('Approving IMPLICIT plan ("yes")...')
            await self._send_approval("yes")

            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for IMPLICIT execution result from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant IMPLICIT Execution Reply (Test 2)]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")

            await self._print_selected_states(
                "[EnvExplorer State After IMPLICIT Execution (Test 2)]",
                tokens=["light308", "lights_308"],
            )

            # Small delay between tests
            await asyncio.sleep(2.0)

            # Test 3: EXPLICIT GOAL - artifact ID specified
            explicit_query = "In lab308, toggle light308."
            logger.info('TEST 3: Sending EXPLICIT request (artifact ID): "%s"', explicit_query)
            print("\n" + "=" * 60)
            print("[TEST 3: EXPLICIT Goal - 'toggle light308' (artifact ID specified)]")
            print("-" * 60)
            print("Expected: LLM should classify as intent_type='explicit'")
            print("Expected logging: 'Intent Type Classified: EXPLICIT (exact artifact ID specified)'")
            print("=" * 60 + "\n")

            proposal = await self._ask_user_assistant(explicit_query, timeout_s=plan_timeout_s)
            if proposal is None:
                print("\nERROR: Timed out waiting for EXPLICIT plan proposal from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant EXPLICIT Plan Proposal (Test 3)]")
            print("-" * 60)
            _print_red_line(proposal)
            print("=" * 60 + "\n")

            logger.info('Approving EXPLICIT plan ("yes")...')
            await self._send_approval("yes")

            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for EXPLICIT execution result from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant EXPLICIT Execution Reply (Test 3)]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")

            await self._print_selected_states(
                "[EnvExplorer State After EXPLICIT Execution (Test 3)]",
                tokens=["light308", "lights_308"],
            )

            print("\n" + "=" * 60)
            print("[Intent Type Classification Test Complete]")
            print("-" * 60)
            print("Review the logs above to verify:")
            print("  1. IMPLICIT request (vague) logged: 'Intent Type Classified: IMPLICIT'")
            print("  2. IMPLICIT request (vague) logged: 'Intent Type Classified: IMPLICIT'")
            print("  3. EXPLICIT request (artifact ID) logged: 'Intent Type Classified: EXPLICIT'")
            print("=" * 60 + "\n")

        async def _run_toggle_only_test_sequence(self) -> None:
            """
            Sequence 6: Toggle Light Only Test (Quick Test)
            - Tests single EXPLICIT goal: "In lab308, toggle light308."
            - Verifies correct intent_type classification and execution
            """
            try:
                plan_timeout_s = float(os.getenv("AMI_PLAN_TIMEOUT_S", "180"))
            except Exception:
                plan_timeout_s = 180.0

            try:
                exec_timeout_s = float(os.getenv("AMI_EXEC_TIMEOUT_S", "180"))
            except Exception:
                exec_timeout_s = 180.0

            logger.info(demo("Running manual test sequence=toggle-only-test"))

            print("\n" + "=" * 60)
            print("[Toggle Light Only Test - Sequence 6]")
            print("=" * 60 + "\n")

            # Single Test: EXPLICIT GOAL - artifact ID specified
            explicit_query = "In lab308, toggle light308."
            logger.info('TEST: Sending EXPLICIT request (artifact ID): "%s"', explicit_query)
            print("\n" + "=" * 60)
            print("[TEST: EXPLICIT Goal - 'toggle light308' (artifact ID specified)]")
            print("-" * 60)
            print("Expected: LLM should classify as intent_type='explicit'")
            print("Expected logging: 'Intent Type Classified: EXPLICIT (exact artifact ID specified)'")
            print("=" * 60 + "\n")

            proposal = await self._ask_user_assistant(explicit_query, timeout_s=plan_timeout_s)
            if proposal is None:
                print("\nERROR: Timed out waiting for EXPLICIT plan proposal from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant EXPLICIT Plan Proposal]")
            print("-" * 60)
            _print_red_line(proposal)
            print("=" * 60 + "\n")

            logger.info('Approving EXPLICIT plan ("yes")...')
            await self._send_approval("yes")

            exec_reply = await self._wait_user_assistant_reply(timeout_s=exec_timeout_s)
            if exec_reply is None:
                print("\nERROR: Timed out waiting for EXPLICIT execution result from UserAssistant.\n")
                return

            print("\n" + "=" * 60)
            print("[UserAssistant EXPLICIT Execution Reply]")
            print("-" * 60)
            _print_red_line(exec_reply)
            print("=" * 60 + "\n")

            await self._print_selected_states(
                "[EnvExplorer State After EXPLICIT Execution]",
                tokens=["light308", "lights_308"],
            )

            print("\n" + "=" * 60)
            print("[Toggle Test Complete]")
            print("-" * 60)
            print("Review the logs above to verify:")
            print("  - EXPLICIT request (artifact ID) logged: 'Intent Type Classified: EXPLICIT'")
            print("=" * 60 + "\n")


async def main():
    args = _parse_args(sys.argv[1:])
    if args.demo:
        _enable_demo_only_logging()
        _strip_demo_prefix_from_logs()

    xmpp_server = os.getenv("SPADE_SERVER", "localhost")
    password = os.getenv("SPADE_PASSWORD", "password")

    # Require LLM API key (both UA and solver will call it).
    _require_env("OPENAI_API_KEY")

    yggdrasil_url = os.getenv("YGGDRASIL_URL", "http://localhost:8080/").strip()

    explorer_jid = f"env_explorer@{xmpp_server}"
    assistant_jid = f"user_assistant@{xmpp_server}"
    solver_jid = f"interaction_solver@{xmpp_server}"
    orchestrator_jid = f"orchestrator@{xmpp_server}"

    # Optional: clear embedded Experience Engine signifier storage before starting (makes the run reproducible).
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
    run_intent_type_test_sequence = sequence == "intent-type-test"
    run_toggle_only_test_sequence = sequence == "toggle-only-test"
    run_startup_sequence = sequence == "startup"

    logger.info(demo("Running manual test sequence=%s"), sequence)
    logger.info(demo("EnvExplorer entrypoint (Yggdrasil URL)=%s"), yggdrasil_url)
    await _reset_lab308_state()

    # CLI convenience: keep the internal demo sequence env toggles working.
    if args.pause_for_sensor:
        os.environ["PAUSE_FOR_SENSOR"] = "1"

    # Load configuration from files using ConfigLoader
    env_config = ConfigLoader.merge_configs(
        ConfigLoader.load_with_env_vars("ami_agents/config/agents.yaml"),
        ConfigLoader.load_with_env_vars("ami_agents/config/environment.yaml")
    )

    # Override specific demo settings
    env_config["yggdrasil_url"] = yggdrasil_url
    env_config["yggdrasil"] = {"url": yggdrasil_url}
    env_config["discovery"] = {
        "notify_on_discovery_complete": True,
        "notify_agents": [solver_jid],
    }

    # Override signifier settings from CLI args if provided
    if args.signifier_matcher or args.signifier_min_similarity is not None:
        if "signifiers" not in env_config:
            env_config["signifiers"] = {}
        if args.signifier_matcher:
            env_config["signifiers"]["matcher_version"] = args.signifier_matcher
        if args.signifier_min_similarity is not None:
            env_config["signifiers"]["min_similarity"] = float(args.signifier_min_similarity)

    # UserAssistant and Solver configs (new agents.yaml-compatible structure)
    # Get model from configuration hierarchy: config -> env -> fallback
    llm_config = env_config.get("llm", {})
    provider_name = llm_config.get("default_provider", "openai")
    provider_cfg = llm_config.get("providers", {}).get(provider_name, {})

    model = os.getenv("OPENAI_MODEL") or provider_cfg.get("model", "gpt-4")
    # This model name is typically served via OpenRouter's OpenAI-compatible API.
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        base_url = "https://openrouter.ai/api/v1" if ":" in model or "/" in model else "https://api.openai.com/v1"

    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip() or (
        "high" if model.startswith("o") and "openai.com" in base_url else ""
    )
    api_timeout = os.getenv("OPENAI_TIMEOUT", "").strip() or os.getenv("OPENAI_HTTP_TIMEOUT", "").strip()
    try:
        api_timeout_s = float(api_timeout) if api_timeout else (120.0 if model.startswith("o") and "openai.com" in base_url else 30.0)
    except Exception:
        api_timeout_s = 120.0 if model.startswith("o") and "openai.com" in base_url else 30.0

    try:
        planning_timeout_s = float(os.getenv("AMI_PLANNING_TIMEOUT", "").strip() or ("180" if model.startswith("o") else "90"))
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
                    **({} if model.startswith("o") else {"temperature": float(os.getenv("OPENAI_TEMPERATURE") or str(provider_cfg.get("temperature", 0.7)))}),
                    **({} if model.startswith("o") else {"max_tokens": int(os.getenv("OPENAI_MAX_TOKENS") or str(provider_cfg.get("max_tokens", 1500)))}),
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
                }
            },
            "retry": {"timeout": api_timeout_s},
        },
        "planning": {
            "timeout": planning_timeout_s,
            "llm_planning": {
                "model": model,
                **({} if model.startswith("o") else {"temperature": float(os.getenv("OPENAI_TEMPERATURE") or str(provider_cfg.get("temperature", 0.7)))}),
                **({} if model.startswith("o") else {"max_tokens": int(os.getenv("OPENAI_MAX_TOKENS") or str(provider_cfg.get("max_tokens", 1500)))}),
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            },
            "context_gathering": {"timeout": 60},
        },
    }

    explorer = EnvExplorerAgent(explorer_jid, password, env_config, hmas_client=DummyHMASClient())

    assistant = None
    if not run_reuse_flow:
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
        target_jids={"explorer": explorer_jid},
    )

    orchestrator = None
    if not run_startup_sequence:
        orchestrator = OrchestratorAgent(
            orchestrator_jid,
            password,
            assistant_jid=assistant_jid,
            explorer_jid=explorer_jid,
            solver_jid=solver_jid,
            run_reuse_flow=run_reuse_flow,
            run_demo_sequence=run_demo_sequence,
            run_reuse_demo_sequence=run_reuse_demo_sequence,
            run_intent_type_test_sequence=run_intent_type_test_sequence,
            run_toggle_only_test_sequence=run_toggle_only_test_sequence,
            yggdrasil_url=yggdrasil_url,
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

        # InteractionSolver must start FIRST to register behaviors that can receive
        # the ENV_DISCOVERY_COMPLETE notification sent by EnvExplorer during discovery.
        logger.info("Starting InteractionSolver...")
        await solver.start(auto_register=True)

        logger.info("Starting EnvExplorer...")
        await explorer.start(auto_register=True)

        if assistant:
            logger.info("Starting UserAssistant...")
            await assistant.start(auto_register=True)

        # Wait for discovery to finish before sending the user query (so tool calls see real capabilities).
        for _ in range(600):
            if getattr(explorer, "discovery_complete", False):
                break
            await asyncio.sleep(0.5)
        if not getattr(explorer, "discovery_complete", False):
            raise RuntimeError("EnvExplorer discovery did not complete in time.")

        if run_startup_sequence:
            logger.info("Startup-only sequence complete (no interactions).")
            if args.hold:
                await asyncio.get_running_loop().run_in_executor(None, input, "Press Enter to stop agents... ")
        else:
            # Pre-warm the UserAssistant execution engine so plan execution has no initial "cold start" delay.
            # This avoids a long environment exploration phase right after the user approves a plan.
            prewarm_enabled = os.getenv("AMI_PREWARM_EXECUTION", "").strip().lower() not in ("0", "false", "no", "off")
            if assistant and prewarm_enabled:
                logger.info(demo("Pre-warming UserAssistant execution engine (YggdrasilIntegration)..."))
                integ_logger = logging.getLogger("ami_agents.environment.integration.integration_engine")
                prev_level = integ_logger.level
                try:
                    # Keep console output readable: EnvExplorer already prints the discovery story.
                    integ_logger.setLevel(logging.WARNING)
                    await assistant.ensure_execution_engine_ready()
                    logger.info(demo("UserAssistant execution engine ready."))
                finally:
                    integ_logger.setLevel(prev_level)

            logger.info("Starting Orchestrator (drives the test)...")
            assert orchestrator is not None
            await orchestrator.start(auto_register=True)

            while orchestrator.is_alive():
                await asyncio.sleep(0.5)

    finally:
        logger.info("Stopping agents...")
        if orchestrator:
            try:
                await orchestrator.stop()
            except Exception:
                pass
        try:
            await solver.stop()
        except Exception:
            pass
        if assistant:
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
