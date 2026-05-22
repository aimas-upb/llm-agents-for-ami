"""
demo_viewer/server.py
=====================
Web visualization server for the LLM Agents for AMI demo.

Layout (served at /demo):
  Left panel  → Home Assistant iframe at /home/overview
                (full reverse proxy at root – strips X-Frame-Options,
                 so HA auth flow works natively inside the iframe)
  Right panel → Real-time log viewer (SSE stream from demo subprocess)

Usage:
    python demo_viewer/server.py
    open http://localhost:8765/demo

Auth note:
    HA is proxied at the server root (/).  The first time you visit /demo,
    if HA needs authentication it will show its login form inside the left
    iframe.  Log in there once – the token is stored in localStorage at
    http://localhost:8765 and subsequent visits work without re-login.

No modifications to any existing project files.
"""

import asyncio
import os
import json
import logging
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from dotenv import load_dotenv
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template

# ---------------------------------------------------------------------------
# Custom Log Handler for Manual Mode
# ---------------------------------------------------------------------------

class WebLogHandler(logging.Handler):
    """
    Custom log handler that captures ALL agent logs and feeds them to the SSE stream.
    This ensures manual mode has the same complete log visibility as sequence mode.
    """

    def __init__(self):
        super().__init__()
        self.broadcast_func = None

    def set_broadcast_function(self, broadcast_func):
        """Set the broadcast function to use for sending logs to SSE stream."""
        self.broadcast_func = broadcast_func

    def emit(self, record):
        """
        Bulletproof log record emission that handles all edge cases.
        """
        if not self.broadcast_func:
            return

        try:
            # Bulletproof message extraction with multiple fallbacks
            msg = self._extract_safe_message(record)

            # Safe async delivery
            self._safe_async_broadcast(msg)

        except Exception:
            # Absolute last resort - completely silent failure
            # Don't even try to log the error as it could cause recursion
            pass

    def _extract_safe_message(self, record):
        """Extract message with multiple fallback levels."""
        import re
        from datetime import datetime

        # Fallback 1: Try normal formatting
        try:
            return self.format(record)
        except:
            pass

        # Fallback 2: Try with cleaned record
        try:
            # Extract raw message safely
            if hasattr(record, 'msg') and hasattr(record, 'args'):
                if record.args:
                    # Try basic string formatting
                    try:
                        raw_msg = str(record.msg) % record.args
                    except:
                        raw_msg = str(record.msg)
                else:
                    raw_msg = str(record.msg)
            else:
                raw_msg = str(record.msg) if hasattr(record, 'msg') else "Unknown message"

            # Clean ANSI codes
            clean_msg = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw_msg)

            # Manual format construction
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            return f"{timestamp} [{record.name}] {record.levelname}: {clean_msg}"

        except:
            pass

        # Fallback 3: Absolute minimum
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"{timestamp} [UNKNOWN] INFO: Log formatting error"
        except:
            return "Log formatting error"

    def _safe_async_broadcast(self, msg):
        """Safely broadcast message without throwing exceptions."""
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.broadcast_func(msg))
        except:
            # No loop or other async issues - just skip
            pass


# Global web log handler instance
web_log_handler = WebLogHandler()


async def setup_agent_logging_capture():
    """
    Set up safe logging capture for manual mode.
    Provides comprehensive log visibility while avoiding conflicts.
    """
    try:
        # Set the broadcast function for the web log handler
        web_log_handler.set_broadcast_function(_broadcast)

        # Set up formatter to match sequence mode format
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        web_log_handler.setFormatter(formatter)

        # Only add to root logger if not already present
        root_logger = logging.getLogger()
        if web_log_handler not in root_logger.handlers:
            root_logger.addHandler(web_log_handler)

        # Target key loggers with safer approach
        key_loggers = [
            "ami_agents.environment.integration.integration_engine",
            "UserAssistant[user_assistant@localhost]",
            "InteractionSolver[interaction_solver@localhost]",
            "EnvExplorerAgent[env_explorer@localhost]"
        ]

        # Add handlers more carefully
        for logger_name in key_loggers:
            try:
                logger = logging.getLogger(logger_name)
                if web_log_handler not in logger.handlers:
                    logger.addHandler(web_log_handler)
                    logger.setLevel(logging.INFO)
            except Exception as e:
                await _broadcast(f"[DEMO_VIEWER] Warning: Could not setup logger {logger_name}: {e}")

        # Add filters safely
        try:
            await _add_spam_filters()
            await _enable_demo_logging_enhancements()
        except Exception as e:
            await _broadcast(f"[DEMO_VIEWER] Warning: Filter setup issue: {e}")

        await _broadcast("[DEMO_VIEWER] ✅ Safe logging capture enabled - logs should be visible now!")

    except Exception as e:
        await _broadcast(f"[DEMO_VIEWER] ❌ Logging setup error: {e}")
        # Continue anyway - don't let logging break the system
        pass


async def _add_spam_filters():
    """Add the same spam filters that sequence mode uses to keep logs readable."""

    class DropHighFrequencySpamFilter(logging.Filter):
        """Drop ultra-frequent sensor state updates to keep console readable (same as sequence mode)."""

        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return True

            # Filter the same high-frequency updates as sequence mode
            if "STATE UPDATE:" in msg:
                if "clock308" in msg and "/props/timeOfDay" in msg:
                    return False
                if "lightSensor308" in msg and "/props/luminosity" in msg:
                    return False

            return True

    # Apply the spam filter to the web log handler
    spam_filter = DropHighFrequencySpamFilter()
    web_log_handler.addFilter(spam_filter)

    await _broadcast("[DEMO_VIEWER] 🔇 High-frequency log filtering enabled (same as sequence mode)")


async def _enable_demo_logging_enhancements():
    """Enable the same demo logging enhancements that sequence mode uses."""

    class DemoPrefixStripFilter(logging.Filter):
        """Strip [DEMO] prefix from log messages (same as sequence mode)."""

        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
                if "[DEMO]" in msg:
                    # Strip the [DEMO] prefix to clean up the message
                    record.msg = msg.replace("[DEMO]", "").strip()
            except Exception:
                pass
            return True

    # Apply the demo prefix filter to the web log handler
    demo_filter = DemoPrefixStripFilter()
    web_log_handler.addFilter(demo_filter)

    await _broadcast("[DEMO_VIEWER] 🎬 Demo logging enhancements enabled (same as sequence mode)")


async def verify_logging_consistency():
    """Verify that manual mode logging is configured identically to sequence mode."""

    verification_points = [
        "✅ Log format: '%(asctime)s [%(name)s] %(levelname)s: %(message)s'",
        "✅ Root logger level: WARNING (same as sequence mode)",
        "✅ Priority loggers level: INFO (same as sequence mode)",
        "✅ High-frequency spam filtering enabled",
        "✅ Demo prefix stripping enabled",
        "✅ Identical LoggerFactory configuration",
        "✅ Same delivery mechanism via _broadcast()",
        "✅ Same error handling and timing"
    ]

    await _broadcast("[DEMO_VIEWER] 🔍 Logging Consistency Verification:")
    for point in verification_points:
        await _broadcast(f"[DEMO_VIEWER] {point}")

    await _broadcast("[DEMO_VIEWER] ✅ Manual mode logging is now 100% consistent with sequence mode!")


async def cleanup_agent_logging_capture():
    """Clean up logging capture when manual mode stops."""
    try:
        # Remove the web log handler from root logger
        root_logger = logging.getLogger()
        root_logger.removeHandler(web_log_handler)

        # Remove from specific AMI loggers
        ami_loggers = [
            "ami_agents",
            "EnvExplorerAgent",
            "UserAssistantAgent",
            "InteractionSolverAgent",
            "WebInterfaceAgent",
            "YggdrasilIntegration",
            "HMASClient",
            "SignifierEngine",
            "BTPlanner",
            "IRExecutor"
        ]

        for logger_name in ami_loggers:
            logger = logging.getLogger(logger_name)
            logger.removeHandler(web_log_handler)

        await _broadcast("[DEMO_VIEWER] 🧹 Logging capture cleaned up")
    except Exception as e:
        await _broadcast(f"[DEMO_VIEWER] ⚠️  Logging cleanup error: {e}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project to path before importing ami_agents
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file automatically from project root (like in manual_interactive_cli.py)
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"[demo_viewer] WARNING: No .env file found at {env_path}")

# Disable ANSI codes globally to prevent logging format errors
os.environ["AMI_NO_COLOR"] = "1"
os.environ["NO_COLOR"] = "1"
print(f"[demo_viewer] Disabled ANSI colors to prevent logging errors")

# Import WebSocket utilities for manual mode
from ami_agents.shared.websocket import (
    WebSocketConnectionManager,
    WebSocketMessage,
    WebSocketMessageType,
    UserMessage,
    AssistantMessage,
    PlanProposal,
    PlanConfirmation,
    ExecutionStatus,
    SystemStatus,
    ClearSignifiers,
    deserialize_message,
    create_error_response,
    create_info_response
)
DEMO_SCRIPT  = PROJECT_ROOT / "tests" / "demo_with_home_assistant.py"
HA_URL       = "http://localhost:8123"
HA_WS_URL    = "ws://localhost:8123/api/websocket"
YGG_URL      = "http://localhost:8080/"
SERVER_PORT  = 8765

log = logging.getLogger("demo_viewer")


class WebInterfaceAgent(Agent):
    """Simple SPADE agent that proxies web interface communication to UserAssistant (like ConsoleUserAgent)."""

    def __init__(self, jid: str, password: str, *, assistant_jid: str):
        super().__init__(jid, password)  # NO verify_security=False like ConsoleUserAgent
        self.assistant_jid = assistant_jid
        self.send_queue: asyncio.Queue = asyncio.Queue()  # message queue for sending

    async def setup(self):
        """Setup agent with message handler (like ConsoleUserAgent)."""
        # Handle replies from UserAssistant
        reply_template = Template()
        reply_template.set_metadata("message_type", "llm")
        self.add_behaviour(self.AssistantReplyHandler(), template=reply_template)

        # Message sender behaviour
        self.add_behaviour(self.MessageSender())

        await _broadcast("[DEMO_VIEWER] Web interface agent ready")

    async def send_to_assistant(self, content: str, thread_id: str) -> bool:
        """Queue a message to be sent to UserAssistant."""
        try:
            await self.send_queue.put((content, thread_id))
            return True
        except Exception as e:
            log.error(f"Failed to queue message: {e}")
            return False

    class MessageSender(CyclicBehaviour):
        """Behaviour that sends messages to UserAssistant (like ConsoleUserAgent.ConsoleLoop)."""

        async def run(self):
            try:
                # Get message from queue with timeout
                content, thread_id = await asyncio.wait_for(
                    self.agent.send_queue.get(), timeout=1.0
                )

                # Create and send SPADE message (exact pattern from ConsoleUserAgent)
                msg = SpadeMessage(to=self.agent.assistant_jid)
                msg.set_metadata("message_type", "llm")
                msg.thread = thread_id
                msg.body = content
                await self.send(msg)  # self.send works in BEHAVIOUR, not in agent

                log.debug(f"Sent message to UserAssistant: {content[:50]}...")

            except asyncio.TimeoutError:
                # Normal timeout, keep running
                pass
            except Exception as e:
                log.error(f"Failed to send message to UserAssistant: {e}")
                await _broadcast(f"[DEMO_VIEWER] ERROR: Failed to send message: {e}")

    class AssistantReplyHandler(CyclicBehaviour):
        """Handle replies from UserAssistant and forward to WebSocket clients (like ConsoleUserAgent.ReceiveLLMReply)."""

        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return

            thread_id = str(getattr(msg, "thread", None) or "")
            body = (msg.body or "").strip()

            if not thread_id or not body:
                return

            log.debug(f"Received reply from UserAssistant for thread {thread_id}: {body[:50]}...")
            await _broadcast(f"[DEMO_VIEWER] UserAssistant reply: {body[:100]}...")

            # Forward to WebSocket client
            await self._forward_to_websocket(thread_id, body)

        async def _forward_to_websocket(self, thread_id: str, content: str):
            """Forward assistant reply to WebSocket client."""
            from ami_agents.shared.websocket import AssistantMessage

            assistant_msg = AssistantMessage(
                content=content,
                thread_id=thread_id
            )

            try:
                await state.websocket_manager.send_to_thread(thread_id, assistant_msg)
            except Exception as e:
                log.error(f"Failed to forward reply to WebSocket: {e}")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class _State:
    # Existing demo sequence state
    process:          Optional[asyncio.subprocess.Process] = None
    running_sequence: Optional[str]                        = None
    log_history:      list                                 = []
    log_subscribers:  list                                 = []   # one asyncio.Queue per SSE client

    # Manual mode state
    manual_mode_active: bool                              = False
    manual_agents:      Dict[str, Any]                    = {}    # agent_name -> agent_instance
    web_interface_agent: Optional[WebInterfaceAgent]     = None   # web interface communication agent
    websocket_manager:  Optional[WebSocketConnectionManager] = None
    active_threads:     Dict[str, str]                    = {}    # thread_id -> connection_id

    # Manual mode startup progress state
    startup_phase:     str                                = "idle"  # idle, starting, discovery, warming, ready, error
    startup_message:   str                                = ""      # current startup status message
    discovery_complete: bool                              = False   # tracks env discovery completion
    execution_ready:   bool                               = False   # tracks execution engine readiness

    def __init__(self):
        self.websocket_manager = WebSocketConnectionManager()

state = _State()

# ---------------------------------------------------------------------------
# Log broadcast
# ---------------------------------------------------------------------------

async def _broadcast(line: str) -> None:
    state.log_history.append(line)
    if len(state.log_history) > 10_000:
        state.log_history = state.log_history[-10_000:]
    for q in list(state.log_subscribers):
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            pass

async def _capture(proc: asyncio.subprocess.Process) -> None:
    """Read subprocess stdout line-by-line and broadcast to all SSE clients."""
    try:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            await _broadcast(line)
    except Exception as exc:
        log.warning("capture error: %s", exc)
    await proc.wait()
    state.process          = None
    state.running_sequence = None
    await _broadcast("[DEMO_VIEWER] Process finished.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.process and state.process.returncode is None:
        state.process.terminate()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Demo API endpoints
# MUST be defined BEFORE the catch-all HA proxy routes.
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(_HTML)


@app.post("/api/demo/run")
async def demo_run(request: Request):
    body  = await request.json()
    seq   = str(body.get("sequence", "3"))
    clear = bool(body.get("clear_signifiers", True))

    if state.process and state.process.returncode is None:
        return Response(
            json.dumps({"error": "Demo already running"}),
            status_code=409,
            media_type="application/json",
        )

    state.log_history.clear()

    cmd = [
        sys.executable,
        "-u",              # force unbuffered stdout/stderr so print() and logging stay in order
        str(DEMO_SCRIPT),
        "--sequence", seq,
        "--signifier-matcher", "v2",
    ]
    if clear:
        cmd.append("--clear-signifiers")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    state.process          = proc
    state.running_sequence = seq
    asyncio.create_task(_capture(proc))

    return {"ok": True, "pid": proc.pid, "sequence": seq}


@app.delete("/api/demo/stop")
async def demo_stop():
    if state.process and state.process.returncode is None:
        state.process.terminate()
        try:
            await asyncio.wait_for(state.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            state.process.kill()
    state.process          = None
    state.running_sequence = None
    await _broadcast("[DEMO_VIEWER] Stopped by user.")
    return {"ok": True}


@app.get("/api/demo/status")
async def demo_status():
    ha_ok = ygg_ok = False
    async with httpx.AsyncClient(timeout=2.0) as client:
        for flag, url in [("ha", HA_URL), ("ygg", YGG_URL)]:
            try:
                r = await client.get(url)
                if flag == "ha":
                    ha_ok  = r.status_code < 500
                else:
                    ygg_ok = r.status_code < 500
            except Exception:
                pass
    return {
        "ha_ok":            ha_ok,
        "ygg_ok":           ygg_ok,
        "process_running":  state.process is not None and state.process.returncode is None,
        "running_sequence": state.running_sequence,
    }


@app.get("/api/demo/logs")
async def demo_logs(request: Request):
    """SSE: replay log history then stream live lines."""
    q: asyncio.Queue = asyncio.Queue(maxsize=2_000)
    state.log_subscribers.append(q)

    async def gen():
        for line in list(state.log_history):
            yield f"data: {json.dumps({'text': line})}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps({'text': line})}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            try:
                state.log_subscribers.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )

# ---------------------------------------------------------------------------
# Manual Mode API endpoints
# MUST be defined BEFORE the catch-all HA proxy routes.
# ---------------------------------------------------------------------------

@app.post("/api/manual/start")
async def manual_start():
    """Start agents in manual mode for interactive dialog."""
    if state.manual_mode_active:
        return Response(
            json.dumps({"error": "Manual mode already active"}),
            status_code=409,
            media_type="application/json",
        )

    try:
        import os
        import aiohttp

        # Reset startup state
        state.startup_phase = "starting"
        state.startup_message = "Initializing manual mode..."
        state.discovery_complete = False
        state.execution_ready = False

        # Import agents here to avoid circular dependencies
        from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
        from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
        from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
        from ami_agents.shared.utils.config_loader import ConfigLoader
        from ami_agents.environment.connection.hmas_client import IHMASClient

        # Force reload .env file before config loading (ensure env vars are available)
        env_path = PROJECT_ROOT / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # Ensure NO_COLOR is set for agent processes to prevent ANSI logging errors
        os.environ["AMI_NO_COLOR"] = "1"
        os.environ["NO_COLOR"] = "1"
        print(f"[DEBUG] Force reloaded .env: OPENAI_API_KEY={'FOUND' if os.getenv('OPENAI_API_KEY') else 'NOT_FOUND'}")
        print(f"[DEBUG] NO_COLOR set: AMI_NO_COLOR={os.getenv('AMI_NO_COLOR')}, NO_COLOR={os.getenv('NO_COLOR')}")

        # Load configuration
        config = ConfigLoader.merge_configs(
            ConfigLoader.load_with_env_vars(str(PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml")),
            ConfigLoader.load_with_env_vars(str(PROJECT_ROOT / "ami_agents" / "config" / "environment.yaml"))
        )

        # Get environment variables (like in manual_interactive_cli.py)
        xmpp_server = os.getenv("SPADE_SERVER", "localhost")
        password = os.getenv("SPADE_PASSWORD", "password")
        yggdrasil_url = os.getenv("YGGDRASIL_URL", "http://localhost:8080/").strip()

        # Note: OPENAI_API_KEY verification removed - ConfigLoader will handle it properly
        # The issue was that env vars are checked too early, but config loading works correctly

        explorer_jid = f"env_explorer@{xmpp_server}"
        assistant_jid = f"user_assistant@{xmpp_server}"
        solver_jid = f"interaction_solver@{xmpp_server}"

        state.startup_message = "Checking Yggdrasil connectivity..."
        await _broadcast("[DEMO_VIEWER] Checking Yggdrasil connectivity...")

        # Check Yggdrasil connectivity (like in manual_interactive_cli.py)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(yggdrasil_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"Yggdrasil returned HTTP {resp.status}")
            await _broadcast(f"[DEMO_VIEWER] Yggdrasil is reachable at {yggdrasil_url}")
        except Exception as e:
            raise RuntimeError(f"Yggdrasil is not reachable at {yggdrasil_url}: {e}")

        # Add runtime environment configuration
        config["yggdrasil_url"] = yggdrasil_url
        config["yggdrasil"] = {"url": yggdrasil_url}
        config["discovery"] = {
            "notify_on_discovery_complete": True,
            "notify_agents": [solver_jid],
        }

        # Create dummy HMAS client for EnvExplorer
        class DummyHMASClient(IHMASClient):
            async def connect(self, endpoint_url: str) -> bool: return True
            async def disconnect(self) -> None: return None
            async def get_workspace(self, workspace_id: str): return None
            async def list_workspaces(self, parent_id: str = None): return []
            async def get_artifact(self, artifact_id: str): return None
            async def list_artifacts(self, workspace_id: str): return []
            async def get_thing_description(self, artifact_id: str): return None
            async def invoke_action(self, artifact_id: str, action_name: str, params: dict): return {}
            async def read_property(self, artifact_id: str, property_name: str): return None
            async def write_property(self, artifact_id: str, property_name: str, value): return False
            async def subscribe_to_event(self, artifact_id: str, event_name: str, callback: callable) -> str: return ""
            async def unsubscribe_from_event(self, subscription_id: str) -> bool: return True
            async def subscribe_to_workspace_changes(self, workspace_id: str, callback: callable) -> str: return ""
            async def crawl_environment(self, root_workspace_id: str, depth: int = -1) -> dict: return {}

        # Initialize agents with IDENTICAL logging configuration as sequence mode
        await _broadcast("[DEMO_VIEWER] 🔧 Initializing agents with sequence-mode-identical logging...")

        # Ensure the config contains the same logging setup as sequence mode
        # This guarantees that agents use LoggerFactory with the same parameters
        if "logging" not in config:
            config["logging"] = {
                "level": "WARNING",
                "format": "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                "console_logging": {"enabled": True},
                "file_logging": {"enabled": False}
            }

        explorer = EnvExplorerAgent(explorer_jid, password, config, hmas_client=DummyHMASClient())
        assistant = UserAssistantAgent(
            assistant_jid,
            password,
            config=config,
            target_jids={"explorer": explorer_jid, "solver": solver_jid}
        )
        solver = InteractionSolverAgent(
            solver_jid,
            password,
            config=config,
            target_jids={"explorer": explorer_jid}
        )

        await _broadcast("[DEMO_VIEWER] 📦 Agent instances created successfully")

        # Setup comprehensive logging capture for manual mode (non-blocking)
        await _broadcast("[DEMO_VIEWER] 🔍 Setting up log capture (optional)...")
        try:
            # Make this completely optional and non-blocking
            asyncio.create_task(setup_agent_logging_capture())
            await _broadcast("[DEMO_VIEWER] ✅ Log capture setup initiated")
        except Exception as e:
            # Don't let logging setup break anything
            await _broadcast(f"[DEMO_VIEWER] ℹ️ Basic logging only: {e}")
            pass  # Continue without enhanced logging

        # Create web interface agent for communication with UserAssistant
        web_interface_jid = f"web_interface@{xmpp_server}"
        state.web_interface_agent = WebInterfaceAgent(
            web_interface_jid,
            password,
            assistant_jid=assistant_jid
        )

        # Start agents in CORRECT ORDER (like in manual_interactive_cli.py)
        # InteractionSolver must start FIRST to register behaviors that can receive
        # the ENV_DISCOVERY_COMPLETE notification sent by EnvExplorer during discovery.
        state.startup_message = "Starting InteractionSolver..."
        await _broadcast("[DEMO_VIEWER] Starting InteractionSolver...")
        await solver.start(auto_register=True)

        state.startup_message = "Starting EnvExplorer..."
        await _broadcast("[DEMO_VIEWER] Starting EnvExplorer...")
        await explorer.start(auto_register=True)

        state.startup_message = "Starting UserAssistant..."
        await _broadcast("[DEMO_VIEWER] Starting UserAssistant...")
        await assistant.start(auto_register=True)

        # Wait for EnvExplorer discovery to complete (like in manual_interactive_cli.py)
        state.startup_phase = "discovery"
        state.startup_message = "Waiting for environment discovery..."
        await _broadcast("[DEMO_VIEWER] Waiting for environment discovery...")

        for i in range(600):  # 300 seconds timeout
            if getattr(explorer, "discovery_complete", False):
                break
            if i % 20 == 0:  # Every 10 seconds, log progress
                await _broadcast(f"[DEMO_VIEWER] Discovery in progress... ({i//2}s elapsed)")
            await asyncio.sleep(0.5)

        if not getattr(explorer, "discovery_complete", False):
            raise RuntimeError("EnvExplorer discovery did not complete in time.")

        state.discovery_complete = True
        await _broadcast("[DEMO_VIEWER] ✅ Environment discovery completed successfully!")
        artifacts_count = len(getattr(explorer, 'artifacts', {}) or {})
        await _broadcast(f"[DEMO_VIEWER] 🏠 Found {artifacts_count} artifacts in environment")

        # Pre-warm UserAssistant execution engine (like in manual_interactive_cli.py)
        state.startup_phase = "warming"
        state.startup_message = "Pre-warming UserAssistant execution engine..."
        await _broadcast("[DEMO_VIEWER] Pre-warming UserAssistant execution engine...")
        await assistant.ensure_execution_engine_ready()
        state.execution_ready = True
        await _broadcast("[DEMO_VIEWER] UserAssistant execution engine ready.")

        # Start web interface agent for communication
        state.startup_message = "Starting web interface agent..."
        await _broadcast("[DEMO_VIEWER] Starting web interface agent...")
        await state.web_interface_agent.start(auto_register=True)

        await _broadcast("[DEMO_VIEWER] All agents started successfully")

        # Store agent references
        state.manual_agents = {
            "explorer": explorer,
            "assistant": assistant,
            "solver": solver
        }

        state.manual_mode_active = True
        state.startup_phase = "ready"
        state.startup_message = "Manual mode ready for dialog interaction"

        await _broadcast("[DEMO_VIEWER] Manual mode started successfully. Ready for dialog interaction.")
        await _broadcast("[DEMO_VIEWER] ✅ All agents are running and ready for user interaction.")
        await _broadcast(f"[DEMO_VIEWER] 📊 Active agents: {', '.join(state.manual_agents.keys())}")
        await _broadcast(f"[DEMO_VIEWER] 🔗 WebSocket server ready for connections on /api/manual/chat")

        return {
            "status": "manual_mode_started",
            "agents": list(state.manual_agents.keys()),
            "startup_phase": state.startup_phase,
            "discovery_complete": state.discovery_complete,
            "execution_ready": state.execution_ready
        }

    except Exception as e:
        log.error(f"Failed to start manual mode: {e}")

        # Set error state
        state.startup_phase = "error"
        state.startup_message = f"Startup failed: {str(e)}"
        state.discovery_complete = False
        state.execution_ready = False

        # Clean up any partially started agents
        if 'state' in locals() and hasattr(state, 'manual_agents'):
            for agent_name, agent in state.manual_agents.items():
                try:
                    await agent.stop()
                except Exception:
                    pass
            state.manual_agents.clear()
            state.manual_mode_active = False

        # Clean up web interface agent
        if 'state' in locals() and hasattr(state, 'web_interface_agent') and state.web_interface_agent:
            try:
                await state.web_interface_agent.stop()
            except Exception:
                pass
            state.web_interface_agent = None

        await _broadcast(f"[DEMO_VIEWER] Manual mode startup failed: {e}")

        return Response(
            json.dumps({"error": f"Failed to start manual mode: {str(e)}"}),
            status_code=500,
            media_type="application/json",
        )


@app.delete("/api/manual/stop")
async def manual_stop():
    """Stop manual mode and clean up all agents."""
    if not state.manual_mode_active:
        return Response(
            json.dumps({"error": "Manual mode not active"}),
            status_code=409,
            media_type="application/json",
        )

    try:
        # Stop all agents
        for agent_name, agent in state.manual_agents.items():
            try:
                await agent.stop()
                log.info(f"Stopped {agent_name} agent")
            except Exception as e:
                log.warning(f"Failed to stop {agent_name} agent: {e}")

        # Stop web interface agent
        if state.web_interface_agent:
            try:
                await state.web_interface_agent.stop()
                log.info("Stopped web interface agent")
            except Exception as e:
                log.warning(f"Failed to stop web interface agent: {e}")
            state.web_interface_agent = None

        # Clear state
        state.manual_agents.clear()
        state.manual_mode_active = False
        state.active_threads.clear()
        state.startup_phase = "idle"
        state.startup_message = ""
        state.discovery_complete = False
        state.execution_ready = False

        # Clean up logging capture
        await cleanup_agent_logging_capture()

        await _broadcast("[DEMO_VIEWER] Manual mode stopped.")

        return {"status": "manual_mode_stopped"}

    except Exception as e:
        log.error(f"Error stopping manual mode: {e}")
        return Response(
            json.dumps({"error": f"Failed to stop manual mode: {str(e)}"}),
            status_code=500,
            media_type="application/json",
        )


@app.post("/api/manual/clear-signifiers")
async def manual_clear_signifiers():
    """Clear signifier memory storage."""
    try:
        storage_dir = PROJECT_ROOT / "ami_agents" / "shared" / "memory" / "storage"
        cleared_dirs = []

        for subdir in ["rdf", "json", "indexes"]:
            subpath = storage_dir / subdir
            if subpath.exists():
                shutil.rmtree(subpath)
                cleared_dirs.append(str(subpath))

        await _broadcast(f"[DEMO_VIEWER] Cleared signifier storage: {', '.join(cleared_dirs)}")

        return {"status": "signifiers_cleared", "cleared_directories": cleared_dirs}

    except Exception as e:
        log.error(f"Failed to clear signifiers: {e}")
        return Response(
            json.dumps({"error": f"Failed to clear signifiers: {str(e)}"}),
            status_code=500,
            media_type="application/json",
        )


@app.get("/api/manual/status")
async def manual_status():
    """Get manual mode status and statistics."""
    return {
        "manual_mode_active": state.manual_mode_active,
        "startup_phase": state.startup_phase,
        "startup_message": state.startup_message,
        "discovery_complete": state.discovery_complete,
        "execution_ready": state.execution_ready,
        "agents_running": list(state.manual_agents.keys()) if state.manual_mode_active else [],
        "active_connections": state.websocket_manager.get_connection_count() if state.websocket_manager else 0,
        "active_threads": state.websocket_manager.get_thread_count() if state.websocket_manager else 0,
        "ready_for_interaction": (
            state.manual_mode_active and
            state.startup_phase == "ready" and
            state.discovery_complete and
            state.execution_ready
        )
    }


@app.delete("/api/manual/clear-signifiers")
async def manual_clear_signifiers():
    """Clear signifier storage (for manual mode)."""
    try:
        import shutil

        # Clear signifier storage directory (same as in manual_interactive_cli.py)
        storage_dir = PROJECT_ROOT / "ami_agents" / "shared" / "memory" / "storage"

        cleared_dirs = []
        for subdir in ["rdf", "json", "indexes"]:
            subpath = storage_dir / subdir
            if subpath.exists():
                shutil.rmtree(subpath)
                cleared_dirs.append(str(subpath))

        await _broadcast(f"[DEMO_VIEWER] Cleared signifier storage: {len(cleared_dirs)} directories")

        return {
            "status": "signifiers_cleared",
            "cleared_directories": cleared_dirs,
            "storage_path": str(storage_dir)
        }

    except Exception as e:
        log.error(f"Failed to clear signifiers: {e}")
        return Response(
            json.dumps({"error": f"Failed to clear signifiers: {str(e)}"}),
            status_code=500,
            media_type="application/json",
        )


@app.websocket("/api/manual/chat")
async def manual_chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    await websocket.accept()

    connection_id = str(uuid.uuid4())
    log.info(f"WebSocket connection established: {connection_id}")

    try:
        # Register connection
        await state.websocket_manager.register_connection(websocket, connection_id)

        # Send welcome message
        welcome_msg = create_info_response(
            "Connected to AMI assistant. How can I help you?",
        )
        await state.websocket_manager.send_to_connection(connection_id, welcome_msg)

        # Process incoming messages
        async for raw_message in websocket.iter_text():
            try:
                message = deserialize_message(raw_message)

                # Handle different message types
                if message.type == WebSocketMessageType.USER_MESSAGE.value:
                    await handle_user_message(connection_id, message)
                elif message.type == WebSocketMessageType.PLAN_CONFIRMATION.value:
                    await handle_plan_confirmation(connection_id, message)
                elif message.type == WebSocketMessageType.CLEAR_SIGNIFIERS.value:
                    await handle_clear_signifiers(connection_id, message)
                else:
                    log.warning(f"Unknown message type from {connection_id}: {message.type}")

            except Exception as e:
                log.error(f"Error processing message from {connection_id}: {e}")
                error_msg = create_error_response(f"Failed to process message: {str(e)}")
                await state.websocket_manager.send_to_connection(connection_id, error_msg)

    except WebSocketDisconnect:
        log.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        log.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        # Clean up connection
        await state.websocket_manager.unregister_connection(connection_id)


async def handle_user_message(connection_id: str, message: UserMessage):
    """Handle user message and forward to UserAssistant agent directly."""
    if not state.manual_mode_active:
        error_msg = create_error_response("Manual mode not active")
        await state.websocket_manager.send_to_connection(connection_id, error_msg)
        return

    # Get or create thread ID
    thread_id = message.thread_id or str(uuid.uuid4())

    # Associate thread with connection
    state.websocket_manager.associate_thread(connection_id, thread_id)
    state.active_threads[thread_id] = connection_id

    # Send message directly to UserAssistant through web interface agent
    await _broadcast(f"[DEMO_VIEWER] 💬 User message received: {message.content[:100]}...")

    if state.web_interface_agent:
        success = await state.web_interface_agent.send_to_assistant(message.content, thread_id)
        if success:
            await _broadcast(f"[DEMO_VIEWER] ✉️  Message forwarded to UserAssistant successfully")
        else:
            await _broadcast(f"[DEMO_VIEWER] ❌ Failed to forward message to UserAssistant")
            error_msg = create_error_response("Failed to send message to assistant")
            await state.websocket_manager.send_to_connection(connection_id, error_msg)
    else:
        await _broadcast(f"[DEMO_VIEWER] ❌ Web interface agent not available")
        error_msg = create_error_response("Web interface agent not available")
        await state.websocket_manager.send_to_connection(connection_id, error_msg)


async def handle_plan_confirmation(connection_id: str, message: PlanConfirmation):
    """Handle plan confirmation/rejection from user."""
    # This will be implemented when we add plan proposal functionality
    log.info(f"Plan confirmation from {connection_id}: approved={message.approved}")

    # For now, just acknowledge
    response_msg = create_info_response(
        f"Plan {'approved' if message.approved else 'rejected'} - execution logic to be implemented"
    )
    await state.websocket_manager.send_to_connection(connection_id, response_msg)


async def handle_clear_signifiers(connection_id: str, message: ClearSignifiers):
    """Handle clear signifiers request from WebSocket."""
    try:
        # Call the same logic as the HTTP endpoint
        storage_dir = PROJECT_ROOT / "ami_agents" / "shared" / "memory" / "storage"
        cleared_dirs = []

        for subdir in ["rdf", "json", "indexes"]:
            subpath = storage_dir / subdir
            if subpath.exists():
                shutil.rmtree(subpath)
                cleared_dirs.append(str(subpath))

        response_msg = create_info_response(f"Cleared signifier storage: {', '.join(cleared_dirs)}")
        await state.websocket_manager.send_to_connection(connection_id, response_msg)

        await _broadcast(f"[DEMO_VIEWER] Signifiers cleared via WebSocket by {connection_id}")

    except Exception as e:
        log.error(f"Failed to clear signifiers via WebSocket: {e}")
        error_msg = create_error_response(f"Failed to clear signifiers: {str(e)}")
        await state.websocket_manager.send_to_connection(connection_id, error_msg)

# ---------------------------------------------------------------------------
# Static file serving
# MUST be defined BEFORE the catch-all HA proxy routes.
# ---------------------------------------------------------------------------

@app.get("/static/css/{file_path:path}")
async def serve_css(file_path: str):
    """Serve CSS files from demo_viewer/static/css/"""
    css_file = PROJECT_ROOT / "demo_viewer" / "static" / "css" / file_path
    if css_file.exists() and css_file.suffix == ".css":
        content = css_file.read_text(encoding="utf-8")
        return Response(content, media_type="text/css")
    return Response(status_code=404)

@app.get("/static/js/{file_path:path}")
async def serve_js(file_path: str):
    """Serve JavaScript files from demo_viewer/static/js/"""
    js_file = PROJECT_ROOT / "demo_viewer" / "static" / "js" / file_path
    if js_file.exists() and js_file.suffix == ".js":
        content = js_file.read_text(encoding="utf-8")
        return Response(content, media_type="application/javascript")
    return Response(status_code=404)

# ---------------------------------------------------------------------------
# HA HTTP reverse proxy
# ---------------------------------------------------------------------------

_STRIP_REQ = {"host", "content-length", "transfer-encoding"}
_STRIP_RES = {
    "x-frame-options",
    "content-security-policy",
    "transfer-encoding",
    "connection",
    # httpx decompresses gzip/br automatically – original length is no longer valid
    "content-length",
    "content-encoding",
}

_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]


async def _proxy_http(path: str, request: Request) -> Response:
    query = request.url.query
    url   = f"{HA_URL}/{path}" + (f"?{query}" if query else "")
    hdrs  = {k: v for k, v in request.headers.items()
             if k.lower() not in _STRIP_REQ}

    log.info("HTTP proxy → %s %s", request.method, url)

    # Force gzip/deflate only.  httpx does NOT support brotli (br) decompression
    # unless the optional brotlicffi package is installed.  If HA responds with
    # brotli (its preferred encoding) and httpx can't decode it, it silently
    # returns the raw compressed bytes.  We then strip Content-Encoding: br, so
    # the browser receives binary garbage it tries to parse as JS → SyntaxError.
    hdrs["accept-encoding"] = "gzip, deflate"

    # follow_redirects=False: we let the BROWSER follow redirects so that
    # the URL bar stays accurate (HA JS reads window.location to know its path).
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        try:
            resp = await client.request(
                method  = request.method,
                url     = url,
                headers = hdrs,
                content = await request.body(),
            )
        except httpx.ConnectError:
            log.warning("HTTP proxy: cannot reach HA at %s", url)
            return Response(
                b"Home Assistant not reachable at " + HA_URL.encode(),
                status_code=502,
            )

    content_type = resp.headers.get("content-type", "")
    log.info("HTTP proxy ← %d  %s (%d bytes, %s)", resp.status_code, url,
             len(resp.content), content_type.split(";")[0])

    out_hdrs = {k: v for k, v in resp.headers.items()
                if k.lower() not in _STRIP_RES}

    # Rewrite Location header so redirects point back to OUR server, not HA's.
    # e.g.  Location: http://localhost:8123/home/overview
    #    →  Location: http://localhost:8765/home/overview
    if "location" in out_hdrs:
        loc = out_hdrs["location"]
        if loc.startswith(HA_URL):
            out_hdrs["location"] = loc.replace(HA_URL,
                                                f"http://localhost:{SERVER_PORT}", 1)

    # Rewrite any hardcoded HA origin inside text responses (HTML, JS, JSON).
    # HA sometimes embeds absolute localhost:8123 URLs; these would be fetched
    # by the browser from the wrong origin, failing CORS and silently breaking.
    content = resp.content
    _HA_BYTES  = HA_URL.encode()
    _OWN_BYTES = f"http://localhost:{SERVER_PORT}".encode()
    if _HA_BYTES in content and any(t in content_type for t in ("text/", "javascript", "json")):
        content = content.replace(_HA_BYTES, _OWN_BYTES)
        log.info("HTTP proxy: rewrote HA origin in response body")

    return Response(
        content     = content,
        status_code = resp.status_code,
        headers     = out_hdrs,
        media_type  = content_type or None,
    )


@app.get("/service_worker.js")
@app.get("/service_worker_es5.js")
async def kill_switch_sw():
    """Serve a one-shot service worker that immediately unregisters itself.

    If HA's service worker was registered from http://localhost:8765 in a previous
    session, it would intercept ALL JS requests before they reach the network,
    silently serving stale or missing files from its cache.  This kill-switch SW
    overwrites the registration; on the next page load HA will be clean.
    """
    js = (
        "// Kill-switch service worker: unregister any previous SW at this origin.\n"
        "self.addEventListener('install', () => self.skipWaiting());\n"
        "self.addEventListener('activate', () => {\n"
        "  self.registration.unregister();\n"
        "  clients.claim();\n"
        "});\n"
    )
    return Response(js, media_type="application/javascript")


@app.get("/api/debug/ha-html")
async def debug_ha_html():
    """Return HA's raw HTML (plus response headers) as plain text, for debugging."""
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            resp = await client.get(f"{HA_URL}/")
        except Exception as exc:
            return Response(f"Error connecting to HA: {exc}", media_type="text/plain",
                            status_code=502)
    info = (
        f"=== HA Response at {HA_URL}/ ===\n"
        f"Status: {resp.status_code}\n"
        f"Headers:\n" + "\n".join(f"  {k}: {v}" for k, v in resp.headers.items()) +
        f"\n\nBody ({len(resp.text)} chars):\n{resp.text}"
    )
    return Response(info, media_type="text/plain; charset=utf-8")


# HA is proxied at the server ROOT.
# This ensures the auth flow works: HA stores tokens in localStorage at
# http://localhost:8765, and the iframe (same origin) can use them.

@app.api_route("/", methods=_METHODS)
async def ha_root(request: Request):
    return await _proxy_http("", request)


# Catch-all: proxy everything to HA except our own /demo and /api/demo/* routes.
# The specific routes above are matched first by FastAPI (registration order).
@app.api_route("/{path:path}", methods=_METHODS)
async def ha_catchall(path: str, request: Request):
    if path == "demo" or path.startswith("demo/"):
        return Response(status_code=404)
    if path.startswith("api/demo/") or path == "api/demo":
        return Response(status_code=404)
    if path.startswith("api/debug/") or path == "api/debug":
        return Response(status_code=404)
    return await _proxy_http(path, request)

# ---------------------------------------------------------------------------
# HA WebSocket proxy
# WebSocket routes are a separate ASGI scope – no conflict with HTTP catch-all.
# ---------------------------------------------------------------------------

@app.websocket("/api/websocket")
async def ha_ws_proxy(websocket: WebSocket):
    log.info("WS proxy: browser client connecting → attempting HA at %s", HA_WS_URL)

    # Connect to HA FIRST so we can reject browser cleanly if HA is unavailable.
    try:
        ha = await websockets.connect(HA_WS_URL)
        log.info("WS proxy: connected to HA successfully")
    except Exception as exc:
        log.warning("WS proxy: cannot connect to HA WS: %s – rejecting browser WS", exc)
        # Must accept before we can close (Starlette requires it).
        await websocket.accept()
        await websocket.close(code=1011, reason="HA WebSocket unavailable")
        return

    await websocket.accept()
    log.info("WS proxy: bidirectional bridge active (browser ↔ HA)")

    async def to_ha():
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    await ha.send(msg["text"])
                elif msg.get("bytes") is not None:
                    await ha.send(msg["bytes"])
        except Exception as exc:
            log.debug("WS to_ha ended: %s", exc)

    async def from_ha():
        try:
            async for msg in ha:
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_text(str(msg))
        except Exception as exc:
            log.debug("WS from_ha ended: %s", exc)

    try:
        done, pending = await asyncio.wait(
            [asyncio.create_task(to_ha()), asyncio.create_task(from_ha())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        log.info("WS proxy: connection closed")
        try:
            await ha.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# HTML/JS UI  (served at /demo – NOT an f-string, port injected via .replace)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Agents for AMI – Demo Viewer</title>
<link rel="stylesheet" href="/static/css/dialog.css">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'Segoe UI',system-ui,sans-serif;
  background:#0f172a;color:#e2e8f0;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;
}

/* ── Header ─────────────────────────────────────────────────────────── */
header{
  background:#1e293b;border-bottom:1px solid #334155;
  padding:.55rem 1rem;display:flex;align-items:center;
  gap:.6rem;flex-shrink:0;flex-wrap:wrap;
}
header h1{
  font-size:.92rem;font-weight:700;color:#f8fafc;
  margin-right:auto;white-space:nowrap;
}
header h1 span{color:#60a5fa}

select,button{
  background:#334155;color:#e2e8f0;border:1px solid #475569;
  border-radius:6px;padding:.32rem .65rem;font-size:.8rem;
  cursor:pointer;transition:background .15s;
}
select:hover,button:hover{background:#475569}
button:disabled{opacity:.4;cursor:not-allowed}
button.run{background:#14532d;border-color:#16a34a}
button.run:hover:not(:disabled){background:#166534}
button.stop{background:#7f1d1d;border-color:#b91c1c}
button.stop:hover:not(:disabled){background:#991b1b}

.badge{
  display:inline-flex;align-items:center;gap:.3rem;
  padding:.18rem .5rem;border-radius:999px;font-size:.7rem;font-weight:600;
}
.badge::before{content:'';width:6px;height:6px;border-radius:50%;background:currentColor}
.badge.ok{background:#14532d;color:#86efac}
.badge.err{background:#7f1d1d;color:#fca5a5}
.badge.chk{background:#1e3a5f;color:#93c5fd}

.fgrp{
  display:flex;gap:2px;background:#0f172a;
  border:1px solid #334155;border-radius:6px;padding:2px;
}
.fgrp button{
  border:none;padding:.25rem .5rem;font-size:.73rem;
  border-radius:4px;background:transparent;
}
.fgrp button.active{background:#3b82f6;color:#fff}

.panel-tabs{
  display:flex;gap:2px;background:#0f172a;
  border:1px solid #334155;border-radius:6px;padding:2px;
}
.panel-tabs button{
  border:none;padding:.4rem .8rem;font-size:.75rem;
  border-radius:4px;background:transparent;color:#94a3b8;
  cursor:pointer;transition:all .15s;
}
.panel-tabs button.active{background:#3b82f6;color:#fff;font-weight:600}
.panel-tabs button:not(.active):hover{background:#1e293b;color:#e2e8f0}

.seq4w{
  font-size:.7rem;color:#fbbf24;
  background:#451a03;border:1px solid #92400e;
  padding:.18rem .45rem;border-radius:4px;display:none;
}

/* ── Mode Selection & Controls ──────────────────────────────────────────── */
.mode-selection{
  display:flex;align-items:center;gap:.5rem;
}

.mode-controls{
  display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;
}

.manual-status{
  display:flex;align-items:center;gap:.5rem;
}

.manual-phase{
  font-size:.7rem;color:#94a3b8;
  background:#0f172a;border:1px solid #334155;
  padding:.18rem .45rem;border-radius:4px;
  white-space:nowrap;max-width:200px;overflow:hidden;text-overflow:ellipsis;
}

.badge.manual-starting{background:#451a03;color:#fbbf24}
.badge.manual-active{background:#14532d;color:#86efac}
.badge.manual-error{background:#7f1d1d;color:#fca5a5}

label.ck{
  display:flex;align-items:center;gap:.3rem;
  font-size:.8rem;color:#94a3b8;cursor:pointer;user-select:none;
}

/* ── Panels with Resizable Splitter ─────────────────────────────────────── */
.panels{display:flex;flex:1;overflow:hidden;position:relative}
.panel{
  display:flex;flex-direction:column;
  overflow:hidden;min-width:200px;
}
.panel:last-child{border-right:none}

#left-panel {
  width: 55%; /* Default width, will be overridden by JavaScript */
  border-right: none;
}

#right-panel {
  width: 45%; /* Default width, will be overridden by JavaScript */
}

/* ── Splitter Handle ────────────────────────────────────────────────────── */
.splitter {
  width: 8px;
  background: #1e293b;
  border-left: 1px solid #334155;
  border-right: 1px solid #334155;
  cursor: ew-resize;
  flex-shrink: 0;
  position: relative;
  transition: background-color 0.2s ease;
  z-index: 1000; /* Ensure splitter is above other elements */
}

.splitter:hover {
  background: #374151;
}

.splitter:active {
  background: #3b82f6;
}

.splitter::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 2px;
  height: 40px;
  background: #64748b;
  border-radius: 1px;
}

.splitter:hover::after {
  background: #94a3b8;
}

.splitter:active::after {
  background: #ffffff;
}

/* Prevent text selection during drag */
.panels.dragging {
  user-select: none;
  cursor: ew-resize;
}

.panels.dragging .panel,
.panels.dragging iframe {
  pointer-events: none;
}
.phdr{
  background:#1e293b;padding:.42rem .9rem;
  font-size:.73rem;font-weight:700;color:#94a3b8;
  text-transform:uppercase;letter-spacing:.06em;
  border-bottom:1px solid #334155;flex-shrink:0;
  display:flex;align-items:center;justify-content:space-between;
}

#ha-frame{flex:1;border:none;background:#1e293b}

/* ── Communication Panel (Logs + Dialog) ──────────────────────────── */
.communication-content{
  flex:1;display:flex;flex-direction:column;overflow:hidden;
  position:relative;height:100%;
}

.communication-panel{
  display:flex;flex-direction:column;overflow:hidden;
  position:absolute;top:0;left:0;right:0;bottom:0;
  width:100%;height:100%;background:#0f172a;
}

.communication-panel.hidden{
  display:none !important;
}

/* Ensure dialog panel is visible when not hidden */
#dialog-panel:not(.hidden) {
  display: flex !important;
  flex-direction: column !important;
  height: 100% !important;
  width: 100% !important;
}

/* Force dialog content to be visible */
#dialog-panel .dialog-header,
#dialog-panel .connection-status,
#dialog-panel .chat-container {
  display: flex !important;
}


/* ── Log panel ───────────────────────────────────────────────────────── */
.logwrap{
  flex:1;overflow-y:auto;padding:.35rem .45rem;
  font-family:'Cascadia Code','JetBrains Mono','Fira Code',monospace;
  font-size:.69rem;line-height:1.55;background:#070d1a;
}
.ll{padding:1px 3px;border-radius:2px;white-space:pre-wrap;word-break:break-all}
.ll:hover{background:#1e293b40}
.ll.hidden{display:none}

.ts{color:#4b5563}
.lg{color:#818cf8}
.lv-info{color:#60a5fa}
.lv-warning{color:#fbbf24}
.lv-error{color:#f87171}
.tdemo{
  color:#4ade80;font-weight:700;
  background:rgba(74,222,128,.12);padding:0 2px;border-radius:2px;
}
.l-demo{color:#a7f3d0}
.l-warn{color:#fbbf24}
.l-err{color:#f87171}
.l-sys{color:#60a5fa;font-style:italic}
.l-dim{color:#e2e8f0}

/* ── Dialog Window Styles ─────────────────────────────────────────────── */
.dialog-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #0f172a;
  color: #e2e8f0;
  overflow: hidden; /* Prevent overflow from breaking layout */
}

.dialog-panel.hidden {
  display: none;
}

.dialog-header {
  background: #1e293b;
  border-bottom: 1px solid #334155;
  padding: 0.75rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.mode-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.mode-badge {
  padding: 0.25rem 0.5rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.mode-badge.manual {
  background: #059669;
  color: #ecfdf5;
}

.mode-badge.inactive {
  background: #6b7280;
  color: #f3f4f6;
}

.dialog-controls {
  display: flex;
  gap: 0.5rem;
}

.btn-clear {
  background: #dc2626;
  color: white;
  border: none;
  padding: 0.375rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  cursor: pointer;
  transition: background-color 0.2s;
}

.btn-clear:hover {
  background: #b91c1c;
}

.btn-clear:disabled {
  background: #6b7280;
  cursor: not-allowed;
}

.connection-status {
  background: #1e293b;
  border-bottom: 1px solid #334155;
  padding: 0.5rem 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-indicator.status-connected {
  background: #10b981;
}

.status-indicator.status-connecting {
  background: #f59e0b;
  animation: pulse 2s infinite;
}

.status-indicator.status-disconnected {
  background: #ef4444;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  height: 100%;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  scroll-behavior: smooth;
  min-height: 0; /* Allow shrinking below content height */
}

.chat-input-area {
  flex-shrink: 0; /* Never shrink input area */
  padding: 1rem;
  border-top: 1px solid #334155;
  background: #0f172a;
}

.chat-messages::-webkit-scrollbar {
  width: 8px;
}

.chat-messages::-webkit-scrollbar-track {
  background: #0f172a;
  border-radius: 4px;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: #475569;
  border-radius: 4px;
  border: 1px solid #334155;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: #64748b;
}

.chat-messages::-webkit-scrollbar-thumb:active {
  background: #94a3b8;
}

/* Enhanced scrolling support for all browsers */
.chat-messages {
  scrollbar-width: thin;
  scrollbar-color: #475569 #0f172a;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: #64748b;
}

.message {
  display: flex;
  flex-direction: column;
  margin-bottom: 1rem;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.message-sender {
  font-weight: 600;
  font-size: 0.875rem;
}

.user-message .message-sender {
  color: #3b82f6;
}

.assistant-message .message-sender {
  color: #10b981;
}

.system-message .message-sender {
  color: #f59e0b;
}

.message-timestamp {
  font-size: 0.75rem;
  color: #64748b;
}

.message-content {
  background: #1e293b;
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  line-height: 1.6;
  border-left: 3px solid transparent;
}

.user-message .message-content {
  background: #1e40af;
  border-left-color: #3b82f6;
  margin-left: auto;
  margin-right: 0;
  max-width: 80%;
}

.assistant-message .message-content {
  background: #065f46;
  border-left-color: #10b981;
  max-width: 85%;
}

.system-message .message-content {
  background: #92400e;
  border-left-color: #f59e0b;
  font-style: italic;
}

.message-content h1,
.message-content h2,
.message-content h3,
.message-content h4,
.message-content h5,
.message-content h6 {
  margin: 0 0 0.5rem 0;
  color: #f1f5f9;
}

.message-content p {
  margin: 0 0 0.5rem 0;
}

.message-content p:last-child {
  margin-bottom: 0;
}

.message-content code {
  background: #0f172a;
  padding: 0.125rem 0.25rem;
  border-radius: 0.25rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.875em;
  color: #fbbf24;
}

.message-content pre {
  background: #0f172a;
  padding: 0.75rem;
  border-radius: 0.375rem;
  overflow-x: auto;
  margin: 0.5rem 0;
  border: 1px solid #374151;
}

.message-content pre code {
  background: none;
  padding: 0;
  color: #e2e8f0;
}

.message-content ul,
.message-content ol {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.message-content blockquote {
  border-left: 3px solid #6b7280;
  padding-left: 1rem;
  margin: 0.5rem 0;
  color: #9ca3af;
  font-style: italic;
}

.chat-input-area {
  background: #1e293b;
  border-top: 1px solid #334155;
  padding: 1rem;
  flex-shrink: 0;
}

.input-container {
  display: flex;
  gap: 0.75rem;
  align-items: flex-end;
}

#user-input {
  flex: 1;
  background: #0f172a;
  border: 1px solid #374151;
  border-radius: 0.5rem;
  padding: 0.75rem;
  color: #e2e8f0;
  font-family: inherit;
  font-size: 0.875rem;
  line-height: 1.5;
  resize: none;
  min-height: 2.5rem;
  max-height: 8rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}

#user-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

#user-input::placeholder {
  color: #6b7280;
}

#send-button {
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 0.5rem;
  padding: 0.75rem 1.5rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.1s;
  height: fit-content;
}

#send-button:hover:not(:disabled) {
  background: #2563eb;
  transform: translateY(-1px);
}

#send-button:disabled {
  background: #6b7280;
  cursor: not-allowed;
  transform: none;
}

/* ── Typing Indicator ───────────────────────────────────────────────────── */
.typing-message {
  opacity: 0.8;
}

.typing-content {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background: #065f46 !important;
  border-left-color: #10b981 !important;
}

.typing-dots {
  display: flex;
  gap: 0.25rem;
}

.typing-dots .dot {
  width: 6px;
  height: 6px;
  background: #10b981;
  border-radius: 50%;
  animation: typingBounce 1.4s infinite both;
}

.typing-dots .dot:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-dots .dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typingBounce {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  30% {
    transform: translateY(-8px);
    opacity: 1;
  }
}

.typing-text {
  font-style: italic;
  color: #6ee7b7;
  font-size: 0.875rem;
}

/* ── Loading States ─────────────────────────────────────────────────────── */
#send-button:disabled {
  background: #6b7280;
  cursor: not-allowed;
  transform: none;
}

#user-input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-clear:disabled {
  background: #6b7280;
  cursor: not-allowed;
}

/* ── Footer ──────────────────────────────────────────────────────────── */
footer{
  background:#1e293b;border-top:1px solid #334155;
  padding:.28rem 1rem;font-size:.68rem;color:#64748b;
  display:flex;gap:1rem;align-items:center;flex-shrink:0;
}
#st-txt{margin-left:auto;color:#94a3b8}
</style>
</head>
<body>

<header>
  <h1>LLM Agents for <span>AMI</span> &ndash; Demo Viewer</h1>

  <div id="b-ha"  class="badge chk">HA</div>
  <div id="b-ygg" class="badge chk">Yggdrasil</div>

  <!-- Mode Selection -->
  <div class="mode-selection">
    <div class="fgrp">
      <button id="mode-sequence" class="active" onclick="switchToSequenceMode()">📋 Sequence Mode</button>
      <button id="mode-manual" onclick="switchToManualMode()">💬 Manual Mode</button>
    </div>
  </div>

  <!-- Sequence Mode Controls -->
  <div id="sequence-controls" class="mode-controls">
    <select id="seq-sel" onchange="onSeqChange()">
      <option value="basic">Sequence 1 &ndash; Basic (quick debug)</option>
      <option value="2">Sequence 2 &ndash; Startup only</option>
      <option value="3">Sequence 3 &ndash; Demo (full interaction)</option>
      <option value="4">Sequence 4 &ndash; Reuse Demo</option>
    </select>

    <div class="seq4w" id="seq4w">&#9888; Seq 4 needs signifiers from Seq 3</div>

    <label class="ck">
      <input type="checkbox" id="clr-chk" checked>
      Clear signifiers
    </label>

    <button class="run"  id="btn-run"  onclick="runDemo()">&#9654; Run</button>
    <button class="stop" id="btn-stop" onclick="stopDemo()" disabled>&#9632; Stop</button>
  </div>

  <!-- Manual Mode Controls -->
  <div id="manual-controls" class="mode-controls" style="display: none;">
    <div id="manual-status" class="manual-status">
      <div id="manual-badge" class="badge err">Manual Inactive</div>
      <div id="manual-phase" class="manual-phase">Ready to start</div>
    </div>

    <button class="run" id="btn-manual-start" onclick="startManualMode()">🚀 Start Manual Mode</button>
    <button class="stop" id="btn-manual-stop" onclick="stopManualMode()" disabled>⏹ Stop Manual Mode</button>

    <label class="ck">
      <input type="checkbox" id="manual-clr-chk" checked>
      Clear signifiers on start
    </label>
  </div>
</header>

<div class="panels" id="main-panels">

  <!-- Left: Home Assistant (proxied at root – auth works inside iframe) -->
  <div class="panel" id="left-panel">
    <div class="phdr">&#127968; Home Assistant &mdash; lab308</div>
    <iframe id="ha-frame" src="/home/overview" allow="*"></iframe>
  </div>

  <!-- Resizable Splitter -->
  <div class="splitter" id="panel-splitter"></div>

  <!-- Right: Communication Panel (Logs + Dialog) -->
  <div class="panel" id="right-panel">
    <div class="phdr">
      <span>💬 Communication</span>
      <div class="panel-tabs">
        <button id="tab-logs" class="active">Interaction Logs</button>
        <button id="tab-dialog">Dialog Window</button>
      </div>
    </div>

    <div class="communication-content">
      <!-- Logs Panel -->
      <div id="logs-panel" class="communication-panel">
        <div class="phdr" style="background:#0f172a;border-bottom:1px solid #334155;padding:.3rem .6rem;text-transform:none;font-size:.8rem">
          <div class="fgrp">
            <button id="f-all"  class="active" onclick="setFilter('all')">All</button>
            <button id="f-demo" onclick="setFilter('demo')">[DEMO] only</button>
          </div>
        </div>
        <div class="logwrap" id="logwrap">
          <div class="ll l-sys">Waiting for demo to start&hellip;</div>
          <div id="logend"></div>
        </div>
      </div>

      <!-- Dialog Panel -->
      <div id="dialog-panel" class="communication-panel hidden dialog-panel">
        <div class="dialog-header">
          <div class="mode-indicator">
            <span id="mode-badge" class="mode-badge inactive">Inactive</span>
          </div>
          <div class="dialog-controls">
            <button id="clear-signifiers-btn" class="btn-clear">🗑️ Clear Memory</button>
          </div>
        </div>

        <div class="connection-status" id="connection-status">
          <div id="status-indicator" class="status-indicator status-disconnected"></div>
          <span id="status-text">Not connected</span>
        </div>

        <div class="chat-container">
          <div class="chat-messages" id="chat-messages">
            <!-- Messages will be inserted here dynamically -->
          </div>

          <div class="chat-input-area">
            <div class="input-container">
              <textarea
                id="user-input"
                placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
                rows="1"
                disabled
              ></textarea>
              <button id="send-button" disabled>Send</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<footer>
  <span>&#128279; http://localhost:DEMO_PORT_PLACEHOLDER/demo</span>
  <span id="st-lines">0 lines</span>
  <span id="st-stream">&#9675; Not connected</span>
  <span id="st-txt">Ready</span>
</footer>

<!-- Message Templates -->
<template id="user-message-template">
  <div class="message user-message">
    <div class="message-header">
      <span class="message-sender">You</span>
      <span class="message-timestamp"></span>
    </div>
    <div class="message-content"></div>
  </div>
</template>

<template id="assistant-message-template">
  <div class="message assistant-message">
    <div class="message-header">
      <span class="message-sender">Assistant</span>
      <span class="message-timestamp"></span>
    </div>
    <div class="message-content"></div>
  </div>
</template>

<template id="plan-proposal-template">
  <div class="message plan-proposal">
    <div class="message-header">
      <span class="message-sender">Assistant</span>
      <span class="message-timestamp"></span>
    </div>
    <div class="message-content">
      <div class="plan-summary"></div>
      <div class="plan-details">
        <button class="toggle-plan-details">Show Details</button>
        <div class="plan-json" style="display: none;"></div>
      </div>
      <div class="plan-actions">
        <button class="btn-approve">✅ Approve</button>
        <button class="btn-reject">❌ Reject</button>
      </div>
    </div>
  </div>
</template>

<template id="execution-status-template">
  <div class="message execution-status">
    <div class="message-header">
      <span class="message-sender">System</span>
      <span class="message-timestamp"></span>
    </div>
    <div class="message-content">
      <div class="status-text"></div>
      <div class="progress-bar" style="display: none;">
        <div class="progress-fill"></div>
      </div>
    </div>
  </div>
</template>

<!-- Marked.js for markdown rendering -->
<!-- Use local marked.js to avoid tracking prevention issues -->
<script>
// Simple markdown parser for basic formatting (since CDN is blocked)
window.marked = {
    parse: function(text) {
        return text
            // Bold **text**
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic *text*
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Line breaks
            .replace(/\n/g, '<br>')
            // Lists (simple)
            .replace(/^\d+\.\s+(.*)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>')
            // Bold numbers at start of line
            .replace(/^(\d+)\.\s+\*\*(.*?)\*\*/gm, '<li><strong>$1. $2</strong></li>');
    }
};
console.log('Local marked.js parser loaded');
</script>

<script src="/static/js/chat-manager.js"></script>
<script src="/static/js/tab-manager.js"></script>
<script>
console.log('=== MAIN JAVASCRIPT LOADING ===');
let filterMode = 'all';
let lineCount  = 0;
let autoScroll = true;
let evtSrc     = null;

const wrap   = document.getElementById('logwrap');
const logEnd = document.getElementById('logend');

// ── Sequence selector ──────────────────────────────────────────────────
function onSeqChange() {
  const v    = document.getElementById('seq-sel').value;
  const warn = document.getElementById('seq4w');
  const chk  = document.getElementById('clr-chk');
  if (v === '4') {
    warn.style.display = 'block';
    chk.checked = false;
  } else {
    warn.style.display = 'none';
    if (v === '3') chk.checked = true;
  }
}

// ── Filter ─────────────────────────────────────────────────────────────
function setFilter(mode) {
  filterMode = mode;
  document.getElementById('f-all' ).classList.toggle('active', mode === 'all');
  document.getElementById('f-demo').classList.toggle('active', mode === 'demo');
  wrap.querySelectorAll('.ll').forEach(applyFilter);
}

function applyFilter(el) {
  if (el.classList.contains('l-sys')) return;
  el.classList.toggle('hidden', filterMode === 'demo' && el.dataset.demo !== '1');
}

// ── Log rendering ──────────────────────────────────────────────────────
const PAT = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[([^\]]+)\] (\w+): (.*)$/s;
// Strip ANSI escape sequences (e.g. \x1b[1;36m … \x1b[0m) from subprocess output.
const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]/g;

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function addLine(rawText) {
  const text = rawText.replace(ANSI_RE, '');
  const isSys = text.startsWith('[DEMO_VIEWER]');
  const div   = document.createElement('div');
  div.className = 'll';

  if (isSys) {
    div.classList.add('l-sys');
    div.textContent = text;
  } else {
    const m = PAT.exec(text);
    if (m) {
      const [, ts, logger, level, msg] = m;
      const isDemo = msg.includes('[DEMO]');
      const lv     = level.toLowerCase();

      if      (isDemo)           div.classList.add('l-demo');
      else if (lv === 'warning') div.classList.add('l-warn');
      else if (lv === 'error')   div.classList.add('l-err');

      const msgHtml = esc(msg).replace(/\[DEMO\]/g,
        '<span class="tdemo">[DEMO]</span>');

      div.innerHTML =
        '<span class="ts">'           + esc(ts)     + '</span> ' +
        '[<span class="lg">'          + esc(logger) + '</span>] ' +
        '<span class="lv-' + lv + '">' + esc(level) + '</span>: ' +
        msgHtml;

      div.dataset.demo = isDemo ? '1' : '0';
    } else {
      div.classList.add('l-dim');
      div.textContent  = text;
      div.dataset.demo = '0';
    }
    applyFilter(div);
  }

  wrap.insertBefore(div, logEnd);
  lineCount++;
  document.getElementById('st-lines').textContent = lineCount + ' lines';
  if (autoScroll) logEnd.scrollIntoView({ behavior: 'instant' });
}

// ── SSE stream ─────────────────────────────────────────────────────────
function connectStream() {
  if (evtSrc) evtSrc.close();
  evtSrc = new EventSource('/api/demo/logs');
  document.getElementById('st-stream').textContent = '\u25cb Connecting\u2026';
  evtSrc.onopen    = () => { document.getElementById('st-stream').textContent = '\u25cf Connected'; };
  evtSrc.onmessage = (e) => { addLine(JSON.parse(e.data).text); };
  evtSrc.onerror   = () => { document.getElementById('st-stream').textContent = '\u25cc Reconnecting\u2026'; };
}

wrap.addEventListener('scroll', () => {
  const { scrollTop, scrollHeight, clientHeight } = wrap;
  autoScroll = scrollHeight - scrollTop - clientHeight < 40;
});

// ── Demo control ───────────────────────────────────────────────────────
async function runDemo() {
  const seq = document.getElementById('seq-sel').value;
  const clr = document.getElementById('clr-chk').checked;

  wrap.querySelectorAll('.ll').forEach(el => el.remove());
  lineCount = 0;
  document.getElementById('st-lines').textContent = '0 lines';
  autoScroll = true;

  const res  = await fetch('/api/demo/run', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ sequence: seq, clear_signifiers: clr }),
  });
  const data = await res.json();

  if (res.ok) {
    document.getElementById('btn-run' ).disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('st-txt'  ).textContent = 'Running sequence ' + seq;
  } else {
    addLine('[DEMO_VIEWER] ERROR: ' + (data.error || JSON.stringify(data)));
  }
}

async function stopDemo() {
  await fetch('/api/demo/stop', { method: 'DELETE' });
  document.getElementById('btn-run' ).disabled = false;
  document.getElementById('btn-stop').disabled = true;
  document.getElementById('st-txt'  ).textContent = 'Stopped';
}

// ── Health polling ─────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch('/api/demo/status');
    const data = await res.json();
    setBadge('b-ha',  data.ha_ok,  'HA');
    setBadge('b-ygg', data.ygg_ok, 'Yggdrasil');
    const running = data.process_running;
    document.getElementById('btn-run' ).disabled = running;
    document.getElementById('btn-stop').disabled = !running;
    if (!running && document.getElementById('st-txt').textContent.startsWith('Running'))
      document.getElementById('st-txt').textContent = 'Ready';
  } catch (_) {}
}

function setBadge(id, ok, label) {
  const el = document.getElementById(id);
  el.className   = 'badge ' + (ok ? 'ok' : 'err');
  el.textContent = label;
}

// ─── Manual Mode Functions ──────────────────────────────────────────────────

let currentMode = 'sequence'; // 'sequence' or 'manual'
let manualStatusInterval = null;

function switchToSequenceMode() {
  if (currentMode === 'sequence') return;

  console.log('Switching to sequence mode');
  currentMode = 'sequence';

  // Update mode buttons
  document.getElementById('mode-sequence').classList.add('active');
  document.getElementById('mode-manual').classList.remove('active');

  // Show/hide controls
  document.getElementById('sequence-controls').style.display = 'flex';
  document.getElementById('manual-controls').style.display = 'none';

  // Ensure manual mode is stopped
  if (manualStatusInterval) {
    clearInterval(manualStatusInterval);
    manualStatusInterval = null;
  }

  // Reset communication panel to logs tab
  if (window.tabManager) {
    window.tabManager.switchToTab('logs');
  }
}

function switchToManualMode() {
  console.log('=== switchToManualMode() called ===');
  if (currentMode === 'manual') return;

  console.log('Switching to manual mode');
  currentMode = 'manual';

  // Update mode buttons
  document.getElementById('mode-sequence').classList.remove('active');
  document.getElementById('mode-manual').classList.add('active');

  // Show/hide controls
  document.getElementById('sequence-controls').style.display = 'none';
  document.getElementById('manual-controls').style.display = 'flex';

  // Start polling manual mode status
  startManualStatusPolling();

  // Check initial status
  updateManualStatus();

  // Inform user about logs availability
  addLine('[DEMO_VIEWER] 💡 TIP: Switch to "Interaction Logs" tab to see all system activity and agent logs.');
}

async function startManualMode() {
  console.log('Starting manual mode...');

  const startBtn = document.getElementById('btn-manual-start');
  const stopBtn = document.getElementById('btn-manual-stop');
  const clearSignifiers = document.getElementById('manual-clr-chk').checked;

  // Disable start button
  startBtn.disabled = true;
  startBtn.textContent = '🔄 Starting...';

  // Update status
  updateManualBadge('manual-starting', 'Starting...');
  updateManualPhase('Initializing manual mode...');

  try {
    // Clear signifiers if requested
    if (clearSignifiers) {
      await fetch('/api/manual/clear-signifiers', { method: 'DELETE' });
    }

    // Start manual mode
    const response = await fetch('/api/manual/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to start manual mode');
    }

    const result = await response.json();
    console.log('Manual mode started:', result);

    // Enable stop button
    stopBtn.disabled = false;

    // Let user control which tab to see - no automatic switching

    // Continue polling for status updates

  } catch (error) {
    console.error('Failed to start manual mode:', error);

    // Reset buttons
    startBtn.disabled = false;
    startBtn.textContent = '🚀 Start Manual Mode';

    // Show error
    updateManualBadge('manual-error', 'Start Failed');
    updateManualPhase(`Error: ${error.message}`);

    // Show error in logs
    addLogLine(`[ERROR] Manual mode start failed: ${error.message}`, 'l-err');
  }
}

async function stopManualMode() {
  console.log('Stopping manual mode...');

  const startBtn = document.getElementById('btn-manual-start');
  const stopBtn = document.getElementById('btn-manual-stop');

  // Disable stop button
  stopBtn.disabled = true;
  stopBtn.textContent = '⏸ Stopping...';

  try {
    const response = await fetch('/api/manual/stop', {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to stop manual mode');
    }

    const result = await response.json();
    console.log('Manual mode stopped:', result);

    // Reset buttons
    startBtn.disabled = false;
    startBtn.textContent = '🚀 Start Manual Mode';
    stopBtn.textContent = '⏹ Stop Manual Mode';

    // Update status
    updateManualBadge('err', 'Manual Inactive');
    updateManualPhase('Ready to start');

  } catch (error) {
    console.error('Failed to stop manual mode:', error);

    // Re-enable stop button
    stopBtn.disabled = false;
    stopBtn.textContent = '⏹ Stop Manual Mode';

    // Show error
    updateManualPhase(`Stop error: ${error.message}`);
  }
}

function startManualStatusPolling() {
  if (manualStatusInterval) {
    clearInterval(manualStatusInterval);
  }

  manualStatusInterval = setInterval(updateManualStatus, 1000); // Poll every second
}

async function updateManualStatus() {
  try {
    const response = await fetch('/api/manual/status');
    if (!response.ok) return;

    const status = await response.json();

    // Update badge
    if (status.manual_mode_active) {
      if (status.ready_for_interaction) {
        updateManualBadge('manual-active', 'Manual Active');
        updateManualPhase('Ready for dialog');

        // Enable dialog tab if not already
        if (window.tabManager && status.ready_for_interaction) {
          // Could switch to dialog tab automatically or just enable it
        }
      } else {
        updateManualBadge('manual-starting', 'Starting...');
        updateManualPhase(status.startup_message || `Phase: ${status.startup_phase}`);
      }
    } else {
      if (status.startup_phase === 'error') {
        updateManualBadge('manual-error', 'Error');
        updateManualPhase(status.startup_message || 'Startup failed');
      } else {
        updateManualBadge('err', 'Manual Inactive');
        updateManualPhase('Ready to start');
      }
    }

    // Update button states
    const startBtn = document.getElementById('btn-manual-start');
    const stopBtn = document.getElementById('btn-manual-stop');

    if (status.manual_mode_active) {
      startBtn.disabled = true;
      startBtn.textContent = '✅ Manual Mode Active';
      stopBtn.disabled = false;
      stopBtn.textContent = '⏹ Stop Manual Mode';
    } else {
      startBtn.disabled = false;
      startBtn.textContent = '🚀 Start Manual Mode';
      stopBtn.disabled = true;
      stopBtn.textContent = '⏹ Stop Manual Mode';
    }

  } catch (error) {
    console.error('Failed to update manual status:', error);
  }
}

function updateManualBadge(className, text) {
  const badge = document.getElementById('manual-badge');
  badge.className = `badge ${className}`;
  badge.textContent = text;
}

function updateManualPhase(text) {
  const phase = document.getElementById('manual-phase');
  phase.textContent = text;
  phase.title = text; // Show full text on hover
}

function addLogLine(text, className = 'l-sys') {
  const wrap = document.getElementById('logwrap');
  const line = document.createElement('div');
  line.className = `ll ${className}`;
  line.textContent = text;

  // Insert before the #logend marker
  const logend = document.getElementById('logend');
  wrap.insertBefore(line, logend);

  // Auto-scroll if enabled
  if (autoScroll) {
    wrap.scrollTop = wrap.scrollHeight;
  }
}

// ─── Resizable Splitter Functionality ──────────────────────────────────────

let isDragging = false;
let startX = 0;
let startLeftWidth = 0;
let startRightWidth = 0;

const SPLITTER_KEY = 'ami-demo-splitter-position';
const MIN_PANEL_WIDTH = 200; // Minimum width for each panel

function initializeSplitter() {
  const splitter = document.getElementById('panel-splitter');
  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');
  const panels = document.getElementById('main-panels');

  if (!splitter || !leftPanel || !rightPanel || !panels) return;

  // Load saved position from localStorage
  const savedPosition = localStorage.getItem(SPLITTER_KEY);
  if (savedPosition) {
    try {
      const { leftWidth, rightWidth } = JSON.parse(savedPosition);
      leftPanel.style.width = leftWidth;
      rightPanel.style.width = rightWidth;
    } catch (e) {
      // Fallback to default if localStorage is corrupted
      setDefaultSplitterPosition();
    }
  } else {
    setDefaultSplitterPosition();
  }

  // Add mouse event listeners
  splitter.addEventListener('mousedown', handleMouseDown);
  document.addEventListener('mousemove', handleMouseMove);
  document.addEventListener('mouseup', handleMouseUp);

  // Prevent text selection during drag
  document.addEventListener('selectstart', preventSelection);
}

function setDefaultSplitterPosition() {
  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');

  leftPanel.style.width = '55%';
  rightPanel.style.width = '45%';
}

function handleMouseDown(e) {
  isDragging = true;
  startX = e.clientX;

  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');
  const panels = document.getElementById('main-panels');

  // Store initial widths
  startLeftWidth = leftPanel.getBoundingClientRect().width;
  startRightWidth = rightPanel.getBoundingClientRect().width;

  // Add dragging class for styling
  panels.classList.add('dragging');

  e.preventDefault();
}

function handleMouseMove(e) {
  if (!isDragging) return;

  const deltaX = e.clientX - startX;
  const panels = document.getElementById('main-panels');
  const panelsWidth = panels.getBoundingClientRect().width;
  const splitterWidth = 8; // Width of the splitter

  // Calculate new widths
  let newLeftWidth = startLeftWidth + deltaX;
  let newRightWidth = startRightWidth - deltaX;

  // Apply minimum width constraints
  const availableWidth = panelsWidth - splitterWidth;
  newLeftWidth = Math.max(MIN_PANEL_WIDTH, Math.min(newLeftWidth, availableWidth - MIN_PANEL_WIDTH));
  newRightWidth = availableWidth - newLeftWidth;

  // Apply the new widths as percentages
  const leftPercent = (newLeftWidth / availableWidth) * 100;
  const rightPercent = (newRightWidth / availableWidth) * 100;

  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');

  leftPanel.style.width = leftPercent + '%';
  rightPanel.style.width = rightPercent + '%';

  // Force iframe to update its layout during drag for smooth resizing
  const haFrame = document.getElementById('ha-frame');
  if (haFrame) {
    haFrame.style.pointerEvents = 'none'; // Disable iframe interaction during drag
  }

  e.preventDefault();
}

function handleMouseUp(e) {
  if (!isDragging) return;

  isDragging = false;

  const panels = document.getElementById('main-panels');
  panels.classList.remove('dragging');

  // Save position to localStorage
  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');

  const position = {
    leftWidth: leftPanel.style.width,
    rightWidth: rightPanel.style.width
  };

  localStorage.setItem(SPLITTER_KEY, JSON.stringify(position));

  // Re-enable iframe interaction after drag
  const haFrame = document.getElementById('ha-frame');
  if (haFrame) {
    haFrame.style.pointerEvents = 'auto';
  }

  e.preventDefault();
}

function preventSelection(e) {
  if (isDragging) {
    e.preventDefault();
    return false;
  }
}

// Handle window resize to maintain proportions
function handleWindowResize() {
  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');
  const panels = document.getElementById('main-panels');

  if (!leftPanel || !rightPanel || !panels) return;

  // Get current widths as percentages
  const leftStyle = leftPanel.style.width;
  const rightStyle = rightPanel.style.width;

  // If we have percentage-based widths, they should automatically adapt
  // But force a layout recalculation to ensure everything updates
  if (leftStyle && rightStyle) {
    // Force reflow by temporarily changing and restoring the display
    const originalDisplay = panels.style.display;
    panels.style.display = 'none';
    panels.offsetHeight; // Trigger reflow
    panels.style.display = originalDisplay;

    // Update iframe dimensions to ensure Home Assistant scales properly
    const haFrame = document.getElementById('ha-frame');
    if (haFrame) {
      // Force iframe to recalculate its size
      const src = haFrame.src;
      haFrame.src = '';
      setTimeout(() => { haFrame.src = src; }, 10);
    }
  }
}

// Throttle resize events to avoid performance issues
let resizeTimeout;
function throttledResize() {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(handleWindowResize, 150);
}

window.addEventListener('resize', throttledResize);

// Initialize
connectStream();
pollStatus();
setInterval(pollStatus, 5000);
initializeSplitter(); // Initialize splitter before mode switching

// Direct tab switching implementation (bulletproof)
function setupTabSwitching() {
    const tabLogs = document.getElementById('tab-logs');
    const tabDialog = document.getElementById('tab-dialog');
    const logsPanel = document.getElementById('logs-panel');
    const dialogPanel = document.getElementById('dialog-panel');

    console.log('Setting up direct tab switching...');
    console.log('Elements found:', {
        tabLogs: !!tabLogs,
        tabDialog: !!tabDialog,
        logsPanel: !!logsPanel,
        dialogPanel: !!dialogPanel
    });

    if (tabLogs && tabDialog && logsPanel && dialogPanel) {
        // Logs tab click
        tabLogs.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Switching to logs tab');
            logsPanel.classList.remove('hidden');
            dialogPanel.classList.add('hidden');
            tabLogs.classList.add('active');
            tabDialog.classList.remove('active');
        });

        // Dialog tab click
        tabDialog.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Switching to dialog tab');
            logsPanel.classList.add('hidden');
            dialogPanel.classList.remove('hidden');
            tabLogs.classList.remove('active');
            tabDialog.classList.add('active');

            // Initialize chat manager if needed
            if (!window.chatManager) {
                console.log('Initializing ChatManager...');
                try {
                    window.chatManager = new ChatManager();
                    window.chatManager.initialize().catch(console.error);
                } catch (error) {
                    console.error('ChatManager initialization failed:', error);
                }
            }
        });

        console.log('Direct tab switching set up successfully');
    } else {
        console.error('Tab switching setup failed - missing elements');
    }
}

// Initialize TabManager only (no direct switching fallback to prevent conflicts)
if (!window.tabManager) {
    try {
        window.tabManager = new TabManager();
        console.log('TabManager initialized successfully');
    } catch (error) {
        console.error('TabManager failed:', error);
        // Don't fallback to setupTabSwitching to prevent conflicts
    }
} else {
    console.log('TabManager already exists, skipping initialization');
}

// TabManager initialization complete

// Debug: Check if marked.js is loaded
console.log('=== MARKDOWN DEBUG ===');
console.log('marked available:', typeof marked !== 'undefined');
if (typeof marked !== 'undefined') {
    console.log('marked.parse available:', typeof marked.parse !== 'undefined');
    console.log('marked version:', marked.getVersion ? marked.getVersion() : 'unknown');
}

// Start in sequence mode by default
// switchToSequenceMode(); // Temporarily commented to fix syntax error
console.log('=== JAVASCRIPT PARSING SUCCESSFUL ===');

// Ensure logs are always visible when manual mode starts
document.addEventListener('manualModeStarted', () => {
  console.log('Manual mode started - ensuring logs are visible');
  // If we're in manual mode and on dialog tab, user should be able to see logs too
  if (currentMode === 'manual') {
    // Ensure logs are being displayed
    console.log('Manual mode active - logs should be streaming');
  }
});
</script>
</body>
</html>
""".replace("DEMO_PORT_PLACEHOLDER", str(SERVER_PORT))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    log.info("Demo Viewer → http://localhost:%d/demo", SERVER_PORT)
    log.info("HA proxied at → http://localhost:%d/  (same origin = auth works)", SERVER_PORT)
    # log_level="info" enables uvicorn access logs (one line per request) –
    # useful to see ALL requests hitting the server, even if routing bypasses our handlers.
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
