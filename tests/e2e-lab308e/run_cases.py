"""JSON-driven end-to-end runner for the lab308e Home Assistant scenario.

Each test case is a JSON file that can:
- drive initial HASP state via HTTP requests
- send a user query through the AMI agent stack
- optionally confirm the proposed plan
- assert final HASP state or assistant replies

Run from repo root:
    ~/aiml/env-spade-3/bin/python tests/e2e-lab308e/run_cases.py

Or point at a single case:
    ~/aiml/env-spade-3/bin/python tests/e2e-lab308e/run_cases.py --case tests/e2e-lab308e/cases/reduce_glare.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
import spade
from dotenv import load_dotenv
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)


def _load_adapter_env_defaults() -> None:
    """Load simple HomeAssistant env defaults used by lab308e setup steps."""
    adapter_env = (
        PROJECT_ROOT
        / "ami_agents"
        / "environment"
        / "integration"
        / "HomeAssistant"
        / "prepare-adapter-env.sh"
    )
    if not adapter_env.exists():
        return
    simple_pattern = re.compile(r"""^\s*export\s+([A-Z0-9_]+)=["']([^"'$`]*)["']\s*$""")
    cat_pattern = re.compile(r"""^\s*export\s+([A-Z0-9_]+)=["']?\$\(cat\s+([^)"']+)\)["']?\s*$""")
    for line in adapter_env.read_text(encoding="utf-8").splitlines():
        simple_match = simple_pattern.match(line)
        if simple_match:
            key, value = simple_match.groups()
        else:
            cat_match = cat_pattern.match(line)
            if not cat_match:
                continue
            key, rel_path = cat_match.groups()
            source_path = (adapter_env.parent / rel_path.strip()).resolve()
            if not source_path.exists():
                continue
            value = source_path.read_text(encoding="utf-8").strip()
        if key in os.environ:
            continue
        if key in {
            "AREAS",
            "BASE_WS_URI",
            "HASP_URL",
            "HA_TOKEN",
            "HA_URL",
            "HA_BASE_URL",
            "TD_SOSA_ENV_VAR_OVERRIDES",
            "TD_SOSA_PROPERTY_RANGES",
            "TD_SOSA_SETTLING_TIMES",
        }:
            os.environ[key] = value


_load_adapter_env_defaults()

from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
from ami_agents.environment.connection.hmas_client import IHMASClient
from ami_agents.shared.utils import spade_compat  # noqa: F401
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.shared.utils.demo_log import demo


class LLMCallCounter:
    total_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.total_calls = 0
        cls.input_tokens = 0
        cls.output_tokens = 0

    @classmethod
    def increment(cls, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        cls.total_calls += 1
        cls.input_tokens += int(input_tokens or 0)
        cls.output_tokens += int(output_tokens or 0)

    @classmethod
    def snapshot(cls) -> dict[str, int]:
        return {
            "llm_calls": cls.total_calls,
            "input_tokens": cls.input_tokens,
            "output_tokens": cls.output_tokens,
        }


class PromptDumpState:
    enabled: bool = False
    output_dir: Optional[Path] = None
    case_name: Optional[str] = None
    case_timestamp: Optional[str] = None
    prompt_entries: List[str] = []

    @classmethod
    def configure(cls, *, enabled: bool, output_dir: Optional[Path]) -> None:
        cls.enabled = enabled
        cls.output_dir = output_dir
        if enabled and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def start_case(cls, case_name: str) -> None:
        cls.case_name = case_name
        cls.case_timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S")
        cls.prompt_entries = []

    @classmethod
    def append_call(cls, kwargs: Dict[str, Any]) -> None:
        if not cls.enabled:
            return
        parts: List[str] = []
        call_index = len(cls.prompt_entries) + 1
        parts.append(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        model = kwargs.get("model")
        if model:
            parts.append(f"Model: {model}")
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            parts.append(f"Reasoning effort: {reasoning_effort}")
        response_format = kwargs.get("response_format")
        if response_format:
            parts.append(f"Response format: {json.dumps(response_format, ensure_ascii=False, default=str)}")
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            parts.append("Messages:")
            for idx, message in enumerate(messages, start=1):
                if isinstance(message, dict):
                    role = message.get("role", "unknown")
                    content = message.get("content")
                    parts.append(f"[{idx}] role={role}")
                    if isinstance(content, str):
                        parts.append(content)
                    else:
                        parts.append(json.dumps(content, ensure_ascii=False, indent=2, default=str))
                else:
                    parts.append(json.dumps(message, ensure_ascii=False, indent=2, default=str))
        elif "input" in kwargs:
            parts.append("Input:")
            parts.append(json.dumps(kwargs.get("input"), ensure_ascii=False, indent=2, default=str))
        else:
            parts.append("Raw kwargs:")
            parts.append(json.dumps(kwargs, ensure_ascii=False, indent=2, default=str))

        entry = f"===== LLM Call {call_index} =====\n" + "\n".join(parts).strip() + "\n"
        cls.prompt_entries.append(entry)

    @classmethod
    def write_case_file(cls) -> Optional[Path]:
        if not cls.enabled or cls.output_dir is None or not cls.case_name:
            return None
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in cls.case_name)
        timestamp_prefix = f"{cls.case_timestamp}_" if cls.case_timestamp else ""
        out_path = cls.output_dir / f"{timestamp_prefix}{safe_name}.txt"
        content = "\n\n".join(cls.prompt_entries).strip()
        out_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
        return out_path


def _install_openai_call_counter() -> None:
    try:
        from openai.resources.chat.completions.completions import AsyncCompletions
    except Exception:
        return

    if getattr(AsyncCompletions.create, "_ami_counting_wrapped", False):
        return

    original_create = AsyncCompletions.create

    async def counted_create(self, *args, **kwargs):
        PromptDumpState.append_call(kwargs)
        response = await original_create(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        LLMCallCounter.increment(input_tokens=prompt_tokens, output_tokens=completion_tokens)
        return response

    counted_create._ami_counting_wrapped = True  # type: ignore[attr-defined]
    AsyncCompletions.create = counted_create


def _require_env(var: str) -> str:
    val = os.getenv(var, "").strip()
    if not val:
        raise RuntimeError(f"Missing required environment variable: {var}")
    return val


LAB308E_HARD_RESET_STATES: Dict[str, Dict[str, Any]] = {
    "light.ambient_lights_308e": {
        "state": "off",
        "attributes": {"friendly_name": "ambient_lights_308e", "brightness": 96},
    },
    "light.task_lights_308e": {
        "state": "off",
        "attributes": {"friendly_name": "task_lights_308e", "brightness": 128},
    },
    "light.desk_lamp_308e": {
        "state": "off",
        "attributes": {"friendly_name": "desk_lamp_308e", "brightness": 140},
    },
    "cover.blinds_308e_cover": {
        "state": "closed",
        "attributes": {"friendly_name": "blinds_308e cover", "current_position": 0, "supported_features": 15},
    },
    "cover.blackout_blinds_308e_cover": {
        "state": "open",
        "attributes": {"friendly_name": "blackout_blinds_308e cover", "current_position": 100, "supported_features": 15},
    },
    "cover.window_308e_cover": {
        "state": "closed",
        "attributes": {"friendly_name": "window_308e cover", "current_position": 0, "supported_features": 15},
    },
    "media_player.projector_308e": {
        "state": "off",
        "attributes": {"friendly_name": "projector_308e", "volume_level": 0.15},
    },
    "media_player.display_wall_308e": {
        "state": "on",
        "attributes": {"friendly_name": "display_wall_308e", "volume_level": 0.0},
    },
    "switch.ceiling_fan_308e": {
        "state": "off",
        "attributes": {"friendly_name": "ceiling_fan_308e"},
    },
    "climate.air_conditioner_308e": {
        "state": "cool",
        "attributes": {
            "friendly_name": "air_conditioner_308e",
            "hvac_mode": "cool",
            "temperature": 24,
            "current_temperature": 26,
            "min_temp": 18,
            "max_temp": 30,
            "target_temp_step": 0.5,
        },
    },
    "climate.heater_308e": {
        "state": "off",
        "attributes": {
            "friendly_name": "heater_308e",
            "hvac_mode": "off",
            "temperature": 20,
            "current_temperature": 26,
            "min_temp": 16,
            "max_temp": 28,
            "target_temp_step": 0.5,
        },
    },
    "sensor.person_counter_308e": {
        "state": 3,
        "attributes": {"friendly_name": "person_counter_308e", "unit_of_measurement": "persons"},
    },
    "binary_sensor.presence_sensing_308e": {
        "state": "on",
        "attributes": {"friendly_name": "presence_sensing_308e", "device_class": "occupancy"},
    },
    # Derived sensors are written after relevant actuators/occupancy so the
    # simulator picks up these exact seed values on its next tick.
    "sensor.internal_light_sensing_308e": {
        "state": 180,
        "attributes": {"friendly_name": "internal_light_sensing_308e", "unit_of_measurement": "lx"},
    },
    "sensor.desk_light_sensing_308e": {
        "state": 120,
        "attributes": {"friendly_name": "desk_light_sensing_308e", "unit_of_measurement": "lx"},
    },
    "sensor.glare_sensing_308e": {
        "state": 75,
        "attributes": {"friendly_name": "glare_sensing_308e", "unit_of_measurement": "%"},
    },
    "sensor.external_light_sensing_308e": {
        "state": 5200,
        "attributes": {"friendly_name": "external_light_sensing_308e", "unit_of_measurement": "lx"},
    },
    "sensor.temperature_sensing_308e": {
        "state": 26,
        "attributes": {"friendly_name": "temperature_sensing_308e", "unit_of_measurement": "C"},
    },
    "sensor.humidity_sensing_308e": {
        "state": 62,
        "attributes": {"friendly_name": "humidity_sensing_308e", "unit_of_measurement": "%"},
    },
    "sensor.co2_sensing_308e": {
        "state": 980,
        "attributes": {"friendly_name": "co2_sensing_308e", "unit_of_measurement": "ppm"},
    },
}


def _clear_signifier_storage() -> Path:
    storage_dir = PROJECT_ROOT / "ami_agents" / "shared" / "memory" / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["rdf", "json", "indexes"]:
        subpath = storage_dir / subdir
        if subpath.exists():
            shutil.rmtree(subpath)
        subpath.mkdir(parents=True, exist_ok=True)
    return storage_dir


def _configure_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    for noisy in [
        "spade",
        "slixmpp",
        "httpx",
        "aiohttp.access",
        "spade_llm",
        "spade_llm.providers",
        "ami_agents.shared.ontologies.loader",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("Lab308eE2ERunner")


class DummyHMASClient(IHMASClient):
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


class HeadlessUserAgent(Agent):
    def __init__(self, jid: str, password: str, *, assistant_jid: str):
        super().__init__(jid, password)
        self.assistant_jid = assistant_jid
        self.thread_id = str(uuid.uuid4())
        self.reply_queue: asyncio.Queue[str] = asyncio.Queue()

    async def setup(self):
        template = Template()
        template.set_metadata("message_type", "llm")
        self.add_behaviour(self.ReceiveLLMReply(), template=template)

    def reset_thread(self) -> str:
        self.thread_id = str(uuid.uuid4())
        self.drain_replies()
        return self.thread_id

    async def ask(self, text: str) -> None:
        behaviour = self.SendMessage(text)
        self.add_behaviour(behaviour)
        await behaviour.join()

    async def wait_for_reply(self, timeout_s: float) -> str:
        return await asyncio.wait_for(self.reply_queue.get(), timeout=timeout_s)

    def drain_replies(self) -> None:
        while True:
            try:
                self.reply_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    class ReceiveLLMReply(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return
            if str(getattr(msg, "thread", "") or "") != self.agent.thread_id:
                return
            await self.agent.reply_queue.put(msg.body or "")

    class SendMessage(CyclicBehaviour):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

        async def run(self):
            msg = SpadeMessage(to=self.agent.assistant_jid)
            msg.set_metadata("message_type", "llm")
            msg.thread = self.agent.thread_id
            msg.body = self.text
            await self.send(msg)
            self.kill()


@dataclass
class CaseResult:
    path: Path
    passed: bool
    details: Dict[str, Any]


class ManagedService:
    def __init__(
        self,
        *,
        name: str,
        argv: List[str],
        cwd: Path,
        logger: logging.Logger,
        env: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.logger = logger
        self.env = env
        self.process: Optional[asyncio.subprocess.Process] = None
        self._log_task: Optional[asyncio.Task[None]] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        if self.is_running():
            return
        self.logger.info("Starting managed %s: %s", self.name, " ".join(self.argv))
        self.process = await asyncio.create_subprocess_exec(
            *self.argv,
            cwd=str(self.cwd),
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._log_task = asyncio.create_task(self._drain_stdout())

    async def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            self.logger.info("Stopping managed %s", self.name)
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                self.logger.warning("Managed %s did not terminate; killing it", self.name)
                process.kill()
                await process.wait()
        if self._log_task is not None:
            try:
                await asyncio.wait_for(self._log_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._log_task.cancel()
            self._log_task = None
        self.process = None

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    async def _drain_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            self.logger.info("[%s] %s", self.name, line.decode(errors="replace").rstrip())


class Lab308eHarness:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.xmpp_server = os.getenv("SPADE_SERVER", "localhost")
        self.password = os.getenv("SPADE_PASSWORD", "password")
        self.yggdrasil_url = os.getenv("YGGDRASIL_URL", "http://localhost:8080/").strip().rstrip("/")
        self.ha_base_url = (
            os.getenv("HA_BASE_URL")
            or os.getenv("HA_URL", "ws://localhost:8123/api/websocket")
            .replace("ws://", "http://")
            .replace("wss://", "https://")
            .split("/api/websocket")[0]
        ).strip().rstrip("/")
        self.response_timeout = float(args.response_timeout)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        self.explorer_jid = f"env_explorer@{self.xmpp_server}"
        self.assistant_jid = f"user_assistant@{self.xmpp_server}"
        self.solver_jid = f"interaction_solver@{self.xmpp_server}"
        self.runner_jid = f"e2e_lab308e_runner@{self.xmpp_server}"
        self.prompt_dump_dir = Path(__file__).resolve().parent / "prompt_dumps"

        self.explorer: Optional[EnvExplorerAgent] = None
        self.assistant: Optional[UserAssistantAgent] = None
        self.solver: Optional[InteractionSolverAgent] = None
        self.user: Optional[HeadlessUserAgent] = None
        self.hasp_service: Optional[ManagedService] = None
        self.simulator_service: Optional[ManagedService] = None

    @staticmethod
    def _clarification_followup(first_reply: str, confirmation_text: str) -> Optional[str]:
        reply = (first_reply or "").strip()
        lowered = reply.lower()
        if lowered.startswith("would you like me to ") and reply.endswith("?"):
            action_text = reply[len("Would you like me to ") :].rstrip("?").strip()
            if action_text:
                action_text = action_text[0].upper() + action_text[1:]
                return action_text
        if lowered.startswith("do you want me to ") and reply.endswith("?"):
            action_text = reply[len("Do you want me to ") :].rstrip("?").strip()
            if action_text:
                action_text = action_text[0].upper() + action_text[1:]
                return action_text
        if lowered.startswith("should i ") and reply.endswith("?"):
            action_text = reply[len("Should I ") :].rstrip("?").strip()
            if action_text:
                action_text = action_text[0].upper() + action_text[1:]
                return action_text
        if confirmation_text.strip().lower() not in {"yes", "y"}:
            return confirmation_text
        return None

    @staticmethod
    def _match_clarification_reply(
        assistant_reply: str, clarification_replies: List[Dict[str, Any]]
    ) -> Optional[str]:
        reply = (assistant_reply or "").strip()
        lowered_reply = reply.lower()
        for rule in clarification_replies:
            match_text = str(rule.get("match", "")).strip()
            if not match_text:
                continue
            match_type = str(rule.get("match_type", "contains")).strip().lower()
            matched = False
            if match_type == "contains":
                matched = match_text.lower() in lowered_reply
            elif match_type == "exact":
                matched = reply == match_text
            elif match_type == "regex":
                matched = re.search(match_text, reply) is not None
            else:
                raise ValueError(f"Unsupported clarification match_type: {match_type}")
            if matched:
                return str(rule.get("reply", "")).strip()
        return None

    def _record_assertion_failure(self, details: Dict[str, Any], result: Dict[str, Any]) -> None:
        details["failed_assertion"] = result
        details["failure_stage"] = "assertion"
        self.logger.error("Assertion failed: %s", json.dumps(result, ensure_ascii=True, default=str))

    @staticmethod
    def _attach_prompt_dump(details: Dict[str, Any]) -> None:
        dump_path = PromptDumpState.write_case_file()
        if dump_path is not None:
            details["prompt_dump_path"] = str(dump_path)

    def _capture_plan(self, details: Dict[str, Any]) -> None:
        if not self.assistant or not self.user:
            return
        try:
            conv = self.assistant.get_conversation(self.user.thread_id)
        except Exception:
            return
        plan_json = getattr(conv, "plan_json", None)
        plan_summary = getattr(conv, "plan_summary", None)
        plan_hash = getattr(conv, "plan_hash", None)
        if plan_json:
            details["plan"] = plan_json
        if plan_summary:
            details["plan_summary"] = plan_summary
        if plan_hash:
            details["plan_hash"] = plan_hash

    def _clear_in_memory_case_state(self) -> str:
        if not self.assistant or not self.user:
            return ""

        old_thread = self.user.thread_id
        new_thread = self.user.reset_thread()

        conversations = getattr(self.assistant, "_conversations", None)
        if isinstance(conversations, dict):
            conversations.clear()

        state_memory = getattr(self.assistant, "state_memory", None)
        clear_state_memory = getattr(state_memory, "clear", None)
        if callable(clear_state_memory):
            clear_state_memory()

        self.logger.info(
            "Cleared UserAssistant in-memory case state: old_thread=%s new_thread=%s",
            old_thread,
            new_thread,
        )
        return new_thread

    async def _hard_reset_lab308e(self) -> None:
        token = _require_env("HA_TOKEN")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession(headers=headers) as session:
            for entity_id, payload in LAB308E_HARD_RESET_STATES.items():
                async with session.post(f"{self.ha_base_url}/api/states/{entity_id}", json=payload) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise AssertionError(
                            f"Hard reset failed for {entity_id}: HTTP {resp.status}: {text}"
                        )

    @staticmethod
    def _is_terminal_reply(text: str) -> bool:
        lowered = (text or "").strip().lower()
        terminal_markers = (
            "done! the plan was executed successfully.",
            "execution failed:",
            "i don't have a pending plan right now.",
            "okay, i've discarded that plan.",
        )
        return any(marker in lowered for marker in terminal_markers)

    async def _wait_for_terminal_reply(self, details: Dict[str, Any], timeout_s: float) -> str:
        deadline = asyncio.get_running_loop().time() + timeout_s
        last_reply = ""
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                if last_reply:
                    return last_reply
                raise asyncio.TimeoutError("Timed out waiting for terminal assistant reply")

            reply = await self.user.wait_for_reply(remaining)
            last_reply = reply
            details["transcript"].append({"role": "assistant", "phase": "execution", "text": reply})
            details["final_reply"] = reply
            self._capture_plan(details)
            if self._is_terminal_reply(reply):
                return reply

    @staticmethod
    def _requires_default_temperature(model: str, base_url: str) -> bool:
        model_name = (model or "").strip().lower()
        base = (base_url or "").strip().lower()
        return model_name.startswith("gpt-5") and "api.openai.com" in base

    def _load_config(self) -> Dict[str, Any]:
        config = ConfigLoader.merge_configs(
            ConfigLoader.load_with_env_vars(str(PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml")),
            ConfigLoader.load_with_env_vars(str(PROJECT_ROOT / "ami_agents" / "config" / "environment.yaml")),
        )

        model = self.model_name
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            base_url = "https://openrouter.ai/api/v1" if ":" in model or "/" in model else "https://api.openai.com/v1"
        reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip() or (
            "high" if model.startswith("o") and "openai.com" in base_url else ""
        )

        config["yggdrasil_url"] = self.yggdrasil_url
        config["yggdrasil"] = {"url": self.yggdrasil_url}
        config["discovery"] = {
            "notify_on_discovery_complete": True,
            "notify_agents": [self.solver_jid],
        }
        config.setdefault("llm", {}).setdefault("providers", {}).setdefault("openai", {})
        config["llm"]["providers"]["openai"].update(
            {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": base_url,
                "model": model,
            }
        )
        if self._requires_default_temperature(model, base_url):
            config["llm"]["providers"]["openai"]["temperature"] = 1.0
            config.setdefault("interaction_solver", {}).setdefault("planning", {}).setdefault("llm_planning", {})
            config["interaction_solver"]["planning"]["llm_planning"]["temperature"] = 1.0
            llm_timeout = float(os.getenv("AMI_LLM_TIMEOUT", "180"))
            config.setdefault("llm", {}).setdefault("retry", {})
            config["llm"]["retry"]["timeout"] = llm_timeout
            config.setdefault("timeouts", {}).setdefault("llm", {})
            config["timeouts"]["llm"]["default"] = llm_timeout
            config["timeouts"]["llm"]["reasoning"] = max(
                float(config["timeouts"]["llm"].get("reasoning", 120.0)),
                llm_timeout,
            )
        if reasoning_effort:
            config["llm"]["providers"]["openai"]["reasoning_effort"] = reasoning_effort
        config.setdefault("planning", {})
        config["planning"]["timeout"] = float(os.getenv("AMI_PLANNING_TIMEOUT", "180"))
        bt_max_ticks = int(os.getenv("BT_MAX_TICKS_USER_ASSISTANT", "240"))
        config.setdefault("bt_execution", {}).setdefault("max_ticks", {})
        config["bt_execution"]["max_ticks"]["user_assistant"] = bt_max_ticks
        return config

    def _hasp_port(self) -> int:
        parsed = urlparse(self.yggdrasil_url)
        if parsed.port is not None:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80

    def _build_managed_services(self) -> None:
        ha_dir = PROJECT_ROOT / "ami_agents" / "environment" / "integration" / "HomeAssistant"
        env = os.environ.copy()
        python_path_parts = [str(PROJECT_ROOT), str(ha_dir)]
        existing_python_path = env.get("PYTHONPATH", "").strip()
        if existing_python_path:
            python_path_parts.append(existing_python_path)
        env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

        if self.args.manage_hasp:
            env["PORT"] = str(self._hasp_port())
            env.setdefault("BASE_WS_URI", self.yggdrasil_url + "/")
            self.hasp_service = ManagedService(
                name="hasp",
                argv=[
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "hasp:app",
                    "--host",
                    self.args.hasp_host,
                    "--port",
                    str(self._hasp_port()),
                ],
                cwd=ha_dir,
                logger=self.logger,
                env=env,
            )

        if self.args.manage_simulator:
            self.simulator_service = ManagedService(
                name="simulate_lab308e",
                argv=[sys.executable, str(ha_dir / "simulate_lab308e.py")],
                cwd=ha_dir,
                logger=self.logger,
                env=env,
            )

    async def _wait_for_yggdrasil(self, timeout_s: float = 20.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        last_error = ""
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.yggdrasil_url) as resp:
                        if resp.status < 400:
                            return
                        last_error = f"HTTP {resp.status}"
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(0.5)
        raise RuntimeError(f"Yggdrasil is not reachable at {self.yggdrasil_url}: {last_error}")

    async def _stop_simulator_for_setup(self) -> None:
        if self.simulator_service is not None:
            await self.simulator_service.stop()

    async def _start_simulator_for_execution(self) -> None:
        if self.simulator_service is not None:
            await self.simulator_service.start()

    async def _restart_hasp_for_case(self) -> None:
        if self.hasp_service is None:
            return
        await self.hasp_service.restart()
        await self._wait_for_yggdrasil(timeout_s=float(self.args.service_start_timeout))

    async def start(self) -> None:
        _require_env("OPENAI_API_KEY")

        if self.args.clear_signifiers:
            storage_dir = _clear_signifier_storage()
            self.logger.info("Cleared signifier storage at %s", storage_dir)

        PromptDumpState.configure(enabled=bool(self.args.dump_prompts), output_dir=self.prompt_dump_dir)

        config = self._load_config()
        self._build_managed_services()

        self.logger.info(demo("Starting lab308e E2E harness (Yggdrasil=%s)"), self.yggdrasil_url)
        if self.hasp_service is not None:
            await self.hasp_service.start()
        await self._wait_for_yggdrasil(timeout_s=float(self.args.service_start_timeout))

        self.explorer = EnvExplorerAgent(self.explorer_jid, self.password, config, hmas_client=DummyHMASClient())
        self.assistant = UserAssistantAgent(
            self.assistant_jid,
            self.password,
            config=config,
            target_jids={"explorer": self.explorer_jid, "solver": self.solver_jid},
        )
        self.solver = InteractionSolverAgent(
            self.solver_jid,
            self.password,
            config=config,
            target_jids={"explorer": self.explorer_jid},
        )
        self.user = HeadlessUserAgent(self.runner_jid, self.password, assistant_jid=self.assistant_jid)

        await self.solver.start(auto_register=True)
        await self.explorer.start(auto_register=True)
        await self.assistant.start(auto_register=True)

        for _ in range(600):
            if getattr(self.explorer, "discovery_complete", False):
                break
            await asyncio.sleep(0.5)
        if not getattr(self.explorer, "discovery_complete", False):
            raise RuntimeError("EnvExplorer discovery did not complete in time.")

        if not self.args.skip_prewarm:
            self.logger.info(demo("Pre-warming UserAssistant execution engine..."))
            await self.assistant.ensure_execution_engine_ready()

        await self.user.start(auto_register=True)

    async def stop(self) -> None:
        if self.simulator_service is not None:
            await self.simulator_service.stop()
        for agent in [self.user, self.assistant, self.solver, self.explorer]:
            if not agent:
                continue
            try:
                await agent.stop()
            except Exception:
                pass
        if self.hasp_service is not None:
            await self.hasp_service.stop()
        await asyncio.sleep(1)

    async def run_case(self, case_path: Path) -> CaseResult:
        case = json.loads(case_path.read_text())
        details: Dict[str, Any] = {"case": case, "transcript": [], "assertions": [], "model_name": self.model_name}
        name = case.get("name") or case_path.stem
        case_response_timeout = float(case.get("response_timeout_seconds", self.response_timeout))
        self.logger.info("Running case: %s", name)
        if self.args.clear_signifiers_per_case:
            storage_dir = _clear_signifier_storage()
            new_thread = self._clear_in_memory_case_state()
            self.logger.info(
                "Cleared signifier storage and in-memory conversation state before case %s at %s%s",
                name,
                storage_dir,
                f" (thread={new_thread})" if new_thread else "",
            )
        started_at = datetime.now().astimezone()
        started_perf = time.perf_counter()
        LLMCallCounter.reset()
        PromptDumpState.start_case(name)

        try:
            await self._stop_simulator_for_setup()
            if not self.args.no_hard_reset and case.get("hard_reset", True):
                await self._restart_hasp_for_case()
            if not self.args.no_hard_reset and case.get("hard_reset", True):
                await self._hard_reset_lab308e()
            await self._run_steps(case.get("initial_state") or [], phase="initial_state")
            pre_query_assertions = case.get("pre_query_assertions") or []
            if pre_query_assertions:
                details["pre_query_assertions"] = []
                for assertion in pre_query_assertions:
                    result = await self._evaluate_assertion(assertion, details)
                    details["pre_query_assertions"].append(result)
                    if not result["passed"]:
                        details["failure_stage"] = "initial_state"
                        details["failed_assertion"] = result
                        details["started_at"] = started_at.isoformat()
                        details["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
                        details.update(LLMCallCounter.snapshot())
                        self._attach_prompt_dump(details)
                        return CaseResult(case_path, False, details)

            query = str(case["query"])
            await self.user.ask(query)
            first_reply = await self.user.wait_for_reply(case_response_timeout)
            details["transcript"].append({"role": "assistant", "phase": "proposal", "text": first_reply})
            details["first_reply"] = first_reply
            self._capture_plan(details)
            current_reply = first_reply

            clarification_replies = case.get("clarification_replies") or []
            clarification_steps: List[Dict[str, str]] = []
            max_clarification_turns = int(case.get("max_clarification_turns", max(len(clarification_replies), 1)))

            for clarification_turn in range(1, max_clarification_turns + 1):
                scripted_reply = self._match_clarification_reply(current_reply, clarification_replies)
                if not scripted_reply:
                    break
                self.logger.info(
                    "Matched clarification for case %s; replying with scripted answer: %s",
                    name,
                    scripted_reply,
                )
                clarification_steps.append({"assistant": current_reply, "reply": scripted_reply})
                await self.user.ask(scripted_reply)
                current_reply = await self.user.wait_for_reply(case_response_timeout)
                details["transcript"].append(
                    {"role": "assistant", "phase": f"clarification_{clarification_turn}", "text": current_reply}
                )
                self._capture_plan(details)

            if clarification_steps:
                details["clarification_replies_used"] = clarification_steps

            final_reply = None
            if case.get("auto_confirm", True):
                confirm_text = str(case.get("confirmation_text", "yes"))
                followup_text = self._clarification_followup(current_reply, confirm_text)
                if followup_text and followup_text != confirm_text:
                    self.logger.info(
                        "Clarification reply detected for case %s; resubmitting as goal: %s",
                        name,
                        followup_text,
                    )
                    details["clarification_followup"] = followup_text
                    await self.user.ask(followup_text)
                else:
                    self.user.drain_replies()
                    await self._start_simulator_for_execution()
                    await self.user.ask(confirm_text)
                final_reply = await self._wait_for_terminal_reply(details, case_response_timeout)

            if not details.get("plan"):
                details["failure_stage"] = "clarification"

            settle_seconds = float(case.get("settle_seconds", self.args.settle_seconds))
            if settle_seconds > 0:
                await asyncio.sleep(settle_seconds)

            assertions = case.get("pass_criteria") or []
            for assertion in assertions:
                result = await self._evaluate_assertion(assertion, details)
                details["assertions"].append(result)
                if not result["passed"]:
                    self._record_assertion_failure(details, result)
                    details["started_at"] = started_at.isoformat()
                    details["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
                    details.update(LLMCallCounter.snapshot())
                    self._attach_prompt_dump(details)
                    return CaseResult(case_path, False, details)

            details["started_at"] = started_at.isoformat()
            details["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
            details.update(LLMCallCounter.snapshot())
            self._attach_prompt_dump(details)
            return CaseResult(case_path, True, details)
        except Exception as exc:
            details["error"] = str(exc)
            details.setdefault("failure_stage", "runtime")
            details["started_at"] = started_at.isoformat()
            details["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
            details.update(LLMCallCounter.snapshot())
            self._attach_prompt_dump(details)
            self.logger.exception("Case failed: %s", name)
            return CaseResult(case_path, False, details)

    async def _run_steps(self, steps: List[Dict[str, Any]], *, phase: str) -> None:
        if not steps:
            return
        async with aiohttp.ClientSession() as session:
            for idx, step in enumerate(steps, start=1):
                step_type = step.get("type", "request")
                if step_type == "sleep":
                    await asyncio.sleep(float(step.get("seconds", 1.0)))
                    continue
                if step_type != "request":
                    raise ValueError(f"Unsupported step type in {phase}[{idx}]: {step_type}")

                method = str(step.get("method", "POST")).upper()
                url = self._resolve_url(
                    self._expand_env_placeholders(step.get("url")),
                    self._expand_env_placeholders(step.get("path")),
                )
                json_body = self._expand_env_placeholders(step.get("json"))
                headers = self._expand_env_placeholders(step.get("headers")) or None
                if isinstance(headers, dict):
                    auth_header = str(headers.get("Authorization") or "")
                    if re.fullmatch(r"Bearer\s*", auth_header):
                        raise RuntimeError(
                            f"{phase}[{idx}] uses Authorization: Bearer ${{HA_TOKEN}}, but HA_TOKEN is not set"
                        )
                expected_status = int(step.get("expect_status", 200))
                async with session.request(method, url, json=json_body, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status != expected_status:
                        raise AssertionError(
                            f"{phase}[{idx}] {method} {url} expected HTTP {expected_status}, got {resp.status}: {text}"
                        )

    async def _evaluate_assertion(self, assertion: Dict[str, Any], details: Dict[str, Any]) -> Dict[str, Any]:
        assertion_type = assertion["type"]
        if assertion_type == "any_of":
            children = assertion.get("assertions") or []
            results = []
            for child in children:
                result = await self._evaluate_assertion(child, details)
                results.append(result)
                if result.get("passed"):
                    return {"type": assertion_type, "passed": True, "results": results}
            return {"type": assertion_type, "passed": False, "results": results}

        if assertion_type == "all_of":
            children = assertion.get("assertions") or []
            results = []
            all_passed = True
            for child in children:
                result = await self._evaluate_assertion(child, details)
                results.append(result)
                if not result.get("passed"):
                    all_passed = False
            return {"type": assertion_type, "passed": all_passed, "results": results}

        if assertion_type in {"assistant_reply_contains", "final_reply_contains"}:
            haystack = "\n".join(item["text"] for item in details["transcript"])
            if assertion_type == "final_reply_contains":
                haystack = details["transcript"][-1]["text"] if details["transcript"] else ""
            needle = str(assertion["value"])
            passed = needle in haystack
            return {
                "type": assertion_type,
                "passed": passed,
                "expected": needle,
                "actual": haystack,
            }

        if assertion_type in {"property_equals", "property_gte", "property_lte", "property_in"}:
            timeout_s = float(assertion.get("timeout_seconds", 30.0))
            interval_s = float(assertion.get("poll_interval_seconds", 1.0))
            path = assertion.get("path")
            url = self._resolve_url(assertion.get("url"), path)
            last_value: Any = None
            deadline = asyncio.get_running_loop().time() + timeout_s
            async with aiohttp.ClientSession() as session:
                while True:
                    async with session.get(url) as resp:
                        body = await resp.text()
                        if resp.status >= 400:
                            raise AssertionError(f"Assertion GET {url} failed with HTTP {resp.status}: {body}")
                        try:
                            value = json.loads(body)
                        except json.JSONDecodeError:
                            value = body
                        last_value = value

                    passed = False
                    if assertion_type == "property_equals":
                        passed = last_value == assertion["equals"]
                    elif assertion_type == "property_gte":
                        passed = float(last_value) >= float(assertion["gte"])
                    elif assertion_type == "property_lte":
                        passed = float(last_value) <= float(assertion["lte"])
                    elif assertion_type == "property_in":
                        passed = last_value in list(assertion["values"])

                    if passed:
                        return {
                            "type": assertion_type,
                            "passed": True,
                            "url": url,
                            "actual": last_value,
                            "expected": assertion.get("equals", assertion.get("gte", assertion.get("lte", assertion.get("values")))),
                        }
                    if asyncio.get_running_loop().time() >= deadline:
                        return {
                            "type": assertion_type,
                            "passed": False,
                            "url": url,
                            "actual": last_value,
                            "expected": assertion.get("equals", assertion.get("gte", assertion.get("lte", assertion.get("values")))),
                            "timeout_seconds": timeout_s,
                        }
                    await asyncio.sleep(interval_s)

        raise ValueError(f"Unsupported assertion type: {assertion_type}")

    def _resolve_url(self, url: Optional[str], path: Optional[str]) -> str:
        if isinstance(url, str) and url.strip():
            return url.strip()
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Expected either 'url' or 'path'")
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.yggdrasil_url}/{path.lstrip('/')}"

    def _expand_env_placeholders(self, value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(
                r"\$\{([A-Z0-9_]+)\}",
                lambda match: os.getenv(match.group(1), ""),
                value,
            )
        if isinstance(value, list):
            return [self._expand_env_placeholders(item) for item in value]
        if isinstance(value, dict):
            return {
                self._expand_env_placeholders(key): self._expand_env_placeholders(val)
                for key, val in value.items()
            }
        return value


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JSON-driven lab308e E2E cases.")
    parser.add_argument("--case-dir", default=str(Path(__file__).resolve().parent / "cases"))
    parser.add_argument("--case", help="Run one specific JSON file.")
    parser.add_argument("--case-name", help="Run one specific test by JSON 'name' field.")
    parser.add_argument("--response-timeout", type=float, default=900.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--clear-signifiers", action="store_true")
    parser.add_argument("--clear-signifiers-per-case", action="store_true")
    parser.add_argument("--skip-prewarm", action="store_true")
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--no-hard-reset", action="store_true")
    parser.add_argument("--manage-hasp", action="store_true")
    parser.add_argument("--manage-simulator", action="store_true")
    parser.add_argument("--hasp-host", default="0.0.0.0")
    parser.add_argument("--service-start-timeout", type=float, default=30.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--write-results", action="store_true")
    return parser.parse_args(argv)


def _discover_cases(args: argparse.Namespace) -> List[Path]:
    if args.case:
        return [Path(args.case).resolve()]
    case_dir = Path(args.case_dir).resolve()
    cases = sorted(p for p in case_dir.glob("*.json") if p.is_file())
    if not args.case_name:
        return cases

    target_name = args.case_name.strip()
    matches: List[Path] = []
    for case_path in cases:
        try:
            payload = json.loads(case_path.read_text())
        except Exception:
            continue
        if payload.get("name") == target_name:
            matches.append(case_path)

    if not matches:
        raise ValueError(f"No case found with name {target_name!r} in {case_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple cases found with name {target_name!r}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches


def _append_results_csv(results: List[CaseResult]) -> Path:
    out_path = Path(__file__).resolve().parent / "results.csv"
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(
                [
                    "test_name",
                    "timestamp",
                    "model_name",
                    "passed",
                    "duration_seconds",
                    "llm_calls",
                    "input_tokens",
                    "output_tokens",
                    "plan",
                    "failure_stage",
                ]
            )
        for result in results:
            details = result.details or {}
            case = details.get("case") or {}
            writer.writerow(
                [
                    case.get("name") or result.path.stem,
                    details.get("started_at") or datetime.now().astimezone().isoformat(),
                    details.get("model_name"),
                    "passed" if result.passed else "failed",
                    details.get("duration_seconds"),
                    details.get("llm_calls"),
                    details.get("input_tokens"),
                    details.get("output_tokens"),
                    details.get("plan"),
                    details.get("failure_stage"),
                ]
            )
    return out_path


async def _async_main(args: argparse.Namespace) -> int:
    logger = _configure_logging(args.log_level)
    _install_openai_call_counter()
    cases = _discover_cases(args)
    if not cases:
        logger.error("No case files found.")
        return 2

    harness = Lab308eHarness(args, logger)
    results: List[CaseResult] = []
    try:
        await harness.start()
        for case_path in cases:
            results.append(await harness.run_case(case_path))
    finally:
        await harness.stop()

    failures = [result for result in results if not result.passed]
    print("\n=== lab308e E2E Summary ===")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.path.name}")
        if not result.passed and result.details.get("error"):
            print(f"  error: {result.details['error']}")
    print(f"Passed {len(results) - len(failures)}/{len(results)} cases")

    csv_path = _append_results_csv(results)
    print(f"Appended results to {csv_path}")

    if args.write_results:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            out_path = results_dir / f"{result.path.stem}.result.json"
            out_path.write_text(json.dumps({"passed": result.passed, **result.details}, indent=2))

    return 1 if failures else 0


def main() -> int:
    args = _parse_args(sys.argv[1:])
    return spade.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
