"""Demo: community query flow with real agents.

This script starts two InteractionSolver agents, two EnvExplorer agents,
and one demo client, then runs three scenarios:

- NO_Q: community disabled on both solver agents
- T_OUT: only the querying solver has community enabled; the peer stays silent
- RESP: both solvers have community enabled and the peer responds

Usage:
  conda run -n ami python tests/demo_community_flow.py \
    --intent "Turn on light308" \
    --workspace-id lab308 \
    --intent-type explicit \
    --polls 3 \
    --poll-interval 1.0

Optional:
  --scenario all|no_q|t_out|resp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp
import spade
from dotenv import load_dotenv
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ami_agents.shared.utils import spade_compat  # noqa: F401
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.shared.models.messages import META_CORRELATION_ID, MessageType, serialize_body
from ami_agents.shared.utils.spade_rpc import send_via_router
from ami_agents.shared.utils.demo_log import demo
from ami_agents.shared.utils.logger import LoggerFactory
from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.interaction_solver.utils.community import Community


class StatusClientAgent(Agent):
    async def setup(self):
        return


class StatusResponseWaiter(OneShotBehaviour):
    """Wait for one PLANNING_STATUS_RESPONSE with a matching correlation id."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self.timeout = timeout
        self.response_body: Optional[str] = None

    async def run(self) -> None:
        msg = await self.receive(timeout=self.timeout)
        if msg:
            self.response_body = msg.body


class PlanCreatedWaiter(OneShotBehaviour):
    """Wait for one PLAN_CREATED with a matching correlation id."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self.timeout = timeout
        self.response_body: Optional[str] = None

    async def run(self) -> None:
        msg = await self.receive(timeout=self.timeout)
        if msg:
            self.response_body = msg.body


class _DemoOnlyFilter(logging.Filter):
    """Allow only log entries containing the [DEMO] prefix."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:
            return False
        return "[DEMO]" in msg


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo community planning flow")
    parser.add_argument("--intent", action="append", default=[], help="Goal intent text")
    parser.add_argument("--workspace-id", default=None, help="Workspace ID to scope planning")
    parser.add_argument("--intent-type", default="implicit", choices=["implicit", "explicit"])
    parser.add_argument("--goal-id", default=None, help="Goal id to reuse (optional)")
    parser.add_argument("--polls", type=int, default=3, help="Number of status polls")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between polls")
    parser.add_argument("--skip-yggdrasil-check", action="store_true")
    parser.add_argument("--demo-only", action="store_true", help="Show only [DEMO] log lines")
    parser.add_argument("--scenario", default="all", choices=["all", "no_q", "t_out", "resp"])
    parser.add_argument("--community-timeout", type=float, default=3.0, help="Community query timeout for the demo scenarios")
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs" / "demo_community_flow"), help="Directory for per-scenario log files")
    return parser.parse_args(argv)


def _configure_logging(config: Dict[str, Any], *, demo_only: bool = False) -> None:
    logging_config = config.get("logging", {})
    root = logging.getLogger()
    root.setLevel(getattr(logging, logging_config.get("level", "INFO")))
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console_handler = LoggerFactory.create_console_handler(logging_config)
    if console_handler:
        formatter = logging.Formatter(
            logging_config.get("format", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    file_handler = LoggerFactory.create_file_handler(logging_config)
    if file_handler:
        formatter = logging.Formatter(
            logging_config.get("format", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if demo_only:
        demo_filter = _DemoOnlyFilter()
        root.addFilter(demo_filter)
        for handler in list(getattr(root, "handlers", []) or []):
            handler.addFilter(demo_filter)


def _attach_scenario_log_file(config: Dict[str, Any], scenario_name: str, log_dir: Path) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    scenario_log_path = log_dir / f"{scenario_name.lower()}.log"
    formatter = logging.Formatter(
        config.get("logging", {}).get("format", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    handler = logging.FileHandler(scenario_log_path, mode="w", encoding="utf-8")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    return handler


async def _poll_status(
    agent: Agent,
    solver_jid: str,
    goal_id: str,
    polls: int,
    poll_interval: float,
    logger,
) -> None:
    for idx in range(1, polls + 1):
        corr_id = str(uuid.uuid4())
        try:
            msg = SpadeMessage(to=solver_jid)
            msg.set_metadata("type", MessageType.PLANNING_STATUS_REQUEST.value)
            msg.set_metadata(META_CORRELATION_ID, corr_id)
            msg.body = serialize_body({"goal_id": goal_id})

            template = Template()
            template.set_metadata("type", MessageType.PLANNING_STATUS_RESPONSE.value)
            template.set_metadata(META_CORRELATION_ID, corr_id)
            waiter = StatusResponseWaiter(timeout=5.0)
            agent.add_behaviour(waiter, template=template)
            await send_via_router(agent, msg)
            await waiter.join()

            if waiter.response_body is None:
                logger.warning("Status poll timed out (%d/%d)", idx, polls)
            else:
                logger.info(demo("Status poll %d/%d: %s"), idx, polls, waiter.response_body)
        except Exception as exc:
            logger.warning("Status poll failed: %s", exc)
        await asyncio.sleep(poll_interval)


async def _wait_for_discovery(*explorers: EnvExplorerAgent) -> None:
    for _ in range(120):
        if explorers and all(getattr(explorer, "discovery_complete", False) for explorer in explorers):
            return
        await asyncio.sleep(0.5)


async def _send_goal(
    client: Agent,
    solver_jid: str,
    goal_id: str,
    intents: List[str],
    intent_type: str,
    workspace_id: Optional[str],
    timeout: float = 120.0,
) -> Optional[str]:
    goal_corr_id = str(uuid.uuid4())
    goal_msg = SpadeMessage(to=solver_jid)
    goal_msg.set_metadata("type", MessageType.GOAL_REQUEST.value)
    goal_msg.set_metadata(META_CORRELATION_ID, goal_corr_id)

    payload: Dict[str, Any] = {
        "goal_id": goal_id,
        "intents": intents,
        "intent_type": intent_type,
    }
    if workspace_id:
        payload["workspace_id"] = workspace_id
    goal_msg.body = serialize_body(payload)

    goal_template = Template()
    goal_template.set_metadata("type", MessageType.PLAN_CREATED.value)
    goal_template.set_metadata(META_CORRELATION_ID, goal_corr_id)
    goal_waiter = PlanCreatedWaiter(timeout=timeout)
    client.add_behaviour(goal_waiter, template=goal_template)
    await send_via_router(client, goal_msg)
    await goal_waiter.join()
    return goal_waiter.response_body


async def _run_scenario(
    *,
    name: str,
    client: Agent,
    solver_a: InteractionSolverAgent,
    solver_b: InteractionSolverAgent,
    explorer_a: EnvExplorerAgent,
    explorer_b: EnvExplorerAgent,
    logger,
    intents: List[str],
    intent_type: str,
    workspace_id: Optional[str],
    polls: int,
    poll_interval: float,
    goal_wait_timeout: float,
    community_timeout: float,
    config: Dict[str, Any],
    log_dir: Path,
    solver_a_community_enabled: bool,
    solver_b_community_enabled: bool,
) -> None:
    print(f"\n=== {name} scenario ===")

    scenario_handler = _attach_scenario_log_file(config, name, log_dir)

    solver_a.community_enabled = solver_a_community_enabled
    solver_b.community_enabled = solver_b_community_enabled
    solver_a.community_query_timeout = community_timeout
    solver_b.community_query_timeout = community_timeout
    solver_a.community_min_response_ratio = 0.5
    solver_b.community_min_response_ratio = 0.5

    solver_a.target_jids["explorer"] = str(explorer_a.jid)
    solver_b.target_jids["explorer"] = str(explorer_b.jid)

    goal_id = str(uuid.uuid4())
    logger.info(
        demo(
            "%s: solver_a.community_enabled=%s solver_b.community_enabled=%s timeout=%.1f goal_id=%s"
        ),
        name,
        solver_a.community_enabled,
        solver_b.community_enabled,
        community_timeout,
        goal_id,
    )

    plan_future = asyncio.create_task(
        _send_goal(
            client=client,
            solver_jid=str(solver_a.jid),
            goal_id=goal_id,
            intents=intents,
            intent_type=intent_type,
            workspace_id=workspace_id,
            timeout=goal_wait_timeout,
        )
    )

    await asyncio.sleep(0.2)
    await _poll_status(
        agent=client,
        solver_jid=str(solver_a.jid),
        goal_id=goal_id,
        polls=polls,
        poll_interval=poll_interval,
        logger=logger,
    )

    plan_body = await plan_future
    if plan_body is None:
        logger.warning("PLAN_CREATED timed out for %s", name)
    else:
        logger.info(demo("%s PLAN_CREATED received: %s"), name, plan_body)

    logging.getLogger().removeHandler(scenario_handler)
    scenario_handler.close()


async def main() -> None:
    args = _parse_args(sys.argv[1:])

    xmpp_server = os.getenv("SPADE_SERVER", "localhost")
    password = os.getenv("SPADE_PASSWORD", "password")
    yggdrasil_url = os.getenv("YGGDRASIL_URL", "http://localhost:8080/").strip()

    explorer_a_jid = os.getenv("EE_A_JID") or f"env_explorer_a@{xmpp_server}"
    explorer_b_jid = os.getenv("EE_B_JID") or f"env_explorer_b@{xmpp_server}"
    solver_a_jid = os.getenv("IS_A_JID") or f"interaction_solver_a@{xmpp_server}"
    solver_b_jid = os.getenv("IS_B_JID") or f"interaction_solver_b@{xmpp_server}"
    client_jid = f"goal_community_demo@{xmpp_server}"

    try:
        config = ConfigLoader.merge_configs(
            ConfigLoader.load_with_env_vars(str(PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml")),
            ConfigLoader.load_with_env_vars(str(PROJECT_ROOT / "ami_agents" / "config" / "environment.yaml")),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load agent configuration: {exc}")

    _configure_logging(config, demo_only=args.demo_only)
    logger = LoggerFactory.get_logger("GoalCommunityDemo", config.get("logging", {}))
    log_dir = Path(args.log_dir)

    config["yggdrasil_url"] = yggdrasil_url
    config["yggdrasil"] = {"url": yggdrasil_url}
    config["discovery"] = {
        "notify_on_discovery_complete": True,
        "notify_agents": [solver_a_jid, solver_b_jid],
    }

    if not args.skip_yggdrasil_check:
        logger.info(demo("Checking Yggdrasil reachable at %s ..."), yggdrasil_url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(yggdrasil_url) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"Yggdrasil returned HTTP {resp.status}")
        except Exception as exc:
            raise RuntimeError(f"Yggdrasil is not reachable at {yggdrasil_url}: {exc}") from exc

    intents = args.intent or ["Turn on a light"]
    goal_wait_timeout = 120.0
    community_timeout = float(args.community_timeout)

    explorer_a = EnvExplorerAgent(explorer_a_jid, password, config)
    explorer_b = EnvExplorerAgent(explorer_b_jid, password, config)
    solver_a = InteractionSolverAgent(
        solver_a_jid,
        password,
        config=config,
        target_jids={"explorer": explorer_a_jid},
    )
    solver_b = InteractionSolverAgent(
        solver_b_jid,
        password,
        config=config,
        target_jids={"explorer": explorer_b_jid},
    )
    client = StatusClientAgent(client_jid, password)

    # Both solvers should know about the same community.
    shared_community = Community(
        community_id="demo-community",
        member_ids=[solver_a_jid, solver_b_jid],
    )
    solver_a.communities = [shared_community]
    solver_b.communities = [shared_community]

    try:
        logger.info("Starting InteractionSolver A...")
        await solver_a.start(auto_register=True)

        logger.info("Starting InteractionSolver B...")
        await solver_b.start(auto_register=True)

        logger.info("Starting EnvExplorer A...")
        await explorer_a.start(auto_register=True)

        logger.info("Starting EnvExplorer B...")
        await explorer_b.start(auto_register=True)

        logger.info("Starting demo client...")
        await client.start(auto_register=True)

        await _wait_for_discovery(explorer_a, explorer_b)

        scenarios: List[Tuple[str, bool, bool]]
        if args.scenario == "no_q":
            scenarios = [("NO_Q", False, False)]
        elif args.scenario == "t_out":
            scenarios = [("T_OUT", True, False)]
        elif args.scenario == "resp":
            scenarios = [("RESP", True, True)]
        else:
            scenarios = [
                ("NO_Q", False, False),
                ("T_OUT", True, False),
                ("RESP", True, True),
            ]

        for scenario_name, solver_a_enabled, solver_b_enabled in scenarios:
            await _run_scenario(
                name=scenario_name,
                client=client,
                solver_a=solver_a,
                solver_b=solver_b,
                logger=logger,
                intents=intents,
                intent_type=args.intent_type,
                workspace_id=args.workspace_id,
                polls=args.polls,
                poll_interval=args.poll_interval,
                goal_wait_timeout=goal_wait_timeout,
                community_timeout=community_timeout,
                config=config,
                log_dir=log_dir,
                solver_a_community_enabled=solver_a_enabled,
                solver_b_community_enabled=solver_b_enabled,
                explorer_a=explorer_a,
                explorer_b=explorer_b,
            )
            await asyncio.sleep(1.0)

    finally:
        logger.info("Stopping agents...")
        for agent in [client, solver_a, solver_b, explorer_a, explorer_b]:
            try:
                await agent.stop()
            except Exception:
                pass
        await asyncio.sleep(1)


if __name__ == "__main__":
    spade.run(main())
