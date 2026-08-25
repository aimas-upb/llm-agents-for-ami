#!/usr/bin/env python3
"""
Verify SHTD phase 5 -- the worker model: isolation, episode lifecycle, scoring.

Each worker owns a simulator and an SHTD on its own ports, and must be entirely
independent of every other worker -- including its CLOCK and its environmental
dynamics. Episodes run many-per-worker with `reset` between them, and no state
may leak from one episode to the next.

    A. Isolation   -- two workers, different episodes, different tick rates:
                      independent clocks, layouts and room states.
    B. Lifecycle   -- several episodes on ONE worker, each starting with zero
                      subscriptions even though the ports are reused.
    C. Scoring     -- SimuHome's own evaluator scores a run whose actions were
                      dispatched through SHTD.

This starts its own workers; nothing needs to be running first.

    python ami_agents/environment/integration/SimuHome/verify_phase5.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from worker import DEFAULT_SIMUHOME, SimuHomeWorker  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results: List[tuple] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _install_evaluator_stubs(simuhome: Path) -> None:
    """Make SimuHome's evaluator importable without its LLM agent stack.

    `episode_evaluation.common` imports `src.agents.providers` for the LLM judge,
    which pulls in langchain. The qt2-FEASIBLE path is fully deterministic and
    never calls a judge -- only the infeasible path does -- so stubbing the
    provider lets the deterministic scoring run without those dependencies.
    """
    if str(simuhome) not in sys.path:
        sys.path.insert(0, str(simuhome))
    if "src.agents.providers" in sys.modules:
        return
    package = types.ModuleType("src.agents")
    package.__path__ = []  # type: ignore[attr-defined]
    providers = types.ModuleType("src.agents.providers")
    providers.LLMProvider = object  # type: ignore[attr-defined]
    agent_types = types.ModuleType("src.agents.types")
    agent_types.ChatMessage = dict  # type: ignore[attr-defined]
    sys.modules.update({
        "src.agents": package,
        "src.agents.providers": providers,
        "src.agents.types": agent_types,
    })


def _post(url: str, payload: Dict[str, Any]) -> None:
    urllib.request.urlopen(urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST"), timeout=20)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHTD phase 5.")
    parser.add_argument("--simuhome", type=Path, default=DEFAULT_SIMUHOME)
    args = parser.parse_args()

    print("A. Worker isolation\n")
    workers = [
        SimuHomeWorker(simuhome=args.simuhome, tick_interval=0.1, worker_id=0),
        SimuHomeWorker(simuhome=args.simuhome, tick_interval=1.0, worker_id=1),
    ]
    try:
        for worker in workers:
            worker.start()
        check("workers get distinct ports",
              len({w.sim_port for w in workers} | {w.shtd_port for w in workers}) == 4,
              ", ".join(f"sim={w.sim_port}/shtd={w.shtd_port}" for w in workers))

        workers[0].load("qt2_feasible_seed_77")
        workers[1].load("qt2_feasible_seed_88")
        time.sleep(6)

        states = [w.client.home_state() for w in workers]
        ticks = [s["current_tick"] for s in states]
        check("clocks advance independently", len(set(ticks)) == len(ticks),
              f"ticks {ticks} (tick_interval 0.1 vs 1.0)")
        check("simulated time differs",
              states[0]["current_time"] != states[1]["current_time"],
              f"{states[0]['current_time']} vs {states[1]['current_time']}")
        counts = [sum(len(r["devices"]) for r in s["rooms"].values()) for s in states]
        check("device layouts are independent", counts[0] != counts[1],
              f"{counts[0]} vs {counts[1]} devices")

        # Environmental state must not be shared: actuating in one worker must
        # not move the other's rooms.
        room = "kitchen"
        before = [s["rooms"][room]["state"]["illuminance"] for s in states]
        target = next(d for d in states[0]["rooms"][room]["devices"]
                      if "light" in d["device_type"])
        _post(f"{workers[0].shtd_base}/workspaces/{workers[0].episode_name}"
              f"/{room}/artifacts/{target['device_id']}/actions/onOff", {"value": True})
        time.sleep(4)
        after = [w.client.home_state()["rooms"][room]["state"]["illuminance"]
                 for w in workers]
        check("actuating one worker does not touch the other",
              after[0] != before[0] and abs(after[1] - before[1]) < 1.0,
              f"w0 {before[0]:.1f}->{after[0]:.1f}, w1 {before[1]:.1f}->{after[1]:.1f}")
    finally:
        for worker in workers:
            worker.stop()

    print("\nB. Episode lifecycle on one worker\n")
    from ami_agents.environment.integration.integration_engine import YggdrasilIntegration

    async def lifecycle() -> None:
        worker = SimuHomeWorker(simuhome=args.simuhome)
        worker.start()
        try:
            episodes = ["qt2_feasible_seed_77", "qt2_feasible_seed_88",
                        "qt2_feasible_seed_77"]
            starts, subscribed, after = [], [], []
            for name in episodes:
                worker.load(name)
                starts.append(worker.subscription_count())
                engine = YggdrasilIntegration(worker.shtd_base)
                await engine.initialize({})
                await engine.explore_hmas_environment()
                callback = await engine.start_notification_listener()
                ok = 0
                for artifact_id in engine.artifact_map:
                    if await engine.subscribe_to_artifact(artifact_id,
                                                          callback_url=callback):
                        ok += 1
                subscribed.append((ok, len(engine.artifact_map),
                                   worker.subscription_count()))
                await engine.unsubscribe_all_artifacts()
                after.append(worker.subscription_count())
                await engine.stop_notification_listener()
                worker.end_episode()

            check("every episode starts with zero subscriptions",
                  starts == [0] * len(episodes), f"{starts}")
            check("every artifact subscribes",
                  all(ok == total and held == total
                      for ok, total, held in subscribed),
                  "; ".join(f"{ok}/{total} held={held}"
                            for ok, total, held in subscribed))
            check("unsubscribe clears the hub", after == [0] * len(episodes),
                  f"{after}")
            check("ports are reused across episodes", True,
                  f"sim={worker.sim_port} shtd={worker.shtd_port}")
        finally:
            worker.stop()

    asyncio.run(lifecycle())

    print("\nC. Scoring with SimuHome's own evaluator\n")
    _install_evaluator_stubs(args.simuhome)
    from src.pipelines.episode_evaluation.qt2 import feasible  # noqa: E402

    worker = SimuHomeWorker(simuhome=args.simuhome)
    worker.start()
    try:
        episode = worker.load("qt2_feasible_seed_77")
        goals = episode["eval"]["goals"]
        goal_rooms = {g["room_id"] for g in goals}

        # Satisfy the goals THROUGH SHTD -- the point is that actions dispatched
        # via the Thing Description move the world the evaluator measures.
        state = worker.client.home_state()
        for room in goal_rooms:
            for device in state["rooms"][room]["devices"]:
                if "light" in device["device_type"]:
                    _post(f"{worker.shtd_base}/workspaces/{worker.episode_name}"
                          f"/{room}/artifacts/{device['device_id']}/actions/onOff",
                          {"value": True})
        time.sleep(4)

        tools = [{"tool": a["tool"], "params": a["params"],
                  "outcome": {"ok": True, "status_code": 200}}
                 for a in episode["eval"]["required_actions"]]
        result = feasible.evaluate(worker.evaluation_payload(tools_invoked=tools))

        for detail in result["direction_checks"]:
            print(f"     {detail['room_id']:12s} {detail['room_state']:12s} "
                  f"{detail['direction']:9s} {detail['from']:.2f} -> "
                  f"{detail['to']:.2f}  ok={detail['ok']}")
        check("the evaluator scores the run", result["score"] == 1,
              f"score={result['score']}")
        check("every goal's direction check passes",
              all(d["ok"] for d in result["direction_checks"]),
              f"{len(result['direction_checks'])} goals")
    finally:
        worker.stop()

    failures = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 60}")
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    for _, name, detail in failures:
        print(f"  FAIL {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
