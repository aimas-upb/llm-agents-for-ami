#!/usr/bin/env python3
"""
Bring up a SimuHome simulator loaded with one benchmark episode, plus the
inspector, in a single command.

    python ami_agents/environment/integration/SimuHome/run_inspector.py \\
        --episode qt2_feasible_seed_77

Then open the printed URL. Ctrl-C stops both processes.

This is the phase-1 "prove the loop" path made repeatable: start the server,
POST the episode's `initial_home_config` verbatim, wait for it to be live, and
serve a read-only view of it.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SIMUHOME_REPO = REPO.parent / "SimuHome"
DEFAULT_BENCHMARK = SIMUHOME_REPO / "data" / "benchmark"

sys.path.insert(0, str(HERE))
from sim_client import SimuHomeClient  # noqa: E402


def free_port() -> int:
    """Ask the OS for an unused port, as SimuHome's own harness does."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_healthy(client: SimuHomeClient, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.health():
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a SimuHome episode with the inspector.")
    parser.add_argument("--episode", default="qt2_feasible_seed_77",
                        help="benchmark episode name, with or without .json")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK,
                        help="SimuHome benchmark directory")
    parser.add_argument("--sim-port", type=int, default=0, help="0 = allocate a free port")
    parser.add_argument("--port", type=int, default=8098, help="inspector port")
    parser.add_argument("--simuhome", type=Path, default=SIMUHOME_REPO,
                        help="path to the SimuHome repo")
    args = parser.parse_args()

    name = args.episode if args.episode.endswith(".json") else f"{args.episode}.json"
    episode_path = args.benchmark / name
    if not episode_path.is_file():
        print(f"Episode not found: {episode_path}", file=sys.stderr)
        return 2
    if not (args.simuhome / "src" / "simulator" / "api" / "app.py").is_file():
        print(f"SimuHome repo not found at {args.simuhome}", file=sys.stderr)
        return 2

    sim_port = args.sim_port or free_port()
    sim_base = f"http://127.0.0.1:{sim_port}/api"

    env = os.environ.copy()
    env["SERVER_PORT"] = str(sim_port)
    print(f"[run] starting simulator on {sim_port}", flush=True)
    sim = subprocess.Popen(
        [sys.executable, "-m", "src.simulator.api.app"],
        cwd=str(args.simuhome), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    inspector: Optional[subprocess.Popen] = None
    try:
        client = SimuHomeClient(sim_base)
        if not wait_healthy(client):
            print("simulator did not become healthy", file=sys.stderr)
            return 1

        config = json.loads(episode_path.read_text(encoding="utf-8"))["initial_home_config"]
        print(f"[run] loading {name}: {len(config.get('rooms', {}))} rooms", flush=True)
        client.reset(config)

        state = client.home_state()
        devices = sum(len(r.get("devices", [])) for r in (state.get("rooms") or {}).values())
        print(f"[run] loaded: {len(state.get('rooms') or {})} rooms, {devices} devices, "
              f"clock {state.get('current_time')}", flush=True)

        inspector = subprocess.Popen(
            [sys.executable, str(HERE / "inspector.py"),
             "--sim", sim_base, "--port", str(args.port)],
        )
        print(f"\n  Inspector:  http://127.0.0.1:{args.port}/")
        print(f"  Simulator:  {sim_base}")
        print("\n  Ctrl-C to stop both.\n", flush=True)

        signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
        while True:
            if sim.poll() is not None:
                print("simulator exited", file=sys.stderr)
                return 1
            if inspector.poll() is not None:
                print("inspector exited", file=sys.stderr)
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[run] stopping", flush=True)
        return 0
    finally:
        for proc in (inspector, sim):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
