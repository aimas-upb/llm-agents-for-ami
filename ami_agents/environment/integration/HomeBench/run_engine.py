#!/usr/bin/env python3
"""Run one HomeBench home by hand: the engine with the home loaded, and HBTD in front.

The HomeBench counterpart of ``SimuHome/run_inspector.py``: the engine on a fixed
port, so a console runner or a browser can be pointed at it, with the home of your
choice already loaded. Nothing is started by the bench; this is for debugging a
single home or episode without a worker.

Worked example (home 42, the home of ``hb1_feasible_seed_1``)::

    python ami_agents/environment/integration/HomeBench/run_engine.py --home 42 \\
        --sim-port 8099 --hbtd-port 8097
    # engine  http://127.0.0.1:8099/api   (GET /api/home/state shows the home)
    # HBTD    http://127.0.0.1:8097/      (the crawler's root; the home workspace is home42)
    # then, for the console runner:  YGGDRASIL_URL=http://127.0.0.1:8097

``--engine-only`` leaves HBTD out. Ctrl-C stops what was started.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

if __package__ in {None, ""}:  # allow `python .../HomeBench/run_engine.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ami_agents.environment.integration.HomeBench.worker import (
    DEFAULT_BENCHMARK, HERE, HOST, _terminate,
)
from ami_agents.environment.integration.HomeBench.sim_client import HomeBenchClient

LOGGER = logging.getLogger("run_engine")

DEFAULT_SIM_PORT = 8099
DEFAULT_HBTD_PORT = 8097
STARTUP_SECONDS = 60.0


def _wait(probe, what: str) -> None:
    """Poll ``probe()`` until it is true, or raise ``TimeoutError`` naming ``what``."""
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        try:
            if probe():
                return
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(0.3)
    raise TimeoutError(f"{what} did not come up within {STARTUP_SECONDS:.0f}s")


def run(home: int, benchmark: Path, sim_port: int, hbtd_port: Optional[int]) -> int:
    """Start the engine on ``sim_port`` with home ``home`` loaded, and HBTD on
    ``hbtd_port`` unless it is None; block until interrupted. Returns the exit code."""
    home_file = benchmark / "homes" / f"home{home}.json"
    if not home_file.is_file():
        LOGGER.error("no such home file: %s (run tests/homebench/convert.py first)", home_file)
        return 1
    processes: List[subprocess.Popen] = []
    try:
        # 1. The engine, preloaded with the home.
        processes.append(subprocess.Popen(
            [sys.executable, str(HERE / "simulator.py"), "--host", HOST,
             "--port", str(sim_port), "--home", str(home_file)]))
        sim_base = f"http://{HOST}:{sim_port}/api"
        _wait(HomeBenchClient(sim_base).health, "engine")
        LOGGER.info("engine  %s  home%d", sim_base, home)

        # 2. HBTD in front of it, bound to the same home id the worker would use.
        if hbtd_port is not None:
            processes.append(subprocess.Popen(
                [sys.executable, str(HERE / "hbtd.py"), "--sim", sim_base,
                 "--home", f"home{home}", "--host", HOST, "--port", str(hbtd_port)]))
            LOGGER.info("HBTD    http://%s:%d/  (home workspace home%d)", HOST, hbtd_port, home)

        # 3. Until interrupted, or until one of them dies.
        while all(p.poll() is None for p in processes):
            time.sleep(1.0)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            _terminate(process)


def main() -> int:
    """CLI: see the module docstring."""
    parser = argparse.ArgumentParser(description="Run one HomeBench home by hand.")
    parser.add_argument("--home", type=int, required=True, help="home id, 0-99")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK,
                        help="the converter's output directory")
    parser.add_argument("--sim-port", type=int, default=DEFAULT_SIM_PORT)
    parser.add_argument("--hbtd-port", type=int, default=DEFAULT_HBTD_PORT)
    parser.add_argument("--engine-only", action="store_true", help="do not start HBTD")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return run(args.home, args.benchmark, args.sim_port,
               None if args.engine_only else args.hbtd_port)


if __name__ == "__main__":
    raise SystemExit(main())
