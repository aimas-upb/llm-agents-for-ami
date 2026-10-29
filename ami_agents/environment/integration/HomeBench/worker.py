#!/usr/bin/env python3
"""One isolated HomeBench worker: an engine, an HBTD, and the episode lifecycle.

The same shape and API as ``SimuHome/worker.py`` (`SimuHomeWorker`), so the bench
starts either benchmark's worker the same way. A worker is two processes on free
ports, started and reaped together:

    engine   ``simulator.py``: one process, one home, the gold-consistent state
    HBTD     ``hbtd.py``: one process, projecting THAT engine's home as TDs
    episodes many per worker; ``load`` resets the engine and restarts HBTD

**HBTD is restarted for every episode, as SimuHome's SHTD is.** The home id is baked
in at process start and is the workspace segment of every IRI; a restart also clears
the hub's subscription table and the state watcher's baseline (both per process).
It costs a second or two against 20 s or more of LLM time per episode, and it removes
a place where HomeBench could behave differently from SimuHome.

**The home id is ``home<id>``**, one of the 100 real homes, so the same id repeats
across the many episodes of one home (SimuHome uses the episode's own name).

Worked example::

    with HomeBenchWorker(benchmark=Path("tests/homebench/data")) as worker:
        episode = worker.load("hb1_feasible_seed_1")     # home 17, say
        worker.shtd_base                                 # http://127.0.0.1:<port>
        worker.client.home_state()                       # engine = the home's initial state
        ...                                              # run the agents against shtd_base
        worker.load("hb2_mixed_seed_3")                  # ends episode 1 first: HBTD stopped,
                                                         # engine reset, HBTD back on home<id2>
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

if __package__ in {None, ""}:  # allow `python .../HomeBench/worker.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ami_agents.environment.integration.HomeBench.engine import load_home_file
from ami_agents.environment.integration.HomeBench.sim_client import HomeBenchClient
from ami_agents.environment.integration.HomeBench.td_builder import (
    DEFAULT_ENVIRONMENT, ENVIRONMENT_MODES,
)

LOGGER = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_BENCHMARK = REPO / "tests" / "homebench" / "data"
HOST = "127.0.0.1"


# -- helpers ------------------------------------------------------------------------


def free_port() -> int:
    """Ask the OS for an unused port, as SimuHome's own harness does."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _get_json(url: str, timeout: float = 20.0) -> Any:
    """GET ``url`` and decode its JSON body (``None`` for an empty one)."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read() or b"null")


def _terminate(process: Optional[subprocess.Popen]) -> None:
    """Stop ``process`` politely, then by force after ten seconds; no-op if gone."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


# -- the worker ---------------------------------------------------------------------


class HomeBenchWorker:
    """An engine + HBTD pair, isolated from every other worker.

    Use as a context manager so both processes are always reaped (see the module
    docstring). Attributes the bench reads: ``client`` (a ``HomeBenchClient`` on
    the engine), ``shtd_base`` (HBTD's URL; the name matches ``SimuHomeWorker`` so
    ``Target`` needs no change), ``episode`` and ``episode_name`` of the loaded one.
    """

    def __init__(
        self,
        benchmark: Path = DEFAULT_BENCHMARK,
        poll_interval: float = 1.0,
        startup_timeout: float = 90.0,
        worker_id: int = 0,
        environment: str = DEFAULT_ENVIRONMENT,
    ) -> None:
        """Pick the two ports; start nothing (``start`` does).

        ``benchmark`` is the converter's output directory (``homes/``, ``episodes/``).
        ``environment`` (``on`` | ``off``) is passed to every HBTD this worker starts:
        the environment model, or the flat graph.
        """
        if environment not in ENVIRONMENT_MODES:
            raise ValueError(
                f"environment must be one of {ENVIRONMENT_MODES}, got {environment!r}")
        self.benchmark = Path(benchmark)
        self.environment = environment
        self.poll_interval = poll_interval
        self.startup_timeout = startup_timeout
        self.worker_id = worker_id

        # 1. Two free ports: no two workers can collide.
        self.sim_port = free_port()
        self.shtd_port = free_port()
        self.sim_base = f"http://{HOST}:{self.sim_port}/api"
        self.shtd_base = f"http://{HOST}:{self.shtd_port}"

        # 2. The engine's client, and the (not yet started) processes.
        self.client = HomeBenchClient(self.sim_base)
        self._sim: Optional[subprocess.Popen] = None
        self._shtd: Optional[subprocess.Popen] = None
        self.episode: Optional[Dict[str, Any]] = None
        self.episode_name: Optional[str] = None
        self.home_id: Optional[str] = None

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> "HomeBenchWorker":
        """Start the engine and wait until it answers; HBTD waits for the first ``load``."""
        # 1. The engine, empty: a home is loaded per episode.
        self._sim = subprocess.Popen(
            [sys.executable, str(HERE / "simulator.py"), "--host", HOST,
             "--port", str(self.sim_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Wait for it. HBTD's home id is set per episode, so it starts unbound
        #    and is started on the first load (starting it here would fix the wrong id).
        if not self._wait_for(self.client.health):
            self.stop()
            raise RuntimeError(f"engine did not start on {self.sim_port}")
        return self

    def _start_hbtd(self, home: str) -> None:
        """(Re)start HBTD bound to the engine and to the home id ``home``."""
        # 1. One HBTD at a time: the previous one (and its hub state) goes first.
        self._stop_hbtd()

        # 2. A fresh process on the same port, pointed at this worker's engine.
        self._shtd = subprocess.Popen(
            [sys.executable, str(HERE / "hbtd.py"),
             "--sim", self.sim_base, "--home", home,
             "--host", HOST, "--port", str(self.shtd_port),
             "--poll-interval", str(self.poll_interval),
             "--environment", self.environment],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Ready when its status route answers.
        def ready() -> bool:
            """True when HBTD's status route answers with a body."""
            return _get_json(f"{self.shtd_base}/_shtd/status", timeout=3) is not None

        if not self._wait_for(ready):
            self.stop()
            raise RuntimeError(f"HBTD did not start on {self.shtd_port}")

    def _wait_for(self, predicate: Callable[[], Any], interval: float = 0.5) -> bool:
        """Poll ``predicate`` until it is true or ``startup_timeout`` passes."""
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return True
            except Exception:  # noqa: BLE001 - not up yet: any failure means "try again"
                pass
            time.sleep(interval)
        return False

    def _stop_hbtd(self) -> None:
        """Stop HBTD (its subscriptions die with the process)."""
        _terminate(self._shtd)
        self._shtd = None

    def stop(self) -> None:
        """Stop both processes."""
        self._stop_hbtd()
        _terminate(self._sim)
        self._sim = None

    def __enter__(self) -> "HomeBenchWorker":
        """Start the worker (context-manager entry)."""
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        """Stop the worker (context-manager exit)."""
        self.stop()

    # -- episodes -------------------------------------------------------------------

    def _read_episode(self, name: str) -> Dict[str, Any]:
        """The episode ``name`` with its home merged in as ``initial_home_config``.

        The same two reads as ``tests/homebench/convert.py:load_episode``, repeated
        here so that code under ``ami_agents/`` does not import from ``tests/``.
        """
        path = self.benchmark / "episodes" / f"{name.removesuffix('.json')}.json"
        if not path.is_file():
            raise FileNotFoundError(f"episode not found: {path}")
        episode = json.loads(path.read_text(encoding="utf-8"))
        episode["initial_home_config"] = load_home_file(
            self.benchmark / "homes" / f"home{episode['home_id']}.json")
        return episode

    def load(self, episode_name: str) -> Dict[str, Any]:
        """Load one episode; ends the previous one cleanly first.

        The engine is reset to the episode's home and HBTD is restarted bound to
        ``home<id>``. Returns the episode (with ``initial_home_config``).
        """
        # 1. The previous episode ends first: subscriptions gone, HBTD stopped.
        if self.episode_name is not None:
            self.end_episode()

        # 2. The episode, and the engine reset to its home.
        episode = self._read_episode(episode_name)
        self.client.reset(dict(episode["initial_home_config"]))
        self.episode = episode
        self.episode_name = episode_name.removesuffix(".json")
        self.home_id = f"home{episode['home_id']}"

        # 3. HBTD, bound to this home: its id is the workspace segment of every IRI.
        self._start_hbtd(self.home_id)
        return episode

    def end_episode(self, engine: Optional[Any] = None) -> None:
        """Clean up before the next episode.

        Subscriptions must not survive an episode: a notification listener binds a
        fixed port, so the next episode would inherit them and be woken by devices it
        never subscribed to (observed on SimuHome, 22 stale subscriptions). Two layers,
        because the first can fail:

        1. If an ``engine`` (the agents' ``YggdrasilIntegration``) is given, ask it to
           unsubscribe properly: a real WebSub ``hub.mode=unsubscribe`` per artifact.
        2. Stop HBTD regardless. Subscriptions live in that process, so none survive
           even if the agent crashed mid-episode or never unsubscribed.
        """
        # 1. The polite layer.
        if engine is not None:
            try:
                import asyncio
                coro = engine.unsubscribe_all_artifacts()
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(coro)
                else:
                    loop.create_task(coro)
            except Exception as exc:  # noqa: BLE001 - cleanup must not fail a run
                LOGGER.warning("[worker %s] unsubscribe failed, restarting HBTD anyway: %s",
                               self.worker_id, exc)

        # 2. The sure layer.
        self._stop_hbtd()
        self.episode = None
        self.episode_name = None
        self.home_id = None

    def subscription_count(self) -> int:
        """How many subscriptions HBTD currently holds; 0 between episodes.

        A stopped HBTD answers nothing, which also reads as 0: to prove a *running*
        hub is empty, call this after ``load`` (a fresh process).
        """
        try:
            return int(_get_json(f"{self.shtd_base}/_shtd/subscriptions")["count"])
        except Exception:  # noqa: BLE001 - not serving: no subscriptions
            return 0

    # -- results --------------------------------------------------------------------

    def final_home_state(self) -> Dict[str, Any]:
        """The world as the episode left it: what the evaluators score."""
        return self.client.home_state()

    def evaluation_payload(self, tools_invoked: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The payload SimuHome's evaluators take, filled from this worker.

        Same keys as ``SimuHomeWorker.evaluation_payload``; HomeBench's feasible
        check is plain equality on the final state, so no baseline run is needed.
        """
        if self.episode is None:
            raise RuntimeError("no episode loaded")
        return {
            "episode": self.episode,
            "client": self.client,
            "final_home_state": self.final_home_state(),
            "tools_invoked": tools_invoked,
        }


def main() -> int:
    """Smoke test: two episodes in one worker; the engine follows each home."""
    import argparse

    parser = argparse.ArgumentParser(description="HomeBench worker smoke test.")
    parser.add_argument("--episodes", nargs="+",
                        default=["hb1_feasible_seed_1", "hb1_feasible_seed_2"])
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with HomeBenchWorker(benchmark=args.benchmark) as worker:
        for name in args.episodes:
            worker.load(name)
            state = worker.final_home_state()
            devices = sum(len(r["devices"]) for r in state["rooms"].values())
            LOGGER.info("%s: %s engine=%d hbtd=%d rooms=%d devices=%d subscriptions=%d",
                        name, worker.home_id, worker.sim_port, worker.shtd_port,
                        len(state["rooms"]), devices, worker.subscription_count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
