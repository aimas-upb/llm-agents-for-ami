#!/usr/bin/env python3
"""
One isolated SimuHome worker: a simulator, an SHTD, and the episode lifecycle.

**Isolation is the whole point.** Several workers run at once, and each must be
completely independent of the others -- including its CLOCK and its
environmental dynamics. That holds because `Home` is module-level state in
`src/simulator/api/routes.py`, so one process serves exactly one home: its tick
counter, simulated time, tick interval and room states all live in that process.

Verified rather than assumed: two simulators started side by side, given
different episodes and different tick intervals, reported 81 vs 9 ticks and
different room layouts after the same wall-clock interval.

    ports        allocated by binding :0, so two workers never collide
    simulator    one process, one home, one clock
    SHTD         one process, projecting THAT simulator only
    episodes     many per worker, `reset` between them

**Subscriptions are cleaned up between episodes.** They outlive the agent
process otherwise, and a notification listener binds a fixed port -- so a later
episode reusing that port would inherit the previous one's subscriptions and be
woken by devices it never subscribed to. Observed during phase-3 testing: 22
stale subscriptions from an earlier run delivering into a new one.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sim_client import SimuHomeClient, SimuHomeError  # noqa: E402

REPO = HERE.parents[3]
DEFAULT_SIMUHOME = REPO.parent / "SimuHome"
DEFAULT_BENCHMARK = DEFAULT_SIMUHOME / "data" / "benchmark"


class _ServerExited(Exception):
    """A server subprocess ended before it was ready."""
    def __init__(self, returncode: Optional[int]):
        super().__init__(returncode)
        self.returncode = returncode


def free_port() -> int:
    """Ask the OS for an unused port, as SimuHome's own harness does."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_json(url: str, timeout: float = 20.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read() or b"null")


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 20.0) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            try:
                return json.loads(body or b"null")
            except json.JSONDecodeError:
                return body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return None


class SimuHomeWorker:
    """A simulator + SHTD pair, isolated from every other worker.

    Use as a context manager so both processes are always reaped:

        with SimuHomeWorker() as worker:
            for episode in episodes:
                worker.load(episode)
                ...
                result = worker.finish_episode()
    """

    def __init__(
        self,
        simuhome: Path = DEFAULT_SIMUHOME,
        benchmark: Path = DEFAULT_BENCHMARK,
        tick_interval: Optional[float] = None,
        poll_interval: float = 1.0,
        startup_timeout: Optional[float] = None,
        worker_id: int = 0,
    ):
        self.simuhome = Path(simuhome)
        self.benchmark = Path(benchmark)
        self.tick_interval = tick_interval
        self.poll_interval = poll_interval
        # Default 90 s; SIMUHOME_STARTUP_TIMEOUT raises it on a loaded node,
        # where SHTD's TD build can take longer than that.
        self.startup_timeout = float(startup_timeout if startup_timeout is not None
                                     else os.environ.get("SIMUHOME_STARTUP_TIMEOUT", 90.0))
        self.worker_id = worker_id

        self.sim_port = free_port()
        self.shtd_port = free_port()
        self.sim_base = f"http://127.0.0.1:{self.sim_port}/api"
        self.shtd_base = f"http://127.0.0.1:{self.shtd_port}"

        self.client = SimuHomeClient(self.sim_base)
        self._sim: Optional[subprocess.Popen] = None
        self._shtd: Optional[subprocess.Popen] = None
        self.episode: Optional[Dict[str, Any]] = None
        self.episode_name: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "SimuHomeWorker":
        """Start the simulator. One failed start (the process exits, or is not
        ready in ``startup_timeout``) is retried once on fresh ports. Its output
        is kept in ``SHTD_LOG_DIR`` (default: the system temp dir), and its tail
        is part of the error: a start that fails silently cannot be diagnosed."""
        if not (self.simuhome / "src" / "simulator" / "api" / "app.py").is_file():
            raise FileNotFoundError(f"SimuHome not found at {self.simuhome}")
        last_error = ""
        for attempt in range(2):
            if attempt:
                self.stop()
                self._reassign_ports()
            env = os.environ.copy()
            env["SERVER_PORT"] = str(self.sim_port)
            log_path = self._log_dir() / f"sim_w{self.worker_id}_{self.sim_port}.log"
            with open(log_path, "w", encoding="utf-8") as log:
                # Through sim_launcher.py: the same app under uvicorn, bound to
                # the loopback interface (SimuHome's own entry point binds
                # 0.0.0.0), and a log line naming whoever terminates it (the
                # simulator is shut down by a signal every few minutes on the
                # shared dgxa100 nodes; see the launcher's docstring).
                self._sim = subprocess.Popen(
                    [sys.executable, str(HERE / "sim_launcher.py"), str(self.sim_port)],
                    cwd=str(self.simuhome), env=env,
                    stdout=log, stderr=subprocess.STDOUT)

            def ready() -> bool:
                if self._sim is not None and self._sim.poll() is not None:
                    raise _ServerExited(self._sim.returncode)
                return bool(self.client.health())

            try:
                if self._wait_for(ready):
                    # SHTD's home id is set per episode, so it starts unbound and
                    # is restarted on the first load. Starting it here would fix
                    # the wrong id.
                    return self
                last_error = f"not ready after {self.startup_timeout:.0f}s"
            except _ServerExited as exc:
                last_error = f"exited with code {exc.returncode}"
            last_error += f" on port {self.sim_port}" + self._log_tail(log_path)
        self.stop()
        raise RuntimeError(f"simulator did not start: {last_error}")

    def _reassign_ports(self) -> None:
        """Fresh ports for both servers: a lingering socket blocks the old one."""
        self.sim_port = free_port()
        self.shtd_port = free_port()
        self.sim_base = f"http://127.0.0.1:{self.sim_port}/api"
        self.shtd_base = f"http://127.0.0.1:{self.shtd_port}"
        self.client = SimuHomeClient(self.sim_base)

    @staticmethod
    def _log_dir() -> Path:
        log_dir = Path(os.environ.get("SHTD_LOG_DIR") or tempfile.gettempdir())
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    @staticmethod
    def _log_tail(log_path: Path, lines: int = 8) -> str:
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-lines:]
        except OSError:
            tail = []
        return ("; output tail:\n  " + "\n  ".join(tail)) if tail else ""

    def _start_shtd(self, home: str) -> None:
        """Start SHTD for ``home``. One failed start (the server exits, or is
        not ready in ``startup_timeout``) is retried once on a fresh port:
        the port is reused across episodes, so a lingering socket from the
        previous SHTD, or a slow start on a loaded node, must not end a
        long run. The server's stderr is kept in ``SHTD_LOG_DIR`` (default:
        the system temp dir), and its tail is part of the error."""
        self._stop_shtd()
        last_error = ""
        for attempt in range(2):
            if attempt:
                # A fresh simulator and fresh ports: SHTD blocks on a wedged
                # simulator, and a lingering socket blocks the old port.
                self.stop()
                self._reassign_ports()
                self.start()
            log_path = self._log_dir() / f"shtd_w{self.worker_id}_{self.shtd_port}.log"
            with open(log_path, "w", encoding="utf-8") as log:
                self._shtd = subprocess.Popen(
                    [sys.executable, str(HERE / "shtd.py"),
                     "--sim", self.sim_base, "--home", home,
                     "--host", "127.0.0.1", "--port", str(self.shtd_port),
                     "--poll-interval", str(self.poll_interval)],
                    stdout=log, stderr=subprocess.STDOUT)

            def ready() -> bool:
                if self._shtd is not None and self._shtd.poll() is not None:
                    raise _ServerExited(self._shtd.returncode)
                try:
                    return _get_json(f"{self.shtd_base}/_shtd/status", timeout=3) is not None
                except Exception:
                    return False

            try:
                if self._wait_for(ready):
                    return
                last_error = f"not ready after {self.startup_timeout:.0f}s"
            except _ServerExited as exc:
                last_error = f"exited with code {exc.returncode}"
            self._stop_shtd()
            last_error += f" on port {self.shtd_port} (home {home})" + self._log_tail(log_path)
        self.stop()
        raise RuntimeError(f"SHTD did not start: {last_error}")

    def _wait_for(self, predicate, interval: float = 0.5) -> bool:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return True
            except _ServerExited:
                raise
            except Exception:
                pass
            time.sleep(interval)
        return False

    def _stop_shtd(self) -> None:
        if self._shtd is not None and self._shtd.poll() is None:
            self._shtd.terminate()
            try:
                self._shtd.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._shtd.kill()
        self._shtd = None

    def stop(self) -> None:
        self._stop_shtd()
        if self._sim is not None and self._sim.poll() is None:
            self._sim.terminate()
            try:
                self._sim.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._sim.kill()
        self._sim = None

    def __enter__(self) -> "SimuHomeWorker":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- episodes ----------------------------------------------------------

    def load(self, episode_name: str) -> Dict[str, Any]:
        """Load one episode. Ends the previous one cleanly first."""
        if self.episode_name is not None:
            self.end_episode()

        name = episode_name if episode_name.endswith(".json") else f"{episode_name}.json"
        path = self.benchmark / name
        if not path.is_file():
            raise FileNotFoundError(f"episode not found: {path}")

        episode = json.loads(path.read_text(encoding="utf-8"))
        config = dict(episode["initial_home_config"])
        if self.tick_interval is not None:
            # Each worker may run its own clock rate; the simulator keeps it
            # per-process, so this does not affect any other worker.
            config["tick_interval"] = self.tick_interval

        self.client.reset(config)
        self.episode = episode
        self.episode_name = path.stem

        # SHTD is a stateless projection, but its home id is baked in at start,
        # and that id is the workspace segment every IRI is built from.
        self._start_shtd(path.stem)
        return episode

    def end_episode(self, engine: Optional[Any] = None) -> None:
        """Clean up before the next episode.

        Subscriptions must not survive an episode. A notification listener binds
        a fixed port, so the next episode would inherit them and be woken by
        devices it never subscribed to -- observed during phase-3 testing, where
        22 subscriptions from an earlier run kept delivering into a later one.

        Two layers, because the first can fail:

        1. If an `engine` is given, ask it to unsubscribe properly -- a real
           WebSub `hub.mode=unsubscribe` per artifact. This exercises the path an
           agent uses in production and leaves the hub's own registry consistent.
        2. Restart SHTD regardless. Subscriptions live in that process, so this
           guarantees zero survive even if the agent crashed mid-episode or
           never unsubscribed at all.
        """
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
                print(f"[worker {self.worker_id}] unsubscribe failed, "
                      f"restarting SHTD anyway: {exc}", flush=True)

        self._stop_shtd()
        self.episode = None
        self.episode_name = None

    def subscription_count(self) -> int:
        """How many subscriptions SHTD currently holds. 0 between episodes."""
        try:
            return int(_get_json(f"{self.shtd_base}/_shtd/subscriptions")["count"])
        except Exception:
            return 0

    # -- results -----------------------------------------------------------

    def final_home_state(self) -> Dict[str, Any]:
        """The world as the episode left it -- what the evaluator scores."""
        return self.client.home_state()

    def evaluation_payload(self, tools_invoked: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The payload SimuHome's own evaluator expects.

        Its scoring is COUNTERFACTUAL: it resets the simulator, fast-forwards a
        baseline to the same tick with no agent acting, and asks whether each
        goal's variable moved further than the baseline did on its own. So it
        needs the final state captured BEFORE it takes the simulator back, and
        it must have the simulator to itself while it runs.
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
    """Smoke test: run two workers side by side and show they are independent."""
    import argparse
    parser = argparse.ArgumentParser(description="SimuHome worker smoke test.")
    parser.add_argument("--episodes", nargs="+",
                        default=["qt2_feasible_seed_77", "qt2_feasible_seed_88"])
    parser.add_argument("--simuhome", type=Path, default=DEFAULT_SIMUHOME)
    args = parser.parse_args()

    workers = [SimuHomeWorker(simuhome=args.simuhome, worker_id=i)
               for i in range(len(args.episodes))]
    try:
        for worker in workers:
            worker.start()
        for worker, episode in zip(workers, args.episodes):
            worker.load(episode)
            print(f"worker {worker.worker_id}: {episode} "
                  f"sim={worker.sim_port} shtd={worker.shtd_port}")
        time.sleep(6)
        for worker in workers:
            state = worker.final_home_state()
            devices = sum(len(r["devices"]) for r in state["rooms"].values())
            print(f"  worker {worker.worker_id}: tick={state['current_tick']:5d} "
                  f"clock={state['current_time']} rooms={len(state['rooms'])} "
                  f"devices={devices}")
        ticks = [w.final_home_state()["current_tick"] for w in workers]
        print(f"\nindependent clocks: {len(set(ticks)) == len(ticks)}")
        return 0
    finally:
        for worker in workers:
            worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
