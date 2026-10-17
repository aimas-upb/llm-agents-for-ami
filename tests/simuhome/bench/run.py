"""Run SimuHome episodes through our agents' reasoning, one record per episode.

    conda run -n ami-agents python -m tests.simuhome.bench.run \\
        --variant tests/simuhome/bench/variants/default.yaml \\
        --episodes qt3_feasible,qt4-2_* --limit 5 --workers 2 --out tests/simuhome/results/bench

One episode, for debugging, printing every step (and scoring it):

    conda run -n ami-agents python -m tests.simuhome.bench.run \\
        --episode qt4-2_feasible_seed_1 --debug [--stop-after planning] \\
        [--sim http://127.0.0.1:14000/api --shtd http://localhost:8097]

Each worker is a process with its own simulator and SHTD (started and stopped
by the run, on free ports), so its clock and home are its own and the py_trees
blackboard is never shared. `--sim/--shtd` use servers you already run instead
(one worker; SHTD keeps the home id it was started with).

Output: <out>/<variant>/<episode>.json per episode, and run.json. Score them
with `evaluate.py`; summarise with `report.py`.

Costs: about 3-6 LLM calls per episode for the steps (segmentation, structuring
per intent, planning when an LLM is needed, the reply), none with local models.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import logging
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_BENCHMARK = PROJECT_ROOT.parent / "SimuHome" / "data" / "benchmark"
AGENTS_YAML = PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml"
DEFAULT_OUT = PROJECT_ROOT / "tests" / "simuhome" / "results" / "bench"


# --------------------------------------------------------------------------
# Which episodes
# --------------------------------------------------------------------------

def _seeds(spec: Optional[str]) -> Optional[set]:
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def select_episodes(benchmark: Path, patterns: str, seeds: Optional[str],
                    limit: Optional[int]) -> List[str]:
    """Episode names matching any pattern ('qt3_feasible', 'qt4-2_*', a full
    name), filtered by seed, at most ``limit`` per (type, case)."""
    names = sorted(p.stem for p in benchmark.glob("qt*_seed_*.json"))
    wanted = _seeds(seeds)
    chosen, per_group = [], {}
    for name in sorted(names, key=lambda n: (n.rsplit("_seed_", 1)[0],
                                             int(n.rsplit("_seed_", 1)[1]))):
        group, seed = name.rsplit("_seed_", 1)
        if not any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(group, p) or name == p
                   for p in [p.strip() for p in patterns.split(",") if p.strip()]):
            continue
        if wanted is not None and int(seed) not in wanted:
            continue
        if limit is not None and per_group.get(group, 0) >= limit:
            continue
        per_group[group] = per_group.get(group, 0) + 1
        chosen.append(name)
    return chosen


# --------------------------------------------------------------------------
# One worker: a simulator + SHTD, episodes in turn
# --------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _logger(debug: bool):
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING,
                        format="%(levelname)s %(name)s %(message)s")
    if not debug:
        for name in ("httpx", "openai", "py_trees"):
            logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("bench")


def run_worker(worker_id: int, names: List[str], settings: Dict[str, Any]) -> List[str]:
    """Run ``names`` on one simulator; the record paths written."""
    from ami_agents.environment.integration.SimuHome.sim_client import SimuHomeClient
    from ami_agents.environment.integration.SimuHome.worker import SimuHomeWorker
    from ami_agents.shared.utils.config_loader import ConfigLoader

    from .models import load_variant
    from .pipeline import Options, Target, run_episode

    logger = _logger(settings["debug"])
    config = ConfigLoader.load_with_env_vars(str(AGENTS_YAML))
    variant = load_variant(settings["variant"])
    out = Path(settings["out"]) / variant.name
    out.mkdir(parents=True, exist_ok=True)
    benchmark = Path(settings["benchmark"])

    memory = None
    if settings["memory"] == "run":
        from .memory import Memory
        memory = Memory(out / f"memory_worker{worker_id}")

    cache = None
    if settings.get("cache"):
        from .pipeline import StepCache
        cache = StepCache(settings["cache"])

    options = Options(context=settings["context"], step_seconds=settings["step_seconds"],
                      exec_timeout=settings["exec_timeout"], stop_after=settings["stop_after"],
                      memory=memory, cache=cache, debug=settings["debug"])

    written = []
    worker = None
    try:
        if settings.get("sim"):
            sim = SimuHomeClient(settings["sim"])
            status = json.loads(urllib.request.urlopen(
                f"{settings['shtd'].rstrip('/')}/_shtd/status", timeout=10).read())
            fixed = (settings["shtd"].rstrip("/"), status["home"])
        else:
            worker = SimuHomeWorker(benchmark=benchmark, worker_id=worker_id).start()
            sim = worker.client
            fixed = None

        for count, name in enumerate(names, 1):
            path = out / f"{name}.json"
            if settings["skip_existing"] and path.exists():
                continue
            episode = json.loads((benchmark / f"{name}.json").read_text(encoding="utf-8"))
            if worker is not None:
                worker.load(name)               # SHTD bound to this episode's id
                target = Target(sim, worker.shtd_base, name)
            else:
                target = Target(sim, *fixed)
            started = time.perf_counter()
            record = asyncio.run(run_episode(name, episode, variant, config, target,
                                             options, logger))
            record["wall_seconds"] = time.perf_counter() - started
            path.write_text(json.dumps(record, default=str), encoding="utf-8")
            written.append(str(path))
            print(f"[w{worker_id} {count}/{len(names)}] {name}: {record['outcome']}"
                  + (f" ({record['error_step']}: {record['error']})" if record["error"]
                     else "") + f"  {record['wall_seconds']:.1f}s", flush=True)
    finally:
        if worker is not None:
            worker.stop()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", help="variant YAML (default: the agents' models)")
    parser.add_argument("--episodes", default="qt*",
                        help="comma-separated patterns: qt3_feasible, qt4-2_*, full names")
    parser.add_argument("--episode", help="one episode, in the foreground (debugging)")
    parser.add_argument("--seeds", help="e.g. 1-10,15")
    parser.add_argument("--limit", type=int, help="at most N episodes per (type, case)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory", choices=("off", "run"), default="off")
    parser.add_argument("--context", choices=("none", "prepend"), default="none",
                        help="prepend: '(It is <base_time>. I am in the <room>.)' "
                             "before the query (test-only)")
    parser.add_argument("--step-seconds", type=float, default=10.0,
                        help="lockstep step, virtual seconds")
    parser.add_argument("--exec-timeout", type=float, default=60.0,
                        help="immediate plans: real seconds before they count as failed")
    parser.add_argument("--stop-after", choices=("segmentation", "structuring", "planning"))
    parser.add_argument("--debug", action="store_true", help="print every step")
    parser.add_argument("--sim", help="an already running simulator API, e.g. "
                                      "http://127.0.0.1:14000/api")
    parser.add_argument("--shtd", help="its SHTD, e.g. http://localhost:8097")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cache", help="reuse segmentation + structuring results kept in "
                                        "this directory (sweeps share one)")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="with --episode: do not score it")
    args = parser.parse_args()
    if bool(args.sim) != bool(args.shtd):
        parser.error("--sim and --shtd go together")

    benchmark = Path(args.benchmark)
    names = [args.episode.removesuffix(".json")] if args.episode else \
        select_episodes(benchmark, args.episodes, args.seeds, args.limit)
    if not names:
        print("No episodes match.")
        return 1
    settings = {"variant": args.variant, "out": args.out, "benchmark": str(benchmark),
                "memory": args.memory, "context": args.context,
                "step_seconds": args.step_seconds, "exec_timeout": args.exec_timeout,
                "stop_after": args.stop_after, "debug": args.debug or bool(args.episode),
                "sim": args.sim, "shtd": args.shtd, "skip_existing": args.skip_existing,
                "cache": args.cache}

    from .models import load_variant
    variant = load_variant(args.variant)
    out = Path(args.out) / variant.name
    out.mkdir(parents=True, exist_ok=True)
    workers = 1 if (args.episode or args.sim) else max(1, args.workers)
    (out / "run.json").write_text(json.dumps({
        "variant": variant.as_dict(), "episodes": names, "settings": settings,
        "workers": workers, "commit": _git_commit(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    print(f"{len(names)} episode(s), variant {variant.name!r}, {workers} worker(s) -> {out}")

    started = time.perf_counter()
    if workers == 1:
        written = run_worker(0, names, settings)
    else:
        chunks = [names[i::workers] for i in range(workers)]
        written = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for paths in pool.map(run_worker, range(workers), chunks,
                                  [settings] * workers):
                written += paths
    print(f"done: {len(written)} record(s) in {time.perf_counter() - started:.0f}s")

    if args.episode and written and not args.no_evaluate and not args.stop_after:
        # Scored at once, in its own process (SimuHome's evaluator imports as `src`).
        subprocess.run([sys.executable, "-m", "tests.simuhome.bench.evaluate",
                        "--records", written[0], "--explain"], cwd=PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
