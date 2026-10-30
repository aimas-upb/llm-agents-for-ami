"""Run SimuHome or HomeBench episodes through our agents' reasoning, one record per episode.

    conda run -n ami-agents python -m tests.simuhome.bench.run \\
        --variant tests/simuhome/bench/variants/default.yaml \\
        --episodes qt3_feasible,qt4-2_* --limit 5 --workers 2 --out tests/simuhome/results/bench

One episode, for debugging, printing every step (and scoring it):

    conda run -n ami-agents python -m tests.simuhome.bench.run \\
        --episode qt4-2_feasible_seed_1 --debug [--stop-after planning] \\
        [--sim http://127.0.0.1:14000/api --shtd http://localhost:8097]

The benchmark is `--suite simuhome` (the default) or `--suite homebench`
(`suites/`): the episodes, the worker, the TD builder, the scoring and the
planning conditions come from the suite. HomeBench episodes are `hb<N>_<case>_seed_<n>`
(N parts; case feasible / infeasible / mixed), converted by `tests/homebench/convert.py`:

    conda run -n ami-agents python -m tests.simuhome.bench.run --suite homebench \\
        --episode hb1_feasible_seed_1 --variant <variant.yaml> --stop-after planning

Each worker is a process with its own simulator and SHTD (started and stopped
by the run, on free ports), so its clock and home are its own and the py_trees
blackboard is never shared. `--sim/--shtd` use servers you already run instead
(one worker; SHTD keeps the home id it was started with). For HomeBench the
simulator is the engine and SHTD is HBTD (`ami_agents/.../HomeBench/worker.py`),
and a worker's episodes are kept together by home.

Output: <out>/<variant>/<episode>.json per episode, and run.json. Score them
with `evaluate.py`; summarise with `report.py`. The run itself ends with a
table of the time spent per processing stage (segmentation, structuring,
planning, profiling, guarding, execution), summed over its episodes
(`meter.Meter.stages`).

Costs: about 3-6 LLM calls per episode for the steps (segmentation, structuring
per intent, planning when an LLM is needed, the reply), none with local models.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import logging
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

from .suites import (  # noqa: E402
    DEFAULT_ENVIRONMENT, DEFAULT_SUITE, ENVIRONMENT_MODES, SUITE_NAMES, Suite, get_suite,
)

AGENTS_YAML = PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml"


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


def _group_and_seed(name: str) -> Tuple[str, int, str]:
    """``(group, seed, suffix)`` of an episode name: ``hb1_feasible_seed_2631_fixed``
    is ``("hb1_feasible", 2631, "_fixed")`` -- a corrected HomeBench episode keeps its
    seed number, with a suffix after it."""
    group, rest = name.rsplit("_seed_", 1)
    digits = re.match(r"\d+", rest)
    return group, int(digits.group()), rest[digits.end():]


def select_episodes(benchmark: Path, patterns: str, seeds: Optional[str],
                    limit: Optional[int], suite: Optional[Suite] = None) -> List[str]:
    """Episode names of ``suite`` (SimuHome when None) matching any pattern
    ('qt3_feasible', 'qt4-2_*', 'hb2_*', a full name), filtered by seed, at most
    ``limit`` per (type, case)."""
    names = (suite or get_suite(DEFAULT_SUITE)).names(benchmark)
    wanted = _seeds(seeds)
    chosen, per_group = [], {}
    for name in sorted(names, key=_group_and_seed):
        group, seed, _ = _group_and_seed(name)
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


def group_by_home(names: List[str], suite: Suite, benchmark: Path) -> List[str]:
    """``names`` with the episodes of one home next to each other (HomeBench's
    hundred homes repeat across its episodes); the order within a home, and of
    the homes by first appearance, is kept. A suite without homes to group by
    (SimuHome: one home per episode) gets ``names`` back unchanged."""
    if suite.home_key is None:
        return list(names)
    by_home: Dict[str, List[str]] = {}
    for name in names:
        by_home.setdefault(suite.home_key(benchmark, name), []).append(name)
    return [name for group in by_home.values() for name in group]


def split_for_workers(names: List[str], workers: int, suite: Suite,
                      benchmark: Path) -> List[List[str]]:
    """One list of episodes per worker. SimuHome's are dealt out round-robin, as
    ever. HomeBench's are dealt out by home -- a home's episodes all go to one
    worker, the largest homes first, each to the worker with the fewest episodes --
    and a worker left without episodes is not started."""
    # 1. No homes to keep together: round-robin.
    if suite.home_key is None:
        return [names[i::workers] for i in range(workers)]
    # 2. Whole homes, largest first, to the lightest worker.
    by_home: Dict[str, List[str]] = {}
    for name in names:
        by_home.setdefault(suite.home_key(benchmark, name), []).append(name)
    chunks: List[List[str]] = [[] for _ in range(workers)]
    for home in sorted(by_home, key=lambda h: (-len(by_home[h]), h)):
        min(chunks, key=len).extend(by_home[home])
    return [chunk for chunk in chunks if chunk]


# --------------------------------------------------------------------------
# The planning conditions
# --------------------------------------------------------------------------

def apply_guidelines(config: Dict[str, Any], path: Optional[Path]) -> None:
    """Point the ISA's planning conditions at the guidelines file ``path`` (None:
    leave the agents' own: the repo file, or `PLANNING_GUIDELINES_PATH`).

    ``config`` is agents.yaml as the bench loaded it; `isa_agent` hands its
    `interaction_solver` section to the planning behaviours, and the power-on guard
    and the condition guard read `planning.guidelines.path` from it
    (`power_on_guard.guidelines_path`), so this is the one place the choice has to land.
    """
    if path is None:
        return
    planning = config.setdefault("interaction_solver", {}).setdefault("planning", {})
    planning.setdefault("guidelines", {})["path"] = str(path)


# --------------------------------------------------------------------------
# The stage summary, printed at the end of a run
# --------------------------------------------------------------------------

def stage_summary(paths: List[str]) -> str:
    """The time per processing stage over the records at ``paths``: total and
    mean wall seconds, share of the pipeline's time, LLM calls, their seconds
    and tokens. Read back from the records, so workers in other processes
    count too; a record without stages (an older one) is skipped."""
    from .meter import STAGES

    # 1. Each record's stages, summed.
    totals = {name: {"seconds": 0.0, "calls": 0, "llm_seconds": 0.0, "tokens_in": 0,
                     "tokens_out": 0} for name in (*STAGES, "other")}
    episodes = 0
    for path in paths:
        try:
            stages = json.loads(Path(path).read_text(encoding="utf-8"))["meter"]["stages"]
        except (OSError, KeyError, TypeError, ValueError):
            continue
        episodes += 1
        for name, entry in stages.items():
            for field, value in entry.items():
                totals.setdefault(name, dict.fromkeys(entry, 0))[field] += value
    if not episodes:
        return "no stage timings recorded"

    # 2. The table: one row per stage that ran, then the total.
    pipeline = sum(t["seconds"] for t in totals.values()) or 1.0
    lines = [f"== time per stage ({episodes} episode(s)) ==",
             f"{'stage':<13}{'total s':>9}{'mean s':>9}{'share':>7}{'LLM calls':>11}"
             f"{'LLM s':>8}{'tokens in/out':>17}"]
    for name, t in totals.items():
        if not t["seconds"] and not t["calls"]:
            continue
        lines.append(f"{name:<13}{t['seconds']:>9.1f}{t['seconds'] / episodes:>9.2f}"
                     f"{100 * t['seconds'] / pipeline:>6.0f}%{t['calls']:>11}"
                     f"{t['llm_seconds']:>8.1f}{t['tokens_in']:>9}/{t['tokens_out']:<7}")
    calls = sum(t["calls"] for t in totals.values())
    lines.append(f"{'total':<13}{pipeline:>9.1f}{pipeline / episodes:>9.2f}{'':>7}{calls:>11}")
    return "\n".join(lines)


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
    """Run ``names`` on one simulator (SimuHome's, or HomeBench's engine); the
    record paths written."""
    from ami_agents.shared.utils.config_loader import ConfigLoader

    from .models import load_variant
    from .pipeline import Options, Target, run_episode

    # 1. The suite, and the agents' config with the suite's planning conditions.
    suite = get_suite(settings.get("suite"))
    logger = _logger(settings["debug"])
    config = ConfigLoader.load_with_env_vars(str(AGENTS_YAML))
    apply_guidelines(config, suite.guidelines_path(settings.get("guidelines")))
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
                      memory=memory, cache=cache, debug=settings["debug"],
                      execution=settings.get("execution", "ticks"),
                      dry_step_seconds=float(settings.get("dry_step_seconds", 1.0)),
                      real_read_seconds=float(settings.get("real_read_seconds", 10.0)),
                      environment=settings.get("environment") if suite.environment else None)

    # 2. The simulator: one the run starts (a worker), or one already running.
    written = []
    worker = None
    try:
        if settings.get("sim"):
            sim = suite.client_cls(settings["sim"])
            status = json.loads(urllib.request.urlopen(
                f"{settings['shtd'].rstrip('/')}/_shtd/status", timeout=10).read())
            fixed = (settings["shtd"].rstrip("/"), status["home"])
        else:
            # The environment mode goes only to a worker whose suite has the option
            # (SimuHomeWorker has no such argument).
            extra = {"environment": settings["environment"]} if suite.environment else {}
            worker = suite.worker_cls(benchmark=benchmark, worker_id=worker_id,
                                      **extra).start()
            sim = worker.client
            fixed = None

        # 3. Each episode: loaded by the suite, served by the worker, run, recorded.
        for count, name in enumerate(names, 1):
            path = out / f"{name}.json"
            if settings["skip_existing"] and path.exists():
                continue
            episode = suite.load_episode(benchmark, name)
            if worker is not None:
                worker.load(name)               # TD server bound to this episode's home id
                # SimuHome's home id is the episode's name; HomeBench's is home<id>.
                target = Target(sim, worker.shtd_base, getattr(worker, "home_id", None) or name)
            else:
                target = Target(sim, *fixed)
            started = time.perf_counter()
            record = asyncio.run(run_episode(name, episode, variant, config, target,
                                             options, logger, suite))
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


def main(execution: str = "ticks", out: Optional[Path] = None) -> int:
    """The bench CLI. ``execution`` and ``out`` are the defaults of
    `--execution` and `--out`: `run_scheduled.py` calls this with
    "scheduled" and its own results directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=SUITE_NAMES, default=DEFAULT_SUITE,
                        help="the benchmark: simuhome (default) or homebench")
    parser.add_argument("--variant", help="variant YAML (default: the agents' models)")
    parser.add_argument("--episodes",
                        help="comma-separated patterns: qt3_feasible, qt4-2_*, hb2_*, full "
                             "names (default: the suite's all: qt* / hb*)")
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
    parser.add_argument("--execution", choices=("ticks", "scheduled"), default=execution,
                        help="ticks: the trees ticked against the simulator (immediate / "
                             "lockstep); scheduled: a dry run of the trees submitted as "
                             "SimuHome workflows (scheduling.py)")
    parser.add_argument("--dry-step-seconds", type=float, default=1.0,
                        help="scheduled: the dry run's step, virtual seconds")
    parser.add_argument("--real-read-seconds", type=float, default=10.0,
                        help="scheduled: how long a value read from the simulator is reused")
    parser.add_argument("--exec-timeout", type=float, default=60.0,
                        help="immediate plans: real seconds before they count as failed")
    parser.add_argument("--stop-after", choices=("segmentation", "structuring", "planning"))
    parser.add_argument("--debug", action="store_true", help="print every step")
    parser.add_argument("--sim", help="an already running simulator API, e.g. "
                                      "http://127.0.0.1:14000/api")
    parser.add_argument("--shtd", help="its SHTD, e.g. http://localhost:8097")
    parser.add_argument("--guidelines",
                        help="the planning conditions: homebench (HomeBench's default: none) "
                             "or default (the agents' own file, as deployed)")
    parser.add_argument("--environment", choices=ENVIRONMENT_MODES,
                        help="HomeBench: on (default) serves the room environments and "
                             "action effects, off the flat graph; recorded in run.json. "
                             "With --sim/--shtd the server you started decides what is "
                             "served, this only sets the TDs the bench builds itself")
    parser.add_argument("--out", help="records directory (default: <suite results>/bench)")
    parser.add_argument("--benchmark", help="the benchmark directory (default: the suite's)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cache", help="reuse segmentation + structuring results kept in "
                                        "this directory (sweeps share one)")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="with --episode: do not score it")
    args = parser.parse_args()
    if bool(args.sim) != bool(args.shtd):
        parser.error("--sim and --shtd go together")

    # 1. The suite, and what it offers: scheduled execution, the guidelines choices.
    suite = get_suite(args.suite)
    if args.execution == "scheduled" and not suite.scheduled:
        parser.error(f"--execution scheduled is SimuHome's: the {suite.title} suite has "
                     "no timed episodes, so the trees are ticked (--execution ticks)")
    try:
        suite.guidelines_path(args.guidelines)
    except ValueError as exc:
        parser.error(str(exc))
    if args.environment and not suite.environment:
        parser.error(f"--environment is for a suite with room environments: the "
                     f"{suite.title} suite has none to switch")
    args.out = args.out or str(out or suite.results_root / "bench")

    # 2. The episodes: one, or the suite's selection, kept together by home.
    benchmark = Path(args.benchmark) if args.benchmark else suite.default_benchmark
    names = [args.episode.removesuffix(".json")] if args.episode else \
        group_by_home(select_episodes(benchmark, args.episodes or suite.default_episodes,
                                      args.seeds, args.limit, suite), suite, benchmark)
    if not names:
        print("No episodes match.")
        return 1
    settings = {"suite": suite.name, "guidelines": args.guidelines,
                "variant": args.variant, "out": args.out, "benchmark": str(benchmark),
                "memory": args.memory, "context": args.context,
                "step_seconds": args.step_seconds, "exec_timeout": args.exec_timeout,
                "stop_after": args.stop_after, "debug": args.debug or bool(args.episode),
                "sim": args.sim, "shtd": args.shtd, "skip_existing": args.skip_existing,
                "cache": args.cache, "execution": args.execution,
                "dry_step_seconds": args.dry_step_seconds,
                "real_read_seconds": args.real_read_seconds}
    # The mode is recorded (run.json) only for a suite that has it.
    if suite.environment:
        settings["environment"] = args.environment or DEFAULT_ENVIRONMENT

    from .models import load_variant
    variant = load_variant(args.variant)
    out = Path(args.out) / variant.name
    out.mkdir(parents=True, exist_ok=True)
    workers = 1 if (args.episode or args.sim) else max(1, args.workers)
    (out / "run.json").write_text(json.dumps({
        "variant": variant.as_dict(), "episodes": names, "settings": settings,
        "workers": workers, "commit": _git_commit(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    print(f"{len(names)} episode(s), variant {variant.name!r}, {workers} worker(s), "
          f"execution {args.execution} -> {out}")

    started = time.perf_counter()
    if workers == 1:
        written = run_worker(0, names, settings)
    else:
        chunks = split_for_workers(names, workers, suite, benchmark)
        written = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            for paths in pool.map(run_worker, range(len(chunks)), chunks,
                                  [settings] * len(chunks)):
                written += paths
    print(f"done: {len(written)} record(s) in {time.perf_counter() - started:.0f}s")

    if args.episode and written and not args.no_evaluate and not args.stop_after:
        # Scored at once, in its own process (SimuHome's evaluator imports as `src`).
        command = [sys.executable, "-m", "tests.simuhome.bench.evaluate",
                   "--records", written[0], "--explain"]
        if suite.name != DEFAULT_SUITE:
            command += ["--suite", suite.name]
        if args.benchmark:
            command += ["--benchmark", args.benchmark]
        subprocess.run(command, cwd=PROJECT_ROOT)

    # The time per stage, last, so it ends the output.
    print("\n" + stage_summary(written), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
