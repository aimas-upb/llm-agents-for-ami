"""The evaluation harness: run SimuHome episodes with scheduled execution, score
them, and compute the statistics -- from one config file.

    conda run -n ami-agents python -m tests.simuhome.bench.harness \\
        --config tests/simuhome/bench/variants/harness.example.yaml \\
        [--out tests/simuhome/results/harness] [--episodes qt3_*] [--limit 5] \\
        [--workers 4] [--no-judge] [--dry-run | --stats-only]

The config is a variant file (`models.py`: one model per phase, each on OpenAI
or a local OpenAI-compatible server through `provider:`), plus how to run it:

    name: nano-openai                  # records under <out>/<name>/
    providers: {local: {base_url: http://localhost:11434/v1}}
    models:
      segmentation: {provider: openai, model: gpt-5-nano, reasoning_effort: low}
      ...                              # structuring, planning.explicit/general,
                                       # profiling, guarding, response, judge
    run:                               # run_scheduled.py options
      episodes: "qt*"                  # patterns: qt3_feasible, qt4-2_*, full names
      limit: null                      # at most N episodes per (type, case)
      workers: 4
    evaluate: {judge: true}            # false: infeasible scored without the LLM judge

Flow, each step its own process (as `sweep.py` runs them):

    1. run         run_scheduled.py --variant <config> --skip-existing
                   -> <out>/<name>/<episode>.json  (an interrupted run resumes)
    2. evaluate    evaluate.py --skip-scored  -> <episode>.eval.json
    3. statistics  stats.py, in-process       -> <out>/<name>/stats/

Records are reused by name: give a changed config a new `name`, or its old
records are kept as they are.

`--dry-run` prints the commands and what they would cost; `--stats-only`
recomputes the statistics from the records already on disk (no LLM calls).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .run import DEFAULT_BENCHMARK, PROJECT_ROOT, select_episodes

DEFAULT_OUT = PROJECT_ROOT / "tests" / "simuhome" / "results" / "harness"
# The run_scheduled.py options a config's `run:` section may set.
RUN_OPTIONS = ("episodes", "seeds", "limit", "workers", "context", "memory", "step_seconds",
               "dry_step_seconds", "real_read_seconds", "exec_timeout")
# Rough LLM calls per episode for the agents' steps, and per judged episode.
CALLS_PER_EPISODE = (3, 6)
JUDGE_CALLS = 3


# --------------------------------------------------------------------------
# The plan of a harness run
# --------------------------------------------------------------------------

def run_options(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """The run options: the config's `run:` section, under the command line's
    own flags (a flag given wins)."""
    options = dict(config.get("run") or {})
    for key in ("episodes", "seeds", "limit", "workers"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    options.setdefault("episodes", "qt*")
    return options


def commands(config_path: Path, records: Path, out: Path, options: Dict[str, Any],
             judge: bool, benchmark: Path) -> List[List[str]]:
    """The run and evaluate command lines, in order."""
    # 1. The run, resuming: episodes already recorded are skipped.
    run = [sys.executable, "-m", "tests.simuhome.bench.run_scheduled",
           "--variant", str(config_path), "--out", str(out), "--benchmark", str(benchmark),
           "--skip-existing"]
    for key in RUN_OPTIONS:
        if options.get(key) is not None:
            run += [f"--{key.replace('_', '-')}", str(options[key])]

    # 2. The scoring, of the records not scored since they were written.
    evaluate = [sys.executable, "-m", "tests.simuhome.bench.evaluate",
                "--records", str(records), "--benchmark", str(benchmark), "--skip-scored"]
    if not judge:
        evaluate.append("--no-judge")
    return [run, evaluate]


def estimate(names: List[str], records: Path, judge: bool) -> str:
    """What the run would cost: the episodes still to run, and a rough count of
    LLM calls (the steps' and the judge's)."""
    todo = [n for n in names if not (records / f"{n}.json").exists()]
    infeasible = sum(1 for n in todo if "_infeasible_" in n)
    low, high = (len(todo) * c for c in CALLS_PER_EPISODE)
    judged = infeasible * JUDGE_CALLS if judge else 0
    return (f"{len(names)} episode(s) selected, {len(names) - len(todo)} already recorded, "
            f"{len(todo)} to run: ~{low}-{high} LLM calls for the steps"
            + (f" + ~{judged} judge calls ({infeasible} infeasible)" if judge else "")
            + " (none for steps on a local model)")


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

def main() -> int:
    """Run, score and summarise one config's episodes (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="harness config (a variant file)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--episodes", help="override run.episodes, e.g. qt3_*,qt4-1_*")
    parser.add_argument("--seeds", help="override run.seeds, e.g. 1-10,15")
    parser.add_argument("--limit", type=int, help="override run.limit (per type and case)")
    parser.add_argument("--workers", type=int, help="override run.workers")
    parser.add_argument("--no-judge", action="store_true",
                        help="score infeasible episodes without the LLM judge")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="print the commands and the estimate, run nothing")
    mode.add_argument("--stats-only", action="store_true",
                      help="only the statistics, from the records on disk")
    args = parser.parse_args()

    from .models import load_variant
    from .stats import compute

    # 1. The config, its variant (every phase's model resolved) and where its
    #    records go -- the directory run.py names after the variant.
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    variant = load_variant(str(config_path))
    out, benchmark = Path(args.out).resolve(), Path(args.benchmark)
    records = out / variant.name
    options = run_options(config, args)
    judge = bool((config.get("evaluate") or {}).get("judge", True)) and not args.no_judge
    title = f"SimuHome harness: {variant.name}"

    # 2. Statistics only: nothing to run.
    if args.stats_only:
        return 0 if compute(records, title) else 1

    # 3. What would run: the models per phase, the commands, the estimate.
    seeds = options.get("seeds")
    names = select_episodes(benchmark, str(options["episodes"]),
                            None if seeds is None else str(seeds), options.get("limit"))
    if not names:
        print("No episodes match.")
        return 1
    print(f"config {config_path.name} -> {records}")
    for step, spec in variant.specs.items():
        print(f"  {step:<18} {spec.label():<28} {'local ' + spec.base_url if spec.local else 'OpenAI'}")
    steps = commands(config_path, records, out, options, judge, benchmark)
    for command in steps:
        print("  $ " + " ".join(command[1:]))
    print(estimate(names, records, judge))
    if args.dry_run:
        return 0

    # 4. Run, then score; a failed step still lets the next one see what was
    #    written (a partial run is scored and summarised as far as it got).
    for command in steps:
        print(f"\n=== {command[2]}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    # 5. The statistics over every record of this config.
    print(f"\n=== statistics", flush=True)
    return 0 if compute(records, title) else 1


if __name__ == "__main__":
    sys.exit(main())
