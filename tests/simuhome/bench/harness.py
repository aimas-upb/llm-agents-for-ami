"""The evaluation harness: run SimuHome or HomeBench episodes, score them, and
compute the statistics -- from one config file.

    conda run -n ami-agents python -m tests.simuhome.bench.harness \\
        --config tests/simuhome/bench/variants/harness.example.yaml \\
        [--out tests/simuhome/results/harness] [--episodes qt3_*] [--limit 5] \\
        [--workers 4] [--no-judge] [--dry-run | --stats-only]

The config is a variant file (`models.py`: one model per phase, each on OpenAI
or a local OpenAI-compatible server through `provider:`), plus how to run it:

    suite: simuhome                    # simuhome (default) | homebench
    name: nano-openai                  # records under <out>/<name>/
    providers: {local: {base_url: http://localhost:11434/v1}}
    models:
      segmentation: {provider: openai, model: gpt-5-nano, reasoning_effort: low}
      ...                              # structuring, planning.explicit/general,
                                       # profiling, guarding, response, judge
    run:                               # run_scheduled.py options
      episodes: "qt*"                  # patterns: qt3_feasible, qt4-2_*, full names
      limit: null                      # at most N episodes per (type, case)
      guidelines: homebench            # homebench only: homebench | default (below)
      environment: on                  # homebench only: on | off (room environments, effects)
      workers: 4
    evaluate: {judge: true}            # false: infeasible scored without the LLM judge
                                       # (SimuHome only: HomeBench has no judge)

Flow, each step its own process (as `sweep.py` runs them):

    1. run         run_scheduled.py (SimuHome) / run.py --suite homebench (HomeBench:
                   no timed episodes, so the trees are ticked, never scheduled)
                   --variant <config> --skip-existing
                   -> <out>/<name>/<episode>.json  (an interrupted run resumes)
    2. evaluate    evaluate.py --skip-scored  -> <episode>.eval.json
    3. statistics  stats.py, in-process       -> <out>/<name>/stats/

Records are reused by name: give a changed config a new `name`, or its old
records are kept as they are.

`--dry-run` prints the commands and what they would cost (the episodes
selected, the LLM calls and the judge calls); `--stats-only` recomputes the
statistics from the records already on disk (no LLM calls).

HomeBench (`suite: homebench`, configs in `tests/homebench/variants/`; the data from
`tests/homebench/convert.py`): `run.guidelines` picks the ISA's planning conditions,
`homebench` (none: the Matter rules do not apply to HomeBench) or `default` (as
deployed). Example, nothing run:

    conda run -n ami-agents python -m tests.simuhome.bench.harness \\
        --config tests/homebench/variants/homebench.quick.yaml --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .run import PROJECT_ROOT, select_episodes
from .suites import DEFAULT_SUITE, ENVIRONMENT_MODES, SUITE_NAMES, Suite, get_suite

# The run.py options a config's `run:` section may set.
RUN_OPTIONS = ("episodes", "seeds", "limit", "workers", "context", "memory", "step_seconds",
               "dry_step_seconds", "real_read_seconds", "exec_timeout", "guidelines",
               "environment")


# --------------------------------------------------------------------------
# The plan of a harness run
# --------------------------------------------------------------------------

def run_options(config: Dict[str, Any], args: argparse.Namespace,
                suite: Optional[Suite] = None) -> Dict[str, Any]:
    """The run options: the config's `run:` section, under the command line's
    own flags (a flag given wins); every episode of the suite when none is named."""
    options = dict(config.get("run") or {})
    for key in ("episodes", "seeds", "limit", "workers"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    options.setdefault("episodes", (suite or get_suite(DEFAULT_SUITE)).default_episodes)
    return options


def check_environment(options: Dict[str, Any], suite: Suite) -> None:
    """Raise ValueError for an ``environment`` option the suite lacks or a value it
    does not offer, so a typo never runs with the other mode (``off`` read as ``on``)."""
    mode = options.get("environment")
    if mode is None:
        return
    if not suite.environment:
        raise ValueError(f"the {suite.title} suite has no environment option")
    if str(mode) not in ENVIRONMENT_MODES:
        # YAML reads a bare on / off as a boolean, hence the hint.
        raise ValueError(f"environment must be one of {', '.join(ENVIRONMENT_MODES)}, "
                         f"got {mode!r} (quote it in YAML: environment: 'off')")


def commands(config_path: Path, records: Path, out: Path, options: Dict[str, Any],
             judge: bool, benchmark: Path,
             suite: Optional[Suite] = None) -> List[List[str]]:
    """The run and evaluate command lines, in order."""
    suite = suite or get_suite(DEFAULT_SUITE)
    # 1. The run, resuming: episodes already recorded are skipped. SimuHome's is
    #    the scheduled runner; a suite without scheduled execution (HomeBench)
    #    runs the tick-based one, with its suite named.
    module = "run_scheduled" if suite.scheduled else "run"
    run = [sys.executable, "-m", f"tests.simuhome.bench.{module}",
           "--variant", str(config_path), "--out", str(out), "--benchmark", str(benchmark),
           "--skip-existing"]
    if suite.name != DEFAULT_SUITE:
        run += ["--suite", suite.name]
    for key in RUN_OPTIONS:
        if options.get(key) is not None:
            run += [f"--{key.replace('_', '-')}", str(options[key])]

    # 2. The scoring, of the records not scored since they were written.
    evaluate = [sys.executable, "-m", "tests.simuhome.bench.evaluate",
                "--records", str(records), "--benchmark", str(benchmark), "--skip-scored"]
    if suite.name != DEFAULT_SUITE:
        evaluate += ["--suite", suite.name]
    if not judge and suite.judge:
        evaluate.append("--no-judge")
    return [run, evaluate]


def estimate(names: List[str], records: Path, judge: bool,
             suite: Optional[Suite] = None) -> str:
    """What the run would cost: the episodes still to run, and a count of LLM
    calls (the steps' and the judge's), from the suite's per-episode estimate:
    a range for SimuHome, exact for HomeBench (one segmentation call and one
    structuring call per part). A suite without a judge (HomeBench) shows none."""
    suite = suite or get_suite(DEFAULT_SUITE)
    judge = judge and suite.judge
    # 1. What is still to run, and its calls by the suite's estimate.
    todo = [n for n in names if not (records / f"{n}.json").exists()]
    estimates = [suite.estimate(n) for n in todo]
    low, high = sum(e[0] for e in estimates), sum(e[1] for e in estimates)
    judged_episodes = sum(1 for e in estimates if e[2])
    judged = sum(e[2] for e in estimates) if judge else 0
    calls = f"{low}-{high}" if low != high or not todo else f"{low}"
    # 2. The line.
    return (f"{len(names)} episode(s) selected, {len(names) - len(todo)} already recorded, "
            f"{len(todo)} to run: ~{calls} LLM calls for the steps"
            + (f" + ~{judged} judge calls ({judged_episodes} {suite.judged_label})"
               if judge else "")
            + " (none for steps on a local model)")


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

def main() -> int:
    """Run, score and summarise one config's episodes (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="harness config (a variant file)")
    parser.add_argument("--suite", choices=SUITE_NAMES,
                        help="the benchmark (default: the config's `suite:`, else simuhome)")
    parser.add_argument("--out", help="results root (default: <suite results>/harness)")
    parser.add_argument("--benchmark", help="the benchmark directory (default: the suite's)")
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
    try:
        suite = get_suite(args.suite or config.get("suite"))
        options = run_options(config, args, suite)
        suite.guidelines_path(options.get("guidelines"))
        check_environment(options, suite)
    except ValueError as exc:
        print(f"config {config_path.name}: {exc}")
        return 2
    out = Path(args.out).resolve() if args.out else suite.results_root / "harness"
    benchmark = Path(args.benchmark) if args.benchmark else suite.default_benchmark
    records = out / variant.name
    # A suite without a judge (HomeBench) never has one, whatever the config says.
    judge = bool((config.get("evaluate") or {}).get("judge", True)) and not args.no_judge \
        and suite.judge
    title = f"{suite.title} harness: {variant.name}"

    # 2. Statistics only: nothing to run.
    if args.stats_only:
        return 0 if compute(records, title) else 1

    # 3. What would run: the models per phase, the commands, the estimate.
    seeds = options.get("seeds")
    names = select_episodes(benchmark, str(options["episodes"]),
                            None if seeds is None else str(seeds), options.get("limit"), suite)
    if not names:
        print("No episodes match." + ("" if suite.names(benchmark) else
              f" {suite.title} has no episodes under {benchmark}: for HomeBench, convert "
              "the dataset first (tests/homebench/convert.py)."))
        return 1
    print(f"config {config_path.name}"
          + (f" ({suite.title})" if suite.name != DEFAULT_SUITE else "") + f" -> {records}")
    for step, spec in variant.specs.items():
        if step == "judge" and not judge:       # no judge runs (--no-judge, or HomeBench)
            continue
        print(f"  {step:<18} {spec.label():<28} {'local ' + spec.base_url if spec.local else 'OpenAI'}")
    steps = commands(config_path, records, out, options, judge, benchmark, suite)
    for command in steps:
        print("  $ " + " ".join(command[1:]))
    print(estimate(names, records, judge, suite))
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
