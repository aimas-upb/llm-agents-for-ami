"""The evaluation harness: run lab308e through HASP, score it, and compute
statistics -- copied from tests/simuhome/bench/harness.py.

    python tests/hasp/hasp_lab308e_harness.py --runs 3 [--dry-run | --stats-only]

Same variant YAML, model-per-phase configuration, run/evaluate/statistics flow,
record format, resumption, reasoning pipeline, and meters as the SimuHome
harness. The backend discovers HASP TDs and executes the trees in real time;
HA cannot fast-forward or schedule SimuHome workflows. Scores use the five
lab308e fixture assertions, captured while HASP is running.

Environment: HA_TOKEN, HA_URL, HA_VIRTUAL_DIR and the configured model API keys.
Each invocation uses a new timestamped output folder by default. Pass --out
to resume an existing folder or rebuild its statistics with --stats-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .run import DEFAULT_BENCHMARK, PROJECT_ROOT, select_episodes
from tests.simuhome.bench.models import ModelSpec, load_variant
from tests.simuhome.bench.stats import compute

DEFAULT_OUT = PROJECT_ROOT / "tests" / "hasp" / "results" / "harness"
# The run.py options a config's `run:` section may set.
RUN_OPTIONS = ("episodes", "runs", "limit", "workers", "context", "memory", "exec_timeout",
               "ha_url", "virtual_dir", "hasp_port", "startup_timeout", "debug", "llm_api",
               "structuring_concurrency", "daylight")
DEFAULT_CONFIG = PROJECT_ROOT / "tests/hasp/bench/variants/harness.example.yaml"
LEGEND = """Groups are lab308e scenarios and repetitions. Scores come from the lab308e
fixture assertions over HASP, not SimuHome evaluators or an LLM judge. Score -1
indicates an infrastructure error and is excluded from success rates and cost means.
Timing, LLM calls, tokens, and goal metrics use the shared reasoning benchmark meter.
Discovery, home setup, and assertion polling are excluded from phase timings.
"""
# Rough LLM calls per episode for the agents' steps, and per judged episode.
CALLS_PER_EPISODE = (3, 6)
JUDGE_CALLS = 3


def report_legend(records: Path, variant) -> str:
    """Include the saved run's model choices when rebuilding historical reports."""
    # Read historical settings rather than the current command-line defaults.
    specs = variant.specs
    daylight_note = ""
    metadata = records / "run.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        daylight = (saved.get("settings") or {}).get("daylight")
        if daylight is not None:
            start = "local noon (daylight) for every trial" if daylight else "real local time of day"
            daylight_note = f"Simulated start: {start}.\n\n"
        steps = (saved.get("variant") or {}).get("steps")
        if steps:
            # Reports written before API selection used Chat Completions.
            specs = {step: ModelSpec(**{"api": "chat-completions", **spec})
                     for step, spec in steps.items()}

    def cell(value):
        """Escape saved configuration values for Markdown table cells."""
        return str(value).replace("|", "\\|").replace("\n", " ")

    # Keep clock conditions alongside the model configuration for comparisons.
    lines = ["## Model configuration", "", "| Phase | Model | Provider / endpoint | API |",
             "|---|---|---|---|"]
    for step, spec in specs.items():
        if step != "judge":
            provider = "local " + spec.base_url if spec.local else "OpenAI"
            lines.append(f"| {cell(step)} | {cell(spec.label())} | {cell(provider)} | {cell(spec.api)} |")
    return LEGEND + "\n" + daylight_note + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The plan of a harness run
# --------------------------------------------------------------------------

def run_options(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """The run options: the config's `run:` section, under the command line's
    own flags (a flag given wins)."""
    options = dict(config.get("run") or {})
    for key in RUN_OPTIONS:
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)
    options.setdefault("episodes", "*")
    options.setdefault("runs", 1)
    options.setdefault("workers", 1)
    return options


def commands(config_path: Path, records: Path, out: Path, options: Dict[str, Any],
             judge: bool, benchmark: Path) -> List[List[str]]:
    """The run and evaluate command lines, in order."""
    # 1. The run, resuming: episodes already recorded are skipped.
    run = [sys.executable, "-m", "tests.hasp.bench.run",
           "--variant", str(config_path), "--out", str(out), "--benchmark", str(benchmark),
           "--skip-existing"]
    for key in RUN_OPTIONS:
        if options.get(key) is not None:
            if key == "daylight":
                run.append("--daylight" if options[key] else "--no-daylight")
            elif key == "debug":
                if options[key]:
                    run.append("--debug")
            else:
                run += [f"--{key.replace('_', '-')}", str(options[key])]

    # 2. The scoring, of the records not scored since they were written.
    evaluate = [sys.executable, "-m", "tests.hasp.bench.evaluate",
                "--records", str(records), "--skip-scored"]
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
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="harness config (a variant file)")
    parser.add_argument("--out", help=f"output root (default: {DEFAULT_OUT}/<timestamp>)")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--episodes", help="scenario patterns, e.g. make_*,reduce_glare")
    parser.add_argument("--runs", "-n", type=int, help="repetitions of each scenario")
    parser.add_argument("--ha-url")
    parser.add_argument("--virtual-dir")
    parser.add_argument("--hasp-port", type=int)
    parser.add_argument("--startup-timeout", type=float)
    parser.add_argument("--exec-timeout", type=float)
    parser.add_argument("--daylight", action=argparse.BooleanOptionalAction, default=None,
                        help="start each trial at simulated local noon (default: real time of day)")
    parser.add_argument("--llm-api", choices=("auto", "chat-completions", "responses"),
                        help="LLM API for all phases (default: model config or auto)")
    parser.add_argument("--memory", choices=("off", "run"))
    parser.add_argument("--context", choices=("none", "prepend"))
    parser.add_argument("--debug", action="store_true", default=None)
    parser.add_argument("--limit", type=int, help="maximum number of selected scenarios")
    parser.add_argument("--workers", type=int, help="override run.workers")
    parser.add_argument("--structuring-concurrency", type=int,
                        help="maximum simultaneous structuring behaviours per agent")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="print the commands and the estimate, run nothing")
    mode.add_argument("--stats-only", action="store_true",
                      help="only the statistics, from the records on disk")
    args = parser.parse_args()
    if args.stats_only and args.out is None:
        parser.error("--stats-only requires --out pointing to an existing run's output root")

    # 1. The config, its variant (every phase's model resolved) and where its
    #    records go -- the directory run.py names after the variant.
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    options = run_options(config, args)
    variant = load_variant(str(config_path), api=options.get("llm_api"))
    out = (Path(args.out) if args.out is not None else
           DEFAULT_OUT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    benchmark = Path(args.benchmark)
    records = out / variant.name
    judge = False
    if options["runs"] < 1 or options["workers"] != 1:
        parser.error("--runs must be positive and --workers must be 1 for a shared HA home")
    if options.get("structuring_concurrency") is not None and int(options["structuring_concurrency"]) < 1:
        parser.error("--structuring-concurrency must be positive")
    title = f"HASP lab308e harness: {variant.name}"

    # 2. Statistics only: nothing to run.
    if args.stats_only:
        return 0 if compute(records, title, legend=report_legend(records, variant)) else 1

    # 3. What would run: the models per phase, the commands, the estimate.
    names = select_episodes(benchmark, str(options["episodes"]),
                            int(options["runs"]), options.get("limit"))
    if not names:
        print("No episodes match.")
        return 1
    print(f"config {config_path.name} -> {records}")
    for step, spec in variant.specs.items():
        if step == "judge":
            continue
        print(f"  {step:<18} {spec.label():<28} {'local ' + spec.base_url if spec.local else 'OpenAI'} ({spec.api})")
    steps = commands(config_path, records, out, options, judge, benchmark)
    for command in steps:
        print("  $ " + " ".join(command[1:]))
    print(estimate(names, records, judge))
    if args.dry_run:
        return 0

    # 4. Run, then score; a failed step still lets the next one see what was
    #    written (a partial run is scored and summarised as far as it got).
    failed = False
    for command in steps:
        print(f"\n=== {command[2]}", flush=True)
        failed |= subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode != 0

    # 5. The statistics over every record of this config.
    print(f"\n=== statistics", flush=True)
    report = compute(records, title, legend=report_legend(records, variant))
    return 0 if report and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
