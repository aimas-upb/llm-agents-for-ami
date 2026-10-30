"""Run lab308e requests through the shared reasoning benchmark, over HASP."""
from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import logging
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_BENCHMARK = PROJECT_ROOT / "tests/e2e-lab308e/cases"
DEFAULT_OUT = PROJECT_ROOT / "tests/hasp/results/harness"

from tests.hasp.hasp_lab308e_benchmark import positive_int, positive_float, write_json, redact
from tests.simuhome.bench.models import load_variant
from tests.simuhome.bench.pipeline import Options, Target, run_episode
from tests.simuhome.bench.run import stage_summary
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.agents.user_assistant.utils.processing import (
    DEFAULT_STRUCTURING_CONCURRENCY, processing_limit,
)
from .backend import Lab308eBackend


def select_episodes(benchmark, patterns, runs, limit=None):
    """The five scenario stems, each repeated N times with stable record IDs."""
    if int(runs) < 1:
        raise ValueError("runs must be positive")
    choices = [p.strip() for p in patterns.split(",") if p.strip()]
    cases = [p.stem for p in sorted(Path(benchmark).glob("*.json"))
             if any(fnmatch.fnmatch(p.stem, pattern) for pattern in choices)]
    return [f"{case}_run_{iteration:03d}" for iteration in range(1, int(runs) + 1)
            for case in cases[:limit]]


async def run(args):
    """Run selected trials with recorded model choices and effective structuring limits."""
    variant = load_variant(args.variant, api=args.llm_api)
    names = select_episodes(Path(args.benchmark), args.episodes, args.runs, args.limit)
    if not names:
        raise ValueError("No scenarios match")
    out = Path(args.out).resolve() / variant.name
    out.mkdir(parents=True, exist_ok=True)
    pending = [name for name in names if not args.skip_existing or not (out / f"{name}.json").exists()]
    print(f"{len(names)} trial(s), {len(pending)} to run, variant {variant.name!r} -> {out}", flush=True)
    if not pending:
        return 0
    if not os.getenv("HA_TOKEN", "").strip():
        raise ValueError("Set HA_TOKEN in the environment or repository .env")
    for spec in variant.specs.values():
        if not spec.local and not os.getenv(spec.api_key_env or "OPENAI_API_KEY"):
            # The lab suite never calls a judge model.
            if spec is not variant["judge"]:
                raise ValueError(f"Set {spec.api_key_env or 'OPENAI_API_KEY'} for the configured model")
    config = ConfigLoader.load_with_env_vars(str(PROJECT_ROOT / "ami_agents/config/agents.yaml"))
    config["environment"] = {"clock": {"source": "system"}}
    # Read the shared UA limit unless the harness/CLI explicitly overrides it.
    ua_config = config.setdefault("user_assistant", {})
    if getattr(args, "structuring_concurrency", None) is not None:
        ua_config.setdefault("processing", {})["structuring_concurrency"] = args.structuring_concurrency
    concurrency = processing_limit(ua_config, "structuring_concurrency", DEFAULT_STRUCTURING_CONCURRENCY)
    logger = logging.getLogger("HASPBench")
    logging.basicConfig(level=logging.INFO if args.debug else logging.WARNING)
    backend = Lab308eBackend(args, out)
    options = Options(debug=args.debug, exec_timeout=args.exec_timeout, context=args.context)
    if args.memory == "run":
        from tests.simuhome.bench.memory import Memory
        options.memory = Memory(out / "memory")
    write_json(out / "run.json", {"variant": variant.as_dict(), "episodes": names,
                                 "backend": "hasp-lab308e", "suite": "hasp",
                                 "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "settings": {"runs": args.runs, "memory": args.memory,
                                              "exec_timeout": args.exec_timeout,
                                              "daylight": args.daylight,
                                              "structuring_concurrency": concurrency}})
    paths = []
    failed = False
    try:
        await backend.start()
        print(f"HASP: {backend.hasp}", flush=True)
        for index, name in enumerate(pending, 1):
            scenario = name.rsplit("_run_", 1)[0]
            case = json.loads((Path(args.benchmark) / f"{scenario}.json").read_text())
            backend.case = case
            episode = {"query": case["query"], "initial_home_config": {}, "user_location": "lab308e",
                       "meta": {"query_type": scenario, "case": "feasible"}}
            started = time.perf_counter()
            print(f"\n[{index}/{len(pending)}] {name}: {case['query']}", flush=True)
            record = await run_episode(name, episode, variant, config,
                                       Target(backend, backend.hasp, "lab308e", backend=backend), options, logger)
            record["wall_seconds"] = time.perf_counter() - started
            record["scenario"] = scenario
            record["fixture"] = case
            checked = time.perf_counter()
            try:
                record["evaluation"] = await backend.check(record)
            except Exception as exc:
                record["evaluation"] = {"score": -1, "error_type": redact(str(exc)), "check": "lab308e"}
            record["evaluation_seconds"] = time.perf_counter() - checked
            # Keep the post-check state too: a check may poll during settling.
            try:
                record["checked_home_state"] = backend.home_state()
            except Exception as exc:
                record["evaluation"] = {"score": -1, "error_type": redact(str(exc)), "check": "lab308e"}
            path = out / f"{name}.json"
            write_json(path, record)
            paths.append(str(path))
            score = record["evaluation"]["score"]
            failed |= score != 1
            print(f"{name}: {record['outcome']}, score={score}, {record['wall_seconds']:.1f}s", flush=True)
    finally:
        await backend.stop()
        if paths:
            print("\n" + stage_summary(paths), flush=True)
    return 1 if failed else 0


def main(argv=None):
    """Parse CLI limits and run the shared pipeline over the HASP backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--episodes", default="*")
    parser.add_argument("--runs", "-n", type=positive_int, default=1)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--workers", type=int, choices=[1], default=1)
    parser.add_argument("--structuring-concurrency", type=positive_int,
                        help="maximum simultaneous structuring behaviours (default: agents.yaml)")
    parser.add_argument("--ha-url", default=os.getenv("HA_URL", "http://localhost:8123"))
    parser.add_argument("--virtual-dir", type=Path, default=os.getenv("HA_VIRTUAL_DIR"))
    parser.add_argument("--ha-yaml-path", default="/config/custom_components/virtual/lab308e.yaml")
    parser.add_argument("--hasp-port", type=int)
    parser.add_argument("--startup-timeout", type=positive_float, default=120)
    parser.add_argument("--daylight", action=argparse.BooleanOptionalAction, default=False,
                        help="start each trial at simulated local noon (default: real time of day)")
    parser.add_argument("--exec-timeout", type=positive_float, default=60,
                        help="maximum behavior-tree execution seconds (default: 60)")
    parser.add_argument("--llm-api", choices=("auto", "chat-completions", "responses"),
                        help="override the LLM API for every model phase")
    parser.add_argument("--context", choices=("none", "prepend"), default="none")
    parser.add_argument("--memory", choices=("off", "run"), default="off")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if args.hasp_port is not None and not 1 <= args.hasp_port <= 65535:
        parser.error("--hasp-port must be between 1 and 65535")
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
