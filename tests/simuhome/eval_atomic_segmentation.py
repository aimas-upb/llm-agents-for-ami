"""
Evaluate ATOMIC_SEGMENTATION_SYSTEM_PROMPT against the SimuHome benchmark.

Runs the User Assistant's first LLM stage over SimuHome episodes and compares,
per query-type family, the number of extracted atomic intents against the number
of goals the benchmark specifies in ``eval.goals``.

For the qt4-* families (temporally conditioned requests) two comparisons are
reported, because a SimuHome goal there bundles a ``targets`` list and the two
counts genuinely diverge (qt4-2 has 50 goals but 100 targets):

    atomic intents vs goals      — the granularity the prompt targets
    atomic intents vs targets    — the device-level action count

Neither is declared "correct" for qt4: the prompt deliberately keeps one
independent scheduled block as one atomic intent, so it should track goals, and
the target delta is reported as a diagnostic.

This is NOT a pytest test: the default cloud route makes paid LLM calls.
Use --base-url with --model to select an explicit local Ollama endpoint;
--repetitions repeats every selected episode independently (default: 1).

Run with:
    conda run -n ami-agents python tests/simuhome/eval_atomic_segmentation.py --limit 3
    conda run -n ami-agents python tests/simuhome/eval_atomic_segmentation.py \
        --out tests/simuhome/results/atomic_seg/
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ami_agents.agents.user_assistant.prompts import ATOMIC_SEGMENTATION_SYSTEM_PROMPT
from ami_agents.agents.user_assistant.utils import loose_json_loads
from ami_agents.agents.user_assistant.utils.llm_client import (
    LLMClientConfig,
    build_behaviour_llm_client,
    build_llm_call_kwargs,
)

# Mirrors ami_agents/environment/integration/SimuHome/worker.py's default.
DEFAULT_BENCHMARK = PROJECT_ROOT.parent / "SimuHome" / "data" / "benchmark"

FAMILIES = ["qt1", "qt2", "qt3", "qt4-1", "qt4-2", "qt4-3"]
QT4_FAMILIES = {"qt4-1", "qt4-2", "qt4-3"}
CASES = ["feasible", "infeasible"]

INTENT_TYPES = ["GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST"]
QUALIFIERS = [
    "explicit", "incomplete", "ambiguous",
    "logical_dependency", "temporal_dependency",
    "achievement", "maintenance",
]

# No live EnvExplorer runs during the evaluation, so the {capabilities} slot gets
# a fixed note instead of a discovered environment dump. Recorded in the run
# metadata so a result set is reproducible.
CAPABILITIES_CTX = (
    "A SimuHome smart home with several rooms (kitchen, living room, bathroom, "
    "bedroom, study room, utility room, office, dining room). Rooms contain "
    "devices such as lights and dimmable lights, air conditioners, heat pumps, "
    "fans, air purifiers, dehumidifiers, humidifiers, washers, dryers, "
    "dishwashers, refrigerators and freezers. Devices can be powered on and off "
    "and configured (mode, fan speed, brightness/level, temperature setpoint, "
    "operational state), and rooms expose readings such as temperature, "
    "humidity, illuminance and air quality."
)


# ---------------------------------------------------------------- episode I/O

def load_episodes(
    benchmark_dir: Path,
    families: List[str],
    cases: List[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """Load episodes grouped by (query_type, case), honouring --limit per group."""
    if not benchmark_dir.is_dir():
        raise SystemExit(f"Benchmark directory not found: {benchmark_dir}")

    by_group: Dict[tuple, List[Dict[str, Any]]] = {}
    for path in sorted(benchmark_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping {path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        meta = data.get("meta") or {}
        # query_type is hyphenated for qt4 ("qt4-1"), so never split the filename
        # on "_" to recover it.
        query_type = str(meta.get("query_type") or "")
        case = str(meta.get("case") or "")
        if query_type not in families or case not in cases:
            continue
        by_group.setdefault((query_type, case), []).append(
            {"path": path, "episode": data}
        )

    episodes: List[Dict[str, Any]] = []
    for key in sorted(by_group):
        group = by_group[key]
        episodes.extend(group[:limit] if limit else group)
    return episodes


def reference_counts(episode: Dict[str, Any], query_type: str) -> tuple:
    """(n_goals, n_targets) from eval.goals; n_targets is None outside qt4."""
    goals = ((episode.get("eval") or {}).get("goals")) or []
    n_goals = len(goals)
    if query_type not in QT4_FAMILIES:
        return n_goals, None
    n_targets = sum(
        len(g.get("targets") or []) for g in goals if isinstance(g, dict)
    )
    return n_goals, n_targets


# ------------------------------------------------------------------ LLM stage

def parse_intents(raw: str) -> List[Dict[str, Any]]:
    """Parse the model's JSON reply into validated intent dicts.

    Raises ValueError on anything unusable, so the caller can record the episode
    as an error rather than silently scoring it as zero intents.
    """
    parsed = loose_json_loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"response is not a JSON object: {raw[:200]!r}")
    intents_data = parsed.get("intents")
    if not isinstance(intents_data, list):
        raise ValueError(f"'intents' is not a list: {raw[:200]!r}")

    intents: List[Dict[str, Any]] = []
    for item in intents_data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        intent_type = str(item.get("type") or "").strip()
        reason = str(item.get("reason") or "").strip()
        raw_qualifiers = item.get("qualifiers")
        qualifiers = [
            q.strip() for q in raw_qualifiers
            if isinstance(q, str) and q.strip()
        ] if isinstance(raw_qualifiers, list) else []
        if text and intent_type in INTENT_TYPES and reason:
            intents.append({
                "text": text,
                "type": intent_type,
                "qualifiers": qualifiers,
                "reason": reason,
            })

    if not intents:
        raise ValueError(f"no valid intents in response: {raw[:200]!r}")
    return intents


async def segment_one(llm_cfg, call_kwargs, prompt: str, query: str, capture=None) -> List[Dict[str, Any]]:
    response = await llm_cfg.client.chat.completions.create(
        model=llm_cfg.model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        **call_kwargs,
    )
    if capture is not None:
        capture["raw_response"] = response.model_dump(mode="json")
    if response.choices[0].finish_reason == "length":
        raise ValueError("Segmentation response truncated by the output token limit")
    return parse_intents((response.choices[0].message.content or "").strip())


async def run_episode(
    entry: Dict[str, Any],
    llm_cfg,
    call_kwargs: Dict[str, Any],
    prompt: str,
    semaphore: asyncio.Semaphore,
    progress: Dict[str, int],
    repetition: int = 1,
    row_file=None,
) -> Dict[str, Any]:
    path: Path = entry["path"]
    episode: Dict[str, Any] = entry["episode"]
    meta = episode.get("meta") or {}
    query_type = str(meta.get("query_type") or "")
    case = str(meta.get("case") or "")
    query = str(episode.get("query") or "")
    n_goals, n_targets = reference_counts(episode, query_type)

    row: Dict[str, Any] = {
        "file": path.name,
        "repetition": repetition,
        "query_type": query_type,
        "case": case,
        "query": query,
        "n_goals": n_goals,
        "n_targets": n_targets,
        "n_intents": None,
        "duration_s": None,
        "intents": [],
        "type_counts": {},
        "qualifier_counts": {},
        "error": None,
    }

    async with semaphore:
        last_error = None
        for _ in range(2):  # one retry, then record the episode as an error
            row["attempts"] = _ + 1
            started = time.monotonic()
            try:
                intents = await segment_one(llm_cfg, call_kwargs, prompt, query, row)
                row["duration_s"] = round(time.monotonic() - started, 3)
                row["intents"] = intents
                row["n_intents"] = len(intents)
                row["type_counts"] = dict(Counter(i["type"] for i in intents))
                row["qualifier_counts"] = dict(
                    Counter(q for i in intents for q in i["qualifiers"])
                )
                last_error = None
                break
            except Exception as exc:  # API failure or unusable response
                row["duration_s"] = round(time.monotonic() - started, 3)
                last_error = f"{type(exc).__name__}: {exc}"
        if last_error:
            row["error"] = last_error

    progress["done"] += 1
    if row_file is not None:
        row_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        row_file.flush()
    status = "ERR" if row["error"] else f"{row['n_intents']} intents"
    took = f" [{row['duration_s']:.1f}s]" if row["duration_s"] is not None else ""
    print(
        f"  [{progress['done']}/{progress['total']}] {path.name} "
        f"({query_type}/{case}, repetition={repetition}): {status} vs {n_goals} goals"
        + (f" / {n_targets} targets" if n_targets is not None else "")
        + took
    )
    return row


# -------------------------------------------------------------------- summary

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per family x case, plus a per-family total across cases."""
    summary: Dict[str, Any] = {}
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row['query_type']}/{row['case']}", []).append(row)
        groups.setdefault(f"{row['query_type']}/all", []).append(row)

    for key in sorted(groups):
        group = groups[key]
        query_type = group[0]["query_type"]
        ok = [r for r in group if r["error"] is None]
        errors = len(group) - len(ok)

        durations = sorted(
            r["duration_s"] for r in group if r["duration_s"] is not None)

        block: Dict[str, Any] = {
            "episodes": len(group),
            "unique_scenarios": len({r["file"] for r in group}),
            "llm_errors": errors,
            "total_intents": sum(r["n_intents"] for r in ok),
            "total_goals": sum(r["n_goals"] for r in ok),
            "type_counts": dict(sum(
                (Counter(r["type_counts"]) for r in ok), Counter())),
            "qualifier_counts": dict(sum(
                (Counter(r["qualifier_counts"]) for r in ok), Counter())),
        }

        if durations:
            block["latency_s"] = {
                "calls": len(durations),
                "mean": round(statistics.fmean(durations), 2),
                "median": round(statistics.median(durations), 2),
                "min": round(durations[0], 2),
                "max": round(durations[-1], 2),
                # index-based p95 so it needs no interpolation on tiny samples
                "p95": round(durations[min(len(durations) - 1,
                                           int(round(0.95 * (len(durations) - 1))))], 2),
                "total": round(sum(durations), 1),
            }

        if ok:
            goal_deltas = [r["n_intents"] - r["n_goals"] for r in ok]
            block["vs_goals"] = {
                "delta": block["total_intents"] - block["total_goals"],
                "exact_match_rate": round(
                    sum(1 for d in goal_deltas if d == 0) / len(ok), 4),
                "mean_abs_error": round(
                    statistics.fmean(abs(d) for d in goal_deltas), 4),
                "delta_distribution": dict(sorted(Counter(goal_deltas).items())),
            }

            if query_type in QT4_FAMILIES:
                with_targets = [r for r in ok if r["n_targets"] is not None]
                if with_targets:
                    target_deltas = [
                        r["n_intents"] - r["n_targets"] for r in with_targets]
                    block["total_targets"] = sum(
                        r["n_targets"] for r in with_targets)
                    block["vs_targets"] = {
                        "delta": block["total_intents"] - block["total_targets"],
                        "exact_match_rate": round(
                            sum(1 for d in target_deltas if d == 0)
                            / len(with_targets), 4),
                        "mean_abs_error": round(
                            statistics.fmean(abs(d) for d in target_deltas), 4),
                        "delta_distribution": dict(
                            sorted(Counter(target_deltas).items())),
                    }

        summary[key] = block
    return summary


def print_report(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("ATOMIC SEGMENTATION vs SIMUHOME REFERENCE COUNTS")
    print("=" * 78)

    for key in sorted(summary):
        block = summary[key]
        print(f"\n{key}  ({block['episodes']} episodes, "
              f"{block['llm_errors']} LLM errors)")
        print(f"  intents={block['total_intents']}  goals={block['total_goals']}"
              + (f"  targets={block['total_targets']}"
                 if "total_targets" in block else ""))

        for label, metric_key in (("vs goals  ", "vs_goals"),
                                  ("vs targets", "vs_targets")):
            metric = block.get(metric_key)
            if not metric:
                continue
            print(f"  {label}: delta={metric['delta']:+d}  "
                  f"exact={metric['exact_match_rate']:.1%}  "
                  f"MAE={metric['mean_abs_error']:.2f}  "
                  f"dist={metric['delta_distribution']}")

        lat = block.get("latency_s")
        if lat:
            print(f"  latency    : mean={lat['mean']}s median={lat['median']}s "
                  f"p95={lat['p95']}s min={lat['min']}s max={lat['max']}s "
                  f"(n={lat['calls']})")
        if block["type_counts"]:
            print(f"  types      : {block['type_counts']}")
        if block["qualifier_counts"]:
            print(f"  qualifiers : {block['qualifier_counts']}")

    print("\n" + "=" * 78)
    print("delta = extracted atomic intents - reference count; "
          "dist maps delta -> episodes.")
    print("=" * 78 + "\n")


# ----------------------------------------------------------------------- main

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate atomic segmentation against SimuHome episodes")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK,
                        help=f"SimuHome benchmark directory (default: {DEFAULT_BENCHMARK})")
    parser.add_argument("--families", default=",".join(FAMILIES),
                        help="Comma-separated query types (default: all)")
    parser.add_argument("--case", choices=["feasible", "infeasible", "both"],
                        default="both", help="Episode case (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes per family/case group")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent LLM calls (default: 8)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory for rows.jsonl and summary.json")
    parser.add_argument("--base-url", help="Explicit OpenAI-compatible URL, e.g. http://127.0.0.1:11434/v1")
    parser.add_argument("--model", help="Model tag; required with --base-url")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Independent evaluations per selected scenario (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for --base-url (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Output token limit for --base-url (default: 4096)")
    parser.add_argument("--request-timeout", type=float, default=180,
                        help="Seconds per API attempt for --base-url (default: 180)")
    args = parser.parse_args()
    if args.repetitions < 1 or args.concurrency < 1 or args.max_tokens < 1 or args.request_timeout <= 0:
        parser.error("repetitions, concurrency, max-tokens and request-timeout must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if bool(args.base_url) != bool(args.model):
        parser.error("--base-url and --model must be supplied together")

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"Unknown families {unknown}; valid: {FAMILIES}")
    cases = CASES if args.case == "both" else [args.case]

    episodes = load_episodes(args.benchmark_dir, families, cases, args.limit)
    if not episodes:
        raise SystemExit("No episodes matched the given filters")

    if args.base_url:
        # Explicit local route: never inherit a real API key or institutional proxy.
        client = AsyncOpenAI(
            api_key="ollama", base_url=args.base_url,
            timeout=args.request_timeout, max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False),
        )
        llm_cfg = LLMClientConfig(client, args.model, args.base_url,
                                  args.temperature, None, None)
        call_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    else:
        # Preserve the live UA configuration for existing cloud evaluations.
        from ami_agents.shared.utils.config_loader import ConfigLoader
        agents_config = ConfigLoader.load_with_env_vars(
            str(PROJECT_ROOT / "ami_agents" / "config" / "agents.yaml"))
        ua_config = agents_config.get("user_assistant", {}) or {}
        llm_cfg = build_behaviour_llm_client(ua_config, "atomic_segmentation")
        call_kwargs = build_llm_call_kwargs(llm_cfg)
    prompt = ATOMIC_SEGMENTATION_SYSTEM_PROMPT.format(capabilities=CAPABILITIES_CTX)

    print(f"Benchmark : {args.benchmark_dir}")
    print(f"Episodes  : {len(episodes)}  "
          f"(families={families}, cases={cases}, limit={args.limit})")
    print(f"Model     : {llm_cfg.model}  kwargs={call_kwargs}")
    print(f"Concurrency: {args.concurrency}; repetitions: {args.repetitions}; "
          f"evaluations: {len(episodes) * args.repetitions}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0, "total": len(episodes) * args.repetitions}
    wall_started = time.monotonic()
    row_file = None
    try:
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            row_file = (args.out / "rows.jsonl").open("x", encoding="utf-8")
            (args.out / "segmentation_prompt.txt").write_text(prompt, encoding="utf-8")
        rows = await asyncio.gather(*[
            run_episode(entry, llm_cfg, call_kwargs, prompt, semaphore, progress,
                        repetition, row_file)
            for repetition in range(1, args.repetitions + 1)
            for entry in episodes
        ])
    finally:
        if row_file is not None:
            row_file.close()
        await llm_cfg.client.close()
    wall_seconds = round(time.monotonic() - wall_started, 1)

    summary = summarize(list(rows))
    print_report(summary)
    print(f"Wall clock: {wall_seconds}s for {len(rows)} episodes "
          f"at concurrency {args.concurrency}\n")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        rows_path = args.out / "rows.jsonl"
        summary_path = args.out / "summary.json"
        summary_path.write_text(json.dumps({
            "run": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "benchmark_dir": str(args.benchmark_dir),
                "families": families,
                "cases": cases,
                "limit": args.limit,
                "model": llm_cfg.model,
                "base_url": llm_cfg.base_url,
                "repetitions": args.repetitions,
                "unique_scenarios": len(episodes),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "call_kwargs": call_kwargs,
                "concurrency": args.concurrency,
                "wall_seconds": wall_seconds,
                "capabilities_ctx": CAPABILITIES_CTX,
            },
            "summary": summary,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {rows_path}\nWrote {summary_path}\n")

    return 1 if any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
