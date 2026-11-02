"""Build one self-contained HTML report over SimuHome / HomeBench harness runs: an
overview across configs, a page per config with the `summary.md` tables and a failure
catalogue, and an episode viewer that opens every trajectory.

    python -m tests.simuhome.bench.viz.build_report \\
        --roots RESULTS/1 [RESULTS/2 ...] [--pooled DIR] \\
        [--reference DIR[=label] ...] [--benchmark SimuHome/data/benchmark] \\
        --out report.html [--title "..."] [--no-records]

Standard library only, no LLM calls. The numbers come from `stats.py` (so they match
`stats/summary.md`); the failure categories from `failures.py`; the page is
`static/report.css` + `static/report.js` around one JSON blob of the aggregates and one
compact JSON block per episode record, parsed only when its trajectory is opened.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.simuhome.bench.viz import failures, loader  # noqa: E402
from tests.simuhome.bench.viz.loader import CASES, PHASES, TYPES, Config, stats  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
# The request types whose attempts the grid repeats; their between-attempt spread is shown.
REPEATED_TYPES = ("qt2", "qt4-1", "qt4-2", "qt4-3")
# The columns of the light per-episode rows the page lists and filters on.
EPISODE_COLUMNS = ("episode", "type", "case", "outcome", "error_step", "score", "error_type",
                   "total_s", "wall_s", "total_calls", "total_tok_in", "total_tok_out",
                   "goals_total", "goals_verified", "check_rate", "execution_check", "judge",
                   "plan_entries", "plans_run", "plans_success", "goals_refused", "bucket")
# `run.json` settings worth a line in the config header.
SETTINGS = ("execution", "context", "memory", "exec_timeout", "step_seconds", "dry_step_seconds",
            "environment", "suite")


# --------------------------------------------------------------------------
# Per-config aggregates
# --------------------------------------------------------------------------

def episode_id(config: Config, attempt_label: str, episode: str) -> str:
    return f"{config.name}|{attempt_label}|{episode}"


def _rate(rows: List[Dict[str, Any]]) -> Optional[float]:
    scored = [r["score"] for r in rows if r.get("score") in (0, 1)]
    return round(sum(scored) / len(scored), 3) if scored else None


def spread(config: Config) -> List[Dict[str, Any]]:
    """The success rate per attempt, its range and standard deviation, for the
    repeated types (qt2, qt4-x) by case and overall; empty below two attempts."""
    if len(config.attempts) < 2:
        return []
    out = []
    types_present = sorted({r["type"] for r in config.rows}, key=stats.type_key)
    for qt in [t for t in REPEATED_TYPES if t in types_present]:
        for case in [c for c in CASES if any(r["case"] == c for r in config.rows)] + ["all"]:
            per_attempt = {}
            for attempt in config.attempts:
                rows = [r for r in attempt.rows if r["type"] == qt and case in ("all", r["case"])]
                if rows:
                    per_attempt[attempt.label] = _rate(rows)
            rates = [v for v in per_attempt.values() if v is not None]
            if len(rates) < 2:
                continue
            pooled = _rate([r for r in config.rows if r["type"] == qt and case in ("all", r["case"])])
            out.append({"type": qt, "case": case, "attempts": per_attempt,
                        "min": min(rates), "max": max(rates),
                        "sd": round(statistics.stdev(rates), 3), "pooled": pooled})
    return out


def failure_catalogue(config: Config) -> Dict[str, Any]:
    """Every failed or errored episode's category and explanation, and the counts per
    category by request type and by case."""
    cases: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {}
    by_case: Dict[str, Dict[str, int]] = {}
    for attempt in config.attempts:
        for (record, evaluation), row in zip(attempt.pairs, attempt.rows):
            category, explanation = failures.categorise(record, evaluation)
            row["category"] = category
            row["explain"] = explanation
            if not category:
                continue
            cases.append({"id": episode_id(config, attempt.label, record["episode"]),
                          "episode": record["episode"], "attempt": attempt.label,
                          "type": row["type"], "case": row["case"], "category": category,
                          "explain": explanation, "score": row["score"],
                          "error_type": row.get("error_type"), "outcome": row.get("outcome")})
            counts.setdefault(category, {}).setdefault(row["type"], 0)
            counts[category][row["type"]] += 1
            by_case.setdefault(category, {}).setdefault(row["case"], 0)
            by_case[category][row["case"]] += 1
    cases.sort(key=lambda c: (failures.CATEGORY_NAMES.index(c["category"]),
                              stats.type_key(c["type"]), c["case"], c["episode"], c["attempt"]))
    return {"cases": cases, "by_type": counts, "by_case": by_case}


def _episode_rows(config: Config) -> List[Dict[str, Any]]:
    out = []
    for attempt in config.attempts:
        for row in attempt.rows:
            light = {k: row.get(k) for k in EPISODE_COLUMNS if k in row}
            light.update({"id": episode_id(config, attempt.label, row["episode"]),
                          "attempt": attempt.label, "category": row.get("category"),
                          "explain": row.get("explain")})
            out.append(light)
    return out


def _settings(config: Config) -> Dict[str, Any]:
    settings = (config.run or {}).get("settings") or {}
    return {k: settings.get(k) for k in SETTINGS if settings.get(k) is not None}


def config_payload(config: Config, slot: int) -> Dict[str, Any]:
    """Everything the page needs about one config: its metadata, the pooled and
    per-attempt summaries (the `summary.md` groups and display tables), the HomeBench
    bucket table when there is one, the failure catalogue and the episode list."""
    catalogue = failure_catalogue(config)          # sets category / explain on the rows
    pooled = stats.aggregate(config.rows)
    attempts = []
    for attempt in config.attempts:
        summary = stats.aggregate(attempt.rows)
        attempts.append({"label": attempt.label, "n": len(attempt.rows),
                         "dir": str(attempt.directory) if attempt.directory else None,
                         "summary": summary, "tables": stats.tables(summary)})
    buckets = stats.bucket_rows(config.rows)
    return {
        "key": config.name, "title": config.title, "reference": config.reference, "slot": slot,
        "gpu": config.gpu, "workers": (config.run or {}).get("workers"),
        "commit": (config.run or {}).get("commit"), "started": (config.run or {}).get("started"),
        "planned": len((config.run or {}).get("episodes") or []) or None,
        "settings": _settings(config), "models": config.models,
        "n": len(config.rows), "attempts": attempts,
        "summary": pooled, "tables": stats.tables(pooled),
        "buckets": stats.official_table(buckets) if buckets else [],
        "notes": stats.homebench_notes(config.rows) if buckets else [],
        "spread": spread(config), "failures": catalogue, "episodes": _episode_rows(config),
    }


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def _json_script(payload: Any) -> str:
    """JSON safe inside a <script> element: a closing tag or comment opener in a
    string would otherwise end the block early."""
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
    return text.replace("</", "<\\/").replace("<!--", "<\\!--")


def _ground_truth(benchmark: Optional[Path], episode: str) -> Optional[Dict[str, Any]]:
    """The answer key of a SimuHome episode (`eval.goals`, the user location), when the
    benchmark directory is given and has the episode."""
    if benchmark is None:
        return None
    path = benchmark / f"{episode}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {"eval": data.get("eval"), "user_location": data.get("user_location"),
            "query": data.get("query")}


def record_blocks(configs: List[Config], benchmark: Optional[Path]) -> Tuple[List[str], int]:
    """One <script type="application/json"> per episode record, compacted; and the
    bytes the compaction removed (snapshots, final states) over every record."""
    blocks, saved = [], 0
    for config in configs:
        for attempt in config.attempts:
            for record, evaluation in attempt.pairs:
                compact = loader.compact_record(record, evaluation)
                truth = _ground_truth(benchmark, record["episode"])
                if truth:
                    compact["ground_truth"] = truth
                saved += sum(v.get("bytes", 0) for v in compact["_stripped"].values())
                ident = episode_id(config, attempt.label, record["episode"])
                blocks.append(f'<script type="application/json" class="rec" data-id="{_attr(ident)}">'
                              f"{_json_script(compact)}</script>")
    return blocks, saved


def _attr(text: str) -> str:
    return (text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def build(configs: List[Config], title: str, roots: List[Path], pooled: Optional[Path],
          benchmark: Optional[Path], with_records: bool) -> str:
    """The whole page as a string."""
    slot = 0
    payloads = []
    for config in configs:
        if config.reference:
            payloads.append(config_payload(config, -1))        # references wear gray
        else:
            if slot >= 8:
                print(f"warning: more than 8 local configs; {config.name} shares the last "
                      "colour slot (fold or split the report)", file=sys.stderr)
            payloads.append(config_payload(config, min(slot, 7)))
            slot += 1
    blocks, saved = record_blocks(configs, benchmark) if with_records else ([], 0)
    data = {
        "title": title,
        "built": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "roots": [str(r) for r in roots], "pooled": str(pooled) if pooled else None,
        "benchmark": str(benchmark) if benchmark else None,
        "phases": list(PHASES), "types": list(TYPES), "cases": list(CASES),
        "repeated_types": list(REPEATED_TYPES),
        "categories": failures.CATEGORIES,
        "legend": stats.LEGEND,
        "records_embedded": with_records, "stripped_bytes": saved,
        "configs": payloads,
    }
    css = (STATIC / "report.css").read_text(encoding="utf-8")
    js = (STATIC / "report.js").read_text(encoding="utf-8")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{_attr(title)}</title>\n<style>\n{css}\n</style>\n</head>\n<body>\n"
        "<div id=\"app\"><noscript>This report needs JavaScript to render its charts and the "
        "episode viewer.</noscript></div>\n"
        f'<script type="application/json" id="data">{_json_script(data)}</script>\n'
        + "\n".join(blocks)
        + f"\n<script>\n{js}\n</script>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _reference(spec: str) -> Tuple[Path, Optional[str]]:
    """"DIR" or "DIR=label"."""
    path, _, label = spec.partition("=")
    return Path(path).expanduser(), (label or None)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True,
                        help="results roots, one per attempt (each holds <config>/ directories)")
    parser.add_argument("--pooled", type=Path,
                        help="a pool_attempts.py directory; its .a<N> records are read as attempts")
    parser.add_argument("--reference", action="append", default=[], metavar="DIR[=LABEL]",
                        help="a records directory (or root) shown as a reference row; repeatable")
    parser.add_argument("--benchmark", type=Path,
                        help="SimuHome's data/benchmark, to show each episode's answer key")
    parser.add_argument("--out", type=Path, required=True, help="the HTML file to write")
    parser.add_argument("--title", default="SimuHome evaluation report")
    parser.add_argument("--no-records", action="store_true",
                        help="leave the episode records out (summary-only page, much smaller)")
    args = parser.parse_args(argv)

    configs = loader.discover(args.roots, args.pooled, [_reference(r) for r in args.reference])
    if not configs:
        print("no records found under the roots given", file=sys.stderr)
        return 1
    html = build(configs, args.title, args.roots, args.pooled, args.benchmark,
                 not args.no_records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size = args.out.stat().st_size
    for config in configs:
        kind = "reference" if config.reference else "config"
        attempts = ", ".join(f"{a.label} ({len(a.rows)})" for a in config.attempts)
        print(f"{kind} {config.title}: attempts {attempts}; GPU {config.gpu.get('gpu') or '-'}")
    print(f"written {args.out} ({size / 1e6:.1f} MB, {sum(len(c.rows) for c in configs)} episode rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
