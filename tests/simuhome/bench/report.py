"""Success rates and time statistics across the bench's variants.

    conda run -n ami-agents python -m tests.simuhome.bench.report \\
        --root tests/simuhome/results/bench [--variants a,b]

Reads every <root>/<variant>/<episode>.json with its .eval.json and writes, in
<root>:

    rates.csv     variant, type, case: n, evaluated, correct, infra, rate
                  (SimuHome's all-or-nothing score); feasible also the partial
                  success -- checks, checks_ok, check_rate, mean_episode_success
                  and the failed checks by class: missing, wrong_value,
                  wrong_timing; infeasible also: execution-check rate, judge rate
    steps.csv     variant, type, step: n, mean / median / p95 / max seconds,
                  LLM calls, LLM seconds, tokens in / out (per episode)
    outcomes.csv  variant, type, case, outcome: count
    report.md     the same, as tables

Steps: segmentation, structuring, state_read, planning (with
planning.achievement and planning.dependency for dependency structures),
profiling and guarding (a dependency plan's, timed inside its planning; the
simple plans' guard steps), execution (real seconds; virtual seconds
alongside), response. Discovery is
listed apart: it is offline in a deployment. A step's LLM calls include its
sub-steps' (structuring.intent under structuring).

Infrastructure failures (score -1) are left out of the rates' denominators, as
SimuHome's `aggregate_results` does.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

STEPS = ("segmentation", "structuring", "state_read", "planning", "planning.achievement",
         "planning.dependency", "profiling", "guarding", "execution", "response")


def _p95(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def load(root: Path, variants: Optional[Iterable[str]]) -> Dict[str, List[Dict[str, Any]]]:
    """{variant: [(record, evaluation or None)]}."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if variants and directory.name not in variants:
            continue
        rows = []
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".eval.json") or path.name in ("run.json",):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if "episode" not in record:
                continue
            evaluation_path = path.with_name(path.stem + ".eval.json")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) \
                if evaluation_path.exists() else None
            rows.append({"record": record, "evaluation": evaluation})
        if rows:
            out[directory.name] = rows
    return out


def _step_values(record: Dict[str, Any], step: str) -> Optional[Dict[str, float]]:
    """One episode's total for a step, or None when the step did not run."""
    meter = record.get("meter") or {}
    seconds = [s["seconds"] for s in meter.get("steps", []) if s["step"] == step]
    calls = [c for c in meter.get("calls", [])
             if c["step"] == step or c["step"].startswith(step + ".")]
    if not seconds:
        return None
    out = {"seconds": sum(seconds), "calls": len(calls),
           "llm_seconds": sum(c["seconds"] for c in calls),
           "tokens_in": sum(c.get("tokens_in") or 0 for c in calls),
           "tokens_out": sum(c.get("tokens_out") or 0 for c in calls)}
    if step == "execution":
        virtual = [s.get("virtual_seconds") for s in meter.get("steps", [])
                   if s["step"] == "execution" and s.get("virtual_seconds") is not None]
        out["virtual_seconds"] = sum(virtual) if virtual else None
    return out


def rates(data) -> List[Dict[str, Any]]:
    rows = []
    for variant, items in data.items():
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for item in items:
            record = item["record"]
            groups.setdefault((record["query_type"], record["case"]), []).append(item)
        for (qt, case), group in sorted(groups.items()):
            scores = [(i["evaluation"] or {}).get("score") for i in group]
            evaluated = [s for s in scores if s is not None and s >= 0]
            row = {"variant": variant, "type": qt, "case": case, "n": len(group),
                   "evaluated": len(evaluated), "correct": sum(evaluated),
                   "infra": sum(1 for s in scores if s == -1),
                   "unscored": sum(1 for s in scores if s is None),
                   "rate": round(sum(evaluated) / len(evaluated), 3) if evaluated else None}
            # Partial success (feasible): every expected value, classified.
            parts = [(i["evaluation"] or {}).get("breakdown") for i in group]
            parts = [p for p in parts if p and p.get("total")]
            if parts:
                total = sum(p["total"] for p in parts)
                row.update({
                    "checks": total,
                    "checks_ok": sum(p["ok"] for p in parts),
                    "check_rate": round(sum(p["ok"] for p in parts) / total, 3),
                    "mean_episode_success": round(statistics.mean(p["success"] for p in parts), 3),
                    "missing": sum(p["missing"] for p in parts),
                    "wrong_value": sum(p["wrong_value"] for p in parts),
                    "wrong_timing": sum(p["wrong_timing"] for p in parts)})
            if case == "infeasible":
                checks = [((i["evaluation"] or {}).get("execution_check") or {}).get("score")
                          for i in group if (i["evaluation"] or {}).get("score", -1) >= 0]
                judged = [((i["evaluation"] or {}).get("judge") or {}).get("score")
                          for i in group if (i["evaluation"] or {}).get("score", -1) >= 0]
                checks = [c for c in checks if c is not None]
                judged = [j for j in judged if j is not None and j >= 0]
                row["execution_check_rate"] = round(sum(checks) / len(checks), 3) \
                    if checks else None
                row["judge_rate"] = round(sum(judged) / len(judged), 3) if judged else None
            rows.append(row)
    return rows


def step_stats(data) -> List[Dict[str, Any]]:
    rows = []
    for variant, items in data.items():
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            by_type.setdefault(item["record"]["query_type"], []).append(item["record"])
            by_type.setdefault("all", []).append(item["record"])
        for qt, records in sorted(by_type.items()):
            for step in STEPS + ("discovery",):
                if step == "discovery":
                    values = [{"seconds": r["discovery_seconds"], "calls": 0,
                               "llm_seconds": 0.0, "tokens_in": 0, "tokens_out": 0}
                              for r in records if r.get("discovery_seconds") is not None]
                else:
                    values = [v for v in (_step_values(r, step) for r in records) if v]
                if not values:
                    continue
                seconds = [v["seconds"] for v in values]
                row = {"variant": variant, "type": qt, "step": step, "n": len(values),
                       "mean_s": round(statistics.mean(seconds), 3),
                       "median_s": round(statistics.median(seconds), 3),
                       "p95_s": round(_p95(seconds), 3), "max_s": round(max(seconds), 3),
                       "calls": round(statistics.mean(v["calls"] for v in values), 2),
                       "llm_s": round(statistics.mean(v["llm_seconds"] for v in values), 3),
                       "tokens_in": round(statistics.mean(v["tokens_in"] for v in values)),
                       "tokens_out": round(statistics.mean(v["tokens_out"] for v in values))}
                if step == "execution":
                    virtual = [v["virtual_seconds"] for v in values
                               if v.get("virtual_seconds") is not None]
                    row["virtual_mean_s"] = round(statistics.mean(virtual), 1) \
                        if virtual else None
                rows.append(row)
    return rows


def outcomes(data) -> List[Dict[str, Any]]:
    counts: Dict[tuple, int] = {}
    for variant, items in data.items():
        for item in items:
            record = item["record"]
            outcome = record.get("outcome") or "?"
            if outcome == "error":
                outcome = f"error@{record.get('error_step')}"
            key = (variant, record["query_type"], record["case"], outcome)
            counts[key] = counts.get(key, 0) + 1
    return [{"variant": v, "type": t, "case": c, "outcome": o, "count": n}
            for (v, t, c, o), n in sorted(counts.items())]


def _csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(none)\n"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(f) is None else str(row.get(f))
                                       for f in fields) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="the directory holding variant dirs")
    parser.add_argument("--variants", help="comma-separated subset")
    args = parser.parse_args()
    root = Path(args.root)
    data = load(root, args.variants.split(",") if args.variants else None)
    if not data:
        print("No records.")
        return 1
    rate_rows, step_rows, outcome_rows = rates(data), step_stats(data), outcomes(data)
    _csv(root / "rates.csv", rate_rows)
    _csv(root / "steps.csv", step_rows)
    _csv(root / "outcomes.csv", outcome_rows)
    report = ("# Bench report\n\n## Success rates\n\n" + _table(rate_rows)
              + "\n## Time per step (seconds per episode)\n\n" + _table(step_rows)
              + "\n## Outcomes\n\n" + _table(outcome_rows))
    (root / "report.md").write_text(report, encoding="utf-8")
    print(_table(rate_rows))
    print(f"written: {root / 'report.md'}, rates.csv, steps.csv, outcomes.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
