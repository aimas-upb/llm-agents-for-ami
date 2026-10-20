"""Statistics over one variant's scored episodes: cost per phase, goals
achieved, SimuHome's score -- averaged by request type and feasibility.

    conda run -n ami-agents python -m tests.simuhome.bench.stats \\
        --records tests/simuhome/results/harness/<variant>

Reads every <episode>.json with its <episode>.eval.json (`evaluate.py`) and
writes <records>/stats/:

    episodes.csv   one row per episode, every metric below
    summary.csv    one row per group (type x case, type, case, overall)
    summary.json   the same, as JSON
    summary.md     three tables (score and goals; seconds per phase; tokens per
                   phase), with a legend of every column

Per episode (`episode_metrics`):

    cost       per phase -- segmentation, structuring, planning, profiling,
               guarding, execution, and "other" (state reads, the reply) --
               the wall seconds, LLM calls, input and output tokens, read from
               the record's `meter.stages` (`meter.Meter.stages`: profiling and
               guarding are taken out of planning, cached steps left out);
               discovery is not counted (offline in a deployment)
    goals      SimuHome-verified (feasible): the eval goals, and those whose
               every check passed (`evaluate.breakdown`); our execution (all):
               the goals structured, the plan entries, the plans run and those
               that ended SUCCESS, the goals refused (impossible / clarify)
    score      SimuHome's verdict (1 / 0 / -1 infrastructure), the check pass
               rate (feasible), the execution check and the judge (infeasible)

Example -- qt4-3_feasible_seed_45, one dependency plan for two eval goals:

    breakdown  goal 0: 2 checks ok; goal 1: 1 check ok
    row        goals_total 2, goals_verified 2, multi_goal True,
               plan_entries 1, plans_run 1, plans_success 1, score 1

Groups (`aggregate`): (type, case) for each of qt1 ... qt4-3 and feasible /
infeasible, then (type, all), (all, case), (all, all). Rates leave out
infrastructure failures (score -1), as SimuHome's `aggregate_results` does;
so do the cost means (a run cut short by a dead API costs nothing, which would
flatter the means).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .meter import STAGES
from .report import _csv, _p95, _table

# The phases reported, in pipeline order; "other" holds the state reads and the reply.
PHASES = (*STAGES, "other")
# The request types, in SimuHome's order; a type not listed sorts after these.
TYPES = ("qt1", "qt2", "qt3", "qt4-1", "qt4-2", "qt4-3")
CASES = ("feasible", "infeasible")
# Plan outcomes that refuse a goal rather than act on it (`pipeline._entry_outcome`).
REFUSED = ("impossible", "clarify")
# Files in a records directory that are not episode records.
NOT_RECORDS = ("run.json", "summary.json")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load(directory: Path) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Every (record, evaluation) in ``directory``; the evaluation is None for
    a record not scored yet. Files that are not episode records (run.json,
    the .eval.json files themselves, anything without an "episode") are
    skipped."""
    out = []
    for path in sorted(directory.glob("*.json")):
        # 1. Episode records only.
        if path.name.endswith(".eval.json") or path.name in NOT_RECORDS:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "episode" not in record:
            continue
        # 2. Its score, when evaluate.py has written one.
        scored = path.with_name(path.stem + ".eval.json")
        evaluation = json.loads(scored.read_text(encoding="utf-8")) \
            if scored.exists() else None
        out.append((record, evaluation))
    return out


# --------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------

def _cost(record: Dict[str, Any]) -> Dict[str, Any]:
    """The cost columns: seconds, calls and tokens per phase, and their totals.
    Every value is None for a record without stage timings (written before the
    meter kept them)."""
    stages = (record.get("meter") or {}).get("stages")
    row: Dict[str, Any] = {}

    # 1. Per phase; a phase that never ran is a zero, not a gap.
    for phase in PHASES:
        entry = (stages or {}).get(phase) or {}
        for column, field in (("s", "seconds"), ("calls", "calls"),
                              ("tok_in", "tokens_in"), ("tok_out", "tokens_out")):
            row[f"{phase}_{column}"] = (entry.get(field) or 0) if stages else None

    # 2. The totals over the phases (the request's end-to-end cost).
    for column in ("s", "calls", "tok_in", "tok_out"):
        row[f"total_{column}"] = sum(row[f"{p}_{column}"] for p in PHASES) \
            if stages else None
    row["wall_s"] = record.get("wall_seconds")
    return row


def _verified_goals(evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """SimuHome-verified goals, from the evaluation's per-check breakdown: the
    eval goals, those whose every check is "ok", and the check pass rate.
    None for an episode without a breakdown (infeasible, or not scored)."""
    breakdown = (evaluation or {}).get("breakdown") or {}
    checks = breakdown.get("checks") or []
    if not checks:
        return {"goals_total": None, "goals_verified": None, "goals_verified_frac": None,
                "multi_goal": None, "check_rate": None}

    # A goal is reached only when all of its checks are.
    by_goal: Dict[Any, List[bool]] = {}
    for check in checks:
        by_goal.setdefault(check.get("goal"), []).append(check.get("class") == "ok")
    total = len(by_goal)
    verified = sum(1 for oks in by_goal.values() if all(oks))
    return {"goals_total": total, "goals_verified": verified,
            "goals_verified_frac": round(verified / total, 3),
            "multi_goal": total > 1, "check_rate": breakdown.get("success")}


def _executed_goals(record: Dict[str, Any]) -> Dict[str, Any]:
    """Goals as our agents handled them: structured, planned, run, run to
    SUCCESS, refused. A goal stopped for a clarification at structuring has
    no plan entry, so the two refusal counts never overlap."""
    structures = record.get("structures") or []
    plans = record.get("plans") or []
    executed = (record.get("execution") or {}).get("plans") or []
    clarified = sum(1 for s in structures if s.get("clarification"))
    return {"goal_intents": len(structures), "plan_entries": len(plans),
            "plans_run": len(executed),
            "plans_success": sum(1 for p in executed if p.get("status") == "SUCCESS"),
            "goals_refused": clarified + sum(1 for p in plans
                                             if p.get("outcome") in REFUSED)}


def _score(record: Dict[str, Any], evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """SimuHome's verdict and its parts. None for a record not scored; the
    judge's -1 (its own failure) is None, as in `report.rates`."""
    evaluation = evaluation or {}
    judge = (evaluation.get("judge") or {}).get("score")
    return {"score": evaluation.get("score"),
            "execution_check": (evaluation.get("execution_check") or {}).get("score"),
            "judge": judge if judge is not None and judge >= 0 else None,
            "error_type": evaluation.get("error_type")}


def episode_metrics(record: Dict[str, Any],
                    evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One flat row for an episode: who it is, its outcome, its cost per
    phase, its goals (verified and executed) and its score (module docstring)."""
    return {"episode": record["episode"], "type": record["query_type"],
            "case": record["case"], "outcome": record.get("outcome"),
            "error_step": record.get("error_step"),
            **_cost(record), **_verified_goals(evaluation), **_executed_goals(record),
            **_score(record, evaluation)}


# --------------------------------------------------------------------------
# Groups of episodes
# --------------------------------------------------------------------------

def _mean(values: Iterable[Any], digits: int = 3) -> Optional[float]:
    """The mean of the values that are not None; None when there are none."""
    present = [v for v in values if v is not None]
    return round(statistics.mean(present), digits) if present else None


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    """numerator / denominator, None for an empty denominator."""
    return round(numerator / denominator, 3) if denominator else None


def _groups(rows: List[Dict[str, Any]]) -> List[Tuple[Tuple[str, str], List[Dict[str, Any]]]]:
    """The rows by (type, case), (type, all), (all, case), (all, all), in that
    order, types in SimuHome's order; empty groups are left out."""
    types = sorted({r["type"] for r in rows},
                   key=lambda t: (TYPES.index(t) if t in TYPES else len(TYPES), t))
    # 1. Every group key, in report order; "all" matches any type or case.
    keys = [(t, c) for t in types for c in CASES] + [(t, "all") for t in types] + \
        [("all", c) for c in CASES] + [("all", "all")]
    # 2. Each group's members.
    out = []
    for qt, case in keys:
        members = [r for r in rows if qt in ("all", r["type"]) and case in ("all", r["case"])]
        if members:
            out.append(((qt, case), members))
    return out


def summarise(qt: str, case: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One group's summary row (the columns are explained in LEGEND)."""
    scored = [r for r in rows if r["score"] in (0, 1)]
    counted = [r for r in rows if r["score"] != -1]       # infra left out of the means
    out: Dict[str, Any] = {"type": qt, "case": case, "n": len(rows),
                           "n_scored": len(scored),
                           "n_infra": sum(1 for r in rows if r["score"] == -1),
                           "n_error": sum(1 for r in rows if r["outcome"] == "error"),
                           "n_unscored": sum(1 for r in rows if r["score"] is None)}

    # 1. SimuHome's score: the share passed, and the finer check pass rate.
    out["success_rate"] = _ratio(sum(r["score"] for r in scored), len(scored))
    out["check_rate"] = _mean(r["check_rate"] for r in scored)

    # 2. Goals SimuHome verified (feasible): pooled over the group, per request,
    #    and over the multi-goal requests alone.
    goals = [r for r in scored if r["goals_total"]]
    multi = [r for r in goals if r["multi_goal"]]
    out.update({
        "goals_total": sum(r["goals_total"] for r in goals),
        "goals_verified": sum(r["goals_verified"] for r in goals),
        "goals_verified_rate": _ratio(sum(r["goals_verified"] for r in goals),
                                      sum(r["goals_total"] for r in goals)),
        "goals_verified_mean": _mean(r["goals_verified"] for r in goals),
        "goals_per_request": _mean(r["goals_total"] for r in goals),
        "multi_n": len(multi),
        "multi_goals_verified_rate": _ratio(sum(r["goals_verified"] for r in multi),
                                            sum(r["goals_total"] for r in multi)),
        "multi_goals_verified_mean": _mean(r["goals_verified"] for r in multi)})

    # 3. Goals as our agents handled them (every case).
    out.update({
        "goal_intents_mean": _mean(r["goal_intents"] for r in counted),
        "plans_success_rate": _ratio(sum(r["plans_success"] for r in counted),
                                     sum(r["plans_run"] for r in counted)),
        "plans_success_mean": _mean(r["plans_success"] for r in counted),
        "goals_refused_mean": _mean(r["goals_refused"] for r in counted)})

    # 4. Infeasible requests: did nothing on the impossible part / judged right.
    infeasible = [r for r in scored if r["case"] == "infeasible"]
    if infeasible:
        checks = [r["execution_check"] for r in infeasible if r["execution_check"] is not None]
        judged = [r["judge"] for r in infeasible if r["judge"] is not None]
        out["execution_check_rate"] = _ratio(sum(checks), len(checks))
        out["judge_rate"] = _ratio(sum(judged), len(judged))
    else:
        out["execution_check_rate"] = out["judge_rate"] = None

    # 5. Cost: the mean per episode of every phase column, and the spread of
    #    the end-to-end time (median, 95th percentile).
    costed = [r for r in counted if r["total_s"] is not None]
    out["n_costed"] = len(costed)
    for phase in (*PHASES, "total"):
        out[f"{phase}_s"] = _mean(r[f"{phase}_s"] for r in costed)
        for column in ("calls", "tok_in", "tok_out"):
            out[f"{phase}_{column}"] = _mean((r[f"{phase}_{column}"] for r in costed), 1)
    seconds = [r["total_s"] for r in costed]
    out["total_s_median"] = round(statistics.median(seconds), 3) if seconds else None
    out["total_s_p95"] = round(_p95(seconds), 3) if seconds else None
    return out


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The summary rows, one per group (`_groups`)."""
    return [summarise(qt, case, members) for (qt, case), members in _groups(rows)]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

# What each column of the summary means; printed at the top of summary.md.
LEGEND = """\
Groups: `type` is SimuHome's request type (qt1 state questions, qt2 room-state
changes, qt3 device commands, qt4-1/2/3 timed requests), `case` feasible or
infeasible; "all" pools them. An *infrastructure failure* (score -1: the API,
the simulator or the judge failed, not the agents) is left out of every rate
and every cost mean.

| column | meaning |
|---|---|
| n | episodes in the group |
| scored | episodes SimuHome scored 1 or 0 |
| infra | infrastructure failures (score -1) |
| errors | episodes where our code raised (the phase is in episodes.csv) |
| success | share of scored episodes SimuHome passed (all-or-nothing per request) |
| checks | mean share of the expected device values that were right (feasible; partial credit) |
| goals ok | eval goals reached (every check of the goal right) / eval goals, pooled over the group (feasible) |
| ok/request | mean eval goals reached per request (feasible) |
| goals/request | mean eval goals per request (feasible) |
| multi n | feasible requests with more than one eval goal |
| multi goals ok | goals ok, over the multi-goal requests only |
| multi ok/request | ok/request, over the multi-goal requests only |
| plans ok | our plans that ran to SUCCESS / our plans run (our executor's view, every case) |
| refused/request | mean goals our agents answered "impossible" or with a question |
| exec check | infeasible: share that did nothing on the impossible part and said so |
| judge | infeasible: share SimuHome's judge panel passed on the reply |
| *phase* s | mean wall seconds per episode in that phase; profiling and guarding are taken out of planning, so the phases add up to total |
| *phase* in/out | mean input / output tokens per episode in that phase (output includes reasoning tokens) |
| calls | mean LLM calls per episode |
| median, p95 | median and 95th percentile of the total seconds (95% of the episodes took less) |

Phases: segmentation, structuring, planning, profiling, guarding, execution, and
other (the state reads and the reply to the user). Discovery is not counted.
"""


def _label(row: Dict[str, Any]) -> str:
    """A group's name in the tables: "qt3 feasible", "qt3", "feasible", "overall"."""
    if row["type"] == "all":
        return "overall" if row["case"] == "all" else f"all {row['case']}"
    return row["type"] if row["case"] == "all" else f"{row['type']} {row['case']}"


def _pct(value: Optional[float]) -> Optional[str]:
    """A share as a percentage, None kept (an empty cell)."""
    return None if value is None else f"{100 * value:.0f}%"


def tables(summary: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """The three display tables: score and goals, seconds per phase, tokens per
    phase -- one row per group, with readable headers (LEGEND)."""
    score, seconds, tokens = [], [], []
    for row in summary:
        group = {"group": _label(row)}
        # 1. Score and goals.
        score.append({**group, "n": row["n"], "scored": row["n_scored"],
                      "infra": row["n_infra"], "errors": row["n_error"],
                      "success": _pct(row["success_rate"]), "checks": _pct(row["check_rate"]),
                      "goals ok": _pct(row["goals_verified_rate"]),
                      "ok/request": row["goals_verified_mean"],
                      "goals/request": row["goals_per_request"],
                      "multi n": row["multi_n"],
                      "multi goals ok": _pct(row["multi_goals_verified_rate"]),
                      "multi ok/request": row["multi_goals_verified_mean"],
                      "plans ok": _pct(row["plans_success_rate"]),
                      "refused/request": row["goals_refused_mean"],
                      "exec check": _pct(row["execution_check_rate"]),
                      "judge": _pct(row["judge_rate"])})
        # 2. Seconds per phase, then the total's spread.
        seconds.append({**group, **{f"{p} s": None if row[f"{p}_s"] is None
                                    else round(row[f"{p}_s"], 2) for p in (*PHASES, "total")},
                        **{label: None if row[k] is None else round(row[k], 2)
                           for label, k in (("median", "total_s_median"),
                                            ("p95", "total_s_p95"))}})
        # 3. Tokens per phase, as "in / out", and the LLM calls.
        tokens.append({**group, **{f"{p} in/out": None if row[f"{p}_tok_in"] is None
                                   else f"{row[f'{p}_tok_in']:.0f} / {row[f'{p}_tok_out']:.0f}"
                                   for p in (*PHASES, "total")},
                       "calls": row["total_calls"]})
    return {"score": score, "seconds": seconds, "tokens": tokens}


def write(directory: Path, rows: List[Dict[str, Any]],
          summary: List[Dict[str, Any]], title: str) -> Path:
    """episodes.csv, summary.csv, summary.json and summary.md in
    ``directory``/stats; returns that directory."""
    out = directory / "stats"
    out.mkdir(parents=True, exist_ok=True)

    # 1. The machine-readable files.
    _csv(out / "episodes.csv", rows)
    _csv(out / "summary.csv", summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # 2. The readable report: legend, then the three tables.
    shown = tables(summary)
    (out / "summary.md").write_text(
        f"# {title}\n\n{LEGEND}\n## Score and goals\n\n{_table(shown['score'])}"
        f"\n## Seconds per phase (mean per episode)\n\n{_table(shown['seconds'])}"
        f"\n## Tokens per phase (mean per episode, input / output)\n\n"
        f"{_table(shown['tokens'])}", encoding="utf-8")
    return out


def compute(directory: Path, title: Optional[str] = None) -> Optional[Path]:
    """Load, measure, group and write the statistics of the records in
    ``directory``; prints the score table. Returns the stats directory, or
    None when there are no records."""
    # 1. One row per episode.
    pairs = load(directory)
    if not pairs:
        print(f"No records in {directory}.")
        return None
    rows = [episode_metrics(record, evaluation) for record, evaluation in pairs]

    # 2. The groups, written and the headline table shown.
    summary = aggregate(rows)
    out = write(directory, rows, summary, title or f"Bench statistics: {directory.name}")
    print(_table(tables(summary)["score"]))
    unscored = sum(1 for r in rows if r["score"] is None)
    if unscored:
        print(f"{unscored} record(s) not scored yet: run evaluate.py on {directory}")
    print(f"written: {out}/summary.md, summary.csv, summary.json, episodes.csv")
    return out


def main() -> int:
    """The CLI: statistics of one records directory, no LLM calls."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True, help="a variant's records directory")
    args = parser.parse_args()
    return 0 if compute(Path(args.records)) else 1


if __name__ == "__main__":
    sys.exit(main())
