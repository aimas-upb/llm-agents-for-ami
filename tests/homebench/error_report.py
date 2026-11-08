"""Error report of one HomeBench run folder: which part of the pipeline loses the
requests, and why, at the granularity of one gold part.

The run folder (`tests/homebench/results/harness/<variant>`) holds one record and one
`.eval.json` per episode, plus `run.json`. `stats.py` turns them into success rates
(`stats/summary.md`); this script turns them into an *error analysis*, from the stored
data alone (no LLM call, no simulator): every failing part of every request is followed
down the pipeline to the first stage that lost it (`error_taxonomy.py`), and the
verdicts are cut every deterministic way the stored data allows.

    infrastructure -> segmentation -> structuring -> planning -> execution
                                              (and feasibility, evaluation)

Written to ``<run>/stats/`` (next to the files `stats.py` writes, which are not touched):

    errors.md            the report: tables, no plots
    errors.json          every aggregate, machine-readable (to compare two runs)
    errors_episodes.csv  one row per episode: verdict, primary stage and code, extras
    errors_parts.csv     one row per gold part: ok, stage, code, detail

The analyses (each a section of `errors.md`):

    1. verdicts, and who is to blame (pipeline / infrastructure / label or metric)
    2. failing stage and code, per episode and per part, with the part funnel
    3. the same by bucket, case, number of parts, home, request length
    4. structuring quality: format conformance of the goals, field accuracy, confusions
    5. planning: route / source x outcome, refusal reasons, outcome by goal format
    6. execution and side effects: rejected actions, extra actions, collateral changes
    7. infeasible requests: refusal, substitution, accepted-by-home (label suspects)
    8. slices of the feasible parts: device, command, value kind, room, position
    9. LLM health per step, cost of a failure, code co-occurrence, examples

An episode with no `.eval.json` (a partial run) is scored from its record and the
source episode (`tests/homebench/ecai400`) with the same deterministic scorer; the
report says how many were reconstructed. An episode missing from the folder is listed.

Usage::

    venv/Scripts/python.exe -m tests.homebench.error_report \\
        tests/homebench/results/harness/homebench.ecai.M3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tests.homebench import convert, scoring
from tests.homebench.error_taxonomy import (
    CODES, STAGES, EpisodeAnalysis, PartDiagnosis, Tools, analyze_episode,
    bare_class_fields, collect_goals, plan_for,
)
from tests.simuhome.bench import evaluate as bench_evaluate

LOGGER = logging.getLogger(__name__)

DEFAULT_BENCHMARK = Path(__file__).parent / "ecai400"
#: An LLM call that wrote this many output tokens is counted as a runaway generation.
RUNAWAY_TOKENS = 4000
#: Rows kept in the "top N" tables.
TOP = 10

#: Who is to blame for a failing episode, by the stage and code of its primary cause.
BLAME_LABEL_CODES = ("eval_turn_off_reading", "infeas_accepted_on_target")


# -- loading ---------------------------------------------------------------------------


@dataclass
class Run:
    """A run folder, read: records, evaluations and (when found) source episodes."""

    path: Path
    name: str
    variant: Dict[str, Any]
    listed: List[str]
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluations: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=dict)
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reconstructed: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """The JSON object in ``path``, None when missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("cannot read %s: %s", path, exc)
        return None


def reconstruct_evaluation(record: Dict[str, Any], episode: Dict[str, Any]
                           ) -> Dict[str, Any]:
    """The evaluation a record would get from ``evaluate.py``, for a record that has no
    ``.eval.json``: the infrastructure check first, then HomeBench's own deterministic
    rule (`scoring.evaluate_episode`) and the ``homebench`` block."""
    infra = bench_evaluate.infra_error(record)
    if infra:
        return {"score": -1, "error_type": "Infrastructure", "detail": infra}
    result = scoring.evaluate_episode(record, episode)
    result["breakdown"] = bench_evaluate.summarise_checks(result.pop("checks"))
    result["homebench"] = scoring.episode_block(record, episode, result)
    return result


def load_run(run_dir: Path, benchmark_dir: Path) -> Run:
    """Read ``run_dir``: ``run.json``, every record, and for each its evaluation (stored
    or reconstructed) and source episode."""
    meta = _read_json(run_dir / "run.json") or {}
    run = Run(run_dir, run_dir.name, meta.get("variant") or {},
              list(meta.get("episodes") or []))
    for path in sorted(run_dir.glob("hb*.json")):
        if path.name.endswith(".eval.json"):
            continue
        record = _read_json(path)
        if record is None:
            run.unreadable.append(path.name)
            continue
        name = record.get("episode") or path.stem
        run.records[name] = record
        try:
            run.sources[name] = convert.load_episode(benchmark_dir, name)
        except (OSError, ValueError, KeyError):
            LOGGER.debug("no source episode for %s", name)
        eval_path = path.with_name(path.stem + ".eval.json")
        stored = _read_json(eval_path) if eval_path.exists() else None
        if stored is None and name in run.sources:
            try:
                stored = reconstruct_evaluation(record, run.sources[name])
                run.reconstructed.append(name)
            except (ValueError, KeyError) as exc:
                LOGGER.warning("%s: cannot be scored (%s)", name, exc)
        run.evaluations[name] = stored
    return run


def analyze_run(run: Run, tools: Tools) -> List[EpisodeAnalysis]:
    """Every episode of the run analyzed (`error_taxonomy.analyze_episode`)."""
    return [analyze_episode(record, run.evaluations.get(name), run.sources.get(name),
                            tools) for name, record in run.records.items()]


# -- small helpers ---------------------------------------------------------------------


def pct(part: float, whole: float) -> Optional[float]:
    """``part / whole`` as a percentage with one decimal, None for an empty whole."""
    return round(100.0 * part / whole, 1) if whole else None


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """The ``q`` quantile (0-1) of ``values`` by nearest rank, None for none."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def mean(values: Sequence[float]) -> Optional[float]:
    """The mean of ``values`` to two decimals, None for none."""
    return round(sum(values) / len(values), 2) if values else None


def fmt(value: Any) -> str:
    """A table cell: None as ``-``, floats trimmed, the rest as text."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".") if value != int(value) else str(
            int(value))
    return str(value)


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def scored(analyses: Iterable[EpisodeAnalysis]) -> List[EpisodeAnalysis]:
    """The episodes that count in a rate: passed or failed (not infrastructure, not
    unscored), as `stats.py` leaves infrastructure failures out of every rate."""
    return [a for a in analyses if a.verdict in ("pass", "fail")]


def parts_of(analyses: Iterable[EpisodeAnalysis]
             ) -> List[Tuple[EpisodeAnalysis, PartDiagnosis]]:
    """``(episode, part)`` for every part of every scored episode."""
    return [(a, p) for a in scored(analyses) for p in a.parts]


# -- 1. verdicts and blame -------------------------------------------------------------


def overview(run: Run, analyses: Sequence[EpisodeAnalysis]) -> Dict[str, Any]:
    """Counts of the folder and of the verdicts, and the pass rate of the scored."""
    verdicts = Counter(a.verdict for a in analyses)
    done = verdicts["pass"] + verdicts["fail"]
    return {
        "run": run.name, "listed": len(run.listed), "records": len(run.records),
        "missing": sorted(set(run.listed) - set(run.records)),
        "unreadable": run.unreadable, "reconstructed": len(run.reconstructed),
        "verdicts": dict(verdicts), "scored": done,
        "pass_rate": pct(verdicts["pass"], done),
        "strict_or_lenient": pct(sum(1 for a in analyses if a.score_extra == 1), done),
        "models": {step: cfg.get("model") for step, cfg in
                   (run.variant.get("steps") or {}).items()}}


def blame(analyses: Sequence[EpisodeAnalysis]) -> Dict[str, int]:
    """Failing episodes split by whose fault the primary cause is: ``infrastructure``
    (no model answer), ``label_or_metric`` (the home accepted what the label forbids,
    or only the lenient reading passes), or ``pipeline`` (segmentation to feasibility)."""
    out: Counter = Counter()
    for a in analyses:
        if a.verdict == "infra":
            out["infrastructure"] += 1
        elif a.verdict == "fail":
            if a.primary_code in BLAME_LABEL_CODES:
                out["label_or_metric"] += 1
            elif a.primary_stage == "infrastructure":
                out["infrastructure"] += 1
            else:
                out["pipeline"] += 1
    return dict(out)


# -- 2. stages, codes, funnel ----------------------------------------------------------


def stage_codes(analyses: Sequence[EpisodeAnalysis]) -> Dict[str, Any]:
    """Failing episodes by primary stage and code, and failing parts by stage and
    code. An infrastructure episode (score -1) counts under ``infrastructure``."""
    failing = [a for a in analyses if a.verdict in ("fail", "infra")]
    episode_stage = Counter(a.primary_stage for a in failing)
    episode_code = Counter(a.primary_code for a in failing)
    part_pairs = [(a, p) for a, p in parts_of(analyses) if not p.ok and p.code]
    part_stage = Counter(p.stage for _, p in part_pairs)
    part_code = Counter(p.code for _, p in part_pairs)
    # An episode "touches" a code when any of its failing parts has it (multi-label).
    touched = Counter(c for a in failing for c in a.codes)
    return {"failing_episodes": len(failing), "failing_parts": len(part_pairs),
            "episode_stage": dict(episode_stage), "episode_code": dict(episode_code),
            "part_stage": dict(part_stage), "part_code": dict(part_code),
            "episodes_touching_code": dict(touched)}


def funnel(analyses: Sequence[EpisodeAnalysis]) -> List[Dict[str, Any]]:
    """The feasible parts' funnel: how many reached each stage without being lost
    earlier, and how many passed. A part lost at stage S reached every stage up to S."""
    feasible = [p for _, p in parts_of(analyses) if p.feasible]
    order = [s for s in STAGES if s not in ("feasibility", "evaluation")]
    rows = [{"step": "gold parts (feasible)", "parts": len(feasible)}]
    labels = {"infrastructure": "answered by the LLM (no infrastructure failure)"}
    for stage in order:
        lost_before = sum(1 for p in feasible
                          if not p.ok and p.stage in order[:order.index(stage)])
        rows.append({"step": labels.get(stage, f"reached {stage}"),
                     "parts": len(feasible) - lost_before})
    rows.append({"step": "passed", "parts": sum(1 for p in feasible if p.ok)})
    for row in rows:
        row["pct"] = pct(row["parts"], len(feasible))
    return rows


# -- 3. cuts ---------------------------------------------------------------------------


def hb_parts(a: EpisodeAnalysis) -> str:
    """``hb1`` .. ``hb10`` of an episode, by its type."""
    match = re.match(r"(hb\d+)", a.name)
    return match.group(1) if match else "?"


def words_bin(a: EpisodeAnalysis) -> str:
    """The request length in words, binned."""
    n = len(a.query.split())
    return "<=10" if n <= 10 else "11-20" if n <= 20 else "21-40" if n <= 40 else ">40"


def cut(analyses: Sequence[EpisodeAnalysis], key: Callable[[EpisodeAnalysis], Any]
        ) -> List[Dict[str, Any]]:
    """Per group of ``key``: episodes, pass / fail / infra, pass rate of the scored, and
    the primary stages of the failures."""
    groups: Dict[Any, List[EpisodeAnalysis]] = defaultdict(list)
    for a in analyses:
        groups[key(a)].append(a)
    rows = []
    for group, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        verdicts = Counter(a.verdict for a in items)
        done = verdicts["pass"] + verdicts["fail"]
        stages = Counter(a.primary_stage for a in items if a.verdict == "fail")
        rows.append({"group": str(group), "n": len(items), "pass": verdicts["pass"],
                     "fail": verdicts["fail"], "infra": verdicts["infra"],
                     "pass_rate": pct(verdicts["pass"], done),
                     "stages": dict(stages)})
    return rows


def worst(rows: List[Dict[str, Any]], minimum: int = 5, top: int = TOP
          ) -> List[Dict[str, Any]]:
    """The groups with the lowest pass rate among those with ``minimum`` scored
    episodes."""
    eligible = [r for r in rows if r["pass"] + r["fail"] >= minimum
                and r["pass_rate"] is not None]
    return sorted(eligible, key=lambda r: (r["pass_rate"], -r["n"]))[:top]


def segmentation_relation(analyses: Sequence[EpisodeAnalysis]) -> List[Dict[str, Any]]:
    """Pass rate by how the number of intents compares with the number of gold parts:
    fewer (parts merged or dropped), equal, more (a part split)."""
    def relation(a: EpisodeAnalysis) -> str:
        if not a.n_parts:
            return "?"
        return "fewer" if a.n_intents < a.n_parts else (
            "equal" if a.n_intents == a.n_parts else "more")
    return cut(scored(analyses), relation)


# -- 4. structuring --------------------------------------------------------------------


def goal_format(run: Run) -> Dict[str, Any]:
    """Format conformance of every structured goal in the run, and the planner's
    outcome for it: goals with all class fields prefixed (``homeont:`` / ``saref:``)
    against goals with a bare one, named against unnamed (``artifact_name`` null),
    ``parameter_name`` set against NA; and how often each field is null / NA."""
    outcome: Dict[str, Counter] = defaultdict(Counter)
    nulls: Counter = Counter()
    total = 0
    for record in run.records.values():
        goals, _ = collect_goals(record)
        used: set = set()
        for ref in goals:
            total += 1
            plan = plan_for(record, ref.text, used)
            result = (plan or {}).get("outcome") or "no plan entry"
            goal = ref.goal
            key = ("bare class field" if bare_class_fields(goal)
                   else "all prefixed / NA")
            outcome[f"class format: {key}"][result] += 1
            named = goal.get("artifact_name") not in (None, "", "NA", "None")
            outcome["artifact_name: " + ("given" if named else "null")][result] += 1
            param = goal.get("parameter_name") not in (None, "NA", "None", "")
            outcome["parameter_name: " + ("filled" if param else "NA/null")][result] += 1
            for name, value in goal.items():
                if value in (None, "NA", "None", ""):
                    nulls[name] += 1
    rows = []
    for group, counts in sorted(outcome.items()):
        n = sum(counts.values())
        rows.append({"group": group, "goals": n, "plan": counts.get("plan", 0),
                     "plan_rate": pct(counts.get("plan", 0), n),
                     "outcomes": dict(counts)})
    return {"goals": total, "by_format": rows,
            "null_rate": {k: pct(v, total) for k, v in nulls.most_common()}}


def field_accuracy(analyses: Sequence[EpisodeAnalysis]) -> Dict[str, Any]:
    """Of the feasible parts linked to a goal, the share whose goal differs from the
    gold on each field, from the diffs stored on failing parts; and the commonest
    (expected, got) pairs per field."""
    linked = [(a, p) for a, p in parts_of(analyses) if p.feasible and p.linked]
    diffs: Counter = Counter()
    pairs: Dict[str, Counter] = defaultdict(Counter)
    for _, part in linked:
        for fld, (expected, got) in (part.detail.get("diffs") or {}).items():
            diffs[fld] += 1
            pairs[fld][(str(expected), str(got))] += 1
    return {"linked_parts": len(linked),
            "diff_rate": {f: pct(n, len(linked)) for f, n in diffs.most_common()},
            "confusions": {f: [{"expected": e, "got": g, "n": n}
                               for (e, g), n in c.most_common(TOP)]
                           for f, c in pairs.items()}}


# -- 5. planning -----------------------------------------------------------------------

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def plan_stats(run: Run) -> Dict[str, Any]:
    """Plan entries by (source, route) x outcome, and the commonest refusal texts."""
    table: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    for record in run.records.values():
        for plan in record.get("plans") or []:
            key = (str(plan.get("source")), str(plan.get("route")))
            outcome = plan.get("outcome") or "?"
            table[key][outcome] += 1
            text = plan.get("explanation") or ""
            if outcome in ("impossible", "error") and text:
                reasons[_NUMBER.sub("N", _QUOTED.sub("'X'", text))[:110]] += 1
            elif outcome == "error":
                reasons[f"error: {plan.get('error')}"] += 1
    return {"by_route": [{"source": s, "route": r, "n": sum(c.values()),
                          "outcomes": dict(c)} for (s, r), c in sorted(table.items())],
            "reasons": [{"text": t, "n": n} for t, n in reasons.most_common(TOP * 2)]}


# -- 6. execution and side effects -----------------------------------------------------


def side_effects(run: Run, analyses: Sequence[EpisodeAnalysis]) -> Dict[str, Any]:
    """Actions the home refused, actions beyond the gold, and how many *passing*
    episodes also did something the request did not ask for."""
    refused = Counter()
    for record in run.records.values():
        for action in (record.get("execution") or {}).get("actions") or []:
            if not action.get("success"):
                refused[str(action.get("status_code"))] += 1
    done = scored(analyses)
    extras = [a for a in done if a.extra_actions]
    passing_extras = [a for a in extras if a.verdict == "pass"]
    top = Counter(t for a in extras for t in a.extra_actions)
    return {"refused_actions": dict(refused),
            "episodes_with_extra_actions": len(extras),
            "passing_with_extra_actions": len(passing_extras),
            "pct_of_pass_with_extra": pct(
                len(passing_extras), sum(1 for a in done if a.verdict == "pass")),
            "extra_actions_total": sum(len(a.extra_actions) for a in done),
            "top_extra_actions": [{"token": t, "n": n} for t, n in top.most_common(TOP)]}


# -- 7. infeasible parts ---------------------------------------------------------------


def infeasible(analyses: Sequence[EpisodeAnalysis]) -> Dict[str, Any]:
    """Infeasible parts (gold ``error_input``): refused or not, by reason, and what was
    done instead; label suspects are the ones the home accepted."""
    rows = [(a, p) for a, p in parts_of(analyses) if not p.feasible]
    by_reason: Dict[str, Counter] = defaultdict(Counter)
    for _, part in rows:
        by_reason[part.reason or "unknown"]["ok" if part.ok else (part.code or "?")] += 1
    suspects = [{"episode": a.name, "query": a.query[:140], "gold": p.token,
                 "sent": p.detail.get("sent"), "reason": p.reason}
                for a, p in rows if p.label_suspect]
    return {"parts": len(rows), "refused": sum(1 for _, p in rows if p.ok),
            "by_reason": {k: dict(v) for k, v in sorted(by_reason.items())},
            "label_suspects": suspects}


# -- 8. slices of the feasible parts ---------------------------------------------------


def value_kind(part: PartDiagnosis) -> str:
    """``onoff``, ``numeric`` or ``enum`` / ``none``, from the gold token."""
    command = part.command or ""
    if command in ("turn_on", "turn_off"):
        return "onoff"
    match = re.search(r"\(([^)]*)\)", part.token)
    arg = match.group(1) if match else ""
    if not arg:
        return "none"
    return "numeric" if re.fullmatch(r"-?\d+(?:\.\d+)?", arg) else "enum"


def part_cut(analyses: Sequence[EpisodeAnalysis], key: Callable[
        [EpisodeAnalysis, PartDiagnosis], Any], feasible: bool = True
             ) -> List[Dict[str, Any]]:
    """Per group of ``key``: parts, parts passed and the pass rate, and the commonest
    failing code, over the feasible (or infeasible) parts of the scored episodes."""
    groups: Dict[Any, List[PartDiagnosis]] = defaultdict(list)
    for a, p in parts_of(analyses):
        if p.feasible == feasible:
            groups[key(a, p)].append(p)
    rows = []
    for group, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ok = sum(1 for p in items if p.ok)
        codes = Counter(p.code for p in items if not p.ok and p.code)
        top = codes.most_common(1)
        rows.append({"group": str(group), "parts": len(items), "ok": ok,
                     "pass_rate": pct(ok, len(items)),
                     "top_code": top[0][0] if top else None,
                     "top_code_n": top[0][1] if top else 0})
    return rows


def position_decay(analyses: Sequence[EpisodeAnalysis]) -> List[Dict[str, Any]]:
    """Pass rate of the feasible parts by their position in the request (1 = first),
    over requests with at least 4 parts: whether later parts fail more."""
    by_pos: Dict[int, List[bool]] = defaultdict(list)
    for a in scored(analyses):
        if a.n_parts >= 4:
            for p in a.parts:
                if p.feasible:
                    by_pos[p.index + 1].append(p.ok)
    return [{"position": pos, "parts": len(v), "pass_rate": pct(sum(v), len(v))}
            for pos, v in sorted(by_pos.items())]


def partial_credit(analyses: Sequence[EpisodeAnalysis]) -> Dict[str, Any]:
    """Failing episodes by how many parts they got wrong: 1 (a near miss), 2, 3+, and
    the mean share of parts right."""
    failing = [a for a in analyses if a.verdict == "fail" and a.parts]
    wrong = Counter(min(sum(1 for p in a.parts if not p.ok), 3) for a in failing)
    shares = [sum(1 for p in a.parts if p.ok) / len(a.parts) for a in failing]
    return {"failing": len(failing),
            "wrong_parts": {("3+" if k == 3 else str(k)): v
                            for k, v in sorted(wrong.items())},
            "mean_parts_right_pct": pct(sum(shares), len(shares))}


# -- 9. LLM health, cost, co-occurrence --------------------------------------------------


def llm_health(run: Run, runaway: int) -> List[Dict[str, Any]]:
    """Per LLM step: calls, failed calls, seconds and output tokens (median, p95) and
    the calls that wrote ``runaway`` tokens or more."""
    steps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in run.records.values():
        for call in (record.get("meter") or {}).get("calls") or []:
            steps[call.get("step") or "?"].append(call)
    rows = []
    for step, calls in sorted(steps.items()):
        seconds = [c["seconds"] for c in calls if c.get("seconds") is not None]
        out = [c["tokens_out"] for c in calls if c.get("tokens_out") is not None]
        failed = sum(1 for c in calls if not c.get("ok"))
        rows.append({"step": step, "calls": len(calls), "failed": failed,
                     "failed_pct": pct(failed, len(calls)),
                     "s_median": quantile(seconds, 0.5), "s_p95": quantile(seconds, 0.95),
                     "out_median": quantile(out, 0.5), "out_p95": quantile(out, 0.95),
                     "runaway": sum(1 for t in out if t >= runaway)})
    return rows


def failure_cost(run: Run, analyses: Sequence[EpisodeAnalysis]) -> List[Dict[str, Any]]:
    """Mean wall seconds and LLM tokens per episode, by verdict and by the primary stage
    of a failure: whether failing is also expensive."""
    def tokens(name: str) -> Tuple[int, int]:
        calls = (run.records[name].get("meter") or {}).get("calls") or []
        return (sum(c.get("tokens_in") or 0 for c in calls),
                sum(c.get("tokens_out") or 0 for c in calls))
    groups: Dict[str, List[EpisodeAnalysis]] = defaultdict(list)
    for a in analyses:
        if a.verdict == "pass":
            groups["pass"].append(a)
        elif a.verdict in ("fail", "infra"):
            groups[f"{a.verdict}: {a.primary_stage}"].append(a)
    return [{"group": g, "n": len(items),
             "wall_s": mean([a.wall_seconds for a in items if a.wall_seconds]),
             "tokens_in": mean([tokens(a.name)[0] for a in items]),
             "tokens_out": mean([tokens(a.name)[1] for a in items])}
            for g, items in sorted(groups.items())]


def cooccurrence(analyses: Sequence[EpisodeAnalysis]) -> List[Dict[str, Any]]:
    """The commonest pairs of codes in one failing episode (several parts failing for
    different reasons)."""
    pairs: Counter = Counter()
    for a in analyses:
        if a.verdict == "fail" and len(a.codes) > 1:
            for i, first in enumerate(a.codes):
                for second in a.codes[i + 1:]:
                    pairs[(first, second)] += 1
    return [{"codes": f"{x} + {y}", "n": n} for (x, y), n in pairs.most_common(TOP)]


def examples(analyses: Sequence[EpisodeAnalysis], per_code: int
             ) -> Dict[str, List[Dict[str, Any]]]:
    """Up to ``per_code`` failing parts per code: the request, the gold and the stored
    detail, for reading the cases behind a number."""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a, p in parts_of(analyses):
        if not p.ok and p.code and len(out[p.code]) < per_code:
            out[p.code].append({"episode": a.name, "query": a.query[:200],
                                "gold": p.token, "detail": p.detail})
    return dict(out)


# -- assembling ------------------------------------------------------------------------


def build_report(run: Run, analyses: Sequence[EpisodeAnalysis], per_code: int,
                 runaway: int) -> Dict[str, Any]:
    """Every aggregate of the report, as plain data (the content of ``errors.json``)."""
    bucket = cut(analyses, lambda a: a.bucket or "?")
    return {
        "overview": overview(run, analyses), "blame": blame(analyses),
        "stage_codes": stage_codes(analyses), "funnel": funnel(analyses),
        "by_bucket": bucket, "by_case": cut(analyses, lambda a: a.case),
        "by_parts": cut(analyses, hb_parts), "by_home": cut(analyses, lambda a: a.home),
        "by_length": cut(analyses, words_bin),
        "by_label_corrected": cut(analyses, lambda a: a.label_corrected),
        "by_percent_relative": cut(analyses, lambda a: a.percent_relative),
        "segmentation_relation": segmentation_relation(analyses),
        "goal_format": goal_format(run), "field_accuracy": field_accuracy(analyses),
        "plans": plan_stats(run), "side_effects": side_effects(run, analyses),
        "infeasible": infeasible(analyses),
        "by_device": part_cut(analyses, lambda a, p: p.device_type),
        "by_command": part_cut(analyses, lambda a, p: f"{p.device_type}.{p.command}"),
        "by_value_kind": part_cut(analyses, lambda a, p: value_kind(p)),
        "by_room": part_cut(analyses, lambda a, p: p.room or "household"),
        "infeasible_by_device": part_cut(analyses, lambda a, p: p.device_type,
                                         feasible=False),
        "position_decay": position_decay(analyses),
        "partial_credit": partial_credit(analyses),
        "llm_health": llm_health(run, runaway),
        "failure_cost": failure_cost(run, analyses),
        "cooccurrence": cooccurrence(analyses),
        "examples": examples(analyses, per_code)}


def _cut_table(rows: List[Dict[str, Any]], title: str) -> str:
    """A pass-rate table of ``cut`` rows with the stage mix of the failures."""
    body = [[r["group"], r["n"], r["pass"], r["fail"], r["infra"], r["pass_rate"],
             ", ".join(f"{s} {n}" for s, n in sorted(
                 r["stages"].items(), key=lambda kv: STAGES.index(kv[0])
                 if kv[0] in STAGES else 99))] for r in rows]
    return md_table([title, "n", "pass", "fail", "infra", "pass %", "failing stages"],
                    body)


def _part_table(rows: List[Dict[str, Any]], title: str, top: int = 0,
                minimum: int = 5) -> str:
    """A table of ``part_cut`` rows (all, or the ``top`` lowest pass rates among the
    groups with at least ``minimum`` parts)."""
    if top:
        rows = sorted((r for r in rows if r["parts"] >= minimum),
                      key=lambda r: (r["pass_rate"] or 0, -r["parts"]))[:top]
    return md_table([title, "parts", "ok", "pass %", "commonest code", "n"],
                    [[r["group"], r["parts"], r["ok"], r["pass_rate"], r["top_code"],
                      r["top_code_n"]] for r in rows])


def render(data: Dict[str, Any]) -> str:
    """The report as markdown (the content of ``errors.md``)."""
    ov, sc = data["overview"], data["stage_codes"]
    out = [f"# Error analysis: {ov['run']}", ""]
    out.append(
        "Deterministic, from the stored records only (no LLM call). A *part* is one "
        "gold token of a request; each failing part is followed down the pipeline to "
        "the first stage that lost it. A failing episode's *primary stage* is the "
        "earliest stage any of its parts failed in. Rates leave out infrastructure "
        "failures (score -1), as `summary.md` does.\n")
    out += ["## 1. Verdicts and blame", "",
            md_table(["metric", "value"], [
                ["episodes listed in run.json", ov["listed"]],
                ["records in the folder", ov["records"]],
                ["missing from the folder", len(ov["missing"])],
                ["scored from the record (no .eval.json)", ov["reconstructed"]],
                *[[f"verdict: {k}", v] for k, v in sorted(ov["verdicts"].items())],
                ["pass rate of scored (%)", ov["pass_rate"]],
                ["pass rate counting the lenient turn-off reading (%)",
                 ov["strict_or_lenient"]]]), ""]
    if ov["models"]:
        out += ["Models: " + ", ".join(f"{s}={m}" for s, m in ov["models"].items()), ""]
    total = sum(data["blame"].values())
    out += ["Whose fault the failing episodes are (primary cause):", "",
            md_table(["blame", "episodes", "% of failing"],
                     [[k, v, pct(v, total)] for k, v in sorted(
                         data["blame"].items(), key=lambda kv: -kv[1])]), ""]
    out += ["## 2. Where the pipeline loses requests", ""]
    out.append(md_table(
        ["stage", "failing episodes", "% of failing", "failing parts", "% of parts"],
        [[s, sc["episode_stage"].get(s, 0),
          pct(sc["episode_stage"].get(s, 0), sc["failing_episodes"]),
          sc["part_stage"].get(s, 0), pct(sc["part_stage"].get(s, 0), sc["failing_parts"])]
         for s in STAGES if sc["episode_stage"].get(s) or sc["part_stage"].get(s)]))
    out += ["", "By code (episodes: primary code; touching: any failing part has it; "
            "parts: failing parts):", ""]
    codes = sorted(set(sc["episode_code"]) | set(sc["part_code"]),
                   key=lambda c: (-sc["part_code"].get(c, 0), c))
    out.append(md_table(
        ["code", "stage", "episodes (primary)", "episodes touching", "parts", "% of parts",
         "meaning"],
        [[c, CODES[c][0], sc["episode_code"].get(c, 0),
          sc["episodes_touching_code"].get(c, 0), sc["part_code"].get(c, 0),
          pct(sc["part_code"].get(c, 0), sc["failing_parts"]), CODES[c][1]]
         for c in codes]))
    out += ["", "Funnel of the feasible parts (a part lost at a stage still reached it):",
            "", md_table(["step", "parts", "%"],
                         [[r["step"], r["parts"], r["pct"]] for r in data["funnel"]]), ""]
    out += ["## 3. The same, cut by request properties", "",
            _cut_table(data["by_bucket"], "bucket (official)"), "",
            _cut_table(data["by_case"], "case"), "",
            _cut_table(data["by_parts"], "parts in request"), "",
            _cut_table(data["by_length"], "request words"), "",
            "Intents compared with gold parts (fewer = parts merged or dropped):", "",
            _cut_table(data["segmentation_relation"], "intents vs parts"), "",
            _cut_table(data["by_label_corrected"], "label corrected (_fixed)"), "",
            _cut_table(data["by_percent_relative"], "'by N percent' request"), "",
            "Ten worst homes (at least 5 scored episodes):", "",
            _cut_table(worst(data["by_home"]), "home"), ""]
    out += _render_structuring(data) + _render_planning(data)
    out += _render_execution(data) + _render_slices(data) + _render_health(data)
    out += _render_examples(data)
    out += ["## Method and caveats", "",
            "* A feasible part passes when `.eval.json`'s check for it is ok under the "
            "gold's own reading; an infeasible part passes when a refusal covers it "
            "(refusals are pooled, as the scorer does; parts whose own goal was refused "
            "come first). Reason `unknown` is the converter's own label for an "
            "infeasible part whose room and device it could not locate.",
            "* A part is linked to the structured goal that fits it best (device id, "
            "room, device class, action; greedy, one goal per part). Leftover goals are "
            "paired with leftover parts in order, and the field diffs then show it. Parts "
            "of a request that names one device twice can be mislinked.",
            "* Field diffs compare normalized values (prefix and case dropped, on/off as "
            "true/false). A relative goal (`modify`) is not compared on value: its value "
            "is computed at planning time.",
            "* A planner refusal is blamed on structuring when the goal's own format "
            "explains it: bare class names (`Foyer` for `homeont:Foyer`), a filled "
            "`parameter_name`, a null `artifact_name`. Section 4's cross-tab shows the "
            "planner outcome by those formats, which is the evidence for the rule.",
            "* `infeas_accepted_on_target` on an `unexist_attribute` part means the home "
            "accepted the action: the label contradicts the home (label suspect).", ""]
    return "\n".join(out)


def _render_structuring(data: Dict[str, Any]) -> List[str]:
    """Section 4."""
    gf, fa = data["goal_format"], data["field_accuracy"]
    out = ["## 4. Structuring quality", "",
           f"Format conformance of the {gf['goals']} structured goals, and what the "
           "planner did with each:", "",
           md_table(["goal format", "goals", "plan produced", "plan %", "outcomes"],
                    [[r["group"], r["goals"], r["plan"], r["plan_rate"],
                      json.dumps(r["outcomes"])] for r in gf["by_format"]]), "",
           "Share of goals with each field null / NA (%): "
           + ", ".join(f"{k} {v}" for k, v in list(gf["null_rate"].items())[:12]), "",
           f"Field differences from the gold over {fa['linked_parts']} linked feasible "
           "parts (% of linked parts):", "",
           md_table(["field", "% of parts"], [[k, v] for k, v in fa["diff_rate"].items()]),
           ""]
    for fld, rows in fa["confusions"].items():
        if rows:
            out += [f"Commonest confusions, `{fld}` (expected -> got):", "",
                    md_table(["expected", "got", "n"],
                             [[r["expected"], r["got"], r["n"]] for r in rows[:6]]), ""]
    return out


def _render_planning(data: Dict[str, Any]) -> List[str]:
    """Section 5."""
    ps = data["plans"]
    return ["## 5. Planning", "",
            md_table(["source", "route", "entries", "outcomes"],
                     [[r["source"], r["route"], r["n"], json.dumps(r["outcomes"])]
                      for r in ps["by_route"]]), "",
            "Commonest refusal / error texts (quoted names masked):", "",
            md_table(["text", "n"], [[r["text"], r["n"]] for r in ps["reasons"]]), ""]


def _render_execution(data: Dict[str, Any]) -> List[str]:
    """Sections 6 and 7."""
    se, inf = data["side_effects"], data["infeasible"]
    out = ["## 6. Execution and side effects", "",
           md_table(["metric", "value"], [
               ["actions the home refused (by status)", json.dumps(se["refused_actions"])],
               ["episodes with actions beyond the gold", se["episodes_with_extra_actions"]],
               ["... of them passing", se["passing_with_extra_actions"]],
               ["% of passing episodes with extra actions", se["pct_of_pass_with_extra"]],
               ["extra actions in total", se["extra_actions_total"]]]), "",
           md_table(["commonest extra action", "n"],
                    [[r["token"], r["n"]] for r in se["top_extra_actions"]]), "",
           "## 7. Infeasible requests (gold `error_input`)", "",
           f"{inf['parts']} infeasible parts, {inf['refused']} covered by a refusal.", "",
           md_table(["reason", "outcomes"],
                    [[k, json.dumps(v)] for k, v in inf["by_reason"].items()]), ""]
    if inf["label_suspects"]:
        out += [f"Label suspects ({len(inf['label_suspects'])}): the home accepted an "
                "action on the device the label says cannot do it:", "",
                md_table(["episode", "request", "gold", "sent"],
                         [[s["episode"], s["query"], s["gold"], s["sent"]]
                          for s in inf["label_suspects"][:TOP]]), ""]
    return out


def _render_slices(data: Dict[str, Any]) -> List[str]:
    """Section 8."""
    pc = data["partial_credit"]
    return ["## 8. Slices of the feasible parts", "",
            _part_table(data["by_device"], "device type"), "",
            "Ten weakest commands (at least 5 parts):", "",
            _part_table(data["by_command"], "command", top=TOP), "",
            _part_table(data["by_value_kind"], "value kind"), "",
            _part_table(data["by_room"], "room"), "",
            "Position in a request of 4 or more parts:", "",
            md_table(["position", "parts", "pass %"],
                     [[r["position"], r["parts"], r["pass_rate"]]
                      for r in data["position_decay"]]), "",
            f"Failing episodes by number of wrong parts: {json.dumps(pc['wrong_parts'])}; "
            f"mean share of parts right {pc['mean_parts_right_pct']} %.", ""]


def _render_health(data: Dict[str, Any]) -> List[str]:
    """Section 9 (tables)."""
    return ["## 9. LLM health, cost of a failure, co-occurrence", "",
            md_table(["step", "calls", "failed", "failed %", "s median", "s p95",
                      "out tok median", "out tok p95", "runaway"],
                     [[r["step"], r["calls"], r["failed"], r["failed_pct"], r["s_median"],
                       r["s_p95"], r["out_median"], r["out_p95"], r["runaway"]]
                      for r in data["llm_health"]]), "",
            md_table(["verdict / primary stage", "episodes", "wall s", "tokens in",
                      "tokens out"],
                     [[r["group"], r["n"], r["wall_s"], r["tokens_in"], r["tokens_out"]]
                      for r in data["failure_cost"]]), "",
            "Codes that fail together in one episode:", "",
            md_table(["codes", "episodes"],
                     [[r["codes"], r["n"]] for r in data["cooccurrence"]]), ""]


def _render_examples(data: Dict[str, Any]) -> List[str]:
    """The examples section."""
    out = ["## Examples per code", ""]
    for code, items in sorted(data["examples"].items()):
        out += [f"### {code}", ""]
        for item in items:
            out.append(f"* `{item['episode']}`: {item['query']}  \n  gold "
                       f"`{item['gold']}`; {json.dumps(item['detail'])[:300]}")
        out.append("")
    return out


# -- files -----------------------------------------------------------------------------


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """Write ``rows`` under ``header`` to ``path`` (UTF-8, ``\\n`` line ends)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(out: Path, run: Run, analyses: Sequence[EpisodeAnalysis],
                  data: Dict[str, Any]) -> List[Path]:
    """Write the four files into ``out`` and return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "errors.md", out / "errors.json", out / "errors_episodes.csv",
             out / "errors_parts.csv"]
    paths[0].write_text(render(data), encoding="utf-8")
    paths[1].write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    write_csv(paths[2], [
        "episode", "verdict", "case", "type", "bucket", "home", "n_parts", "parts_ok",
        "primary_stage", "primary_code", "codes", "extra_actions", "n_intents", "n_goals",
        "n_dead_structures", "label_corrected", "score", "score_extra", "wall_s", "note"],
        [[a.name, a.verdict, a.case, a.htype, a.bucket, a.home, a.n_parts,
          sum(1 for p in a.parts if p.ok), a.primary_stage, a.primary_code,
          " ".join(a.codes), " ".join(a.extra_actions), a.n_intents, a.n_goals, a.n_dead,
          a.label_corrected, a.score, a.score_extra, a.wall_seconds, a.note]
         for a in analyses])
    write_csv(paths[3], [
        "episode", "part", "token", "feasible", "reason", "device_type", "room",
        "command", "ok", "stage", "code", "linked", "label_suspect", "detail"],
        [[a.name, p.index, p.token, p.feasible, p.reason, p.device_type, p.room,
          p.command, p.ok, p.stage, p.code, p.linked, p.label_suspect,
          json.dumps(p.detail, ensure_ascii=False, default=str)]
         for a in analyses for p in a.parts])
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: analyze one run folder and write its error report."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="a run folder, e.g. "
                        "tests/homebench/results/harness/homebench.ecai.M3")
    parser.add_argument("--episodes", type=Path, default=DEFAULT_BENCHMARK,
                        help="the benchmark folder with episodes/ and homes/ (default: "
                        "tests/homebench/ecai400)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: <run_dir>/stats)")
    parser.add_argument("--examples", type=int, default=3,
                        help="examples kept per code (default 3)")
    parser.add_argument("--runaway-tokens", type=int, default=RUNAWAY_TOKENS,
                        help="output tokens that make a call a runaway generation")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.run_dir.is_dir():
        parser.error(f"{args.run_dir} is not a folder")
    run = load_run(args.run_dir, args.episodes)
    if not run.records:
        parser.error(f"no episode records (hb*.json) in {args.run_dir}")
    analyses = analyze_run(run, Tools.load())
    data = build_report(run, analyses, args.examples, args.runaway_tokens)
    paths = write_outputs(args.out or args.run_dir / "stats", run, analyses, data)
    ov = data["overview"]
    print(f"{run.name}: {ov['records']} records, verdicts {ov['verdicts']}, "
          f"pass rate {ov['pass_rate']} %")
    print("blame:", data["blame"])
    print("written: " + ", ".join(p.name for p in paths) + f" in {paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
