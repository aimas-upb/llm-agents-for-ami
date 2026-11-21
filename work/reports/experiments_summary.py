"""Every ECAI-400 structuring experiment side by side: scores, error types, time.

Reads only what the harness already wrote (`run.json`, the episode records, and
`stats/errors_episodes.csv`, `stats/errors_parts.csv`, `stats/summary.md` that
`python -m tests.homebench.error_report <run>` writes), plus the GENRE server logs kept
under `<run>/_job/`. No LLM call, standard library only:

    python -I work/reports/experiments_summary.py [results/harness dir]

Output is markdown on stdout (the tables of work/reports/experimente_complet.md).
Episodes scored as infrastructure failures are left out of every rate and every
paired comparison, as the harness does; time tables use every record.

`fep.genre-v2` is the full H2 run (job 302815, after the `filter_ids` fix).
`fep.genre-v2-partial` is the first H2 attempt (job 300721) that died after 66 of its
400 episodes; it is kept only to show the speed before and after the fix (`EXTRA`).
"""

import csv
import json
import math
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HARNESS = Path(sys.argv[1] if len(sys.argv) > 1
               else "tests/homebench/results/harness")

# key -> (folder, label). Order is the order of the tables.
REFERENCE = {
    "gpt5m": ("homebench.ecai.gpt-5-mini", "gpt-5-mini (OpenAI, every step)"),
    "gpt5n": ("homebench.ecai.gpt-5-nano", "gpt-5-nano (OpenAI, every step)"),
    "M1": ("homebench.ecai.M1", "M1 (structuring Qwen3-8B)"),
    "M2": ("homebench.ecai.M2", "M2 (structuring Qwen3-8B, planner Qwen3-4B)"),
    "M4": ("homebench.ecai.M4", "M4 (structuring DS-R1-7B, planner Qwen3-4B)"),
    "M3L": ("homebench.ecai.M3", "M3 laptop (structuring DS-R1-7B)"),
    "F": ("homebench.ecai.M3.pv-json-v2", "F json_v2 (laptop)"),
}
FEP = {
    "M3": ("fep400.M3", "M3 control"),
    "G": ("fep.trie-v1", "G iri_trie"),
    "H": ("fep.genre-v1", "H genre"),
    "G2": ("fep.trie-v2", "G2 iri_trie_v2"),
    "H2": ("fep.genre-v2", "H2 genre_v2"),
    "G2M1": ("fep.trie-v2-m1", "G2-M1 iri_trie_v2, structuring Qwen3-8B"),
    "H2M1": ("fep.genre-v2-m1", "H2-M1 genre_v2, structuring Qwen3-8B (4 shards merged)"),
}
EXTRA = {"H2p": ("fep.genre-v2-partial", "H2 first attempt (PARTIAL, slow masks)")}
RUNS = {**REFERENCE, **FEP}
ALL = {**RUNS, **EXTRA}
GROUPS = [("hb1 feasible", lambda r: r["bucket"] == "normal_single"),
          ("hb1 infeasible", lambda r: r["bucket"] == "unexist_single"),
          ("multi feasible", lambda r: r["bucket"] == "normal_multi"),
          ("mixed", lambda r: r["bucket"] == "mix_multi")]
STAGES = ["segmentation", "structuring", "planning", "execution", "other"]
RELATIVE = re.compile(
    r"\b(increase|raise|lower|decrease|reduce|dim|bump|turn (?:up|down))\b.*\bby\b"
    r"|\b(more|less)\b", re.IGNORECASE)
IRI_SLOTS = ("location_class", "artifact_class", "affordance_class")
# sacct of the cluster jobs (queried 2026-11-17, H2 2026-11-20, G2M1 / H2M1 2026-11-21): key -> (job, elapsed, state)
JOBS = {"M3": ("294585", "1:06:48", "COMPLETED"), "G": ("299621", "1:56:15", "COMPLETED"),
        "H": ("300297", "4:45:30", "COMPLETED"), "G2": ("300435", "2:10:12", "COMPLETED"),
        "H2": ("302815", "5:03:22", "COMPLETED"),
        "G2M1": ("303338", "3:18:26", "COMPLETED"),
        "H2M1": ("303360-303363", "2:37:42 + 2:58:32 + 2:47:10 + 2:49:03", "COMPLETED"),
        "H2p": ("300721", "4:33:08", "FAILED (exit 120)")}

_CACHE: dict = {}


def csv_rows(key: str, name: str) -> list:
    path = HARNESS / ALL[key][0] / "stats" / name
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def episodes(key: str) -> dict:
    return {r["episode"]: r for r in csv_rows(key, "errors_episodes.csv")}


def records(key: str) -> dict:
    """Readable episode records of a run (a 0-byte record is skipped)."""
    if key not in _CACHE:
        out = {}
        for path in (HARNESS / ALL[key][0]).glob("*.json"):
            if path.name.endswith(".eval.json") or path.name == "run.json":
                continue
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "episode" in rec:
                out[rec["episode"]] = rec
        _CACHE[key] = out
    return _CACHE[key]


def run_json(key: str) -> dict:
    return json.loads((HARNESS / ALL[key][0] / "run.json").read_text("utf-8"))


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "-"


def wilson(num: int, den: int) -> str:
    """95% Wilson interval of a proportion, in percent."""
    if not den:
        return "-"
    p, z = num / den, 1.96
    centre = (p + z * z / (2 * den)) / (1 + z * z / den)
    half = z * math.sqrt(p * (1 - p) / den + z * z / (4 * den * den)) / (1 + z * z / den)
    return f"{100 * (centre - half):.0f}-{100 * (centre + half):.0f}"


def mcnemar(gained: int, lost: int) -> str:
    """Exact two-sided McNemar p-value of the discordant pairs."""
    n = gained + lost
    if n == 0:
        return "1"
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n
    p = min(1.0, 2 * tail)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def quantile(values: list, q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def scored(eps: dict, pick=lambda r: True) -> list:
    return [k for k, r in eps.items() if r["verdict"] != "infra" and pick(r)]


def passed(eps: dict, keys: list) -> int:
    return sum(eps[k]["verdict"] == "pass" for k in keys)


def official(key: str) -> str:
    """EM / precision / recall / F1 of the `all` bucket, from stats/summary.md."""
    path = HARNESS / ALL[key][0] / "stats" / "summary.md"
    if not path.exists():
        return "- | - | - | -"
    text = path.read_text(encoding="utf-8")
    block = text.split("## Official HomeBench metrics", 1)[-1]
    m = re.search(r"^\| all \| \d+ \| (\d+)% \| (\d+)% \| (\d+)% \| (\d+)% \|", block, re.M)
    return " | ".join(m.groups()) if m else "- | - | - | -"


def inventory() -> None:
    print("\n### Runs\n")
    print("| key | folder | structuring model | decoding | workers | commit | started | "
          "records | scored | infra eps | cluster job |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, (folder, _) in ALL.items():
        rj, eps = run_json(key), episodes(key)
        steps = rj["variant"]["steps"]
        sel = (rj["variant"].get("prompt_selection") or {}).get("structuring") or {}
        decoding = sel.get("decoding", "free")
        if sel.get("context", "text") != "text":
            decoding += f" + context {sel['context']}"
        job = JOBS.get(key)
        print(f"| {key} | `{folder}` | {steps['structuring']['model']} | {decoding} | "
              f"{rj.get('workers')} | `{rj.get('commit')}` | {str(rj.get('started'))[:16]} | "
              f"{len(eps)} | {len(scored(eps))} | {len(eps) - len(scored(eps))} | "
              f"{'%s %s %s' % job if job else '-'} |")


def headline() -> None:
    print("\n### Episode pass rate (infra left out)\n")
    print("| run | scored | pass | 95% CI | " + " | ".join(g for g, _ in GROUPS)
          + " | feasible parts | infeasible parts | EM | precision | recall | F1 |")
    print("|---|---|---|---|" + "---|" * (len(GROUPS) + 6))
    for key in RUNS:
        eps = episodes(key)
        parts = csv_rows(key, "errors_parts.csv")
        keep = scored(eps)
        cells = []
        for _, pick in GROUPS:
            ks = scored(eps, pick)
            cells.append(f"{passed(eps, ks)}/{len(ks)} = {pct(passed(eps, ks), len(ks))}")
        pcell = []
        for flag in ("True", "False"):
            sel = [r for r in parts if r["feasible"] == flag
                   and eps.get(r["episode"], {}).get("verdict") != "infra"]
            pcell.append(f"{pct(sum(r['ok'] == 'True' for r in sel), len(sel))}")
        print(f"| {key} | {len(keep)} | {passed(eps, keep)} = {pct(passed(eps, keep), len(keep))}"
              f" | {wilson(passed(eps, keep), len(keep))} | " + " | ".join(cells)
              + f" | {pcell[0]} | {pcell[1]} | {official(key)} |")


def paired(eps: dict, a: str, b: str, subset=None) -> str:
    both = [k for k in eps[a] if k in eps[b] and (subset is None or k in subset)
            and eps[a][k]["verdict"] != "infra" and eps[b][k]["verdict"] != "infra"]
    gained = sum(eps[a][k]["verdict"] != "pass" and eps[b][k]["verdict"] == "pass"
                 for k in both)
    lost = sum(eps[a][k]["verdict"] == "pass" and eps[b][k]["verdict"] != "pass"
               for k in both)
    return (f"| {a} -> {b} | {len(both)} | {gained} | {lost} | {passed(eps[a], both)} | "
            f"{passed(eps[b], both)} | {mcnemar(gained, lost)} |")


def pairs_table() -> None:
    eps = {k: episodes(k) for k in RUNS}
    print("\n### Paired flips (same episode, two runs; infra left out)\n")
    print("| pair | n | gained | lost | pass A | pass B | McNemar p |")
    print("|---|---|---|---|---|---|---|")
    for a, b in (("M3", "G"), ("M3", "H"), ("M3", "G2"), ("M3", "H2"), ("G", "G2"),
                 ("H", "H2"), ("G", "H"), ("G2", "H2"), ("G", "H2"), ("M3L", "M3"),
                 ("M3L", "F"), ("M3", "F"), ("M1", "M2"), ("M3", "M4"), ("M1", "M3L"),
                 ("gpt5m", "M1"), ("gpt5m", "G2"), ("gpt5m", "H2"),
                 ("M1", "G2M1"), ("M1", "H2M1"), ("G2M1", "H2M1"), ("G2", "G2M1"),
                 ("H2", "H2M1"), ("gpt5m", "G2M1"), ("gpt5m", "H2M1")):
        print(paired(eps, a, b))


def stage_tables() -> None:
    eps = {k: episodes(k) for k in RUNS}
    parts = {k: csv_rows(k, "errors_parts.csv") for k in RUNS}
    keys = list(RUNS)
    print("\n### Failing episodes by the first stage that lost them\n")
    stages = sorted({r["primary_stage"] for e in eps.values() for r in e.values()
                     if r["verdict"] == "fail" and r["primary_stage"]})
    print("| stage | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for stage in stages:
        print(f"| {stage} | " + " | ".join(
            str(sum(r["verdict"] == "fail" and r["primary_stage"] == stage
                    for r in eps[k].values())) for k in keys) + " |")
    print("| infra (verdict) | " + " | ".join(
        str(sum(r["verdict"] == "infra" for r in eps[k].values())) for k in keys) + " |")
    print("| failing episodes | " + " | ".join(
        str(sum(r["verdict"] == "fail" for r in eps[k].values())) for k in keys) + " |")
    counters = {k: Counter(r["code"] for r in parts[k]
                           if r["ok"] != "True" and r["code"]) for k in keys}
    codes = sorted({c for c in counters.values() for c in c},
                   key=lambda c: -sum(ctr[c] for ctr in counters.values()))
    print("\n### Failing parts by error code\n")
    print("| code | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for code in codes[:40]:
        print(f"| `{code}` | " + " | ".join(str(counters[k][code]) for k in keys) + " |")
    print("| all failing parts | " + " | ".join(
        str(sum(counters[k].values())) for k in keys) + " |")
    print("| parts in scored episodes | " + " | ".join(
        str(sum(eps[k].get(r['episode'], {}).get('verdict') != 'infra' for r in parts[k]))
        for k in keys) + " |")


FAMILIES = [
    ("notation: right idea, wrong spelling", {
        "struct_class_prefix", "struct_slot_confusion", "struct_invalid_device_name",
        "struct_wrong_device_name", "struct_parameter_name", "struct_placeholder_echo",
        "struct_device_unnamed"}),
    ("structuring chose wrong (room, device, action, value)", {
        "struct_wrong_room", "struct_wrong_affordance", "struct_wrong_device_class",
        "struct_room_missing", "struct_value_missing", "struct_wrong_value"}),
    ("structuring reply unusable", {"struct_json_invalid"}),
    ("lost between structuring and plan", {
        "seg_part_lost", "plan_unrouted", "plan_explicit_failed", "plan_false_impossible",
        "plan_no_action", "plan_error_other", "plan_false_clarify"}),
    ("executed with a wrong value", {"exec_wrong_value"}),
    ("infeasible part mishandled", {
        "infeas_substituted", "infeas_executed_other", "infeas_silent",
        "struct_false_clarify_device", "struct_false_clarify_attribute",
        "struct_false_clarify_other", "struct_false_clarify_ambiguous"}),
    ("label suspect (home accepted the action)", {"infeas_accepted_on_target"}),
    ("scoring reading (turn off as brightness 0)", {"eval_turn_off_reading"}),
    ("infrastructure (timeout, LLM error, context)", {
        "infra_llm_timeout", "infra_llm_error", "infra_llm_context"}),
]


def family_table() -> None:
    keys = list(RUNS)
    counters = {k: Counter(r["code"] for r in csv_rows(k, "errors_parts.csv")
                           if r["ok"] != "True" and r["code"]) for k in keys}
    known = set().union(*(codes for _, codes in FAMILIES))
    print("\n### Failing parts by family of error\n")
    print("| family | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for name, codes in FAMILIES:
        print(f"| {name} | " + " | ".join(
            str(sum(counters[k][c] for c in codes)) for k in keys) + " |")
    other = {k: sum(v for c, v in counters[k].items() if c not in known) for k in keys}
    if any(other.values()):
        print("| other codes | " + " | ".join(str(other[k]) for k in keys) + " |")


def timing_tables() -> None:
    print("\n### Time per episode (seconds, every record; includes failed calls)\n")
    print("| run | workers | records | mean | median | p95 | max | sum (h) | "
          + " | ".join(f"{s} s" for s in STAGES) + " | LLM calls | structuring s/call |")
    print("|---|---|---|---|---|---|---|---|" + "---|" * (len(STAGES) + 2))
    for key in RUNS:
        recs = list(records(key).values())
        walls = [r["wall_seconds"] for r in recs if r.get("wall_seconds") is not None]
        stage_mean, calls, per_call = {}, [], []
        for s in STAGES:
            vals = [((r.get("meter") or {}).get("stages") or {}).get(s, {}).get("seconds", 0.0)
                    for r in recs]
            stage_mean[s] = st.mean(vals) if vals else 0.0
        for r in recs:
            cs = (r.get("meter") or {}).get("calls") or []
            calls.append(len(cs))
            per_call += [c["seconds"] for c in cs if str(c.get("step", "")).startswith("structuring")]
        print(f"| {key} | {run_json(key).get('workers')} | {len(recs)} | {st.mean(walls):.1f} | "
              f"{st.median(walls):.1f} | {quantile(walls, .95):.1f} | {max(walls):.0f} | "
              f"{sum(walls) / 3600:.2f} | " + " | ".join(f"{stage_mean[s]:.1f}" for s in STAGES)
              + f" | {st.mean(calls):.1f} | {st.mean(per_call) if per_call else 0:.1f} |")
    print("\n### Tokens per episode (mean; output includes reasoning tokens)\n")
    print("| run | total in | total out | structuring in | structuring out | "
          "structuring calls | failed LLM calls |")
    print("|---|---|---|---|---|---|---|")
    for key in RUNS:
        recs = list(records(key).values())
        tin = tout = sin = sout = scalls = failed = 0
        for r in recs:
            m = r.get("meter") or {}
            stg = m.get("stages") or {}
            tin += sum(v.get("tokens_in", 0) for v in stg.values())
            tout += sum(v.get("tokens_out", 0) for v in stg.values())
            sin += stg.get("structuring", {}).get("tokens_in", 0)
            sout += stg.get("structuring", {}).get("tokens_out", 0)
            scalls += stg.get("structuring", {}).get("calls", 0)
            failed += sum(not c.get("ok", True) for c in m.get("calls") or [])
        n = max(1, len(recs))
        print(f"| {key} | {tin / n:.0f} | {tout / n:.0f} | {sin / n:.0f} | {sout / n:.0f} | "
              f"{scalls / n:.1f} | {failed} |")


def job_hours(key: str) -> float:
    """Summed `sacct` Elapsed of the cluster job(s) of a run, in hours (0 if none)."""
    job = JOBS.get(key)
    if not job:
        return 0.0
    return sum(int(h) + int(m) / 60 + int(s) / 3600 for h, m, s in
               (t.split(":") for t in job[1].split(" + ")))


def efficiency_table() -> None:
    """Score and time of every full run in one table (pass per hour, hours per pass)."""
    print("\n### Performance and time of every full run (400 episodes)\n")
    print("| run | scored | passed | pass % | sum of episode times (h) | s / episode | "
          "hours per passed episode | passed per hour | cluster job hours | "
          "passed per job hour | structuring calls / episode | "
          "tokens in / episode | tokens out / episode |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in RUNS:
        eps, recs = episodes(key), list(records(key).values())
        keep = scored(eps)
        ok = passed(eps, keep)
        walls = [r["wall_seconds"] for r in recs if r.get("wall_seconds") is not None]
        hours = sum(walls) / 3600
        stg = [(r.get("meter") or {}).get("stages") or {} for r in recs]
        n = max(1, len(recs))
        tin = sum(v.get("tokens_in", 0) for s in stg for v in s.values()) / n
        tout = sum(v.get("tokens_out", 0) for s in stg for v in s.values()) / n
        calls = sum(s.get("structuring", {}).get("calls", 0) for s in stg) / n
        jh = job_hours(key)
        print(f"| {key} | {len(keep)} | {ok} | {pct(ok, len(keep))} | {hours:.2f} | "
              f"{hours * 3600 / n:.1f} | {hours / ok:.3f} | {ok / hours:.1f} | "
              f"{f'{jh:.2f}' if jh else '-'} | {f'{ok / jh:.1f}' if jh else '-'} | "
              f"{calls:.1f} | {tin:.0f} | {tout:.0f} |")


def same_episodes_timing() -> None:
    """H, the first H2 attempt and the full H2 over the episodes the attempt reached."""
    sub = set(records("H2p"))
    print(f"\n### Cost over the {len(sub)} episodes the first H2 attempt reached\n")
    print("| run | workers | mean wall s | structuring s | structuring s/call | "
          "failed calls | calls/episode |")
    print("|---|---|---|---|---|---|---|")
    for key in ("H", "H2p", "H2"):
        recs = [r for k, r in records(key).items() if k in sub]
        walls = [r["wall_seconds"] for r in recs]
        sec, per_call, failed, calls = [], [], 0, []
        for r in recs:
            m = r.get("meter") or {}
            sec.append((m.get("stages") or {}).get("structuring", {}).get("seconds", 0.0))
            cs = m.get("calls") or []
            calls.append(len(cs))
            per_call += [c["seconds"] for c in cs if str(c.get("step", "")).startswith("structuring")]
            failed += sum(not c.get("ok", True) for c in cs)
        print(f"| {key} | {run_json(key).get('workers')} | {st.mean(walls):.1f} | "
              f"{st.mean(sec):.1f} | {st.mean(per_call):.1f} | {failed} | {st.mean(calls):.1f} |")


GENRE_LOG = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO genre_server genre request: "
                       r"prompt (\d+) tokens, thinking (\d+), free (\d+), answer (\d+)")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ ")


def genre_logs() -> None:
    import datetime as dt
    print("\n### GENRE server logs (H, H2 first attempt, H2, H2-M1): requests, tokens, gaps, failures\n")
    print("| arm | requests answered | requests failed (client gave up, BrokenPipe) | "
          "prompt tok | thinking tok | free tok | answer tok | median gap s | p90 gap s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in ("H", "H2p", "H2", "H2M1"):
        base = HARNESS / ALL[key][0] / "_job"
        # H2M1 ran as four jobs, each with its own server log (one folder per job)
        paths = [base / "genre_server.log"] + sorted(base.glob("*/genre_server.log"))
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        rows, failed, gaps = [], 0, []
        for path in paths:
            stamps = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = GENRE_LOG.match(line)
                if m:
                    rows.append(tuple(int(x) for x in m.groups()[1:]))
                    stamps.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                elif " 500 -" in line:
                    failed += 1
            gaps += [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        cols = [st.mean(c) for c in zip(*rows)]
        print(f"| {key} | {len(rows)} | {failed} | {cols[0]:.0f} | {cols[1]:.0f} | "
              f"{cols[2]:.0f} | {cols[3]:.0f} | {st.median(gaps):.0f} | "
              f"{quantile(gaps, .9):.0f} |")


def null_table() -> None:
    print("\n### Goals the structuring wrote (every goal of every structuring call)\n")
    print("| run | goals | no room | all 3 IRI slots null | no device class | "
          "no affordance class | `set` for a relative request |")
    print("|---|---|---|---|---|---|---|")
    for key in RUNS:
        n = noroom = allnull = nodev = noaff = relset = 0
        for rec in records(key).values():
            relative = bool(RELATIVE.search(rec.get("query", "")))
            for s in rec.get("structures") or []:
                for g in ((s.get("structure") or {}).get("goals") or {}).values():
                    n += 1
                    noroom += not g.get("location_class")
                    nodev += not g.get("artifact_class")
                    noaff += not g.get("affordance_class")
                    allnull += not any(g.get(x) for x in IRI_SLOTS)
                    relset += relative and g.get("goal_effect") == "set"
        print(f"| {key} | {n} | {noroom} | {allnull} | {nodev} | {noaff} | {relset} |")


def decoding_stats() -> None:
    print("\n### What the decodings did (structuring calls with a decoding record)\n")
    for key in ("G", "G2", "H", "H2", "G2M1", "H2M1"):
        calls = n_dec = 0
        slots, how, abst = Counter(), Counter(), Counter()
        extras = Counter()
        for rec in records(key).values():
            for s in rec.get("structures") or []:
                d = s.get("decoding")
                if not d:
                    continue
                n_dec += 1
                calls += d.get("calls") or 0
                for item in d.get("interventions") or []:
                    slots[item["slot"]] += 1
                    how[item["how"]] += 1
                    if item["how"] == "abstain":
                        abst[rec["case"]] += 1
                for item in d.get("changed") or []:
                    slots[item["slot"]] += 1
                for item in d.get("abstained") or []:
                    abst[rec["case"]] += 1
                for field in ("thinking_tokens", "answer_tokens", "free_answer_tokens"):
                    extras[field] += d.get(field) or 0
                extras["thinking_closed"] += bool(d.get("thinking_closed"))
        mean = {k: round(v / n_dec) for k, v in extras.items() if k.endswith("tokens")} \
            if n_dec else {}
        print(f"- **{key}**: {n_dec} structuring calls with a record; {calls} model calls; "
              f"changed/decoded slots {sum(slots.values())} ({dict(slots.most_common(7))}); "
              f"how {dict(how)}; abstentions by case {dict(abst)}"
              + (f"; mean tokens {mean}; thinking cut and closed {extras['thinking_closed']}"
                 if key.startswith("H") else ""))


def relative_table() -> None:
    print("\n### hb1 feasible, relative requests: `goal_effect` written and pass\n")
    print("| run | request | goal_effect | n | pass |")
    print("|---|---|---|---|---|")
    for key in ("M3", "G", "H", "G2", "H2", "M1", "G2M1", "H2M1"):
        eps = episodes(key)
        tally = Counter()
        for ep, row in eps.items():
            if row["bucket"] != "normal_single" or row["verdict"] == "infra":
                continue
            rec = records(key).get(ep)
            if not rec:
                continue
            goals = [g for s in rec.get("structures") or []
                     for g in ((s.get("structure") or {}).get("goals") or {}).values()]
            if len(goals) != 1:
                continue
            kind = "relative" if RELATIVE.search(rec["query"]) else "absolute"
            tally[(kind, goals[0].get("goal_effect"), row["verdict"] == "pass")] += 1
        for kind in ("relative", "absolute"):
            for effect in ("modify", "set", None):
                total = tally[(kind, effect, True)] + tally[(kind, effect, False)]
                if total:
                    ok = tally[(kind, effect, True)]
                    print(f"| {key} | {kind} | {effect} | {total} | {ok} ({pct(ok, total)}) |")


def omitted_keys() -> None:
    """GENRE beam answers that leave an IRI key out (not `null`: absent), and what
    those episodes score. `no_null` closes the `null` exit, not the omission exit."""
    print("\n### GENRE answers that omit a room / device / action key\n")
    print("| arm | structuring calls | calls missing a key | missing room | missing device class "
          "| missing action class | key was in the free answer (any) | episodes affected | "
          "affected: pass | other episodes: pass |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in ("H", "H2", "H2M1"):
        eps = episodes(key)
        calls = bad = free_had = 0
        miss = Counter()
        hit: set = set()
        for ep, rec in records(key).items():
            for st_rec in rec.get("structures") or []:
                d = st_rec.get("decoding") or {}
                answer, free = d.get("answer") or "", d.get("free_answer") or ""
                if not answer:
                    continue
                calls += 1
                gone = [k for k in IRI_SLOTS if f'"{k}"' not in answer]
                for k in gone:
                    miss[k] += 1
                free_had += any(f'"{k}"' in free for k in gone)
                if gone:
                    bad += 1
                    hit.add(ep)
        aff = scored({k: v for k, v in eps.items() if k in hit})
        rest = scored({k: v for k, v in eps.items() if k not in hit})
        print(f"| {key} | {calls} | {bad} ({pct(bad, calls)}) | {miss['location_class']} | "
              f"{miss['artifact_class']} | {miss['affordance_class']} | {free_had} | "
              f"{len(hit)} | {passed(eps, aff)}/{len(aff)} = {pct(passed(eps, aff), len(aff))} | "
              f"{passed(eps, rest)}/{len(rest)} = {pct(passed(eps, rest), len(rest))} |")


def disagreements(keys=("M3", "G", "H", "G2", "H2"),
                  pairs=(("G2", "H2"), ("H2", "G2"))) -> None:
    """Episodes where two arms differ, with the other arms (default: G2 and H2)."""
    eps = {k: episodes(k) for k in keys}

    def cell(key: str, ep: str) -> str:
        r = eps[key].get(ep)
        if not r:
            return "-"
        return "pass" if r["verdict"] == "pass" else             f"{r['verdict']} {r['primary_code']}".strip()

    for a, b in pairs:
        rows = [ep for ep in sorted(eps[a])
                if ep in eps[b] and eps[a][ep]["verdict"] == "pass"
                and eps[b][ep]["verdict"] == "fail"]
        print(f"\n### Episodes {a} passes and {b} fails ({len(rows)})\n")
        print("| episode | case | parts | " + " | ".join(keys) + " |")
        print("|---|---|---|" + "---|" * len(keys))
        for ep in sorted(rows, key=lambda e: (eps[a][e]["case"], eps[a][e]["n_parts"], e)):
            print(f"| {ep} | {eps[a][ep]['case']} | {eps[a][ep]['n_parts']} | "
                  + " | ".join(cell(k, ep) for k in keys) + " |")


def genre_phases() -> None:
    """Why H2 is slow: the work the masks did per structuring call, and where the
    seconds went (`decoding.seconds`: thinking / free answer / beam answer)."""
    print("\n### GENRE masks and phases per structuring call (H, H2 first attempt, H2, H2-M1)\n")
    fields = ("json_steps", "json_widened", "filtered_steps", "masked_steps")
    print("| arm | calls | " + " | ".join(fields) + " | thinking s | free s | "
          "beam s (`genre`) | total s | beam median s | corr(json_widened, beam s) |")
    print("|---|---|" + "---|" * (len(fields) + 6))
    names = ("thinking", "free", "genre")
    for key in ("H", "H2p", "H2", "H2M1"):
        rows, secs = [], []
        for rec in records(key).values():
            for s in rec.get("structures") or []:
                d = s.get("decoding")
                if d and isinstance(d.get("seconds"), dict):
                    rows.append([d.get(f) or 0 for f in fields])
                    secs.append(d["seconds"])
        means = {k: st.mean(s.get(k, 0) for s in secs) for k in names}
        beam = [s.get("genre", 0) for s in secs]
        corr = st.correlation([r[1] for r in rows], beam) if len(rows) > 2 else float("nan")
        print(f"| {key} | {len(rows)} | " + " | ".join(
            f"{st.mean(c):.0f}" for c in zip(*rows)) + " | "
              + " | ".join(f"{means[k]:.1f}" for k in names)
              + f" | {sum(means.values()):.1f} | {st.median(beam):.1f} | {corr:.2f} |")


def main() -> None:
    inventory()
    headline()
    pairs_table()
    stage_tables()
    family_table()
    timing_tables()
    efficiency_table()
    same_episodes_timing()
    genre_logs()
    genre_phases()
    null_table()
    decoding_stats()
    omitted_keys()
    relative_table()
    disagreements()
    disagreements(("M1", "G2M1", "H2M1"),
                  (("M1", "G2M1"), ("G2M1", "M1"), ("G2M1", "H2M1"), ("H2M1", "G2M1"),
                   ("M1", "H2M1"), ("H2M1", "M1")))


if __name__ == "__main__":
    main()
