"""Tables for work/reports/structurare_explicat.md, read from the ECAI-400 fep runs.

Reads only what the harness already wrote (`stats/errors_episodes.csv`,
`stats/errors_parts.csv`, the episode records); no LLM call, stdlib only:

    python -I work/reports/structuring_analysis.py [results/harness dir]

Runs (folder -> arm): fep400.M3 = M3, fep.trie-v1 = G, fep.genre-v1 = H,
fep.trie-v2 = G2. Episodes scored as infrastructure failures are left out of every
rate and of every paired comparison, as the harness does.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HARNESS = Path(sys.argv[1] if len(sys.argv) > 1
               else "tests/homebench/results/harness")
RUNS = {"M3": "fep400.M3", "G": "fep.trie-v1", "H": "fep.genre-v1",
        "G2": "fep.trie-v2"}
RELATIVE = re.compile(
    r"\b(increase|raise|lower|decrease|reduce|dim|bump|turn (?:up|down))\b.*\bby\b"
    r"|\b(more|less)\b", re.IGNORECASE)


def rows(run: str, name: str) -> list:
    with open(HARNESS / RUNS[run] / "stats" / name, encoding="utf-8",
              newline="") as handle:
        return list(csv.DictReader(handle))


def episodes(run: str) -> dict:
    return {r["episode"]: r for r in rows(run, "errors_episodes.csv")}


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "-"


def verdict_table(eps: dict) -> None:
    print("\n### Episode verdicts (infra left out)\n")
    print("| group | n | " + " | ".join(f"{k} pass" for k in RUNS) + " |")
    print("|---|---|" + "---|" * len(RUNS))
    groups = [("all", lambda r: True),
              ("hb1 feasible", lambda r: r["bucket"] == "normal_single"),
              ("hb1 infeasible", lambda r: r["case"] == "infeasible"),
              ("multi feasible", lambda r: r["bucket"] == "normal_multi"),
              ("mixed", lambda r: r["case"] == "mixed")]
    for name, pick in groups:
        cells, n = [], None
        for run in RUNS:
            data = eps[run]
            keep = [k for k, r in data.items() if pick(r) and r["verdict"] != "infra"]
            n = n or len(keep)
            cells.append(f"{sum(data[k]['verdict'] == 'pass' for k in keep)}/{len(keep)}"
                         f" = {pct(sum(data[k]['verdict'] == 'pass' for k in keep), len(keep))}")
        print(f"| {name} | {n} | " + " | ".join(cells) + " |")


def buckets(eps: dict) -> None:
    print("\n### Buckets present\n")
    print(Counter((r["type"], r["case"], r["bucket"]) for r in eps["M3"].values()))


def paired(eps: dict, a: str, b: str) -> None:
    both = [k for k in eps[a] if k in eps[b]
            and eps[a][k]["verdict"] != "infra" and eps[b][k]["verdict"] != "infra"]
    gained = [k for k in both if eps[a][k]["verdict"] != "pass" and eps[b][k]["verdict"] == "pass"]
    lost = [k for k in both if eps[a][k]["verdict"] == "pass" and eps[b][k]["verdict"] != "pass"]
    print(f"| {a} -> {b} | {len(both)} | {len(gained)} | {len(lost)} |"
          f" {sum(eps[a][k]['verdict'] == 'pass' for k in both)} |"
          f" {sum(eps[b][k]['verdict'] == 'pass' for k in both)} |")


def part_codes(parts: dict) -> None:
    print("\n### Failing parts by code (feasible and infeasible parts together)\n")
    counters = {run: Counter(r["code"] for r in data
                             if r["ok"] != "True" and r["code"])
                for run, data in parts.items()}
    codes = sorted({c for ctr in counters.values() for c in ctr},
                   key=lambda c: -max(ctr[c] for ctr in counters.values()))
    print("| code | " + " | ".join(RUNS) + " |")
    print("|---|" + "---|" * len(RUNS))
    for code in codes[:26]:
        print(f"| `{code}` | " + " | ".join(str(counters[r][code]) for r in RUNS) + " |")
    print("| total failing | " + " | ".join(str(sum(counters[r].values())) for r in RUNS) + " |")
    print("\n### Parts passed\n")
    print("| | " + " | ".join(RUNS) + " |")
    print("|---|" + "---|" * len(RUNS))
    for label, flag in (("feasible parts", "True"), ("infeasible parts", "False")):
        cells = []
        for run in RUNS:
            sel = [r for r in parts[run] if r["feasible"] == flag]
            cells.append(f"{sum(r['ok'] == 'True' for r in sel)}/{len(sel)}"
                         f" = {pct(sum(r['ok'] == 'True' for r in sel), len(sel))}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\n### Failing parts by stage\n")
    stages = Counter()
    for run in RUNS:
        for r in parts[run]:
            if r["ok"] != "True" and r["stage"]:
                stages[(r["stage"], run)] += 1
    names = sorted({s for s, _ in stages})
    print("| stage | " + " | ".join(RUNS) + " |")
    print("|---|" + "---|" * len(RUNS))
    for stage in names:
        print(f"| {stage} | " + " | ".join(str(stages[(stage, r)]) for r in RUNS) + " |")


def record(run: str, episode: str) -> dict:
    with open(HARNESS / RUNS[run] / f"{episode}.json", encoding="utf-8") as handle:
        return json.load(handle)


def decoding_stats(eps: dict) -> None:
    print("\n### What the decodings did (per structuring call in the records)\n")
    for run in ("G", "G2", "H"):
        calls = slots = changed = 0
        by_slot, kinds = Counter(), Counter()
        abstain_cases, n_dec = Counter(), 0
        for episode in eps[run]:
            rec = record(run, episode)
            for structure in rec.get("structures") or []:
                dec = structure.get("decoding")
                if not dec:
                    continue
                n_dec += 1
                for item in dec.get("interventions") or []:
                    slots += 1
                    by_slot[item["slot"]] += 1
                    kinds[item["how"]] += 1
                    if item["how"] == "abstain":
                        abstain_cases[rec["case"]] += 1
                calls += dec.get("calls") or 0
                for change in dec.get("changed") or []:
                    changed += 1
                    by_slot[change["slot"]] += 1
                for item in dec.get("abstained") or []:
                    abstain_cases[rec["case"]] += 1
        print(f"- **{run}**: {n_dec} structuring calls with a decoding record; "
              f"{calls} model calls; re-decoded/changed values: "
              f"{slots or changed} (by slot {dict(by_slot)}; by how {dict(kinds)}); "
              f"abstentions by episode case {dict(abstain_cases)}")


def relative_requests(eps: dict) -> None:
    print("\n### hb1 feasible: relative requests, goal_effect written, pass\n")
    print("| run | request | goal_effect | n | pass |")
    print("|---|---|---|---|---|")
    for run in RUNS:
        tally = Counter()
        for episode, row in eps[run].items():
            if row["bucket"] != "normal_single" or row["verdict"] == "infra":
                continue
            rec = record(run, episode)
            goals = [g for s in rec.get("structures") or []
                     for g in ((s.get("structure") or {}).get("goals") or {}).values()]
            if len(goals) != 1:
                continue
            kind = "relative" if RELATIVE.search(rec["query"]) else "absolute"
            effect = goals[0].get("goal_effect")
            tally[(kind, effect, row["verdict"] == "pass")] += 1
        for kind in ("relative", "absolute"):
            for effect in ("modify", "set", None):
                n = tally[(kind, effect, True)] + tally[(kind, effect, False)]
                if n:
                    print(f"| {run} | {kind} | {effect} | {n} | "
                          f"{tally[(kind, effect, True)]} ({pct(tally[(kind, effect, True)], n)}) |")


def main() -> None:
    eps = {run: episodes(run) for run in RUNS}
    parts = {run: rows(run, "errors_parts.csv") for run in RUNS}
    buckets(eps)
    verdict_table(eps)
    print("\n### Paired flips (infra left out)\n")
    print("| pair | n | gained | lost | pass A | pass B |")
    print("|---|---|---|---|---|---|")
    for a, b in (("M3", "G"), ("M3", "H"), ("M3", "G2"), ("G", "G2"), ("H", "G2"),
                 ("G", "H")):
        paired(eps, a, b)
    part_codes(parts)
    decoding_stats(eps)
    relative_requests(eps)
    room_swaps(eps)




def room_swaps(eps: dict) -> None:
    """Episodes where the decoding replaced a room the free reply wrote correctly (the
    name of a room of the home, only the `homeont:` prefix or spacing differing from
    its trie value) by another room, and how they ended. A room counts as one of the
    home when the trie, with a single candidate, ever wrote it."""
    def key(text):
        return re.sub(r"[^a-z0-9]", "", (text or "").split(":")[-1].lower())

    print("\n### A valid room replaced by another room (location_class)\n")
    for run in ("G", "G2"):
        known = set()
        records = {}
        for episode in eps[run]:
            records[episode] = record(run, episode)
            for structure in records[episode].get("structures") or []:
                for item in (structure.get("decoding") or {}).get("interventions") or []:
                    if item["slot"] == "location_class" and item["how"] == "trie" \
                            and item["candidates"] == 1 and item["decoded"]:
                        known.add(key(item["decoded"]))
        swapped = {}
        for episode, rec in records.items():
            for structure in rec.get("structures") or []:
                for item in (structure.get("decoding") or {}).get("interventions") or []:
                    if item["slot"] == "location_class" and item["decoded"] \
                            and key(item["free"]) in known \
                            and key(item["free"]) != key(item["decoded"]):
                        swapped.setdefault(episode, []).append(item)
        table = Counter((eps[run][e]["case"], eps[run][e]["verdict"],
                         "evidence" if any(i.get("evidence") for i in items)
                         else "no evidence") for e, items in swapped.items())
        codes = Counter(eps[run][e]["primary_code"] for e in swapped
                        if eps[run][e]["verdict"] == "fail")
        print(f"- **{run}**: {len(swapped)} episodes; (case, verdict, evidence used): "
              f"{dict(sorted(table.items()))}; primary code of the failing ones: "
              f"{dict(codes.most_common(6))}")
        if run == "G2":
            lost = [e for e in swapped if eps["G"].get(e, {}).get("verdict") == "pass"
                    and eps[run][e]["verdict"] == "fail"]
            print(f"  of them passed in G and fail in G2: {len(lost)}")


if __name__ == "__main__":
    main()
