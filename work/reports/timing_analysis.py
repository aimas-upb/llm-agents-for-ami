"""Where the time goes in the partial local-model HomeBench runs.

Reads the per-episode records under tests/homebench/results/harness/<run>/ and the
Ollama server log, and prints markdown tables: per-run overview, the M1 time
partition, the episodes every run finished, timeouts by goal route, queueing,
model-load cost.

    python -I work/reports/timing_analysis.py

Standard library only. Numbers are over what each run had finished when this was
run (the runs are partial, see the report). The Ollama part is skipped when its
log or the `ollama` command is not there.
"""

import datetime as dt
import glob
import json
import math
import os
import re
import statistics as st
import subprocess
from collections import Counter, defaultdict

BASE = os.path.join("tests", "homebench", "results", "harness")
RUNS = {
    "M1": "homebench.ecai.M1",
    "M3": "homebench.ecai.M3",
    "coder-7b": "homebench.ecai.qwen2.5-coder-7b",
    "qwen3-8b": "homebench.ecai.qwen3-8b",
}
MODELS = ["ds-r1-1.5b-16k", "qwen3-8b-16k", "qwen2.5-coder-7b-16k", "ds-r1-7b-16k"]
HUNG_EXEC_S = 300.0        # an `execution` stage longer than this is a simulator hang
STAGES = ["segmentation", "structuring", "planning", "profiling", "guarding",
          "response", "execution", "other"]
# The routing rule of ami_agents/agents/user_assistant/utils/goal_routing.py
DEPENDENCY_LABELS = {"logical_dependency", "temporal_dependency", "sequential_dependency"}
SPECIFICITY_LABELS = {"explicit", "incomplete", "ambiguous"}
KIND_LABELS = {"achievement", "maintenance"}


def load(run_dir):
    """Finished episode records of one run, keyed by episode id (+ file mtime)."""
    out = {}
    for path in glob.glob(os.path.join(BASE, run_dir, "*.json")):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if "episode" in rec and rec.get("wall_seconds") is not None:
            rec["_mtime"] = os.path.getmtime(path)
            out[rec["episode"]] = rec
    return out


def stage_seconds(rec):
    stages = (rec.get("meter") or {}).get("stages") or {}
    return {s: stages.get(s, {}).get("seconds", 0.0) for s in STAGES}


def calls(rec, prefix=None):
    cs = (rec.get("meter") or {}).get("calls") or []
    return [c for c in cs if prefix is None or c["step"].startswith(prefix)]


def timed_out(rec):
    return any(not c.get("ok", True) for c in calls(rec))


def hung(rec):
    return stage_seconds(rec)["execution"] > HUNG_EXEC_S


def route(qualifiers):
    labels = {str(q).strip().lower() for q in (qualifiers or [])}
    spec, kind = labels & SPECIFICITY_LABELS, labels & KIND_LABELS
    if len(spec) == 1 and kind == {"achievement"} and not labels & DEPENDENCY_LABELS:
        return "simple"
    return "dependency"


def why_dependency(qualifiers):
    labels = {str(q).strip().lower() for q in (qualifiers or [])}
    spec, kind = labels & SPECIFICITY_LABELS, labels & KIND_LABELS
    if not labels:
        return "qualifiers empty"
    if not kind and len(spec) == 1 and labels == spec:
        return "specificity only, no kind label"
    if labels & DEPENDENCY_LABELS:
        return "a dependency label"
    if kind == {"maintenance"}:
        return "maintenance"
    if route(qualifiers) == "simple":
        return "(simple route)"
    return "other ill-formed"


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for the table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    total = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(total, k)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(n1, k) + 1) if p(x) <= p0 * (1 + 1e-9))


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def overview(data):
    print("### Per run\n")
    rows = []
    for name, eps in data.items():
        w = [r["wall_seconds"] for r in eps.values()]
        h = [r for r in eps.values() if hung(r)]
        rows.append([name, len(w), f"{sum(w) / 3600:.2f}", f"{st.mean(w):.0f}",
                     f"{st.median(w):.0f}", f"{pct(w, .9):.0f}", f"{max(w):.0f}",
                     len(h), f"{sum(r['wall_seconds'] for r in h) / 3600:.2f}"])
    table(["run", "episodes", "total h", "mean s", "median s", "p90 s", "max s",
           "simulator hangs", "hang h"], rows)


def partition_m1(eps):
    print("### M1: where its wall time went (all finished episodes)\n")
    parts = defaultdict(float)
    for r in eps.values():
        s = stage_seconds(r)
        parts["simulator hang (execution stage of the hung episodes)"] += s["execution"] if hung(r) else 0
        key = ("structuring, episodes with a timed-out call" if timed_out(r)
               else "structuring, other episodes")
        parts[key] += s["structuring"]
        parts["segmentation"] += s["segmentation"]
        parts["planning, profiling, guarding, response, other, execution (normal)"] += (
            s["planning"] + s["profiling"] + s["guarding"] + s["response"] + s["other"]
            + (0 if hung(r) else s["execution"]))
        parts["unaccounted (harness, discovery, between stages)"] += (
            r["wall_seconds"] - sum(s.values()))
    total = sum(r["wall_seconds"] for r in eps.values())
    rows = [[k, f"{v / 3600:.2f}", f"{100 * v / total:.1f}%"]
            for k, v in sorted(parts.items(), key=lambda kv: -kv[1])]
    rows.append(["total", f"{total / 3600:.2f}", "100%"])
    table(["part", "hours", "share"], rows)
    ok = [r for r in eps.values() if not hung(r)]
    w = [r["wall_seconds"] for r in ok]
    print(f"Without the {len(eps) - len(ok)} hung episodes: {len(ok)} episodes, "
          f"mean {st.mean(w):.0f} s, median {st.median(w):.0f} s, "
          f"{len(ok) and sum(w) / 3600:.2f} h; at that mean 400 episodes = "
          f"{400 * st.mean(w) / 3600:.1f} h.\n")


def by_stage(data):
    print("### Seconds per episode by stage (all finished episodes)\n")
    rows = []
    for name, eps in data.items():
        tot = defaultdict(float)
        for r in eps.values():
            for s, v in stage_seconds(r).items():
                tot[s] += v
        rows.append([name] + [f"{tot[s] / len(eps):.1f}" for s in STAGES]
                    + [f"{sum(r['wall_seconds'] for r in eps.values()) / len(eps):.1f}"])
    table(["run"] + STAGES + ["wall"], rows)


def matched(data):
    ids = sorted(set.intersection(*[set(e) for e in data.values()]))
    print(f"### The {len(ids)} episodes every run finished\n")
    rows = []
    for name, eps in data.items():
        sub = [eps[i] for i in ids]
        w = [r["wall_seconds"] for r in sub]
        cs = [c for r in sub for c in calls(r, "structuring")]
        bad = [c for c in cs if not c.get("ok", True)]
        rows.append([name, f"{st.mean(w):.1f}", f"{st.median(w):.1f}",
                     f"{sum(w) / 60:.1f}", len(cs), len(bad),
                     f"{sum(r['wall_seconds'] for r in sub if timed_out(r)) / 60:.1f}",
                     f"{sum(c['tokens_in'] for c in cs if c.get('ok', True)) / len(sub):.0f}",
                     f"{sum(c['tokens_out'] for c in cs if c.get('ok', True)) / len(sub):.0f}"])
    table(["run", "mean s/ep", "median s/ep", "total min", "structuring calls",
           "timed out", "min in episodes with a timeout", "tok in/ep (ok)", "tok out/ep (ok)"],
          rows)
    clean = [i for i in ids if not any(timed_out(data[n][i]) for n in ("M1", "M3"))]
    print(f"Only the {len(clean)} of these episodes where neither M1 nor M3 had a timed-out call:\n")
    rows = []
    for name, eps in data.items():
        w = [eps[i]["wall_seconds"] for i in clean]
        st_s = [stage_seconds(eps[i])["structuring"] for i in clean]
        seg = [stage_seconds(eps[i])["segmentation"] for i in clean]
        rows.append([name, f"{st.mean(w):.1f}", f"{st.median(w):.1f}",
                     f"{st.mean(seg):.1f}", f"{st.mean(st_s):.1f}"])
    table(["run", "mean s/ep", "median s/ep", "segmentation s/ep", "structuring s/ep"], rows)
    m1 = [c for i in ids for c in calls(data["M1"][i], "structuring")]
    q3 = [c for i in ids for c in calls(data["qwen3-8b"][i], "structuring")]
    a, b = sum(not c.get("ok", True) for c in m1), sum(c.get("ok", True) for c in m1)
    c_, d = sum(not c.get("ok", True) for c in q3), sum(c.get("ok", True) for c in q3)
    print(f"Structuring calls on these episodes, same model (Qwen3-8B): M1 {a} timed out / {b} ok, "
          f"qwen3-8b {c_} / {d}; Fisher exact two-sided p = {fisher_two_sided(a, b, c_, d):.4f}\n")


def routing(data):
    print("### Why the structuring prompt differs: the segmenter's qualifiers decide the route\n")
    rows = []
    for name, eps in data.items():
        reasons = Counter()
        for r in eps.values():
            for i in r.get("intents") or []:
                if isinstance(i, dict):
                    reasons[why_dependency(i.get("qualifiers"))] += 1
        n = sum(reasons.values())
        simple = reasons.pop("(simple route)", 0)
        rows.append([name, n, f"{simple} ({100 * simple / n:.0f}%)"]
                    + [f"{reasons[k]} ({100 * reasons[k] / n:.0f}%)" for k in
                       ("specificity only, no kind label", "qualifiers empty",
                        "a dependency label", "maintenance", "other ill-formed")])
    table(["run", "intents", "simple route", "dependency: specificity only, no kind",
           "dependency: qualifiers empty", "dependency: dependency label",
           "dependency: maintenance", "dependency: other"], rows)
    print("Structuring calls by route (call and intent matched by order and text; "
          "seconds are for calls that did not time out):\n")
    groups = defaultdict(list)
    for name, eps in data.items():
        for r in eps.values():
            steps = [s for s in r["meter"]["steps"] if s["step"] == "structuring.intent"]
            cs = calls(r, "structuring.intent")
            by_text = {i["text"].strip().rstrip(".").lower(): i
                       for i in (r.get("intents") or []) if isinstance(i, dict)}
            if len(steps) != len(cs):
                continue
            for s, c in zip(steps, cs):
                it = by_text.get(s["text"].strip().rstrip(".").lower())
                if it is not None:
                    groups[(name, route(it.get("qualifiers")))].append(c)
    rows = []
    for (name, rt), cs in sorted(groups.items()):
        ok = [c for c in cs if c.get("ok", True)]
        bad = len(cs) - len(ok)
        rows.append([name, rt, len(cs), bad, f"{100 * bad / len(cs):.0f}%",
                     f"{st.mean(c['seconds'] for c in ok):.1f}" if ok else "-",
                     f"{st.mean(c['tokens_in'] for c in ok):.0f}" if ok else "-",
                     f"{st.mean(c['tokens_out'] for c in ok):.0f}" if ok else "-"])
    table(["run", "route", "calls", "timed out", "rate", "mean s (ok)", "mean tok in (ok)",
           "mean tok out (ok)"], rows)


def queueing(data):
    print("### Calls per episode and timeouts (the server runs one request at a time)\n")
    for name in ("M1", "coder-7b", "qwen3-8b"):
        by_k = defaultdict(lambda: [0, 0, 0])
        for r in data[name].values():
            cs = calls(r, "structuring")
            k = len(cs)
            by_k[k][0] += 1
            by_k[k][1] += k
            by_k[k][2] += sum(not c.get("ok", True) for c in cs)
        print(f"{name}: " + "; ".join(f"{k} call{'s' if k != 1 else ''}: {e} ep, {n} calls, "
                                       f"{b} timed out" for k, (e, n, b) in sorted(by_k.items())) + "\n")
    rows = []
    for name, eps in data.items():
        step_s = sum(stage_seconds(r)["structuring"] for r in eps.values())
        call_s = sum(c["seconds"] for r in eps.values() for c in calls(r, "structuring"))
        rows.append([name, f"{step_s / 3600:.2f}", f"{call_s / 3600:.2f}", f"{call_s / step_s:.2f}"])
    table(["run", "structuring wall h", "sum of call seconds h", "ratio (>1: calls overlap)"],
          rows)


def single_calls(data):
    print("### Cost of one structuring call when the episode has a single call "
          "(no queue), by model\n")
    rows = []
    for name, eps in data.items():
        g = defaultdict(list)
        for r in eps.values():
            cs = calls(r, "structuring")
            if len(cs) == 1 and cs[0].get("ok", True):
                g[cs[0]["model"]].append(cs[0])
        for m, cs in g.items():
            rows.append([name, m, len(cs), f"{st.mean(c['tokens_in'] for c in cs):.0f}",
                         f"{st.mean(c['tokens_out'] for c in cs):.0f}",
                         f"{st.median(c['seconds'] for c in cs):.1f}",
                         f"{st.mean(c['seconds'] for c in cs):.1f}",
                         f"{pct([c['seconds'] for c in cs], .9):.1f}"])
    table(["run", "model", "calls", "mean tok in", "mean tok out", "median s", "mean s",
           "p90 s"], rows)


def ollama_log(data):
    print("### Ollama server log\n")
    path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log")
    if not os.path.exists(path):
        print("server.log not found, skipped.\n")
        return
    blob = {}
    try:
        for m in MODELS:
            out = subprocess.run(["ollama", "show", m, "--modelfile"], capture_output=True,
                                 encoding="utf-8", errors="replace", timeout=30).stdout
            sha = re.search(r"sha256-([0-9a-f]{64})", out or "")
            if sha:
                blob[sha.group(1)] = m.replace("-16k", "")
    except (OSError, subprocess.SubprocessError):
        pass
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()
    first = next((l for l in lines if "server config" in l), "")
    cfg = {k: v for k, v in re.findall(r"(OLLAMA_[A-Z_]+):(\S+)", first)}
    print("Server config: " + ", ".join(f"{k}={cfg.get(k)}" for k in (
        "OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_FLASH_ATTENTION",
        "OLLAMA_KV_CACHE_TYPE", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE")) + "\n")
    tre = re.compile(r"^time=(\S+?)[+-]\d\d:\d\d ")
    loads, cur = [], None
    for l in lines:
        m = tre.match(l)
        t = dt.datetime.fromisoformat(m.group(1)) if m else None
        if t and "starting llama-server" in l:
            sha = re.search(r"sha256-([0-9a-f]{64})", l)
            cur = {"t": t, "model": blob.get(sha.group(1), "?") if sha else "?", "ready": None}
            loads.append(cur)
        elif cur and t and 'msg="loaded runners"' in l and cur["ready"] is None:
            cur["ready"] = t
    rows = []
    for name, eps in data.items():
        a = dt.datetime.fromtimestamp(min(r["_mtime"] - r["wall_seconds"] for r in eps.values()))
        b = dt.datetime.fromtimestamp(max(r["_mtime"] for r in eps.values()))
        ls = [x for x in loads if a <= x["t"] <= b and x["ready"]]
        secs = [(x["ready"] - x["t"]).total_seconds() for x in ls]
        per = defaultdict(list)
        for x, s in zip(ls, secs):
            per[x["model"]].append(s)
        wall = sum(r["wall_seconds"] for r in eps.values())
        rows.append([name, len(ls), f"{len(ls) / len(eps):.2f}",
                     "; ".join(f"{m} {st.mean(v):.1f}s x{len(v)}" for m, v in sorted(per.items())),
                     f"{sum(secs) / 60:.1f}", f"{sum(secs) / len(eps):.1f}",
                     f"{100 * sum(secs) / wall:.1f}%"])
    table(["run", "model loads", "loads/episode", "mean load (start -> ready)", "total min",
           "s/episode", "share of run"], rows)
    a = dt.datetime.fromtimestamp(min(r["_mtime"] - r["wall_seconds"] for r in data["M1"].values()))
    b = dt.datetime.fromtimestamp(max(r["_mtime"] for r in data["M1"].values()))
    last_ngen, cutoffs, codes = None, [], Counter()
    for l in lines:
        m = re.search(r"n_gen =\s+(\d+), tg =\s+([\d.]+) t/s", l)
        if m:
            last_ngen = (int(m.group(1)), float(m.group(2)))
        g = re.match(r"\[GIN\] (\d{4})/(\d\d)/(\d\d) - (\d\d):(\d\d):(\d\d) \| (\d+) \|\s*(\S+) \|"
                     r".*chat/completions", l)
        if g and a <= dt.datetime(*map(int, g.groups()[:6])) <= b:
            codes[(int(g.group(7)))] += 1
            if g.group(7) == "500":
                cutoffs.append((g.group(8), last_ngen))
    print(f"M1 window, /v1/chat/completions status codes: {dict(codes)}; "
          f"all 500s lasted {sorted(set(c[0] for c in cutoffs))}.")
    gens = [c[1][0] for c in cutoffs if c[1]]
    rate = [c[1][1] for c in cutoffs if c[1]]
    if gens:
        print(f"At the cut the model was still generating: tokens generated so far "
              f"median {st.median(gens):.0f} (min {min(gens)}, max {max(gens)}), "
              f"decode speed median {st.median(rate):.0f} tok/s; "
              f"{sum(g > 5000 for g in gens)} of {len(gens)} cuts had more than 5000 "
              f"tokens generated (the others were mostly calls waiting in the queue).\n")


def main():
    data = {name: load(d) for name, d in RUNS.items()}
    overview(data)
    partition_m1(data["M1"])
    by_stage(data)
    matched(data)
    routing(data)
    queueing(data)
    single_calls(data)
    ollama_log(data)


if __name__ == "__main__":
    main()
