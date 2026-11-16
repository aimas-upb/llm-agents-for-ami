"""Decoding report of one HomeBench episode: how the trie (`decoding: iri_trie`) decoded
the structuring reply, step by step, for reading and for showing how the method works.

`prompt_report.py` shows what every LLM call receives; this shows what the trie
decoding decided between those calls. It runs ONE episode the way `prompt_report` does
(the harness's command line, stopped after structuring by default), so the folder holds
the prompt report too, then draws the record's `structures[].decoding.trace`
(`iri_trie.TRACE_EVENTS`) as `decoding.md`:

    conda run -n ami-agents python -m tests.homebench.decoding_report \\
        --config tests/homebench/variants/homebench.ecai.M3.pv-trie.yaml \\
        --episode hb1_feasible_seed_4351 [--stop-after structuring] \\
        [--out tests/homebench/results/decoding]

    --record <episode.json>   draw an episode record that is already on disk, running
                              nothing (a harness run of the trie arm, or a report's
                              run/ folder); [--out-file <page.md>]

What `decoding.md` holds, for each structuring reply the trie decoded:

    timeline     every decision in order: the free reply (call 1), each value kept, each
                 value outside its trie, the characters forced, each constrained step with
                 the token taken and its rank, the closing token, the free continuation
    per value    the candidates the value was decoded with, as a prefix tree (the value
                 decoded marked), why those (the slots before it, the later slots used as
                 evidence), the end of the reply the model saw, and per step the 20 tokens
                 the call returned: logprob, probability, allowed by the trie or not, and
                 which was taken; the gap between the model's top token and the one taken
                 (how much the trie overruled the model)
    diff         the free reply against the decoded one

Example, `hb1_feasible_seed_4351` ("Set the air conditioner temperature to 20 degrees in
the guest bedroom"): the free reply writes `affordance_class: homeont:TemperatureSetpoint`,
which the air conditioner does not have; the later `affordance_name: temperature` leaves
one candidate, `saref:SetAbsoluteLevelCommand`, written whole (forced, no step call), then
one call for its closing token and one for the rest of the reply.

A record written before the trace existed has only its `interventions`: they are listed,
with a note that the steps need a run with the current code. Ollama is local, so a run
of an M config costs nothing (about 15 s and ~7 calls of structuring per episode).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ami_agents.agents.user_assistant.utils.iri_trie import THINK_END, IriTrie
from tests.homebench import prompt_report
from tests.homebench.prompt_report import block
from tests.simuhome.bench import run_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "tests" / "homebench" / "results" / "decoding"

# How the method works, said once at the top of every page (the team reads the page,
# not the code).
METHOD = """\
How the trie decoding works (`ami_agents/agents/user_assistant/utils/iri_trie.py`):

1. **Free reply (call 1).** The model answers exactly as without the trie (the control).
2. **Check.** Each value of a slot that names something of the home (`location_class`,
   `artifact_name`, `affordance_class`, ...) is looked up in the trie of the values the
   home allows there, given the slots written before it in the same object (a device of
   the written room, an action of the written device). A value in its trie is **kept**:
   constrained greedy decoding would have written the same.
3. **Re-decode the first invalid value.** The reply is cut at the value's opening quote.
   Its candidates are narrowed by the slots the model wrote *after* it (evidence). Where
   the trie has one branch, characters are **forced** with no call. At each branch, one
   call (the reply so far prefilled as the assistant's message, 1 token, top 20 logprobs)
   and the most probable token the trie allows is **taken**.
4. **Close and continue.** The value is closed with the model's own closing token
   (`",`), then the model continues freely from it, and the check goes on (step 2).
"""


# -- formatting ---------------------------------------------------------------

def tok(text: Optional[str]) -> str:
    """A token or value as an inline code span a table can hold: JSON-quoted so spaces
    and newlines show, pipes escaped."""
    if text is None:
        return "*(none)*"
    shown = json.dumps(text, ensure_ascii=False).replace("|", "\\|")
    return f"`` {shown} ``" if "`" in shown else f"`{shown}`"


def prob(logprob: float) -> str:
    """A logprob as a percentage."""
    return f"{100 * math.exp(logprob):.1f}%"


def slots_text(slots: Dict[str, str]) -> str:
    """Slot values as `slot = value`, comma separated; "none" when empty."""
    return ", ".join(f"`{k}` = {tok(v)}" for k, v in slots.items()) or "none"


# -- the candidates as a prefix tree ------------------------------------------

def _trie_nodes(values: List[str]) -> Dict[str, Any]:
    """A character trie of ``values`` (a complete value marked by the key None)."""
    root: Dict[str, Any] = {}
    for value in values:
        node = root
        for char in value:
            node = node.setdefault(char, {})
        node[None] = {}
    return root


def _compressed(node: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """The children of ``node`` with every one-branch chain merged into one label."""
    out = []
    for char, child in node.items():
        if char is None:
            continue
        label = char
        while len(child) == 1 and None not in child:
            [(next_char, child)] = child.items()
            label += next_char
        out.append((label, child))
    return out


def prefix_tree(values: List[str], decoded: Optional[str]) -> str:
    """The candidates as a prefix tree, one branch per line: where the trie branches is
    where the decoding needs a call; the value decoded is marked."""
    lines: List[str] = []

    def walk(node: Dict[str, Any], indent: str, written: str) -> None:
        # 1. Each merged branch, drawn as a tree; a value that ends here is marked.
        children = _compressed(node)
        for i, (label, child) in enumerate(children):
            last = i == len(children) - 1
            value = written + label
            mark = ""
            if None in child:
                mark = "   <== decoded" if value == decoded else "   (a value)"
            lines.append(f"{indent}{'`-- ' if last else '|-- '}{label}{mark}")
            # 2. Below it, its own branches.
            walk(child, indent + ("    " if last else "|   "), value)

    walk(_trie_nodes(values), "", "")
    return "\n".join(lines)


# -- one constrained step -----------------------------------------------------

def verdict(trie: IriTrie, prefix: str, token: str) -> str:
    """Whether the trie allows ``token`` after ``prefix``, and why not when it does not."""
    if trie.accept(prefix, token) is not None:
        return "allowed"
    if '"' in token:
        return "no: closes an incomplete value"
    return "no: leaves the trie"


def step_table(event: Dict[str, Any], trie: IriTrie) -> List[str]:
    """The tokens one call returned, with what the trie made of each."""
    lines = ["| rank | token | logprob | prob | trie | |", "|---|---|---|---|---|---|"]
    for rank, (token, logprob) in enumerate(event["top"]):
        taken = "**taken**" if rank == event["rank"] else ""
        lines.append(f"| {rank + 1} | {tok(token)} | {logprob:.3f} | {prob(logprob)} | "
                     f"{verdict(trie, event['prefix'], token)} | {taken} |")
    return lines


def overrule_line(event: Dict[str, Any]) -> str:
    """How far the trie overruled the model at a step: the model's top token against the
    one taken (the signal an abstention rule would threshold)."""
    top = event["top"]
    if not top:
        return "The call returned no tokens."
    if event["rank"] is None:
        return (f"None of the {len(top)} tokens is allowed: the value goes to the "
                "`nearest` fallback.")
    if event["rank"] == 0:
        return f"The model's own top token {tok(top[0][0])} is allowed: no overruling."
    taken, best = top[event["rank"]], top[0]
    return (f"The model preferred {tok(best[0])} ({prob(best[1])}); the trie took "
            f"{tok(taken[0])} ({prob(taken[1])}), rank {event['rank'] + 1}: a gap of "
            f"{best[1] - taken[1]:.2f} nats.")


# -- the page -----------------------------------------------------------------

def call_ref(number: Optional[int], pages: Dict[int, str]) -> str:
    """A decoding call's number, linked to its prompt-report page when there is one."""
    if number is None:
        return "-"
    return f"[{number}]({pages[number]})" if number in pages else str(number)


def after_think(text: str) -> str:
    """A reply without the thinking some models write in the content."""
    end = text.rfind(THINK_END)
    return text[end + len(THINK_END):].lstrip("\n") if end >= 0 else text


def timeline(trace: List[Dict[str, Any]], pages: Dict[int, str]) -> List[str]:
    """Every decision in one table, in order (module docstring)."""
    lines = ["| call | what | detail |", "|---|---|---|"]
    for event in trace:
        kind, ref = event["event"], call_ref(event.get("call"), pages)
        # 1. What the free decoding wrote: kept, or found outside its trie.
        if kind == "free":
            lines.append(f"| {ref} | free reply | {len(event['text']):,} chars |")
        elif kind == "kept":
            lines.append(f"| - | kept | `{event['slot']}` = {tok(event['value'])} |")
        elif kind == "invalid":
            lines.append(f"| - | **outside the trie** | `{event['slot']}` = "
                         f"{tok(event['free'])}; {len(event['allowed'])} candidate(s)"
                         + (f", evidence {slots_text(event['evidence'])}"
                            if event["evidence"] else "") + " |")
        # 2. How the value was decoded again, or why it was not.
        elif kind == "abstain":
            lines.append(f"| - | **abstain** | the room has no {tok(event['words'])}: "
                         "the goal is rewritten as `unmatched_device`, nothing decoded |")
        elif kind == "forced":
            lines.append(f"| - | forced | {tok(event['chars'])} (one branch, no call) |")
        elif kind == "step":
            top = event["top"]
            if event["rank"] is None:
                detail = f"no allowed token among {len(top)}"
            else:
                detail = (f"took {tok(top[event['rank']][0])}, rank {event['rank'] + 1}"
                          f" of {len(top)}" + ("" if event["rank"] == 0 else
                                                f"; the model's top was {tok(top[0][0])}"))
            lines.append(f"| {ref} | step after {tok(event['prefix'])} | {detail} |")
        elif kind == "nearest":
            lines.append(f"| {ref} | nearest | the model meant {tok(event['meant'])}, "
                         f"snapped to {tok(event['value'])} |")
        elif kind == "close":
            how = "with the last step's token" if event["call"] is None else \
                "the model's own closing token"
            lines.append(f"| {ref} | close | {tok(event['token'])} ({how}) |")
        # 3. The free continuation, and the end.
        elif kind == "resume":
            lines.append(f"| {ref} | free continuation | {len(event['text']):,} chars |")
        elif kind == "done":
            lines.append(f"| - | done | {len(event['text']):,} chars |")
    return lines


def intervention_section(number: int, events: List[Dict[str, Any]],
                         decoded: Optional[str], pages: Dict[int, str]) -> List[str]:
    """One re-decoded value in detail: its ``events`` from `invalid` to `resume`."""
    invalid = events[0]
    trie = IriTrie(invalid["allowed"])
    # 1. Why this value, and what it may become.
    lines = [f"### Value {number}: `{invalid['slot']}`", "",
             f"The free decoding wrote {tok(invalid['free'])}, not among the "
             f"{len(invalid['allowed'])} value(s) the trie allows here.", "",
             f"- written before it in this object: {slots_text(invalid['before'])}",
             f"- written after it by the model, used as evidence: "
             f"{slots_text(invalid['evidence'])}",
             f"- candidates {'narrowed by the object' if invalid['conditioned'] else 'of the whole home (no path fits the object)'}:",
             "", block(prefix_tree(invalid["allowed"], decoded)), "",
             "The end of the reply the model is given (the value is written right after "
             "it):", "", block("..." + invalid["tail"]), ""]
    # 2. Each event that decoded it.
    for event in events[1:]:
        kind = event["event"]
        if kind == "abstain":
            lines += [f"**Abstained**: {tok(invalid['free'])} is a device kind of the "
                      f"ontology the room does not have, so the trie's nearest device "
                      f"is not put in its place; the goal is rewritten with the IRI "
                      f"slots null and `unmatched_device` = {tok(event['words'])}.", ""]
        elif kind == "forced":
            lines += [f"**Forced** {tok(event['chars'])}: the trie has one branch there, "
                      "so it is written with no call.", ""]
        elif kind == "step":
            lines += [f"**Step, call {call_ref(event['call'], pages)}**, value so far "
                      f"{tok(event['prefix'])}:", "", *step_table(event, trie), "",
                      overrule_line(event), ""]
        elif kind == "nearest":
            lines += [f"**Nearest** (call {call_ref(event.get('call'), pages)}): the "
                      f"model went on to write {tok(event['meant'])}; the closest "
                      f"candidate is {tok(event['value'])}.", ""]
        elif kind == "close":
            lines += [f"**Closed** with {tok(event['token'])}"
                      + (" (it came with the last step's token)." if event["call"] is None
                         else f", the first token starting with a quote that call "
                              f"{call_ref(event['call'], pages)} returned.")
                      + " A bare `\"` where the model writes `\",` would make it skip the "
                      "comma.", ""]
        elif kind == "resume":
            lines += [f"**The model continues freely** (call "
                      f"{call_ref(event['call'], pages)}):", "", block(event["text"]), ""]
    return lines


def interventions(trace: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """The trace's events cut per re-decoded value: each list from an `invalid` to the
    event before the next `invalid` (or the `done`/`kept` that follows its continuation)."""
    groups: List[List[Dict[str, Any]]] = []
    for event in trace:
        if event["event"] == "invalid":
            groups.append([event])
        elif groups and event["event"] in ("abstain", "forced", "step", "nearest",
                                           "close", "resume") and groups[-1][-1][
                                               "event"] not in ("resume", "abstain"):
            groups[-1].append(event)
    return groups


def structure_section(index: int, structure: Dict[str, Any],
                      pages: Dict[int, str]) -> List[str]:
    """The decoding of one structuring reply: summary, timeline, values, diff."""
    decoding = structure["decoding"]
    changes = decoding.get("interventions") or []
    lines = [f"## Structuring reply {index + 1}: {tok(structure.get('text'))}", "",
             f"{decoding.get('calls')} call(s), {len(changes)} value(s) decoded again.", "",
             "| # | slot | free decoding | trie decoding | how | steps | candidates | "
             "evidence |", "|---|---|---|---|---|---|---|---|"]
    for i, change in enumerate(changes, 1):
        lines.append(f"| {i} | `{change['slot']}` | {tok(change['free'])} | "
                     f"{tok(change['decoded'])} | {change['how']} | {change['steps']} | "
                     f"{change['candidates']} | "
                     f"{slots_text(change.get('evidence') or {})} |")
    trace = decoding.get("trace")
    # 1. A record from before the trace: the summary is all there is.
    if not trace:
        return lines + ["", "This record has no trace (it was written before the trace "
                        "existed): run the episode with the current code to see each "
                        "step.", ""]
    # 2. The timeline, the free reply, each value in detail.
    lines += ["", "### Timeline", "", *timeline(trace, pages), "",
              "### The free reply (call 1)", "", block(after_think(trace[0]["text"])), ""]
    for number, group in enumerate(interventions(trace), 1):
        decoded = changes[number - 1]["decoded"] if number <= len(changes) else None
        lines += intervention_section(number, group, decoded, pages)
    # 3. What changed, line by line.
    final = next((e["text"] for e in reversed(trace) if e["event"] == "done"), "")
    diff = difflib.unified_diff(after_think(trace[0]["text"]).splitlines(),
                                after_think(final).splitlines(), "free decoding",
                                "trie decoding", lineterm="", n=2)
    lines += ["### Free reply against decoded reply", "",
              block("\n".join(diff) or "(identical)", "diff"), ""]
    return lines


def decoding_markdown(record: Dict[str, Any], pages: Dict[int, str],
                      source: str) -> str:
    """`decoding.md` of an episode record (module docstring); ``pages`` maps a decoding
    call's number to its prompt-report page, ``source`` says where the record is."""
    decoded = [s for s in record.get("structures") or [] if s.get("decoding")]
    config = record.get("variant") or {}
    lines = [f"# Decoding report: {record.get('episode')}", "",
             f"Record `{source}`. Config `{config.get('name')}`; prompt variants: "
             f"{run_config.prompt_label(run_config.configuration({'variant': config, 'settings': {}}))}.",
             "",
             "## The request", "", block(str(record.get("query") or record.get("input"))),
             "", f"Outcome: `{record.get('outcome')}`.", "", METHOD]
    if not decoded:
        return "\n".join(lines + ["No structuring reply of this record was decoded with "
                                  "the trie (is the config's `decoding` `iri_trie`?)."])
    for index, structure in enumerate(decoded):
        lines += structure_section(index, structure, pages)
    return "\n".join(lines)


# -- linking the calls to the prompt report -----------------------------------

def call_pages(folder: Path, record: Dict[str, Any]) -> Dict[int, str]:
    """Decoding call number -> its page in the prompt report of ``folder``, when that is
    certain: one decoded structure, and as many structuring calls as it counted (with
    several, their calls interleave and cannot be told apart)."""
    calls_file = folder / "calls.json"
    decoded = [s for s in record.get("structures") or [] if s.get("decoding")]
    if not calls_file.exists() or len(decoded) != 1:
        return {}
    calls = json.loads(calls_file.read_text(encoding="utf-8"))["calls"]
    structuring = [c for c in calls if c["step"].startswith("structuring")]
    if len(structuring) != decoded[0]["decoding"].get("calls"):
        return {}
    return {n: f"calls/{prompt_report.file_stem(c)}.md"
            for n, c in enumerate(structuring, 1)}


# -- the CLI ------------------------------------------------------------------

def write_page(record_path: Path, out_file: Path, folder: Optional[Path]) -> None:
    """Write the decoding page of ``record_path`` to ``out_file``; calls linked to the
    prompt report of ``folder`` when given."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    pages = call_pages(folder, record) if folder else {}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(decoding_markdown(record, pages, record_path.as_posix()),
                        encoding="utf-8")
    print(f"read {out_file}")


def run_and_report(args: argparse.Namespace) -> int:
    """Run the episode under the prompt recorder (`prompt_report.report_episode`), then
    write `decoding.md` beside its prompt report."""
    from tests.simuhome.bench import harness
    from tests.simuhome.bench.models import load_variant
    from tests.simuhome.bench.suites import get_suite

    # 1. The config as the harness reads it; only a trie arm has something to show.
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    variant = load_variant(str(config_path))
    chosen = run_config.configuration({"variant": variant.as_dict(), "settings": {}})
    if "decoding=iri_trie" not in run_config.prompt_label(chosen):
        print(f"config {config_path.name} does not use decoding: iri_trie "
              f"({run_config.prompt_label(chosen)}); nothing to show")
        return 2
    suite = get_suite(config.get("suite"))
    options = harness.run_options(config, SimpleNamespace(
        episodes=None, seeds=None, limit=None, workers=None), suite)
    harness.check_environment(options, suite)
    benchmark = (PROJECT_ROOT / config["benchmark"] if config.get("benchmark")
                 else suite.default_benchmark)

    # 2. The episode, run and captured as by the prompt report.
    code, folder = prompt_report.report_episode(
        config_path, variant, suite, options, benchmark, suite.names(benchmark),
        args.episode, args.stop_after, Path(args.out).resolve())
    if folder is None:
        return code

    # 3. The decoding page, from the record the run wrote.
    record_path = folder / "run" / variant.name / f"{folder.name}.json"
    if not record_path.exists():
        print(f"the run wrote no record ({record_path}); see {folder / 'run.log'}")
        return code or 1
    write_page(record_path, folder / "decoding.md", folder)
    return code


def main() -> int:
    """Run one episode and write its decoding report, or draw a record (module
    docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="a trie arm, e.g. "
                        "tests/homebench/variants/homebench.ecai.M3.pv-trie.yaml")
    parser.add_argument("--episode", default="hb1_feasible_seed_4351",
                        help="episode name or pattern (default: %(default)s)")
    parser.add_argument("--stop-after", default="structuring",
                        choices=("segmentation", "structuring", "planning"),
                        help="stop the pipeline there (default: %(default)s)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="reports root")
    parser.add_argument("--record", help="draw this episode record; run nothing")
    parser.add_argument("--out-file", help="with --record: the page to write (default: "
                        "<out>/<run folder>/<episode>.decoding.md)")
    args = parser.parse_args()

    # 1. A record on disk: draw it.
    if args.record:
        record_path = Path(args.record).resolve()
        out_file = Path(args.out_file) if args.out_file else (
            Path(args.out) / record_path.parent.name / f"{record_path.stem}.decoding.md")
        folder = record_path.parents[2] if record_path.parent.parent.name == "run" else None
        write_page(record_path, out_file.resolve(), folder)
        return 0
    # 2. Otherwise run the episode.
    if not args.config:
        parser.error("--config is required (unless --record)")
    return run_and_report(args)


if __name__ == "__main__":
    sys.exit(main())
