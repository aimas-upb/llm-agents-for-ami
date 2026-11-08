"""Prompt report of one HomeBench episode: what every LLM call of the pipeline receives.

Runs ONE episode the way the harness runs it (same config, models, guidelines and
environment mode -- the command line is built by `harness.commands`, with the episode
picked instead of `run.episodes`), keeps every LLM request the agents make, and writes a
folder from which the prompts can be read and compared, as sent, and as the parts they are
made of. Nothing is scored; the run is the harness's, with a recorder around it.

    conda run -n ami-agents python -m tests.homebench.prompt_report \\
        --config tests/homebench/variants/homebench.ecai.M3.yaml \\
        [--episode hb1_feasible_seed_1085] [--stop-after planning] \\
        [--out tests/homebench/results/prompts]

No `--episode`: the three episodes of `PROMPT_EPISODES`, one after another, each with its
own folder under `<out>/<config name>/` and an `index.md` there -- no single episode
brings out every prompt HomeBench can trigger, these three together bring out six of the
eight (`--stop-after`, if given, replaces each one's own stop). `--episode` runs that one
alone: a name or pattern, the simplest match winning (fewest parts, then the case, then
the seed -- not the first file alphabetically, which is the 10-part `hb10_mixed_seed_132`).
The models are the config's, so a local config (the M configs, on Ollama) costs nothing
and an OpenAI one is billed for the episodes' calls (a handful per episode: 1
segmentation, 1 structuring per part, planning, profiling, guards, replies).

    --render <folder>    write the report again from `<folder>/calls.json`, running nothing
                         (after changing how it is written, or to move a capture elsewhere)

The folder, `<out>/<config name>/<episode>/`:

    README.md            the episode, the models per step, and the calls in order, linked
    abstract.md          every call's prompt as an outline of its parts, side by side:
                         which constants it is composed of, what is filled in per call
    catalog.md           every prompt text of the code as composed there, independent of
                         this episode (`prompt_parts.catalog_markdown`)
    calls/NN_<step>__<function>.md       one call: the model and parameters, the outline of
                         each message (name, kind, characters, share, where defined), the
                         full text as sent, the tools / response format, the reply
    calls/NN_...system.txt / .user.txt   the exact text of each message, nothing added, for
                         diffing one prompt design against another
    calls/NN_....json    the whole captured request and response
    calls.json           every call + the run's meta (what `--render` reads)
    run.log              what the run printed; run/ holds its record and run.json

How a message is taken apart is `prompt_parts.py`; how the calls are caught is
`simuhome/bench/prompt_capture.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tests.homebench import prompt_parts
from tests.homebench.prompt_parts import Part

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "tests" / "homebench" / "results" / "prompts"

# What a run without `--episode` covers: (episode, stop after, prompts it brings out).
# No single episode reaches them all, and all three planned on M3 (`ecai.M3`) when chosen.
PROMPT_EPISODES: List[Tuple[str, Optional[str], str]] = [
    ("hb2_feasible_seed_14", None,
     "segmentation, simple structuring, modify-recipe, plan summary"),
    ("hb8_mixed_seed_224_fixed", "planning", "BT planner"),
    ("hb3_feasible_seed_2_fixed", "planning", "dependency structuring"),
]


# -- choosing the episode -----------------------------------------------------------

def natural_key(name: str) -> Tuple[int, str, int, str]:
    """Sort key of an episode name, simplest first: (parts, case, seed, name), so
    ``hb1_feasible_seed_1085`` comes before ``hb10_mixed_seed_132`` (which a plain string
    sort puts first, '0' < '_')."""
    match = re.match(r"hb(\d+)_(\w+?)_seed_(\d+)", name)
    if match is None:
        return (10 ** 6, name, 0, name)
    return (int(match.group(1)), match.group(2), int(match.group(3)), name)


def pick_episode(names: List[str], wanted: Optional[str]) -> str:
    """The episode to run: ``wanted`` (a full name, or a pattern such as ``hb2_mixed_*``,
    the simplest match winning), else the simplest of ``names``. ValueError when
    ``wanted`` matches none -- a typo must not run some other episode."""
    import fnmatch

    # 1. Nothing asked for: the simplest episode.
    ordered = sorted(names, key=natural_key)
    if not wanted:
        return ordered[0]
    # 2. A name or a pattern; the file extension is allowed.
    wanted = wanted.removesuffix(".json")
    matches = [n for n in ordered if n == wanted or fnmatch.fnmatch(n, wanted)]
    if not matches:
        raise ValueError(f"no episode matches {wanted!r}; the benchmark has "
                         f"{len(names)}, e.g. {', '.join(ordered[:3])}")
    return matches[0]


# -- running the episode ------------------------------------------------------------

def run_argv(commands: List[List[str]], episode: str,
             stop_after: Optional[str]) -> List[str]:
    """The `run.py` arguments for one episode, from the harness's own run command: its
    options as the config sets them, minus the resume flag and the episode patterns, plus
    this episode, unscored (`--no-evaluate`)."""
    # 1. The harness's run command is [python, -m, module, ...args]; keep the args.
    args = commands[0][3:]
    kept: List[str] = []
    skip_value = False
    for token in args:
        # 2. Drop what picks other episodes or skips recorded ones.
        if skip_value:
            skip_value = False
        elif token == "--episodes":
            skip_value = True
        elif token != "--skip-existing":
            kept.append(token)
    # 3. This episode only.
    kept += ["--episode", episode, "--no-evaluate"]
    if stop_after:
        kept += ["--stop-after", stop_after]
    return kept


def run_captured(argv: List[str], log_path: Path) -> Tuple[int, List[Dict[str, Any]]]:
    """Run `run.py` in this process with ``argv``, recorder installed; (its exit code, the
    LLM calls made). What the run prints goes to ``log_path``, so the report is not buried
    in the per-step debug output."""
    from tests.simuhome.bench import run
    from tests.simuhome.bench.prompt_capture import PromptRecorder

    recorder = PromptRecorder()
    saved = sys.argv
    sys.argv = ["run"] + argv
    try:
        # The recorder goes on first, so the meter `run_episode` installs sits above it.
        with recorder.installed(), log_path.open("w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = run.main()
    finally:
        sys.argv = saved
    return code, recorder.calls


# -- writing the report -------------------------------------------------------------

def slug(text: str) -> str:
    """``text`` as a file-name piece: letters, digits, dot, dash and underscore."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def fence(text: str) -> str:
    """A code fence longer than any run of backticks in ``text``, so the text sits inside
    it as it is."""
    longest = max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def block(text: str, language: str = "text") -> str:
    """``text`` in a fenced code block, unchanged."""
    marks = fence(text)
    return f"{marks}{language}\n{text}\n{marks}"


def site_text(site: Optional[Dict[str, Any]]) -> str:
    """Where a call was made, `file:line` and the function, for the page."""
    if not site:
        return "unknown"
    return f"`{site['file']}:{site['line']}` in `{site['function']}`"


def file_stem(call: Dict[str, Any]) -> str:
    """``NN_<step>__<function>``: the calls' file name, in the order they started."""
    function = (call.get("site") or {}).get("function", "unknown")
    return f"{call['seq']:02d}_{slug(call['step'])}__{slug(function)}"


def text_of(message: Dict[str, Any]) -> str:
    """A message's content as text: a string as it is; anything else (tool calls, a
    list of parts) as JSON."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps({k: v for k, v in message.items() if k != "role"},
                      ensure_ascii=False, indent=1)


def message_files(messages: List[Dict[str, Any]]) -> List[str]:
    """The text file name of each message: ``system``, ``user`` for the first of a role,
    ``user2`` ... for the following (a retry's feedback)."""
    seen: Dict[str, int] = {}
    names = []
    for message in messages:
        role = str(message.get("role"))
        seen[role] = seen.get(role, 0) + 1
        names.append(role if seen[role] == 1 else f"{role}{seen[role]}")
    return names


def params_text(call: Dict[str, Any]) -> str:
    """The request's parameters in one line each: scalars as they are, the tools and the
    response format by name and size."""
    lines = []
    for key, value in call["request"]["params"].items():
        if isinstance(value, (dict, list)):
            size = len(json.dumps(value, ensure_ascii=False))
            label = (", ".join(t.get("function", t).get("name", "?") for t in value
                               if isinstance(t, dict)) if key == "tools" else "")
            lines.append(f"- `{key}`: {label + ', ' if label else ''}{size:,} chars of JSON "
                         "(full text below)")
        else:
            lines.append(f"- `{key}` = `{value}`")
    return "\n".join(lines) or "- (none)"


def decompose_call(call: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str, Optional[Part]]]:
    """Each message of the call with its file name and its parts (None for a message that
    is a turn of the conversation, not a prompt: the model's own reply, a tool result)."""
    messages = call["request"]["messages"]
    out = []
    for message, name in zip(messages, message_files(messages)):
        role = message.get("role")
        parts = (prompt_parts.decompose_message(text_of(message))
                 if role in ("system", "user") and isinstance(message.get("content"), str)
                 else None)
        out.append((message, name, parts))
    return out


def share_line(part: Part) -> str:
    """How much of a message is fixed in the code and how much is filled per call."""
    static, dynamic = prompt_parts.static_dynamic(part)
    total = max(part.chars, 1)
    return (f"fixed in the code {static:,} chars ({100 * static / total:.1f}%), "
            f"filled per call {dynamic:,} chars ({100 * dynamic / total:.1f}%)")


def call_markdown(call: Dict[str, Any], decomposed) -> str:
    """The page of one call (module docstring)."""
    response = call.get("response") or {}
    usage = response.get("usage") or {}
    chars = sum(len(text_of(m)) for m, _, _ in decomposed)
    lines = [f"# Call {call['seq']:02d}: {call['step']}", "",
             "| | |", "|---|---|",
             f"| step | `{call['step']}` |",
             f"| made at | {site_text(call.get('site'))} |",
             f"| model | `{call['model']}` at {call['base_url'] or 'the default endpoint'}, "
             f"{call['api']} API |",
             f"| started | {call['started']:.1f} s into the run; took {call.get('seconds', 0)} s"
             f"{'' if call['ok'] else '; FAILED: ' + str(call['error'])} |",
             f"| sent | {chars:,} chars in {len(decomposed)} message(s): "
             + ", ".join(f"{n} {len(text_of(m)):,}" for m, n, _ in decomposed) + " |",
             f"| tokens | {usage.get('tokens_in')} in / {usage.get('tokens_out')} out "
             f"(as the server counted)"
             + (f"; {chars / usage['tokens_in']:.2f} chars per input token"
                if usage.get("tokens_in") else "") + " |", ""]

    # 1. The abstract view: each prompt message as the parts it is made of.
    lines += ["## Parts of the prompt", "",
              "What each message is composed of, from the code: constants (defined in a "
              "prompt module), inline glue, and fields filled per call. `chars` are "
              "characters of the text as sent; the model's own token counts are above.", ""]
    for message, name, part in decomposed:
        lines.append(f"### {name} message")
        if part is None:
            lines += ["", f"A {message.get('role')} turn of the conversation (not a prompt).",
                      ""]
            continue
        lines += ["", share_line(part), "", block(prompt_parts.render_tree(part)), ""]

    # 2. The text as sent.
    lines += ["## Text as sent", ""]
    for message, name, _ in decomposed:
        lines += [f"### {name}", "", block(text_of(message)), ""]

    # 3. The rest of the request: tools, response format, other parameters.
    lines += ["## Request parameters", "", params_text(call), ""]
    for key, value in call["request"]["params"].items():
        if isinstance(value, (dict, list)):
            lines += [f"### {key}", "", block(json.dumps(value, ensure_ascii=False, indent=1),
                                              "json"), ""]

    # 4. The reply.
    lines += ["## Reply", ""]
    if not call["ok"]:
        lines += [f"The call failed: `{call['error']}`", ""]
    else:
        lines += [f"Finish reason: `{response.get('finish_reason')}`.", ""]
        if response.get("reasoning"):
            lines += ["### reasoning", "", block(response["reasoning"]), ""]
        if response.get("content") is not None:
            lines += ["### content", "", block(response["content"]), ""]
        for tool in response.get("tool_calls") or []:
            lines += [f"### tool call `{tool['name']}`", "", block(str(tool["arguments"]),
                                                                 "json"), ""]
    return "\n".join(lines)


def outline(part: Part) -> str:
    """One line for a message: its name and size, then its parts one level down, the
    fields marked `{like this}`."""
    inside = ", ".join(f"{c.name if c.kind != 'inline' else 'inline'} {c.chars:,}"
                       for c in part.children)
    return f"**{part.name}** ({part.kind}, {part.chars:,} chars)" + (
        f": {inside}" if inside else "")


def abstract_markdown(meta: Dict[str, Any], calls: List[Dict[str, Any]],
                      decomposed: Dict[int, list]) -> str:
    """`abstract.md`: every call's prompt as an outline, to compare the steps."""
    lines = ["# Prompts as outlines", "",
             f"Episode `{meta['episode']}`, config `{meta['config_name']}`. For each call, "
             "each message as the parts it is made of (one level down; the full tree is in "
             "the call's page). Constants are texts of the prompt modules; a field is "
             "filled in per call.", ""]
    for call in calls:
        lines += [f"## {call['seq']:02d} {call['step']} (`{call['model']}`)", "",
                  f"Made at {site_text(call.get('site'))}.", ""]
        for message, name, part in decomposed[call["seq"]]:
            if part is None:
                lines.append(f"- **{name}**: a {message.get('role')} turn of the conversation")
            else:
                lines.append(f"- {name}: {outline(part)} -- {share_line(part)}")
        lines.append("")
    return "\n".join(lines)


def readme_markdown(meta: Dict[str, Any], calls: List[Dict[str, Any]],
                    stems: Dict[int, str]) -> str:
    """`README.md`: the episode, the models per step, the calls in order."""
    lines = [f"# Prompt report: {meta['episode']}", "",
             f"Config `{meta['config']}` (`{meta['config_name']}`), run {meta['started']}, "
             f"commit `{meta.get('commit')}`.", "",
             "## The request", "", block(str(meta.get("input") or meta.get("query") or "")),
             "", f"Outcome: `{meta.get('outcome')}`"
             + (f"; error: {meta['error']}" if meta.get("error") else "") + ".", "",
             "## Models per step", "", "| step | model |", "|---|---|"]
    lines += [f"| {step} | `{label}` |" for step, label in meta["models"].items()]
    lines += ["", "## LLM calls, in the order they started", "",
              "| # | step | made in | model | chars sent | tokens in/out | s | page |",
              "|---|---|---|---|---|---|---|---|"]
    for call in calls:
        usage = (call.get("response") or {}).get("usage") or {}
        sent = sum(len(text_of(m)) for m in call["request"]["messages"])
        function = (call.get("site") or {}).get("function", "?")
        lines.append(
            f"| {call['seq']} | {call['step']} | `{function}` | `{call['model']}` | {sent:,} | "
            f"{usage.get('tokens_in')}/{usage.get('tokens_out')} | {call.get('seconds', 0)} | "
            f"[page](calls/{stems[call['seq']]}.md) |")
    if not calls:
        lines.append("| | no LLM call was made: see `run.log` | | | | | | |")
    lines += ["", "## Files", "",
              "- `abstract.md`: every call's prompt as an outline of its parts, side by side",
              "- `catalog.md`: every prompt text of the code as composed there",
              "- `calls/`: per call a page (`.md`), each message's exact text (`.system.txt`, "
              "`.user.txt`) and the whole capture (`.json`)",
              "- `calls.json`: all calls and the run's meta; `--render <this folder>` writes "
              "the report again from it", "- `run.log`: what the run printed", ""]
    return "\n".join(lines)


def write_report(directory: Path, meta: Dict[str, Any], calls: List[Dict[str, Any]]) -> None:
    """Write every file of the report (module docstring) into ``directory``."""
    calls_dir = directory / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    # 1. Each call: its page, the exact text of its messages, its capture.
    stems: Dict[int, str] = {}
    decomposed: Dict[int, list] = {}
    for call in calls:
        stem = file_stem(call)
        stems[call["seq"]] = stem
        decomposed[call["seq"]] = decompose_call(call)
        (calls_dir / f"{stem}.md").write_text(call_markdown(call, decomposed[call["seq"]]),
                                              encoding="utf-8")
        for message, name, _ in decomposed[call["seq"]]:
            (calls_dir / f"{stem}.{name}.txt").write_text(text_of(message), encoding="utf-8")
        (calls_dir / f"{stem}.json").write_text(
            json.dumps(call, ensure_ascii=False, indent=1), encoding="utf-8")
    # 2. The pages that read across calls, and the capture the report can be rebuilt from.
    (directory / "README.md").write_text(readme_markdown(meta, calls, stems), encoding="utf-8")
    (directory / "abstract.md").write_text(abstract_markdown(meta, calls, decomposed),
                                           encoding="utf-8")
    (directory / "catalog.md").write_text(prompt_parts.catalog_markdown(), encoding="utf-8")
    (directory / "calls.json").write_text(
        json.dumps({"meta": meta, "calls": calls}, ensure_ascii=False, indent=1),
        encoding="utf-8")


# -- the CLI ------------------------------------------------------------------------

def episode_meta(config_path: Path, variant, episode: str, record_path: Path,
                 started: str) -> Dict[str, Any]:
    """The run's meta for the report: the config, the model per step, and what the
    episode's record says happened (outcome, error, the input, the final reply)."""
    from tests.simuhome.bench.run import _git_commit

    meta: Dict[str, Any] = {
        "episode": episode, "config": config_path.relative_to(PROJECT_ROOT).as_posix()
        if config_path.is_relative_to(PROJECT_ROOT) else config_path.as_posix(),
        "config_name": variant.name, "started": started, "commit": _git_commit(),
        "models": {step: spec.label() for step, spec in variant.specs.items()
                   if step != "judge"}}
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        meta.update({k: record.get(k) for k in ("input", "query", "outcome", "error",
                                                "error_step", "response")})
    return meta


def render_only(folder: Path) -> int:
    """Write the report again from ``folder``/calls.json, running nothing."""
    saved = json.loads((folder / "calls.json").read_text(encoding="utf-8"))
    write_report(folder, saved["meta"], saved["calls"])
    print(f"report rewritten in {folder} ({len(saved['calls'])} call(s))")
    return 0


def report_episode(config_path: Path, variant, suite, options, benchmark,
                   names: List[str], wanted: Optional[str],
                   stop_after: Optional[str], out: Path) -> Tuple[int, Optional[Path]]:
    """Run one episode under the recorder and write its report into ``out``/<config
    name>/<episode>/; (the run's exit code, that folder), or (2, None) when ``wanted``
    matches no episode."""
    from tests.simuhome.bench import harness

    # 1. The episode, and the folder it is reported in.
    try:
        episode = pick_episode(names, wanted)
    except ValueError as exc:
        print(f"config {config_path.name}: {exc}")
        return 2, None
    directory = out / variant.name / episode
    directory.mkdir(parents=True, exist_ok=True)
    print(f"{config_path.name}: episode {episode} -> {directory}", flush=True)

    # 2. The harness's own run command, for this episode, under the recorder. Its
    #    record lands in the report folder (run/), not in the harness's results.
    commands = harness.commands(config_path, directory / "run" / variant.name,
                                directory / "run", options, False, benchmark, suite)
    argv = run_argv(commands, episode, stop_after)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    code, calls = run_captured(argv, directory / "run.log")

    # 3. The report; a run that failed before any call still gets its README.
    meta = episode_meta(config_path, variant, episode,
                        directory / "run" / variant.name / f"{episode}.json", started)
    write_report(directory, meta, calls)
    print(f"{len(calls)} LLM call(s) recorded; outcome {meta.get('outcome')}"
          + (f" ({meta.get('error_step')}: {meta['error']})" if meta.get("error") else ""))
    for call in calls:
        print(f"  {call['seq']:>2} {call['step']:<22} {call['model']:<24} "
              f"{sum(len(text_of(m)) for m in call['request']['messages']):>7,} chars")
    print(f"read {directory / 'README.md'}")
    return code, directory


def index_markdown(config_name: str, rows: List[Tuple[str, str, Optional[str], int,
                                                      Optional[Path]]]) -> str:
    """`index.md` of a default run: the episodes in the order they ran, each with what
    it is there for, how far it ran, and its folder."""
    lines = [f"# Prompt reports: {config_name}", "",
             "The default episodes, one folder each, run in this order. Each is chosen "
             "for the prompts it brings out (`PROMPT_EPISODES`).", "",
             "| episode | stop after | prompts it is for | exit code | report |",
             "|---|---|---|---|---|"]
    for episode, stop_after, purpose, code, folder in rows:
        link = f"[{episode}/README.md]({folder.name}/README.md)" if folder else "not run"
        lines.append(f"| `{episode}` | {stop_after or 'the end'} | {purpose} | {code} | "
                     f"{link} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run one episode, or the default three, with the recorder and write the prompt
    reports (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="harness config, e.g. "
                                         "tests/homebench/variants/homebench.ecai.M3.yaml")
    parser.add_argument("--episode", help="episode name or pattern (default: the three "
                                          "episodes of PROMPT_EPISODES, one after another)")
    parser.add_argument("--stop-after", choices=("segmentation", "structuring", "planning"),
                        help="stop the pipeline there (fewer calls, no execution); with "
                             "no --episode, replaces each default episode's own stop")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="reports root")
    parser.add_argument("--render", help="rewrite the report of this folder from its "
                                         "calls.json; run nothing")
    args = parser.parse_args()
    if args.render:
        return render_only(Path(args.render))
    if not args.config:
        parser.error("--config is required (unless --render)")

    # Imported here: they load the agents' code and the OpenAI client.
    from tests.simuhome.bench import harness
    from tests.simuhome.bench.models import load_variant
    from tests.simuhome.bench.suites import get_suite

    # 1. The config as the harness reads it: variant, suite, run options, benchmark.
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    variant = load_variant(str(config_path))
    suite = get_suite(config.get("suite"))
    options = harness.run_options(config, SimpleNamespace(
        episodes=None, seeds=None, limit=None, workers=None), suite)
    harness.check_environment(options, suite)
    benchmark = (PROJECT_ROOT / config["benchmark"] if config.get("benchmark")
                 else suite.default_benchmark)
    names = suite.names(benchmark)
    out = Path(args.out).resolve()

    # 2. One episode as asked: the report of that one.
    if args.episode:
        return report_episode(config_path, variant, suite, options, benchmark,
                              names, args.episode, args.stop_after, out)[0]

    # 3. No episode: the default ones in turn, so every prompt is under one folder.
    #    A missing episode stops before anything runs; a failed run does not stop the rest.
    for episode, _, _ in PROMPT_EPISODES:
        if episode not in names:
            print(f"config {config_path.name}: default episode {episode!r} is not in the "
                  f"benchmark ({len(names)} episodes)")
            return 2
    rows = []
    worst = 0
    for episode, stop_after, purpose in PROMPT_EPISODES:
        stop = args.stop_after or stop_after
        code, folder = report_episode(config_path, variant, suite, options,
                                      benchmark, names, episode, stop, out)
        rows.append((episode, stop, purpose, code, folder))
        worst = max(worst, code)
    index = out / variant.name / "index.md"
    index.write_text(index_markdown(variant.name, rows), encoding="utf-8")
    print(f"read {index}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
