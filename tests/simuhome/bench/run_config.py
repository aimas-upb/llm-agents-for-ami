"""The configuration a run ran with, stated at the head of every report made from it.

A result is only comparable to another when what each ran with is known: the model of
every step, how the structuring prompt was built, the guidelines, the environment mode.
`run.py` writes all of it into `<records>/run.json`; the reports (`stats.py`'s
`summary.md`, `report.py`'s `report.md`, `tests/homebench/error_report.py`'s `errors.md`,
`tests/homebench/prompt_report.py`'s `README.md`) each read it back through this one
module, so they all say the same thing, in the same words, and none of them has to infer
a setting from the text of a prompt.

    config = load(records_dir)             # run.json -> the configuration, or None
    print(markdown(config))                # the section every report opens with
    print(prompt_label(config))            # one line: "structuring: context=json_v2"

Flow, on a run of `homebench.ecai.M3.pv-json-v2.yaml`:

    1. run.json holds `variant` (the model per step, `prompt_variants` = what differs
       from the default, `prompt_selection` = every axis written out) and `settings`
       (suite, guidelines, environment, memory, context, ...)
    2. `configuration(run)` flattens it into plain rows: models, prompt axes (with the
       option's one-line description, from the live code), settings
    3. `markdown(config)` renders the rows as the report's "Experiment configuration"

A run.json from before `prompt_selection` existed has only the non-default choices;
its axes are filled in with the defaults of the current code, and the section says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ami_agents.shared.utils import prompt_variants

# The run settings worth stating, in reading order: each changes what the agents saw or
# how far the pipeline went. Shown only when the run recorded a value.
SETTINGS_SHOWN = ("suite", "benchmark", "guidelines", "environment", "memory", "context",
                  "execution", "stop_after", "cache", "workers")


# -- reading a run ------------------------------------------------------------

def load(directory: Path) -> Optional[Dict[str, Any]]:
    """The configuration of the run whose records are in ``directory``, from its
    ``run.json``; None when there is no readable one (a run from before it existed)."""
    try:
        run = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return configuration(run) if isinstance(run, dict) else None


def _model_row(step: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """One step's model as a row: the model, where it is served, how it is called."""
    # 1. A reasoning model has an effort, the others a temperature.
    if spec.get("reasoning_effort"):
        call = f"reasoning_effort={spec['reasoning_effort']}"
    else:
        call = f"temperature={spec.get('temperature', 0.0)}"
    # 2. Only what is not the default of the call.
    if spec.get("max_tokens"):
        call += f", max_tokens={spec['max_tokens']}"
    if spec.get("api") and spec["api"] != "auto":
        call += f", api={spec['api']}"
    return {"step": step, "model": spec.get("model"),
            "endpoint": spec.get("base_url") or "OpenAI", "call": call}


def configuration(run: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration of a run as plain rows, from the content of its ``run.json``."""
    variant = run.get("variant") or {}
    settings = run.get("settings") or {}

    # 1. The prompt: every axis. A run.json that has the full selection is read as it
    #    says; an older one has the non-default choices only, so the rest is the
    #    current defaults, and that is said.
    selection = variant.get("prompt_selection")
    reconstructed = selection is None
    if reconstructed:
        try:
            selection = prompt_variants.full_selection(variant.get("prompt_variants"))
        except ValueError:
            selection = {}
    # 2. The models, in the order of the variant.
    models = [_model_row(step, spec) for step, spec in (variant.get("steps") or {}).items()
              if step != "judge"]
    return {
        "variant": variant.get("name"),
        "config_file": settings.get("variant"),
        "commit": run.get("commit"),
        "started": run.get("started"),
        "episodes": len(run.get("episodes") or []),
        "models": models,
        "prompt": prompt_variants.describe_selection(selection),
        "prompt_reconstructed": reconstructed,
        "settings": {key: settings[key] for key in SETTINGS_SHOWN
                     if settings.get(key) is not None},
    }


# -- stating it ---------------------------------------------------------------

def prompt_label(config: Optional[Dict[str, Any]]) -> str:
    """The prompt choices that differ from the default, on one line:
    ``structuring: context=json_v2``; ``default`` when every axis is the default."""
    if not config:
        return "unknown (no run.json)"
    changed: Dict[str, List[str]] = {}
    for row in config["prompt"]:
        if not row["default"]:
            changed.setdefault(row["step"], []).append(f"{row['axis']}={row['option']}")
    return "; ".join(f"{step}: {', '.join(axes)}" for step, axes in changed.items()) \
        or "default (every axis at its default)"


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    """A markdown table; a cell's pipes are escaped, so a description cannot split it."""
    def cells(row: List[Any]) -> str:
        return "| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |"

    return [cells(header), "|" + "---|" * len(header)] + [cells(row) for row in rows]


def markdown(config: Optional[Dict[str, Any]], heading: str = "## Experiment configuration"
             ) -> str:
    """The section a report opens with: what this run ran with, in full. Empty text
    would hide a missing record, so a run without a run.json says that it has none."""
    if not config:
        return (f"{heading}\n\nNot recorded: there is no `run.json` beside the records, "
                "so the configuration of this run cannot be stated.\n")
    # 1. Who and when: the config file, the variant (named after it), the code.
    lines = [heading, "",
             f"Config file `{config['config_file']}`, variant `{config['variant']}`, "
             f"commit `{config['commit']}`, started {config['started']}, "
             f"{config['episodes']} episode(s) listed.", ""]

    # 2. The prompt of every step that has variants: every axis, the default marked, so
    #    "default" is read as what it is rather than as nothing said.
    lines += [f"**Prompt variants**: {prompt_label(config)}.", ""]
    lines += _table(["step", "axis", "option", "default?", "what it does"],
                    [[r["step"], r["axis"], f"`{r['option']}`",
                      "yes" if r["default"] else "**no**", r["summary"]]
                     for r in config["prompt"]])
    if config["prompt_reconstructed"]:
        lines += ["", "This run.json predates the recorded selection: axes it does not "
                  "list are shown as the defaults of the current code."]

    # 3. The models, and where each is served.
    lines += ["", "**Models per step**", ""]
    lines += _table(["step", "model", "endpoint", "call"],
                    [[r["step"], f"`{r['model']}`", r["endpoint"], r["call"]]
                     for r in config["models"]])

    # 4. The run's own settings.
    if config["settings"]:
        lines += ["", "**Run settings**", ""]
        lines += [f"- `{key}`: {value}" for key, value in config["settings"].items()]
    return "\n".join(lines) + "\n"


def markdown_of(directory: Path, heading: str = "## Experiment configuration") -> str:
    """`markdown(load(directory))`: the section for the run whose records are in
    ``directory``."""
    return markdown(load(directory), heading)
