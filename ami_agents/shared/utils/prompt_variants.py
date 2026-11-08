"""Named variants of how an LLM step builds its call: prompt, context, output
constraint, post-processing.

A step (say `structuring`) has a few *axes*, each one decision about how its call is
made: how the home context is rendered, which prompt text is sent, whether the
reply is constrained by a schema, whether the parsed reply is repaired. Every axis
has named *options*, one of them the default, which is what the step did before the
axis existed. A *selection* names one option per axis; an axis left out takes its
default. The selection comes from the agent's config:

    user_assistant:
      prompt_variants:
        structuring: {context: json, constraint: schema}     # prompt, repair: default

Flow, from the step's side (`GoalStructuringBehaviour._structure`):

    1. selection = selection_from_config(self.agent.config, "structuring")
    2. rendered  = selection["context"](context)             # the option, called
    3. system    = selection["prompt"](structure, rendered, qualifiers)

What an option is called with and returns is a contract of its *axis*, written in
the axis's `doc` where the step declares it; this module checks names, not
signatures. A misspelt step, axis or option raises `ValueError` listing what exists,
so a typo never runs as the default and quietly labels the run as something else.

Adding a variant is one function in the step's module, registered on its axis:

    @CONTEXT.option("table")
    def table_context(context: GoalContext) -> RenderedContext: ...

and `table` is then valid in a config. Adding a step is a module that calls
`declare_step`, plus one line in `STEP_MODULES` (the registry imports it on first
use, so the shared package never imports an agent at load time).

Harness side: `normalize` turns a variant file's `prompt_variants:` mapping into the
non-default choices only (so "all defaults" and "no entry" are the same variant, and
hash alike), and `apply_to_config` writes it where the step's agent reads it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

# The agent config key under an agent's section that holds the selections.
CONFIG_KEY = "prompt_variants"

# The steps that have variants, and the module that declares each. Imported on first
# use by `get_step`; the module's `declare_step` call fills the registry.
STEP_MODULES: Dict[str, str] = {
    "structuring": "ami_agents.agents.user_assistant.utils.goal_structuring_variants",
}

# Filled by `declare_step`, when a step's module is imported.
_STEPS: Dict[str, "Step"] = {}


# -- declaring a step ---------------------------------------------------------

@dataclass
class Axis:
    """One decision about how a step builds its call, with its named options.

    ``doc`` states the contract of an option on this axis (what it is called with,
    what it returns); `default` is the option used when a selection names none.
    """

    name: str
    default: str
    doc: str
    options: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def option(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register the function as the option ``name`` of this axis.

        Registering a name twice is an error: two options silently sharing a name
        would make a config mean different things depending on import order."""
        def register(function: Callable[..., Any]) -> Callable[..., Any]:
            if name in self.options:
                raise ValueError(f"axis {self.name!r} already has an option {name!r}")
            self.options[name] = function
            return function
        return register


@dataclass
class Step:
    """A step with variants: the agent whose config holds its selection, and its axes."""

    name: str
    agent: str
    axes: Dict[str, Axis] = field(default_factory=dict)

    def axis(self, name: str, default: str, doc: str) -> Axis:
        """Declare an axis of this step and return it, to register options on."""
        if name in self.axes:
            raise ValueError(f"step {self.name!r} already has an axis {name!r}")
        axis = Axis(name=name, default=default, doc=doc)
        self.axes[name] = axis
        return axis


def declare_step(name: str, agent: str) -> Step:
    """Register a step; called once, at the top of the module that defines its axes.

    ``agent`` is the name of the agent's section in agents.yaml, where
    `prompt_variants.<name>` is read."""
    if name in _STEPS:
        raise ValueError(f"step {name!r} is already declared")
    step = Step(name=name, agent=agent)
    _STEPS[name] = step
    return step


def get_step(name: str) -> Step:
    """The declared step ``name``, importing its module the first time.

    Raises ValueError for a step that has no variants, listing the ones that do."""
    # 1. A known step is declared by importing its module (once).
    if name not in _STEPS and name in STEP_MODULES:
        importlib.import_module(STEP_MODULES[name])
    # 2. Still unknown: a typo, or a step nobody gave variants.
    if name not in _STEPS:
        raise ValueError(f"unknown step {name!r}; steps with prompt variants: "
                         f"{', '.join(sorted(STEP_MODULES))}")
    return _STEPS[name]


# -- a selection --------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    """The option chosen on every axis of one step.

    `chosen` maps each axis to an option name (defaults filled in); indexing by axis
    name gives the option's function, ready to call."""

    step: str
    chosen: Mapping[str, str]
    _step: Step = field(repr=False, compare=False, default=None)  # type: ignore[assignment]

    def __getitem__(self, axis: str) -> Callable[..., Any]:
        """The function of the option chosen on ``axis``."""
        return self._step.axes[axis].options[self.chosen[axis]]

    def non_default(self) -> Dict[str, str]:
        """The axes set to something other than their default: what the variant
        changes, and the only part worth recording or hashing."""
        return {name: option for name, option in self.chosen.items()
                if option != self._step.axes[name].default}

    def label(self) -> str:
        """`default`, or the changed axes as `context=json,constraint=schema`."""
        changed = self.non_default()
        return ",".join(f"{k}={v}" for k, v in changed.items()) or "default"


def resolve(step_name: str, requested: Optional[Mapping[str, Any]] = None) -> Selection:
    """The selection of ``step_name`` for the ``requested`` axis -> option mapping.

    Axes the request leaves out take their defaults. Raises ValueError for an unknown
    step, axis or option (each message lists the valid names), and for an axis whose
    default option was never registered (a declaration bug, caught at first use)."""
    step = get_step(step_name)
    requested = dict(requested or {})

    # 1. Every requested axis and option must exist; YAML scalars arrive as
    #    whatever they parse to, so compare their text.
    for axis_name, option in requested.items():
        if axis_name not in step.axes:
            raise ValueError(f"step {step.name!r} has no axis {axis_name!r}; "
                             f"axes: {', '.join(step.axes)}")
        options = step.axes[axis_name].options
        if str(option) not in options:
            raise ValueError(f"axis {step.name}.{axis_name} has no option {option!r}; "
                             f"options: {', '.join(options)}")

    # 2. Defaults for the rest; a default that is not an option is a bug here.
    chosen: Dict[str, str] = {}
    for axis_name, axis in step.axes.items():
        option = str(requested.get(axis_name, axis.default))
        if option not in axis.options:
            raise ValueError(f"axis {step.name}.{axis_name}: default {option!r} is not "
                             f"a registered option ({', '.join(axis.options)})")
        chosen[axis_name] = option
    return Selection(step=step.name, chosen=chosen, _step=step)


def selection_from_config(config: Optional[Mapping[str, Any]], step: str) -> Selection:
    """The selection of ``step`` from an agent's config (its own section).

    A config without `prompt_variants`, or without the step, is the all-defaults
    selection: the step behaves as it did before variants existed."""
    section = (config or {}).get(CONFIG_KEY) or {}
    return resolve(step, section.get(step))


# -- the harness's side -------------------------------------------------------

def normalize(prompts: Optional[Mapping[str, Mapping[str, Any]]]) -> Dict[str, Dict[str, str]]:
    """A variant file's `prompt_variants:` as the non-default choices per step.

    Validates every step, axis and option (see `resolve`) and drops the steps that
    change nothing, so a file that names only defaults equals one that names nothing."""
    out: Dict[str, Dict[str, str]] = {}
    for step_name, requested in (prompts or {}).items():
        changed = resolve(step_name, requested).non_default()
        if changed:
            out[str(step_name)] = changed
    return out


def apply_to_config(config: Dict[str, Any], prompts: Mapping[str, Mapping[str, str]]) -> None:
    """Write ``prompts`` (a `normalize` result) into ``config`` where each step's
    agent reads it: `<agent>.prompt_variants.<step>`.

    ``config`` is agents.yaml as loaded; steps already set there are replaced whole,
    so the variant file, not agents.yaml, decides every axis of a step it names."""
    for step_name, changed in prompts.items():
        agent = get_step(step_name).agent
        section = config.setdefault(agent, {}).setdefault(CONFIG_KEY, {})
        section[step_name] = dict(changed)


def catalog() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Every step's axes, defaults and option names, for docs and `--dry-run`."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for step_name in STEP_MODULES:
        step = get_step(step_name)
        out[step_name] = {name: {"default": axis.default, "options": list(axis.options),
                                 "doc": axis.doc}
                          for name, axis in step.axes.items()}
    return out


def describe(selection: Selection) -> List[str]:
    """One line per axis: the chosen option and whether it is the default."""
    step = get_step(selection.step)
    return [f"{name}={option}" + ("" if option == step.axes[name].default else " *")
            for name, option in selection.chosen.items()]
