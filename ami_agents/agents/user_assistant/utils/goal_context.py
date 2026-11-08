"""The goal context as the structurer receives it, and the names it may copy.

EnvExplorer answers a goal-context request with a payload that carries the
context twice: `text`, the indented rendering the prompt has always used, and
`context`, the structured view that text is rendered from (see
`env_explorer/utils/actuation_context.py`). `GoalContext` keeps both, so a step can
build its prompt from either; `Vocabulary` reads the names and classes out of the
structured view, which is what an output schema or a repair step needs to know:

    context.data   {"workspaces": [{"title": "Master Bedroom",
                                    "space_class": "homeont:MasterBedroom",
                                    "artifacts": [{"title": "master_bedroom_light",
                                                   "homeont_types": ["homeont:DimmableLight"],
                                                   "actions": [{"title": "brightness",
                                                                "command_types": [...]}]}]}]}
    vocabulary(context.data).room_classes        ["homeont:MasterBedroom"]
    vocabulary(context.data).artifact_names      ["master_bedroom_light"]
    vocabulary(context.data).affordance_names    ["brightness", "onOff"]

A context that came as text only (a stub, an older EnvExplorer) has `data=None`; the
variants that need the structure say so with `require_data`, instead of guessing it
back out of the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


# -- the context --------------------------------------------------------------

@dataclass(frozen=True)
class GoalContext:
    """One goal's home context: the prompt text, and the structure it came from."""

    text: str
    data: Optional[Dict[str, Any]] = None

    def require_data(self, who: str) -> Dict[str, Any]:
        """The structured context, or ValueError naming ``who`` that needs it.

        The error is for a configuration that cannot work (a variant that reads the
        structure, fed by a source that sends text), so it is raised, not papered over."""
        if self.data is None:
            raise ValueError(f"{who} needs the structured goal context, but the context "
                             f"source sent only its text")
        return self.data


# -- the vocabulary -----------------------------------------------------------

@dataclass
class Vocabulary:
    """The names and classes a goal context offers, each list in first-seen order.

    Every field is the set of values a structured reply may put in one slot:
    `room_classes` for `location_class`, `artifact_names` for `artifact_name`, and
    so on. A list is empty when the context has none of that kind (no environment
    variables when the goal needed none), and the caller then leaves the slot free."""

    room_classes: List[str] = field(default_factory=list)
    room_names: List[str] = field(default_factory=list)
    artifact_classes: List[str] = field(default_factory=list)
    artifact_names: List[str] = field(default_factory=list)
    affordance_classes: List[str] = field(default_factory=list)
    affordance_names: List[str] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)
    property_classes: List[str] = field(default_factory=list)
    property_names: List[str] = field(default_factory=list)
    variable_classes: List[str] = field(default_factory=list)
    variable_names: List[str] = field(default_factory=list)
    quantity_kinds: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)


def _add(values: List[str], new: Iterable[Any]) -> None:
    """Append the strings of ``new`` that ``values`` does not hold yet."""
    for value in new:
        if isinstance(value, str) and value and value not in values:
            values.append(value)


def _parameters(schema: Dict[str, Any], out: List[str]) -> None:
    """The parameter names of an input schema, nested parameters included."""
    for parameter in schema.get("parameters") or []:
        _add(out, [parameter.get("parameter_name")])
        _parameters(parameter.get("parameter_schema") or {}, out)


def _workspace(node: Dict[str, Any], vocab: Vocabulary) -> None:
    """Add one workspace, its environment, devices and sub-workspaces to ``vocab``."""
    # 1. The room itself.
    _add(vocab.room_names, [node.get("title")])
    _add(vocab.room_classes, [node.get("space_class")])

    # 2. The room's environment variables, when the goal asked for them.
    environment = node.get("environment") or {}
    for variable in environment.get("variables") or []:
        _add(vocab.variable_names, [variable.get("title")])
        _add(vocab.variable_classes, [variable.get("class")])
        _add(vocab.quantity_kinds, [variable.get("quantity_kind")])
        _add(vocab.units, [variable.get("unit")])

    # 3. Each device: its classes and name, what it does, and what it reports.
    for artifact in node.get("artifacts") or []:
        _add(vocab.artifact_names, [artifact.get("title")])
        _add(vocab.artifact_classes, artifact.get("homeont_types") or [])
        for action in artifact.get("actions") or []:
            _add(vocab.affordance_names, [action.get("title")])
            _add(vocab.affordance_classes, action.get("command_types") or [])
            _parameters(action.get("input") or {}, vocab.parameter_names)
        for prop in artifact.get("properties") or []:
            _add(vocab.property_names, [prop.get("title")])
            _add(vocab.property_classes, [prop.get("class")])

    # 4. Rooms nested under this one.
    for sub in node.get("workspaces") or []:
        _workspace(sub, vocab)


def vocabulary(data: Dict[str, Any]) -> Vocabulary:
    """The `Vocabulary` of a structured goal context (`GoalContext.data`)."""
    vocab = Vocabulary()
    for node in data.get("workspaces") or []:
        _workspace(node, vocab)
    return vocab
