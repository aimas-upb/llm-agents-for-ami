"""The variants of the goal-structuring call (`prompt_variants.structuring`).

Structuring is one LLM call per goal: a system prompt (rules, a guide to the home
context, the context), the request as the user message, a free-text JSON reply. Five
axes say how that call is built (see `shared/utils/prompt_variants.py` for the
mechanism, and the `doc` of each axis below for what its options receive and return):

    context      how the home context is rendered         text (default) | json | json_v2
    prompt       which system prompt is sent              full (default) | focused
    constraint   whether the reply is schema-constrained  none (default) | schema
    decoding     how the reply's tokens are chosen        free (default) | iri_trie
    repair       what is fixed in the parsed reply        none (default) | class_names

With every axis at its default the call is the one structuring always made, byte for
byte. The harness's experiment arms are combinations (`tests/homebench/variants/`):

    A control    (nothing set)
    B            constraint: schema
    C            context: json        (version 1 of the JSON prompt, see below)
    D            prompt: focused
    E            repair: class_names
    F            context: json_v2     (version 2 of the JSON prompt)
    G            decoding: iri_trie   (GENRE-style trie of the home's IRIs, `iri_trie.py`)

Versions of the prompt. The structuring prompt is the rules (what to fill, from what),
the guide to the context's format, and the context. The rules were written for the
text rendering and name things as it writes them (`Artifact`, `name`, `command class`).
Each `context` option therefore fixes both the format of the context and the version of
the prompt that goes with it, and the option name is the version:

    text      the text context, its guide, the rules in its words       (the original)
    json      JSON context + guide v1, but still the rules in the text's words: a model
              is left to translate `name` of an `Artifact` into `title` of `artifacts`
              (kept as it ran: arm C of the finished runs)
    json_v2   JSON context + guide v2 + the rules reworded in the JSON's keys
              (`goal_structuring_prompts.JSON_REWORDINGS`); nothing to translate

The option is in the variant file, the run's `run.json`, and the header of every report
(`tests/simuhome/bench/run_config.py`), so what a result ran with is never inferred
from the prompt text.

Flow, on "Set the brightness of the light to 10 in the master bedroom" with `B`:

    1. rendered = context option(context)          the text of the home + its guide
    2. system   = prompt option(structure, rendered, qualifiers)
    3. format   = constraint option(...)           a json_schema whose enums are this
                                                   home's classes and names, so a reply
                                                   cannot write `DimmableLight` where
                                                   only `homeont:DimmableLight` exists
    4. parsed   = repair option(parsed, context)   applied to the reply, after parsing

Two things are structural limits, not choices. The schema and the focused prompt are
written for the `simple` structure (one goal), which is 99% of the goals of the
ECAI-400 runs; for the `dependency` structure (goals and predicates under open keys
G1, G2, ... that a strict schema cannot name) `constraint: schema` sends no schema and
`prompt: focused` sends the full prompt. And an option that reads the structured
context raises (`GoalContext.require_data`) when the context source sent text only.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ....shared.utils.prompt_variants import declare_step
from ..prompts.goal_structuring_prompts import (
    CONTEXT_GUIDE,
    CONTEXT_GUIDE_JSON,
    CONTEXT_GUIDE_JSON_V2,
    DEVICE_CONVENTIONS,
    DIALECT_JSON,
    DIALECT_TEXT,
    GOAL_CONTEXT_PLACEHOLDER,
    PROMPTS_BY_DIALECT,
    render_focused_prompt,
)
from .goal_context import GoalContext, Vocabulary, vocabulary
from .iri_trie import decode_with_iri_trie

# A home with a device that runs cycles needs the "start" convention in the prompt;
# this class name in the context is how the focused prompt knows.
CYCLE_COMMAND = "SetOperationalStateCommand"

# The name `response_format` carries, for the API's own logs.
SCHEMA_NAME = "goal_structure"

STEP = declare_step("structuring", agent="user_assistant")


@dataclass(frozen=True)
class RenderedContext:
    """The home context as sent: the guide that explains its format, the body, and the
    dialect of the rules that go with it (`DIALECT_TEXT`: the rules in the words of the
    text rendering; `DIALECT_JSON`: reworded in the JSON's keys)."""

    guide: str
    body: str
    dialect: str = DIALECT_TEXT


def _labels(qualifiers: Sequence[str]) -> set:
    """The segmenter's qualifiers, lower-cased (the router's own normalisation)."""
    return {str(q).strip().lower() for q in (qualifiers or [])}


# -- axis: context -----------------------------------------------------------

CONTEXT = STEP.axis(
    "context", "text",
    "How the home context is rendered. An option takes the `GoalContext` and returns "
    "a `RenderedContext` (guide, body, dialect): the guide describes the body's format "
    "and the dialect says which version of the rules goes with it, so the three always "
    "travel together. The option's name is the prompt's version (module docstring).")


@CONTEXT.option("text")
def text_context(context: GoalContext) -> RenderedContext:
    """The indented text EnvExplorer renders, with the guide written for it: what
    structuring always sent."""
    return RenderedContext(CONTEXT_GUIDE, context.text)


def _without_none(value: Any) -> Any:
    """``value`` with every None dropped from its dicts (lists kept whole: an empty
    `observed_by` says that nothing in the home reads the variable)."""
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


def _json_body(data: Dict[str, Any]) -> str:
    """The structured view as indented JSON, nulls dropped: the body of both JSON
    contexts, so the two versions differ in the prompt's words only."""
    return json.dumps(_without_none(data), indent=1, ensure_ascii=False)


@CONTEXT.option("json")
def json_context(context: GoalContext) -> RenderedContext:
    """Version 1 of the JSON prompt: the structured view as indented JSON, with a guide
    to its keys, under the rules still written in the text's words.

    The same facts as the text, under the keys the structured view has; the rules keep
    saying `Artifact`, `name`, `command class`, which the JSON does not have. Kept as
    it ran (arm C); `json_v2` is the version whose rules speak the JSON's keys."""
    data = context.require_data("context=json")
    return RenderedContext(CONTEXT_GUIDE_JSON, _json_body(data))


@CONTEXT.option("json_v2")
def json_v2_context(context: GoalContext) -> RenderedContext:
    """Version 2 of the JSON prompt: the same JSON, its guide v2 and the rules reworded
    in the JSON's keys (`title`, `command_types`, `acts_upon`, ...), so a model has no
    text-format word to translate. Same facts as `json`; a difference in results is the
    wording of the prompt's."""
    data = context.require_data("context=json_v2")
    return RenderedContext(CONTEXT_GUIDE_JSON_V2, _json_body(data), DIALECT_JSON)


# -- axis: prompt ------------------------------------------------------------

PROMPT = STEP.axis(
    "prompt", "full",
    "Which system prompt is sent. An option takes the structure (`simple` | "
    "`dependency`), the `RenderedContext` and the segmenter's qualifiers, and returns "
    "the system prompt text.")


@PROMPT.option("full")
def full_prompt(structure: str, rendered: RenderedContext,
                qualifiers: Sequence[str]) -> str:
    """The structure's own prompt in the rendered context's dialect, the context filled
    in; its guide swapped for the rendered one (a no-op for the text context, whose
    guide it already holds, and for the JSON dialect, which holds its own)."""
    template = PROMPTS_BY_DIALECT[rendered.dialect][structure].replace(
        CONTEXT_GUIDE, rendered.guide)
    return template.replace(GOAL_CONTEXT_PLACEHOLDER, rendered.body)


@PROMPT.option("focused")
def focused_prompt(structure: str, rendered: RenderedContext,
                   qualifiers: Sequence[str]) -> str:
    """Only what an explicit or incomplete achievement goal needs, plus two examples.

    Leaves out the ambiguous-goal rules and shape, and the device conventions unless
    the home has a device running cycles. Any other goal (a dependency structure, an
    ambiguous goal) gets the full prompt: its rules are what that goal needs."""
    # 1. A goal that needs the rules the focused prompt leaves out: the full prompt.
    if structure != "simple" or "ambiguous" in _labels(qualifiers):
        return full_prompt(structure, rendered, qualifiers)
    # 2. The cycle convention only for a home that has a cycle device.
    conventions = DEVICE_CONVENTIONS if CYCLE_COMMAND in rendered.body else ""
    return render_focused_prompt(rendered.guide, rendered.body, conventions,
                                 rendered.dialect)


# -- axis: constraint --------------------------------------------------------

CONSTRAINT = STEP.axis(
    "constraint", "none",
    "Whether the reply is constrained. An option takes the structure, the qualifiers "
    "and the `GoalContext`, and returns the `response_format` to send, or None to "
    "send none.")


@CONSTRAINT.option("none")
def no_constraint(structure: str, qualifiers: Sequence[str],
                  context: GoalContext) -> Optional[Dict[str, Any]]:
    """No `response_format`: the reply is free text, parsed loosely."""
    return None


def _choice(values: Sequence[str], extra: Sequence[str] = ()) -> Dict[str, Any]:
    """A string slot limited to ``values`` (and ``extra``) -- or any string when the
    context has none of that kind -- or null."""
    allowed = list(dict.fromkeys([*values, *extra]))
    base: Dict[str, Any] = {"type": "string", "enum": allowed} if allowed else {"type": "string"}
    return {"anyOf": [base, {"type": "null"}]}


def _fixed(values: Sequence[str]) -> Dict[str, Any]:
    """A string slot that must be one of ``values`` (never null)."""
    return {"type": "string", "enum": list(values)}


def _nullable_string() -> Dict[str, Any]:
    """A free string slot, or null."""
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _object(properties: Dict[str, Any]) -> Dict[str, Any]:
    """An object with exactly these properties, all required: what OpenAI's strict
    mode demands, and what makes a missing field an explicit null."""
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def goal_schema(vocab: Vocabulary) -> Dict[str, Any]:
    """The schema of an explicit or incomplete goal, in the order of the prompt's
    output shape: the slots that name something of the home take that home's names."""
    return _object({
        "goal_specificity": _fixed(["explicit", "incomplete"]),
        "goal_kind": _fixed(["achievement"]),
        "intent_text": {"type": "string"},
        "location_class": _choice(vocab.room_classes),
        "artifact_class": _choice(vocab.artifact_classes),
        "artifact_name": _choice(vocab.artifact_names),
        "device_reference": _choice(["name", "description"]),
        "affordance_class": _choice(vocab.affordance_classes),
        "parent_class": _nullable_string(),
        "affordance_name": _choice(vocab.affordance_names),
        # "NA" (needs no parameter) and null (needs one, unknown) stay distinct.
        "parameter_name": _choice(vocab.parameter_names, extra=["NA"]),
        "goal_effect": _choice(["set", "modify"]),
        "target_value_text": _nullable_string(),
        "target_value_determined": _nullable_string(),
        "target_value_is_percentage": {"type": "boolean"},
        "quantifier": _choice(["all", "one"]),
        "unmatched_device": _nullable_string(),
        "unmatched_action": _nullable_string(),
    })


def ambiguous_goal_schema(vocab: Vocabulary) -> Dict[str, Any]:
    """The schema of an ambiguous goal: no device, the environment variables it implies,
    each taken from this home's variables and rooms."""
    variable = _object({
        "property_name": _choice(vocab.variable_names),
        "property_class": _choice(vocab.variable_classes),
        "property_quantity": _choice(vocab.quantity_kinds),
        "property_measurement_unit": _choice(vocab.units),
        "property_sensed_space": _object({
            "space_class": _choice(vocab.room_classes),
            "space_name": _choice(vocab.room_names),
        }),
        "property_change_direction": _fixed(["increase", "decrease"]),
    })
    return _object({
        "goal_specificity": _fixed(["ambiguous"]),
        "goal_kind": _fixed(["achievement"]),
        "intent_text": {"type": "string"},
        "implied_environment_vars": {"type": "array", "items": variable},
    })


@CONSTRAINT.option("schema")
def schema_constraint(structure: str, qualifiers: Sequence[str],
                      context: GoalContext) -> Optional[Dict[str, Any]]:
    """A strict `json_schema` whose enums are the classes and names of this home.

    The prompt tells the model to copy classes and names from the context; the schema
    makes anything else impossible to write. It does not make a copied value the right
    one (a wrong room is still a room of the home). None for the dependency structure:
    its goals and predicates sit under open keys (G1, G2, ...) that a strict schema
    cannot name."""
    # 1. Only the simple structure has a closed shape.
    if structure != "simple":
        return None
    # 2. The shape follows the segmenter's label: an ambiguous goal names variables,
    #    any other names a device and an action.
    vocab = vocabulary(context.require_data("constraint=schema"))
    schema = (ambiguous_goal_schema(vocab) if "ambiguous" in _labels(qualifiers)
              else goal_schema(vocab))
    return {"type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": schema}}


# -- axis: decoding ----------------------------------------------------------

DECODING = STEP.axis(
    "decoding", "free",
    "How the reply is decoded. An (async) option takes the LLM client config "
    "(`client`, `model`), the messages, the call's kwargs and the `GoalContext`, and "
    "returns a `DecodedReply` (the reply text; a record of what the decoding changed, "
    "or None).")


@dataclass(frozen=True)
class DecodedReply:
    """The reply's text, and what the decoding did (None: one free call, as always)."""

    text: str
    record: Optional[Dict[str, Any]] = None


@DECODING.option("free")
async def free_decoding(llm_cfg: Any, messages: List[Dict[str, str]],
                        kwargs: Dict[str, Any], context: GoalContext) -> DecodedReply:
    """One call, the reply as the model decodes it: what structuring always did."""
    response = await llm_cfg.client.chat.completions.create(
        model=llm_cfg.model, messages=messages, **kwargs)
    return DecodedReply(response.choices[0].message.content or "")


@DECODING.option("iri_trie")
async def iri_trie_decoding(llm_cfg: Any, messages: List[Dict[str, str]],
                            kwargs: Dict[str, Any], context: GoalContext) -> DecodedReply:
    """GENRE-style constrained decoding of the names: each IRI value in the reply is
    decoded under a trie of the values the home allows there, narrowed by what the
    same object already names (a device of the written room, an action of the written
    device); the rest of the reply is free (`utils/iri_trie.py`).

    Needs an endpoint that continues a final assistant message and returns
    `top_logprobs` (Ollama). Not with `constraint: schema`: Ollama cannot apply a
    grammar to a continued message."""
    if "response_format" in kwargs:
        raise ValueError("decoding=iri_trie cannot be combined with a response_format "
                         "(constraint=schema)")
    data = context.require_data("decoding=iri_trie")
    decoded = await decode_with_iri_trie(llm_cfg.client.chat.completions.create,
                                         llm_cfg.model, messages, kwargs, data)
    return DecodedReply(decoded.text, decoded.as_dict())


# -- axis: repair ------------------------------------------------------------

REPAIR = STEP.axis(
    "repair", "none",
    "What is fixed in the parsed reply. An option takes the parsed JSON object and "
    "the `GoalContext`, and returns (the reply, possibly a repaired copy; one log line "
    "per change).")


@REPAIR.option("none")
def no_repair(parsed: Dict[str, Any],
              context: GoalContext) -> Tuple[Dict[str, Any], List[str]]:
    """The reply as the model wrote it."""
    return parsed, []


# The slots that hold a class, and where the home's classes for each are listed. A
# `property_class` is a device property's class or an environment variable's.
CLASS_SLOTS: Dict[str, Tuple[str, ...]] = {
    "location_class": ("room_classes",),
    "space_class": ("room_classes",),
    "artifact_class": ("artifact_classes",),
    "affordance_class": ("affordance_classes",),
    "property_class": ("property_classes", "variable_classes"),
}


def _notation(name: str) -> str:
    """A class name without its prefix, case or spacing: `homeont:MasterBedroom`,
    `MasterBedroom` and `Master bedroom` all become `masterbedroom`."""
    return re.sub(r"[^a-z0-9]", "", name.rsplit(":", 1)[-1].lower())


def canonical_class(value: str, candidates: Sequence[str]) -> str:
    """The home's class that ``value`` writes differently, else ``value`` unchanged.

    A class the home has is kept. Otherwise the one candidate with the same name
    (prefix, case and spacing ignored) is returned; none, or several, leave the value
    alone: a name that fits two classes is a guess, and an unknown one is the model's
    error to keep, not to hide."""
    if value in candidates:
        return value
    matches = {c for c in candidates if _notation(c) == _notation(value)}
    return next(iter(matches)) if len(matches) == 1 else value


def _repair_node(node: Any, vocab: Vocabulary, log: List[str]) -> None:
    """Rewrite, in place, every class slot of ``node`` and of what it holds."""
    # 1. A list (the dependency structure's lists, implied variables): each item.
    if isinstance(node, list):
        for item in node:
            _repair_node(item, vocab, log)
    # 2. An object: a class slot holding a string is canonicalised against the
    #    classes the home lists for that slot; any other value is searched inside,
    #    since goals, predicates and variables nest under other keys.
    elif isinstance(node, dict):
        for key, value in list(node.items()):
            if key in CLASS_SLOTS and isinstance(value, str):
                candidates = [c for field in CLASS_SLOTS[key] for c in getattr(vocab, field)]
                fixed = canonical_class(value, candidates)
                if fixed != value:
                    node[key] = fixed
                    log.append(f"{key}: {value!r} -> {fixed!r}")
            else:
                _repair_node(value, vocab, log)


@REPAIR.option("class_names")
def class_names_repair(parsed: Dict[str, Any],
                       context: GoalContext) -> Tuple[Dict[str, Any], List[str]]:
    """Write each class the reply names as the home writes it: `Foyer` as
    `homeont:Foyer`, `Master bedroom` as `homeont:MasterBedroom`.

    Notation only: the class must exist in the home's context under that name. A class
    in the wrong slot, or one the home lacks, stays as it is. This is what separates
    "the model chose wrongly" from "the model chose right and wrote it differently"."""
    vocab = vocabulary(context.require_data("repair=class_names"))
    repaired = copy.deepcopy(parsed)
    log: List[str] = []
    _repair_node(repaired, vocab, log)
    return repaired, log
