"""Constrained decoding of the names in a structuring reply, with a trie of the home's
valid IRIs (`decoding: iri_trie`).

The idea is GENRE's (De Cao et al., "Autoregressive Entity Retrieval", ICLR 2021,
github.com/facebookresearch/GENRE): a model writes an entity's name token by token,
and at every step only the tokens that keep the written prefix inside a prefix tree
(trie) of the valid names may be chosen, so the name written is always one that
exists. Here the entities are the values of the slots that name something of the
home: `location_class`, `artifact_name`, `affordance_class`, ... (`SLOT_VOCABULARY`).
The rest of the reply (the thinking, the keys, the free-text slots, a `null`) is
decoded as before; only the inside of a quoted IRI value is constrained.

Two things differ from `constraint: schema` (flat enums turned into a grammar):

    1. The trie of a slot is conditioned on what the same JSON object already holds
       (`iri_paths`): once `location_class` is `homeont:MasterBedroom`, the trie of
       `artifact_name` holds only that room's devices, and once the device is
       written, `affordance_name` holds only that device's actions. A flat enum lets
       a reply put a device of another room, or an action of another device.
    2. Only the IRI spans are constrained: the model still chooses `null` (no device
       of the home fits), and reasoning models think unconstrained.

How it runs on Ollama, whose API has no logits processor: the reply is first decoded
greedily as always (the control call). Each IRI value in it is checked against its
trie; a value that is in the trie is exactly what constrained greedy decoding would
have written too (every prefix of it is in the trie, so the greedy token was allowed
at every step), and it is kept. At the first value that is not, the reply is cut at
the opening quote and the value is decoded again, one token per call (the reply so
far prefilled as the assistant's message, `max_tokens=1`, `top_logprobs=20`): the
most probable of the 20 tokens that the trie allows is taken. Then the model
continues freely from the corrected value, and the check goes on. When none of the
20 tokens is allowed, the model's own free continuation of the value is snapped to
the nearest completion in the trie (`nearest`, logged as such): the API shows 20
tokens, not the whole vocabulary. Every re-decoded value is logged, and every decision
(the tokens offered, which was taken, what was forced) is traced for
`tests/homebench/decoding_report.py`.

Whether a value is valid is decided on the slots written before it, as constrained
greedy decoding would; but the trie a value is re-decoded with is also narrowed by
the slots the model wrote after it in the same object (`narrow_by_evidence`): without
that, a guessed `affordance_class` made the correct `affordance_name` after it invalid
and it was overwritten too (`hb1_feasible_seed_4351`, `temperature` -> `mode`).

    reply   ... "location_class": "Master bedroom", "artifact_class": ...
    trie    location_class: {homeont:MasterBedroom, homeont:LaundryRoom}
    step 1  "homeont:" is forced (every room starts so); top tokens "Master" (-0.01),
            "master", "Bed": "Master" is the first the trie allows
    step 2  ... -> "homeont:MasterBedroom" (forced where the trie has one branch)
    close   the model's own closing token, `",` (a bare `"` where it would write `",`
            makes it skip the comma: the token-boundary problem GENRE avoids by
            working on token ids)
    then    the model writes the rest of the reply after the corrected value
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Tuple

from .goal_context import Vocabulary, vocabulary

# The slots whose value names something of the home, and the vocabulary fields that
# list what each may hold (`goal_context.Vocabulary`). A `property_*` slot names a
# device property in a predicate or an environment variable in an ambiguous goal.
SLOT_VOCABULARY: Dict[str, Tuple[str, ...]] = {
    "location_class": ("room_classes",),
    "artifact_class": ("artifact_classes",),
    "artifact_name": ("artifact_names",),
    "affordance_class": ("affordance_classes",),
    "affordance_name": ("affordance_names",),
    "parameter_name": ("parameter_names",),
    "property_name": ("property_names", "variable_names"),
    "property_class": ("property_classes", "variable_classes"),
    "property_quantity": ("quantity_kinds",),
    "property_measurement_unit": ("units",),
    "space_class": ("room_classes",),
    "space_name": ("room_names",),
}

# Values a slot may hold besides the home's names: "NA" is "needs no parameter".
EXTRA_VALUES: Dict[str, Tuple[str, ...]] = {"parameter_name": ("NA",)}

# The opening of a slot's quoted value in the reply, and a slot written with its value.
SLOT_OPEN = re.compile(r'"(%s)"\s*:\s*"' % "|".join(SLOT_VOCABULARY))
SLOT_VALUE = re.compile(r'"(%s)"\s*:\s*"([^"]*)"' % "|".join(SLOT_VOCABULARY))

THINK_END = "</think>"

# What Ollama returns at most per step (it rejects top_logprobs above 20).
TOP_LOGPROBS = 20
# A value longer than this many decoding steps is cut and snapped (no IRI of a home
# is near it; it bounds a runaway loop).
MAX_VALUE_STEPS = 24
# Values re-decoded in one reply at most (a goal has 6 such slots; a dependency
# structure a few goals of them).
MAX_INTERVENTIONS = 24
# Tokens of free continuation the `nearest` fallback reads to know what the model meant.
NEAREST_TOKENS = 24
# Characters of the reply before a re-decoded value kept in the trace (what the model
# sees just before it, for the decoding report).
TRACE_TAIL = 160

_END = ""      # the trie key that marks a complete value


# -- the trie -----------------------------------------------------------------

class IriTrie:
    """A character trie of the values a slot may hold.

    Characters, not the model's tokens: a token is allowed when the value written so
    far plus the token's text is still a prefix of a value, so every tokenization of
    a valid name is accepted (GENRE builds its trie on token ids of one tokenization)."""

    def __init__(self, values: List[str]) -> None:
        self._root: Dict[str, Any] = {}
        self.values = list(dict.fromkeys(values))
        for value in self.values:
            node = self._root
            for char in value:
                node = node.setdefault(char, {})
            node[_END] = {}

    def _node(self, prefix: str) -> Optional[Dict[str, Any]]:
        node = self._root
        for char in prefix:
            node = node.get(char)
            if node is None:
                return None
        return node

    def has_prefix(self, prefix: str) -> bool:
        """True when some value starts with ``prefix``."""
        return self._node(prefix) is not None

    def is_complete(self, value: str) -> bool:
        """True when ``value`` is one of the values."""
        node = self._node(value)
        return node is not None and _END in node

    def is_leaf(self, value: str) -> bool:
        """True when ``value`` is complete and no longer value extends it."""
        node = self._node(value)
        return node is not None and set(node) == {_END}

    def forced_extension(self, prefix: str) -> str:
        """The characters every value starting with ``prefix`` continues with: there is
        nothing to choose until the trie branches or a value ends."""
        node, out = self._node(prefix), []
        while node is not None and len(node) == 1 and _END not in node:
            char = next(iter(node))
            out.append(char)
            node = node[char]
        return "".join(out)

    def completions(self, prefix: str) -> List[str]:
        """The values that start with ``prefix``, in insertion order."""
        return [v for v in self.values if v.startswith(prefix)]

    def accept(self, prefix: str, token: str) -> Optional[Tuple[str, Optional[str]]]:
        """What writing ``token`` after ``prefix`` gives, if the trie allows it.

        Returns (the value, the token from its quote on) when the token closes the
        string on a complete value (`"`, `",`, ...), (the longer prefix, None) when it
        stays inside a value, None when it leaves the trie (an empty token too)."""
        if '"' in token:
            quote = token.index('"')
            value = prefix + token[:quote]
            return (value, token[quote:]) if self.is_complete(value) else None
        if token and self.has_prefix(prefix + token):
            return prefix + token, None
        return None


# -- the valid paths ----------------------------------------------------------

Row = Dict[str, FrozenSet[str]]


def _row(**slots: Any) -> Row:
    """A path row; a slot with no value is left out (it constrains nothing)."""
    return {k: frozenset(v) for k, v in slots.items() if v}


def _parameters(schema: Dict[str, Any]) -> List[str]:
    """The parameter names of an action's input, nested ones included."""
    out: List[str] = []
    for parameter in schema.get("parameters") or []:
        if parameter.get("parameter_name"):
            out.append(parameter["parameter_name"])
        out.extend(_parameters(parameter.get("parameter_schema") or {}))
    return out


def _artifact_rows(artifact: Dict[str, Any], rooms: List[str]) -> List[Row]:
    """One row per action and per property of a device, each with its rooms."""
    base = {"location_class": rooms, "artifact_class": artifact.get("homeont_types") or [],
            "artifact_name": [artifact.get("title")] if artifact.get("title") else []}
    rows = [_row(**base, affordance_name=[a.get("title")] if a.get("title") else [],
                 affordance_class=a.get("command_types") or [],
                 parameter_name=[*_parameters(a.get("input") or {}), "NA"])
            for a in artifact.get("actions") or []]
    rows += [_row(**base, property_name=[p.get("title")] if p.get("title") else [],
                  property_class=[p.get("class")] if p.get("class") else [])
             for p in artifact.get("properties") or []]
    return rows or [_row(**base)]


def _workspace_rows(node: Dict[str, Any], above: List[str]) -> List[Row]:
    """The rows of one workspace, its environment, devices and sub-workspaces. A device
    is in its own room and in every room that holds that room."""
    rooms = above + ([node["space_class"]] if node.get("space_class") else [])
    rows = [_row(space_class=[node.get("space_class")] if node.get("space_class") else [],
                 space_name=[node.get("title")] if node.get("title") else [])]
    for variable in (node.get("environment") or {}).get("variables") or []:
        rows.append(_row(**{key: [variable.get(src)] if variable.get(src) else []
                            for key, src in (("property_name", "title"),
                                             ("property_class", "class"),
                                             ("property_quantity", "quantity_kind"),
                                             ("property_measurement_unit", "unit"))}))
    for artifact in node.get("artifacts") or []:
        rows.extend(_artifact_rows(artifact, rooms))
    for sub in node.get("workspaces") or []:
        rows.extend(_workspace_rows(sub, rooms))
    return [row for row in rows if row]


def iri_paths(data: Dict[str, Any]) -> List[Row]:
    """The valid combinations of the home's names, one row per device action, device
    property, environment variable and room (`GoalContext.data` in)."""
    rows: List[Row] = []
    for node in data.get("workspaces") or []:
        rows.extend(_workspace_rows(node, []))
    return rows


@dataclass
class SlotValues:
    """What a slot may hold here, and whether that was narrowed by the object's other
    slots (False: no path fits them, so the slot's whole vocabulary is allowed)."""

    values: List[str]
    conditioned: bool


def flat_values(slot: str, vocab: Vocabulary) -> List[str]:
    """Every value the home offers for ``slot``, plus its extras: a home whose actions
    all take a direct value still has "NA" for `parameter_name` (as the schema's
    enum), so a parameter name the home lacks cannot be written."""
    values = [v for name in SLOT_VOCABULARY[slot] for v in getattr(vocab, name)]
    return list(dict.fromkeys([*values, *EXTRA_VALUES.get(slot, ())]))


def slot_values(slot: str, written: Dict[str, str], rows: List[Row],
                vocab: Vocabulary) -> SlotValues:
    """The values ``slot`` may take given the ``written`` slots of the same object.

    A row fits when it holds every written slot with the written value; the slot may
    take what the fitting rows hold for it. When none fits (the object mixes a device
    with another room's action, say), the flat vocabulary: the trie never makes a
    reply impossible to write."""
    flat = flat_values(slot, vocab)
    if not flat:
        return SlotValues([], False)
    given = {k: v for k, v in written.items() if k != slot and v in flat_values(k, vocab)}
    fitting = [row for row in rows if slot in row
               and all(k in row and v in row[k] for k, v in given.items())]
    values = [v for v in flat if any(v in row[slot] for row in fitting)]
    if slot in EXTRA_VALUES and values:
        values = list(dict.fromkeys([*values, *EXTRA_VALUES[slot]]))
    return SlotValues(values, True) if values else SlotValues(flat, False)


# -- reading the reply --------------------------------------------------------

def _object_start(text: str, pos: int) -> int:
    """Where the innermost JSON object open at ``pos`` starts (0 if none)."""
    stack: List[int] = []
    in_string = escaped = False
    for i, char in enumerate(text[:pos]):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            stack.append(i)
        elif char == "}" and stack:
            stack.pop()
    return stack[-1] if stack else 0


def _object_end(text: str, start: int) -> int:
    """Where the JSON object opening at ``start`` closes (the end of ``text`` when the
    reply is cut off before it does)."""
    depth = 0
    in_string = escaped = False
    for i in range(start, len(text)):
        char = text[i]
        # 1. Braces inside a string value are text, not structure.
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        # 2. The object closes when its own brace does.
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth <= 0:
                return i
    return len(text)


def written_slots(text: str, pos: int) -> Dict[str, str]:
    """The slot values already written in the object open at ``pos``."""
    return {m.group(1): m.group(2)
            for m in SLOT_VALUE.finditer(text, _object_start(text, pos), pos)}


def later_slots(text: str, value_start: int) -> Dict[str, str]:
    """The slot values the reply writes after the value opening at ``value_start``, in
    the same object: what the model went on to name (the first of each slot)."""
    value_end = text.find('"', value_start)
    if value_end < 0:
        return {}
    end = _object_end(text, _object_start(text, value_start))
    out: Dict[str, str] = {}
    for match in SLOT_VALUE.finditer(text, value_end + 1, end):
        out.setdefault(match.group(1), match.group(2))
    return out


def narrow_by_evidence(slot: str, before: Dict[str, str], after: Dict[str, str],
                       rows: List[Row], vocab: Vocabulary,
                       allowed: SlotValues) -> Tuple[SlotValues, Dict[str, str]]:
    """The values an invalid ``slot`` is decoded with, narrowed by the slots the model
    wrote after it in the same object, and those slots (the evidence used).

    Why: the trie is conditioned left to right, so a value it re-decodes is chosen
    before the slots that follow, and those are then checked against it. In
    `hb1_feasible_seed_4351` the free reply had `affordance_class:
    homeont:TemperatureSetpoint` (invalid) and then `affordance_name: temperature`
    (an action of that air conditioner). Greedy took the `homeont:` branch (the
    model's first token), wrote `homeont:SetModeCommand`, and then the correct
    `temperature` no longer fit and became `mode`. The later slots are the model's
    own choices; a value consistent with them is decoded instead, here
    `saref:SetAbsoluteLevelCommand`, the only class of `temperature`.

    Each later slot is taken in reply order if, with what is taken so far, it leaves
    the slot fewer values than before, but some. One that fits nothing (a device of
    another room, an invented name) or changes nothing (`parameter_name: NA`) is
    passed over. Nothing narrows: ``allowed`` as it is."""
    values, used = allowed.values, {}
    for key, value in after.items():
        # 1. Only another slot whose value names something of the home is evidence.
        if key == slot or key in before or value not in flat_values(key, vocab):
            continue
        # 2. Taken if it leaves the slot fewer values, but some.
        trial = slot_values(slot, {**before, **used, key: value}, rows, vocab)
        if trial.conditioned and trial.values and set(trial.values) < set(values):
            used[key] = value
            values = trial.values
    return SlotValues(values, allowed.conditioned or bool(used)), used


@dataclass
class InvalidSlot:
    """A slot whose value is not in its trie: where the value starts, what was written
    (None: cut off before its closing quote) and the trie it must be decoded with; the
    slots written before it and the later slots that narrowed the trie (evidence)."""

    slot: str
    value_start: int
    written: Optional[str]
    trie: IriTrie
    conditioned: bool
    before: Dict[str, str] = field(default_factory=dict)
    evidence: Dict[str, str] = field(default_factory=dict)


def first_invalid_slot(text: str, pos: int, rows: List[Row],
                       vocab: Vocabulary) -> Optional[InvalidSlot]:
    """The first slot value at or after ``pos`` that its trie does not hold.

    Valid or not is decided on the slots written before (what constrained greedy
    decoding sees, so a kept value is the one it would write); the trie of an invalid
    value is then narrowed by the slots after it (`narrow_by_evidence`)."""
    for match in SLOT_OPEN.finditer(text, pos):
        # 1. What the slot may hold given the object so far.
        slot, start = match.group(1), match.end()
        before = written_slots(text, match.start())
        allowed = slot_values(slot, before, rows, vocab)
        if not allowed.values:
            continue                      # the home has nothing of this kind: free
        # 2. A value already in the trie is kept.
        end = text.find('"', start)
        written = text[start:end] if end >= 0 else None
        if written is not None and written in allowed.values:
            continue
        # 3. The first that is not: its trie, narrowed by what the model wrote next.
        narrowed, evidence = narrow_by_evidence(slot, before, later_slots(text, start),
                                                rows, vocab, allowed)
        return InvalidSlot(slot, start, written, IriTrie(narrowed.values),
                           narrowed.conditioned, before, evidence)
    return None


def _notation(value: str) -> str:
    """A name without case, spacing or punctuation, for the nearest-value fallback."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def nearest(meant: str, candidates: List[str]) -> str:
    """The candidate closest to what the model meant (character similarity)."""
    return max(candidates, key=lambda c: difflib.SequenceMatcher(
        None, _notation(meant), _notation(c)).ratio())


# -- decoding -----------------------------------------------------------------

Create = Callable[..., Awaitable[Any]]


@dataclass
class TrieDecoding:
    """The reply as decoded, and what the trie changed in it: one entry per value
    decoded again (slot, what the free decoding wrote, what the trie decoding wrote,
    how: `trie` or `nearest`, steps, whether the trie was conditioned, the later slots
    that narrowed it), and the trace of every decision, in order (`TRACE_EVENTS`)."""

    text: str
    calls: int = 0
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    abstain: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """The record's `decoding` entry (`abstain` only when the option set it, so a
        record of `iri_trie` keeps the shape it always had)."""
        record = {"option": "iri_trie", "calls": self.calls,
                  "interventions": self.interventions, "trace": self.trace}
        if self.abstain:
            record["abstain"] = True
        return record


# What the trace holds, one event per decision, in the order they were taken (what
# `tests/homebench/decoding_report.py` draws). `call` is the decoding's call number
# (1 is the free reply); `top` the [token, logprob] pairs the call returned.
TRACE_EVENTS = {
    "free": "the free reply (call 1): text",
    "kept": "a slot value in its trie, kept with no call: slot, value",
    "invalid": "a value outside its trie: slot, free, before, evidence, allowed, "
               "conditioned, tail (the reply just before the value)",
    "forced": "characters the trie forces (one branch), written with no call: chars",
    "step": "one constrained step: call, prefix (value so far), top, rank (of the "
            "token taken; None: none allowed)",
    "abstain": "the invalid `artifact_class` names a device kind the room lacks, so the "
               "goal is rewritten as `unmatched_device`, nothing decoded: words",
    "nearest": "no top token allowed: call, meant (free continuation), value",
    "close": "the token that closes the value: token, call and top (None: it came "
             "with the last step's token)",
    "resume": "the model's free continuation after the value: call, text",
    "done": "the reply as decoded: text",
}


def _event(trace: List[Dict[str, Any]], kind: str, **fields: Any) -> None:
    """Append one trace event (`TRACE_EVENTS`)."""
    trace.append({"event": kind, **fields})


def _reasoning(message: Any) -> str:
    """The thinking Ollama returned apart from the content (empty when none)."""
    return getattr(message, "reasoning", None) or getattr(
        message, "reasoning_content", None) or ""


def _prefilled(messages: List[Dict[str, str]], thinking: str,
               text: str) -> List[Dict[str, str]]:
    """The messages with the reply so far as the assistant's unfinished turn: the model
    continues it (Ollama continues a final assistant message)."""
    head = f"<think>{thinking}{THINK_END}" if thinking else ""
    return [*messages, {"role": "assistant", "content": head + text}]


def _step_kwargs(kwargs: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """The call's settings for one constrained step (output length replaced)."""
    out = {k: v for k, v in kwargs.items() if k not in ("max_tokens", "max_completion_tokens")}
    return {**out, **extra}


class _Decoder:
    """One reply's decoding: the calls it makes and what it changed."""

    def __init__(self, create: Create, model: str, messages: List[Dict[str, str]],
                 kwargs: Dict[str, Any], thinking: str, top_k: int,
                 trace: List[Dict[str, Any]]) -> None:
        self.create, self.model, self.messages = create, model, messages
        self.kwargs, self.thinking, self.top_k = kwargs, thinking, top_k
        self.trace = trace
        self.calls = 0

    async def _call(self, text: str, **kwargs: Any) -> Any:
        self.calls += 1
        return await self.create(model=self.model,
                                 messages=_prefilled(self.messages, self.thinking, text),
                                 **kwargs)

    async def _top_tokens(self, text: str) -> List[Tuple[str, float]]:
        """The top tokens after ``text`` with their logprobs, most probable first."""
        response = await self._call(text, **_step_kwargs(
            self.kwargs, max_tokens=1, logprobs=True, top_logprobs=self.top_k))
        logprobs = response.choices[0].logprobs
        content = logprobs.content if logprobs else []
        tops = sorted(content[0].top_logprobs, key=lambda t: -t.logprob) if content else []
        return [(top.token, round(top.logprob, 4)) for top in tops]

    async def _next_token(self, text: str, value: str,
                          trie: IriTrie) -> Optional[Tuple[str, Optional[str]]]:
        """The most probable of the top tokens that the trie allows after ``value``
        (traced with all of them, and the rank of the one taken)."""
        tops = await self._top_tokens(text + value)
        for rank, (token, _) in enumerate(tops):
            accepted = trie.accept(value, token)
            if accepted is not None:
                _event(self.trace, "step", call=self.calls + 1, prefix=value, top=tops,
                       rank=rank)
                return accepted
        _event(self.trace, "step", call=self.calls + 1, prefix=value, top=tops,
               rank=None)
        return None

    async def _closing(self, text: str) -> str:
        """The token that closes a value the trie completed, as the model writes it:
        `",` or `"}` more often than a bare `"`. A bare quote where the model writes
        `",` leaves it believing the comma written, and the reply breaks."""
        tops = await self._top_tokens(text)
        tokens = [t for t, _ in tops if t.startswith('"')]
        closing = tokens[0] if tokens else '"'
        _event(self.trace, "close", token=closing, call=self.calls + 1, top=tops)
        return closing

    async def _nearest(self, text: str, value: str, trie: IriTrie) -> str:
        """No top token fits: what the model goes on to write, snapped to the trie."""
        response = await self._call(text + value, **_step_kwargs(
            self.kwargs, max_tokens=NEAREST_TOKENS, stop=['"']))
        meant = value + (response.choices[0].message.content or "").split('"')[0]
        snapped = nearest(meant, trie.completions(value) or trie.values)
        _event(self.trace, "nearest", call=self.calls + 1, meant=meant, value=snapped)
        return snapped

    async def _value(self, text: str, trie: IriTrie) -> Tuple[str, Optional[str], str, int]:
        """The value after ``text``, the closing token if one was decoded with it, how
        (`trie` | `nearest`) and the constrained steps it took."""
        value, steps = "", 0
        while steps < MAX_VALUE_STEPS:
            # 1. What the trie forces is written without a call.
            forced = trie.forced_extension(value)
            if forced:
                _event(self.trace, "forced", chars=forced)
            value += forced
            if trie.is_leaf(value):
                return value, None, "trie", steps
            # 2. A branch: one call, the most probable allowed token.
            steps += 1
            accepted = await self._next_token(text, value, trie)
            if accepted is None:
                return await self._nearest(text, value, trie), None, "nearest", steps
            value, closing = accepted
            if closing is not None:
                return value, closing, "trie", steps
        snapped = nearest(value, trie.completions(value) or trie.values)
        _event(self.trace, "nearest", call=None, meant=value, value=snapped)
        return snapped, None, "nearest", steps

    async def value(self, text: str, trie: IriTrie) -> Tuple[str, str, str, int]:
        """Decode one value after ``text`` (which ends at the opening quote): the value,
        the token that closes it, how (`trie` | `nearest`) and the constrained steps."""
        value, closing, how, steps = await self._value(text, trie)
        if closing is None:
            closing = await self._closing(text + value)
        else:
            _event(self.trace, "close", token=closing, call=None, top=None)
        return value, closing, how, steps

    async def resume(self, text: str) -> str:
        """The model's free continuation of the reply after ``text``."""
        response = await self._call(text, **self.kwargs)
        continuation = response.choices[0].message.content or ""
        _event(self.trace, "resume", call=self.calls + 1, text=continuation)
        return continuation


def _abstention(text: str, invalid: InvalidSlot, rows: List[Row],
                vocab: Vocabulary) -> Optional[Tuple[str, str, int]]:
    """The reply with the goal object of ``invalid`` rewritten as a goal about a missing
    device, the user's words for it, and where the check goes on; None when ``invalid``
    is not an `artifact_class` naming a device kind the room lacks, or its object is cut
    off (`utils/abstention.py`)."""
    # Imported here: abstention.py reads this module's helpers.
    from .abstention import abstain_object, absent_goal

    if invalid.slot != "artifact_class":
        return None
    words = absent_goal({**invalid.before, "artifact_class": invalid.written or ""},
                        rows, vocab)
    start = _object_start(text, invalid.value_start)
    end = _object_end(text, start)
    if words is None or end >= len(text):
        return None
    rewritten = abstain_object(text[start:end + 1], words)
    return text[:start] + rewritten + text[end + 1:], words, start + len(rewritten)


async def decode_with_iri_trie(create: Create, model: str,
                               messages: List[Dict[str, str]], kwargs: Dict[str, Any],
                               data: Dict[str, Any], top_k: int = TOP_LOGPROBS,
                               abstain: bool = False) -> TrieDecoding:
    """Decode a structuring reply with every IRI value constrained to its trie.

    ``create`` is the client's `chat.completions.create` (an Ollama endpoint: it must
    continue a final assistant message and return `top_logprobs`); ``kwargs`` the
    call's own settings; ``data`` the structured goal context the tries come from.
    With ``abstain`` an `artifact_class` that names a device kind the room does not
    have is not decoded again (the trie would put the room's nearest device in its
    place): its goal becomes one about the missing device (`unmatched_device`)."""
    # 1. The free reply: the control call, and every value it writes in its trie is
    #    what constrained decoding writes as well.
    response = await create(model=model, messages=messages, **kwargs)
    message = response.choices[0].message
    result = TrieDecoding(text=message.content or "", calls=1, abstain=abstain)
    decoder = _Decoder(create, model, messages, kwargs, _reasoning(message), top_k,
                       result.trace)
    _event(result.trace, "free", call=1, text=result.text)
    rows, vocab = iri_paths(data), vocabulary(data)
    end = result.text.rfind(THINK_END)
    pos = end + len(THINK_END) if end >= 0 else 0

    # 2. Each value outside its trie: decoded again under the trie, then the reply
    #    continued freely from it. The values passed over on the way are kept.
    while len(result.interventions) < MAX_INTERVENTIONS:
        invalid = first_invalid_slot(result.text, pos, rows, vocab)
        _trace_kept(result.trace, result.text, pos,
                    invalid.value_start if invalid else len(result.text))
        if invalid is None:
            break
        head = result.text[:invalid.value_start]
        _event(result.trace, "invalid", slot=invalid.slot, free=invalid.written,
               before=invalid.before, evidence=invalid.evidence,
               allowed=invalid.trie.values, conditioned=invalid.conditioned,
               tail=head[-TRACE_TAIL:])
        # 2a. A device the room lacks is reported missing, not replaced.
        missing = _abstention(result.text, invalid, rows, vocab) if abstain else None
        if missing is not None:
            result.text, words, pos = missing
            _event(result.trace, "abstain", words=words)
            result.interventions.append({
                "slot": invalid.slot, "free": invalid.written, "decoded": None,
                "how": "abstain", "steps": 0, "conditioned": invalid.conditioned,
                "candidates": len(invalid.trie.values), "evidence": {},
                "words": words})
            continue
        # 2b. Any other value outside its trie is decoded again.
        value, closing, how, steps = await decoder.value(head, invalid.trie)
        result.interventions.append({
            "slot": invalid.slot, "free": invalid.written, "decoded": value, "how": how,
            "steps": steps, "conditioned": invalid.conditioned,
            "candidates": len(invalid.trie.values), "evidence": invalid.evidence})
        result.text = head + value + closing
        result.text += await decoder.resume(result.text)
        pos = len(head) + len(value) + 1
    result.calls = decoder.calls + 1
    _event(result.trace, "done", text=result.text)
    return result


def _trace_kept(trace: List[Dict[str, Any]], text: str, pos: int, end: int) -> None:
    """Trace the slot values written whole between ``pos`` and ``end``: those the check
    passed over, so kept as the free decoding wrote them (an unconstrained slot too)."""
    for match in SLOT_VALUE.finditer(text, pos, end):
        _event(trace, "kept", slot=match.group(1), value=match.group(2))
