"""The parts an LLM prompt is made of, named from the code that builds it.

A prompt the agents send is assembled from three kinds of material, and for an analysis
of prompt structure the three have to be told apart:

    constants   the prompt texts kept as module-level ALL_CAPS strings in the prompt
                modules (`PROMPT_MODULES`), some of them composed of smaller ones
                (`GOAL_STRUCTURING_SIMPLE_PROMPT` joins `GOAL_SPEC_RULES`,
                `CONTEXT_GUIDE`, ... ; `GOAL_SPEC_RULES` joins five more)
    inline text the glue written inside a template or at the call site
    fields      what is filled in at call time: the home's context, the structured
                goal, the user's words (`{goal_context}`, `{checklist}`, ...)

Nothing here is a second copy of a prompt: the constants are *read from the live modules*
(and their line numbers from the modules' source), and a message sent is explained by
finding those constants inside it. A prompt text changed in the code is therefore
explained as changed, with nothing to keep in step here, except `USER_PATTERNS` and
`FIELD_ORIGINS` below, which describe messages the code builds with f-strings and say
where each field's content comes from. A message that matches none of it is reported
whole, as "built inline" -- visible, not silently misdescribed.

    part = decompose_message(system_text)       # the tree of Parts of one message
    print(render_tree(part))                    # the monospaced outline
    static, dynamic = static_dynamic(part)      # characters fixed in code / filled per call

Flow of `decompose_message`, on a goal-structuring system prompt:

    1. a registered template whose literal pieces match the message from end to end
       (`GOAL_STRUCTURING_SIMPLE_PROMPT` with `{goal_context}` as the one gap) -> its
       literal pieces are explained (constants found inside, inline glue) and the gap's
       text is the field's value
    2. else a user-message pattern (`Request: {request}\\nSegmenter qualifiers: ...`)
    3. else a JSON object -> one field per top-level key
    4. else the whole message is one dynamic field
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The modules whose ALL_CAPS strings are the agents' prompt texts.
PROMPT_MODULES = (
    "ami_agents.agents.interaction_solver.prompts",
    "ami_agents.bt_planning.planning.prompts",
    "ami_agents.agents.user_assistant.prompts.goal_structuring_prompts",
    "ami_agents.agents.user_assistant.prompts.intent_parsing_prompts",
    "ami_agents.agents.user_assistant.prompts.intent_response_prompts",
    "ami_agents.agents.user_assistant.prompts.intent_segmentation_prompts",
)
# A string shorter than this is a label, not a prompt part.
MIN_CHARS = 40
# `{name}`, but not the `{{`/`}}` of an escaped brace (str.format) nor JSON's `{"`.
PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_][a-z_0-9]*)\}(?!\})")
HEADING = re.compile(r"^(#{1,4}) +(.+?)\s*$", re.M)
DYNAMIC_KINDS = ("field", "dynamic")

# Where each field's content comes from, keyed "<template or pattern name>.<field>" (a
# field name alone is not enough: `request` is the user's words in one prompt and a
# structured request in another). Written from the code that fills each one.
FIELD_ORIGINS: Dict[str, str] = {
    "GOAL_STRUCTURING_SIMPLE_PROMPT.goal_context":
        "EnvExplorer's goal context for the request, narrowed to the rooms and devices it "
        "names (user_assistant/pipeline.py: fetch_goal_context)",
    "GOAL_STRUCTURING_DEPENDENCY_PROMPT.goal_context":
        "EnvExplorer's goal context for the request, narrowed to the rooms and devices it "
        "names (user_assistant/pipeline.py: fetch_goal_context)",
    "ENV_STATE_REQUEST_PARSER_PROMPT.home_devices":
        "EnvExplorer's device list for the request (pipeline.py: fetch_goal_context)",
    "ENV_STATE_REQUEST_PARSER_PROMPT.ontology_context":
        "the ontology's class lists (user_assistant/utils/ontology_context.py)",
    "ENV_CAPABILITIES_REQUEST_PARSER_PROMPT.home_devices":
        "EnvExplorer's device list for the request (pipeline.py: fetch_goal_context)",
    "ENV_CAPABILITIES_REQUEST_PARSER_PROMPT.ontology_context":
        "the ontology's capability class lists (ontology_context.py)",
    "BT_PLANNING_SYSTEM_PROMPT.capability_context":
        "the affordances shown by device and name (bt_planning/planning/prompts.py: "
        "format_capability_context)",
    "BT_PLANNING_SYSTEM_PROMPT.signifier_hints":
        "remembered plans that match (prompts.py: format_signifier_hints)",
    "BT_PLANNING_SYSTEM_PROMPT.observable_property_hints":
        "properties that observe the goal's room variable (prompts.py: "
        "format_observable_property_hints)",
    "GOAL_PROFILE_PROMPT.goal_structure":
        "the structuring step's output, GoalStructure.to_wire_dict() "
        "(interaction_solver/utils/profiling.py: ask_profile)",
    "GOAL_PROFILE_PROMPT.specific_goal_context":
        "the devices' context for the profiling questions (profiling.py: ask_profile)",
    "GOAL_PROFILE_PROMPT.checklist":
        "the questions the profiling step asks (profiling.py: ask_profile)",
    "CONDITION_GUARD_PROMPT.request":
        "the request being guarded (interaction_solver/utils/condition_check.py: _ask)",
    "CONDITION_GUARD_PROMPT.plan_actions":
        "the actions of the plan being checked (condition_check.py: _ask)",
    "CONDITION_GUARD_PROMPT.steps":
        "the plan's steps with their conditions (condition_check.py: _step_brief)",
    "CONDITION_GUARD_PROMPT.devices":
        "the devices the steps touch (condition_check.py: device_context)",
    "goal-structuring request.request":
        "the atomic intent's text, from the segmentation step",
    "goal-structuring request.qualifiers":
        "the segmenter's qualifiers for that intent, joined with commas ('none' when empty)",
    "modify-recipe request.request": "the goal's intent text (explicit_goal_llm_planning.py)",
    "modify-recipe request.goal_brief":
        "the structured goal as the planning context briefs it (goal_brief)",
    "modify-recipe request.affordances":
        "the device's affordances: name, type, semantic types, description, input schema",
    "planner request with the structured goal.intent":
        "the goal's intent text (explicit_goal_llm_planning.py: generate_bt(intents=...))",
    "planner request with the structured goal.goal_brief":
        "the structured goal as the planning context briefs it (goal_brief)",
    "planner request, several intents, with the structured goal.intents":
        "the goals' intent texts, one `- ` line each (bt_planner.py: _format_user_message)",
    "planner request, several intents, with the structured goal.goal_brief":
        "the structured goal as the planning context briefs it (goal_brief)",
    "planner request, several intents.intents":
        "the goals' intent texts, one `- ` line each (bt_planner.py: _format_user_message)",
    "query-response request.user_text": "what the user asked",
    "query-response request.raw_data": "the raw query result to phrase",
}
FIELD_DEFAULT = "filled at call time (origin not registered in prompt_parts.FIELD_ORIGINS)"


def origin_of(owner: str, name: str) -> str:
    """Where the content of field ``name`` of template or pattern ``owner`` comes from."""
    return FIELD_ORIGINS.get(f"{owner}.{name}", FIELD_DEFAULT)


@dataclass(frozen=True)
class Constant:
    """One prompt text kept as a module-level constant, where it is defined."""

    name: str
    value: str
    file: str                                  # from the project root, posix
    line: int
    fields: Tuple[str, ...]                    # its `{placeholders}`, in order

    @property
    def source(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class Part:
    """One part of a message, with the parts it is made of.

    kind: template | constant | section | inline | field | pattern | json | dynamic.
    ``text`` is the part's own text in the message (empty for a field not yet filled)."""

    kind: str
    name: str
    source: str
    text: str
    children: List["Part"] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)


# -- the registry of prompt constants -------------------------------------------------

def _definition_lines(path: Path) -> Dict[str, int]:
    """{name: line} of the module-level assignments of a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: Dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lines[target.id] = node.lineno
    return lines


@lru_cache(maxsize=1)
def load_registry() -> Tuple[Constant, ...]:
    """Every prompt constant of `PROMPT_MODULES`, read from the live modules: an ALL_CAPS
    string of at least `MIN_CHARS`, the first definition of a given text winning (a prompt
    re-exported by another module is one constant)."""
    registry: List[Constant] = []
    seen = set()
    for module_name in PROMPT_MODULES:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        lines = _definition_lines(path)
        for name, value in vars(module).items():
            if not (name.isupper() and isinstance(value, str) and len(value) >= MIN_CHARS):
                continue
            if value in seen or name not in lines:
                continue
            seen.add(value)
            registry.append(Constant(
                name=name, value=value, file=path.relative_to(PROJECT_ROOT).as_posix(),
                line=lines[name], fields=tuple(PLACEHOLDER.findall(value))))
    return tuple(registry)


# -- explaining a text by the constants inside it -------------------------------------

def _preview(text: str, width: int = 46) -> str:
    """The start of ``text`` on one line, to name an inline stretch."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width - 3] + "..."


def _gap(text: str, owner: Optional[Constant], unfilled: bool) -> List[Part]:
    """The stretch of ``text`` no constant explains, as inline parts. With ``unfilled``
    (a template shown as defined, not as sent) its `{placeholders}` are fields."""
    if not text:
        return []
    where = f"inline in {owner.name}" if owner else "inline"
    if not text.strip():
        return [Part("inline", "separator", where, text)]
    # 1. As sent: nothing left to split, the placeholders were filled already.
    if not unfilled:
        return [Part("inline", f"inline: {_preview(text)}", where, text,
                     _sections(text, f"section of {owner.name}" if owner else "section"))]
    # 2. As defined: the placeholders are the fields, the pieces between them inline.
    parts: List[Part] = []
    pieces = PLACEHOLDER.split(text)
    for index, piece in enumerate(pieces):
        if index % 2:
            parts.append(Part("field", "{" + piece + "}",
                              origin_of(owner.name if owner else "", piece), ""))
        elif piece.strip():
            parts.append(Part("inline", f"inline: {_preview(piece)}", where, piece))
    return parts


def _sections(text: str, where: str) -> List[Part]:
    """The markdown sections (`## Heading`) of ``text``, when it has at least two: how a
    long prompt such as the segmentation one is organised inside its one constant (or
    inside the inline body of a template). ``where`` is the sections' source."""
    marks = list(HEADING.finditer(text))
    if len(marks) < 2:
        return []
    parts: List[Part] = []
    if text[:marks[0].start()].strip():
        parts.append(Part("section", "(before the first heading)", where,
                          text[:marks[0].start()]))
    ends = [m.start() for m in marks[1:]] + [len(text)]
    for mark, end in zip(marks, ends):
        parts.append(Part("section", f"{mark.group(1)} {mark.group(2)}", where,
                          text[mark.start():end]))
    return parts


def _constant_part(constant: Constant, unfilled: bool = False) -> Part:
    """``constant`` as a part: the smaller constants and glue it is composed of, or, when
    it contains no other constant, its markdown sections."""
    children = _explain(constant.value, constant, {constant.name}, unfilled)
    if not any(child.kind == "constant" for child in children):
        children = _sections(constant.value, f"section of {constant.name}")
    # A constant that is all one inline stretch has nothing to show beneath it.
    if len(children) == 1 and children[0].kind == "inline":
        children = []
    return Part("constant", constant.name, constant.source, constant.value, children)


def _explain(text: str, owner: Optional[Constant], exclude: set,
             unfilled: bool = False) -> List[Part]:
    """The parts of ``text`` in order: the registered constants found inside it (the
    longest first, each claiming its stretch once; what is inside one is explained by the
    recursion), and the inline text between them."""
    # 1. Which constants occur, longest first so a composite claims its stretch before the
    #    pieces it contains do.
    claimed: List[Tuple[int, int, Constant]] = []
    for constant in sorted(load_registry(), key=lambda c: -len(c.value)):
        if constant.fields or constant.name in exclude:
            continue
        start = 0
        while (index := text.find(constant.value, start)) >= 0:
            end = index + len(constant.value)
            if any(index < b and a < end for a, b, _ in claimed):
                start = index + 1
                continue
            claimed.append((index, end, constant))
            start = end
    # 2. In text order, with the glue between them.
    parts: List[Part] = []
    position = 0
    for begin, end, constant in sorted(claimed, key=lambda item: item[0]):
        parts += _gap(text[position:begin], owner, unfilled)
        parts.append(_constant_part(constant))
        position = end
    return parts + _gap(text[position:], owner, unfilled)


# -- explaining a message -------------------------------------------------------------

def _literals(constant: Constant, unescape: bool) -> List[str]:
    """A template's literal pieces, between its placeholders; ``unescape`` turns the
    `{{` / `}}` of a str.format template into the braces it renders as."""
    pieces = PLACEHOLDER.split(constant.value)[0::2]
    return [p.replace("{{", "{").replace("}}", "}") if unescape else p for p in pieces]


def _template_part(constant: Constant, segments: List[Tuple[str, str]], text: str) -> Part:
    """A template as a part, from its ``segments``: ("lit", text) pieces and
    ("field", value) gaps, in order (the field names come from the template)."""
    children: List[Part] = []
    names = iter(constant.fields)
    for kind, value in segments:
        if kind == "lit":
            children += _explain(value, constant, {constant.name})
        else:
            name = next(names)
            children.append(Part("field", "{" + name + "}",
                                 origin_of(constant.name, name), value))
    return Part("template", constant.name, constant.source, text, children)


def _match_template(constant: Constant, text: str) -> Optional[Part]:
    """``text`` as ``constant`` filled in, when its literal pieces match it from end to
    end; None when they do not."""
    for unescape in (False, True):
        pieces = _literals(constant, unescape)
        pattern = "^" + "(.*?)".join(re.escape(p) for p in pieces) + "$"
        match = re.match(pattern, text, re.S)
        if match is None:
            continue
        # 1. The literal pieces and the gaps between them, in order: each gap's text
        #    is what the call filled the placeholder with.
        segments: List[Tuple[str, str]] = []
        for index, piece in enumerate(pieces):
            segments.append(("lit", piece))
            if index < len(constant.fields):
                segments.append(("field", match.group(index + 1)))
        return _template_part(constant, segments, text)
    return None


# (name, where the code builds it, the f-string with its {fields}). Most specific first.
USER_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
    ("goal-structuring request",
     "ami_agents/agents/user_assistant/behaviours/goal_structuring.py: _structure",
     "Request: {request}\nSegmenter qualifiers: {qualifiers}"),
    ("modify-recipe request",
     "ami_agents/agents/interaction_solver/behaviours/explicit_goal_llm_planning.py: "
     "_plan_modify",
     "Request: {request}\n\nStructured goal:\n{goal_brief}\n\nAffordances:\n{affordances}"),
    ("planner request, several intents, with the structured goal",
     "ami_agents/bt_planning/planning/bt_planner.py: _format_user_message, generate_bt",
     "Please handle the following requests:\n{intents}\n\nStructured goal:\n{goal_brief}"),
    ("planner request with the structured goal",
     "ami_agents/bt_planning/planning/bt_planner.py: generate_bt",
     "{intent}\n\nStructured goal:\n{goal_brief}"),
    ("planner request, several intents",
     "ami_agents/bt_planning/planning/bt_planner.py: _format_user_message",
     "Please handle the following requests:\n{intents}"),
    ("query-response request",
     "ami_agents/agents/user_assistant/pipeline.py: format_query_response",
     "User asked: {user_text}\n\nRaw data:\n{raw_data}"),
)


def _match_pattern(text: str) -> Optional[Part]:
    """``text`` as one of the `USER_PATTERNS`, its fields cut out; None for none."""
    for name, source, template in USER_PATTERNS:
        pieces = re.split(r"\{([a-z_]+)\}", template)
        literals, names = pieces[0::2], pieces[1::2]
        match = re.match("^" + "(.*?)".join(re.escape(p) for p in literals) + "$", text, re.S)
        if match is None:
            continue
        children: List[Part] = []
        for index, literal in enumerate(literals):
            if literal:
                children.append(Part("inline", f"inline: {_preview(literal)}", source, literal))
            if index < len(names):
                children.append(Part("field", "{" + names[index] + "}",
                                     origin_of(name, names[index]),
                                     match.group(index + 1)))
        return Part("pattern", name, source, text, children)
    return None


def _json_style(text: str, data: dict) -> Dict[str, object]:
    """The `json.dumps` options that write ``data`` as exactly ``text`` (the calling code
    may use the defaults, a compact form, an indent, sorted keys); {} when none does, in
    which case the sizes of the fields are only as close as the defaults get."""
    for ascii_only in (True, False):
        for separators in (None, (",", ":")):
            for indent in (None, 1, 2):
                for sort_keys in (False, True):
                    style = {"ensure_ascii": ascii_only, "separators": separators,
                             "indent": indent, "sort_keys": sort_keys}
                    if json.dumps(data, **style) == text:
                        return style
    return {}


def _match_json(text: str) -> Optional[Part]:
    """``text`` as a JSON object (the answer prompts send one): a field per top-level
    key, the punctuation and key names between them as inline."""
    # 1. A JSON object, not a bare value or an empty one.
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or not data:
        return None
    # 2. Each value written as the message wrote it, so the sizes add up to the message
    #    (an approximation, clamped, when no known style reproduces it).
    style = _json_style(text, data)
    children = [Part("field", f"{key}", "JSON key of the message",
                     json.dumps(value, **style)) for key, value in data.items()]
    rest = len(text) - sum(c.chars for c in children)
    # 3. What is left is the syntax and the key names; its text is a stand-in of that
    #    size (the message's own characters of it are spread between the fields).
    if rest > 0:
        children.append(Part("inline", "JSON punctuation and key names",
                             "the message's JSON syntax", " " * rest))
    return Part("json", "JSON object", "built with json.dumps in the calling code",
                text, children)


def decompose_message(text: str) -> Part:
    """The tree of parts of one message sent to a model (module docstring, flow)."""
    # 1. A registered template, or a registered constant sent as it is.
    registry = sorted(load_registry(), key=lambda c: -len(c.value))
    for constant in registry:
        if constant.fields:
            part = _match_template(constant, text)
            if part is not None:
                return part
        elif text == constant.value or text.strip() == constant.value.strip():
            return _constant_part(constant)
    # 2. The user-message patterns, then a JSON object, then nothing known.
    return (_match_pattern(text) or _match_json(text)
            or Part("dynamic", "(whole message)",
                    "the calling code sends this as one value; no registered template", text))


def decompose_constant(constant: Constant) -> Part:
    """A prompt constant as defined in the code, its placeholders still open: the catalog
    view, the same tree a message made from it gets, with empty fields."""
    if not constant.fields:
        return _constant_part(constant)
    pieces = PLACEHOLDER.split(constant.value)
    segments = [("lit" if i % 2 == 0 else "field", "" if i % 2 else piece)
                for i, piece in enumerate(pieces)]
    return _template_part(constant, segments, constant.value)


# -- reading the tree -----------------------------------------------------------------

def leaves(part: Part) -> List[Part]:
    """The parts with nothing beneath them, in order."""
    if not part.children:
        return [part]
    return [leaf for child in part.children for leaf in leaves(child)]


def static_dynamic(part: Part) -> Tuple[int, int]:
    """(characters fixed in the code, characters filled in per call) of a message."""
    dynamic = sum(leaf.chars for leaf in leaves(part) if leaf.kind in DYNAMIC_KINDS)
    return part.chars - dynamic, dynamic


def render_tree(part: Part, width: int = 58) -> str:
    """The outline of ``part`` as monospaced text: one line per part, its kind, size
    (characters, share of the message) and where it is defined."""
    total = max(part.chars, 1)
    lines = [f"{'part':<{width}} {'kind':<9} {'chars':>7} {'share':>6}  source"]

    def walk(node: Part, prefix: str, last: bool, top: bool) -> None:
        branch = "" if top else ("`-- " if last else "|-- ")
        label = (prefix + branch + node.name)
        label = label if len(label) <= width else label[:width - 3] + "..."
        lines.append(f"{label:<{width}} {node.kind:<9} {node.chars:>7,} "
                     f"{100 * node.chars / total:>5.1f}%  {node.source}")
        extension = "" if top else ("    " if last else "|   ")
        for index, child in enumerate(node.children):
            walk(child, prefix + extension, index == len(node.children) - 1, False)

    walk(part, "", True, True)
    return "\n".join(lines)


# -- the catalog ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _source_lines() -> Tuple[Tuple[str, int, str], ...]:
    """(file, line number, line) of every line of `ami_agents/`'s Python, read once: the
    catalog looks up each constant's name in all of them."""
    out: List[Tuple[str, int, str]] = []
    for path in sorted((PROJECT_ROOT / "ami_agents").rglob("*.py")):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            out.append((relative, number, line))
    return tuple(out)


def _references(constant: Constant) -> List[str]:
    """Where else the code names ``constant`` (file:line): the places that build a
    prompt from it. A line of an import only moves the name, so those are left out
    (a line starting `from`/`import`, and the names of a parenthesised import list)."""
    pattern = re.compile(rf"\b{re.escape(constant.name)}\b")
    found: List[str] = []
    for relative, number, line in _source_lines():
        if not pattern.search(line) or (relative, number) == (constant.file, constant.line):
            continue
        if re.match(r"\s*(from|import)\b", line) or re.match(r"\s+[A-Z_0-9]+,\s*$", line):
            continue
        found.append(f"{relative}:{number}")
    return found


def catalog_markdown() -> str:
    """Every top-level prompt of the code as markdown: the outline of what it is
    composed of, and where it is used. A constant that is part of another is shown inside
    that one, not on its own."""
    registry = load_registry()
    lines = ["# Prompt catalog (from the code)", "",
             "Every prompt text of the agents, as the code defines it: the constants it is "
             "composed of, its inline glue, and the fields filled in per call (empty here). "
             "Read from the live modules, so it is the code as of now.", ""]
    top = [c for c in registry
           if not any(o is not c and c.value in o.value for o in registry)]
    for constant in top:
        tree = decompose_constant(constant)
        fields = ("fields: " + ", ".join(constant.fields)) if constant.fields else "no fields"
        lines += [f"## {constant.name}", "", f"Defined at `{constant.source}`; {fields}.", ""]
        used = _references(constant)
        lines += [("Named at: " + ", ".join(f"`{u}`" for u in used[:6])
                   if used else "Named elsewhere only through a composite or a table."), ""]
        lines += ["```text", render_tree(tree), "```", ""]
    return "\n".join(lines)
