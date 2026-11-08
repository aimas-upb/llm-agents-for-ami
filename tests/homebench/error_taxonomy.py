"""Where, and why, one HomeBench episode went wrong: a deterministic error taxonomy.

`scoring.py` says whether an episode passed and, coarsely, why (`Goal Failure` or
`Not Reported Infeasible`). This module says which part of our pipeline lost each
part of the request, from data already stored in the run folder, without an LLM call.

An episode record holds the whole trail of the pipeline, and the episode's `meta.gold`
says what should have happened, one *part* per gold token. Each part is followed down
the pipeline and stops at the first stage that lost it:

    infrastructure   the LLM call timed out or was rejected (no model answer at all)
    segmentation     no goal was produced for the part (fewer goals than parts)
    structuring      a goal exists but its room / device / affordance / value is wrong,
                     or a clarifying question was asked for a part that is feasible
    planning         the goal is right but the planner answered impossible / error
    execution        a plan exists but the home did not end up doing the gold call
    feasibility      the part is infeasible (gold `error_input`) and was not refused
    evaluation       the strict gold reading fails, a lenient one (`turn_off`) passes

and carries a *code* naming the failure (`CODES`). The stages are ordered as the
pipeline runs, so the stage of an episode is the earliest stage any of its parts
failed in.

The flow, per episode (`analyze_episode`):

    1. the gold parts: `meta.gold` parsed, plus the part's room / device / reason from
       the source episode when it is available (`build_parts`);
    2. the goals the structuring step produced, each linked to the part it most
       resembles (`link_parts`), the dead structures (error, question) kept apart;
    3. every part diagnosed: a feasible part is ok when the stored check says so,
       otherwise the first failing stage is found (`diagnose_feasible`); an infeasible
       part is ok when a refusal covers it (the scorer pools refusals), otherwise
       `diagnose_infeasible` says what the pipeline did instead;
    4. the episode rolled up: verdict, primary stage and code, extra actions.

Worked example::

    analysis = analyze_episode(record, evaluation, source, Tools.load())
    analysis.verdict          # "fail"
    analysis.primary_stage    # "planning"
    analysis.primary_code     # "plan_no_device_match"
    [p.code for p in analysis.parts]   # [None, "plan_no_device_match"]
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from ami_agents.environment.integration.HomeBench.mappings import (
    MappingTableError, Mappings, load_mappings,
)
from tests.homebench import convert
from tests.simuhome.bench import gold_log as gold_log_module
from tests.simuhome.bench import homebench_metrics as metrics

LOGGER = logging.getLogger(__name__)

MAPPINGS_DIR = Path(__file__).parent / "mappings"

# -- the taxonomy ---------------------------------------------------------------------

#: The stages in pipeline order; an episode's stage is its earliest failing one.
STAGES = ("infrastructure", "segmentation", "structuring", "planning", "execution",
          "feasibility", "evaluation")

#: ``code: (stage, what it means)``. The order is the tie-break for equal counts.
CODES: Dict[str, Tuple[str, str]] = {
    "infra_llm_timeout": ("infrastructure", "an LLM call timed out or failed: no model "
                          "answer"),
    "infra_llm_context": ("infrastructure", "the LLM server rejected the prompt "
                          "(HTTP 400, prompt longer than the context)"),
    "infra_llm_error": ("infrastructure", "an LLM call raised another transport error"),
    "seg_part_lost": ("segmentation", "no goal was produced for the part: the request "
                      "was split in fewer goals than it has parts"),
    "struct_json_invalid": ("structuring", "the structuring answer was not a valid "
                            "JSON object"),
    "struct_no_goal": ("structuring", "a structure came back with no goal"),
    "struct_false_clarify_device": ("structuring", "asked 'I can't find the X' for a "
                                    "device the home has"),
    "struct_false_clarify_attribute": ("structuring", "asked 'X has no Y setting' for "
                                       "an attribute the device has"),
    "struct_false_clarify_ambiguous": ("structuring", "asked 'which device?' although "
                                       "the request names it"),
    "struct_false_clarify_other": ("structuring", "asked another question for a "
                                   "feasible part"),
    "struct_class_prefix": ("structuring", "class names written without their "
                            "`homeont:` / `saref:` prefix, so the planner finds no "
                            "match (format)"),
    "struct_parameter_name": ("structuring", "`parameter_name` filled for an action "
                              "that takes a direct value; the planner refuses it "
                              "(format)"),
    "struct_device_unnamed": ("structuring", "the goal has no `artifact_name`; every "
                              "such goal ends 'unrouted' in the planner"),
    "struct_placeholder_echo": ("structuring", "a field holds the name of a schema "
                                "field (`location_class`) instead of a value"),
    "struct_slot_confusion": ("structuring", "a value sits in the wrong field (a device "
                              "class in `location_class`, a device id in "
                              "`artifact_class`)"),
    "struct_room_missing": ("structuring", "the goal has no room (NA / empty)"),
    "struct_wrong_room": ("structuring", "the goal names another room than the gold"),
    "struct_wrong_device_class": ("structuring", "the goal names another kind of "
                                  "device than the gold"),
    "struct_invalid_device_name": ("structuring", "the goal's device name is not a "
                                   "device of the home (hallucinated)"),
    "struct_wrong_device_name": ("structuring", "the goal names another device of the "
                                 "home than the gold"),
    "struct_wrong_affordance": ("structuring", "the goal asks another action / "
                                "attribute than the gold"),
    "struct_value_missing": ("structuring", "the goal has no target value, the gold "
                             "has one"),
    "struct_wrong_value": ("structuring", "the goal's target value differs from the "
                           "gold's"),
    "plan_missing_entry": ("planning", "a goal was structured but no plan entry exists"),
    "plan_unrouted": ("planning", "no planning path accepts the structure"),
    "plan_unplannable": ("planning", "the goal names no action to plan"),
    "plan_explicit_failed": ("planning", "the explicit planner raised an error"),
    "plan_error_other": ("planning", "the planner ended in another error"),
    "plan_no_device_match": ("planning", "planner: no single device matches the goal's "
                             "class and room (feasible part)"),
    "plan_no_action": ("planning", "planner: the device has no such action (feasible "
                       "part)"),
    "plan_false_impossible": ("planning", "planner answered impossible for another "
                              "reason (feasible part)"),
    "plan_false_clarify": ("planning", "the planner asked a question for a feasible "
                           "part"),
    "exec_plan_failed": ("execution", "a plan ended in a status other than SUCCESS"),
    "exec_action_rejected": ("execution", "the home refused the action sent for the "
                             "part"),
    "exec_wrong_value": ("execution", "the right call was sent with another value"),
    "exec_wrong_command": ("execution", "another command was sent to the right device"),
    "exec_not_run": ("execution", "a plan exists but the executor was not run "
                     "(mode none)"),
    "exec_no_effect": ("execution", "a plan ran, no accepted action matches the gold "
                       "and no better reason is stored"),
    "infeas_accepted_on_target": ("feasibility", "the home accepted an action on the "
                                  "very device the label says cannot do it (label "
                                  "suspect)"),
    "infeas_substituted": ("feasibility", "a different device in the same room was "
                           "operated instead of reporting the missing one"),
    "infeas_executed_other": ("feasibility", "nothing was refused; actions were "
                              "executed elsewhere"),
    "infeas_silent": ("feasibility", "nothing was refused and nothing was executed"),
    "eval_turn_off_reading": ("evaluation", "met only by the lenient turn-off reading "
                              "(score_extra), not by the gold's own"),
    "unattributed": ("evaluation", "failed, but no part could be blamed (should not "
                     "happen)"),
}
CODE_ORDER = {code: i for i, code in enumerate(CODES)}

#: Error text of a structure -> code.
_TIMEOUT = re.compile(r"timed out|timeout", re.IGNORECASE)
_CONTEXT = re.compile(r"Error code: 400|context length|too long", re.IGNORECASE)
_JSON_INVALID = re.compile(r"expected a JSON object|JSON", re.IGNORECASE)
#: A clarifying question -> code.
_QUESTION_KINDS = (
    (re.compile(r"can't find|cannot find|couldn't find", re.IGNORECASE), "device"),
    (re.compile(r"has no .* setting|no .* setting", re.IGNORECASE), "attribute"),
    (re.compile(r"which device", re.IGNORECASE), "ambiguous"),
)
_NO_DEVICE = re.compile(r"No single device", re.IGNORECASE)
_NO_ACTION = re.compile(r"No action|names no action", re.IGNORECASE)
_PARAMETER_NAME = re.compile(r"direct value, not parameter", re.IGNORECASE)
#: A value that is the name of a schema field (the model echoed the template).
_ECHO = re.compile(r"^(location|artifact|affordance|parent)_(class|name)$"
                   r"|^room_class$|^device_class$", re.IGNORECASE)
#: Fields of a goal that carry a class name and must be written ``prefix:Name``.
CLASS_FIELDS = ("location_class", "artifact_class", "affordance_class")
_PREFIXED = re.compile(r"^(homeont|saref):\S+$")
#: Plan codes a bare class name explains: the planner compares the class strings.
_FORMAT_PLAN_CODES = ("plan_no_device_match", "plan_no_action", "plan_false_impossible",
                      "plan_unrouted", "plan_unplannable")
_PLAN_ERRORS = {"unrouted": "plan_unrouted", "unplannable_goal": "plan_unplannable",
                "explicit_planning_failed": "plan_explicit_failed"}
#: Structured-goal field differences in priority order -> code.
_DIFF_CODES = (("placeholder_echo", "struct_placeholder_echo"),
               ("slot_confusion", "struct_slot_confusion"),
               ("room_missing", "struct_room_missing"),
               ("room", "struct_wrong_room"),
               ("device_class", "struct_wrong_device_class"),
               ("device_name_invalid", "struct_invalid_device_name"),
               ("device_name_other", "struct_wrong_device_name"),
               ("affordance", "struct_wrong_affordance"),
               ("value_missing", "struct_value_missing"),
               ("value", "struct_wrong_value"))
#: Placeholders a structuring model writes when it names no device.
_NO_NAME = (None, "", "NA", "None", "null")
#: A link between a part and a goal needs at least a room or a device-class match.
MIN_LINK_SCORE = 3


# -- vocabulary: free-text classes -> table keys ---------------------------------------


def _norm(text: Any) -> str:
    """Lower-case alphanumerics of ``text`` after any ``prefix:`` (``homeont:Foyer``
    -> ``foyer``, ``Dining Room`` -> ``diningroom``)."""
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower().split(":")[-1])


def _table_rows(name: str) -> List[Dict[str, Any]]:
    """The rows of ``mappings/<name>.yaml``."""
    path = MAPPINGS_DIR / f"{name}.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("rows") or []


#: Class names the structuring model uses that the device table does not list.
_DEVICE_ALIASES = {"colorlight": "light", "dimmablelight": "light",
                   "onofflight": "light", "heater": "heating", "rvc": "vacuum_robot",
                   "vacuum": "vacuum_robot", "airconditioner": "air_conditioner",
                   "airpurifier": "air_purifiers", "dehumidifier": "dehumidifiers"}


class Vocabulary:
    """Resolves the room and device-class names a structured goal carries to the keys
    of the mapping tables (``homeont:DiningRoom`` -> ``ding_room``)."""

    def __init__(self) -> None:
        """Read ``room_map`` and ``device_type_map`` once."""
        self._rooms: Dict[str, str] = {}
        for row in _table_rows("room_map"):
            for alias in (row["key"], row.get("title"), row.get("homeont_class")):
                self._rooms[_norm(alias)] = row["key"]
        self._devices: Dict[str, str] = dict(_DEVICE_ALIASES)
        for row in _table_rows("device_type_map"):
            aliases = [row["key"], row.get("title")]
            aliases += [t for t in row.get("td_types") or [] if t.startswith("homeont:")]
            for alias in aliases:
                key = _norm(alias)
                self._devices[key] = row["key"]
                self._devices.setdefault(key.rstrip("s"), row["key"])

    def room(self, text: Any) -> Optional[str]:
        """The room key ``text`` names, or None when it names no room of the tables."""
        return self._rooms.get(_norm(text))

    def device_type(self, text: Any) -> Optional[str]:
        """The device-type key ``text`` names, or None."""
        key = _norm(text)
        return self._devices.get(key) or self._devices.get(key.rstrip("s"))


@dataclass
class Tools:
    """What the analysis reads besides the run: vocabulary and mapping tables."""

    vocabulary: Vocabulary
    mappings: Mappings

    @classmethod
    def load(cls) -> "Tools":
        """The repo's own tables."""
        return cls(Vocabulary(), load_mappings())


def norm_value(value: Any) -> Optional[str]:
    """A value in a comparable form: numbers without a trailing ``.0``, booleans and
    on/off as ``true``/``false``, words lower-cased with ``_`` for spaces; None stays
    None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower().rstrip("%").strip()
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        pass
    return {"on": "true", "off": "false", "yes": "true", "no": "false"}.get(
        text, text.replace(" ", "_"))


# -- gold parts ------------------------------------------------------------------------


@dataclass
class Part:
    """One gold token of an episode and what is known about its target."""

    index: int
    token: str
    feasible: bool
    room: Optional[str] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    command: Optional[str] = None
    action: Optional[str] = None
    value: Any = None
    reason: Optional[str] = None

    @property
    def device_prefix(self) -> Optional[str]:
        """``room.device`` as a gold token starts (``vacuum_robot`` has no room)."""
        if not self.device_type:
            return None
        return self.device_type if self.room in (None, convert.HOUSEHOLD_ROOM) else (
            f"{self.room}.{self.device_type}")


def build_parts(record: Dict[str, Any], source: Optional[Dict[str, Any]],
                tools: Tools) -> List[Part]:
    """The gold parts of ``record``, in gold order.

    A feasible part is read from its gold call (room, device, command, argument) and
    the mapping tables (affordance and value the structuring step should produce). An
    infeasible part takes its room, device and reason (``unexist_device`` /
    ``unexist_attribute``) from ``source``'s goals when the source episode is
    available; a single infeasible request without a source still has its reason in
    the dataset type. Raises ``ValueError`` when the gold does not parse.
    """
    meta = record.get("meta") or {}
    gold = meta.get("gold")
    if not gold:
        raise ValueError("record has no meta.gold")
    htype = meta.get("homebench_type") or ""
    try:
        calls = convert.parse_gold(gold, convert.multi_parts(htype) is not None)
    except convert.ConversionError as exc:
        raise ValueError(str(exc)) from exc
    tokens = metrics.gold_tokens(gold)
    goals = ((source or {}).get("eval") or {}).get("goals") or []
    if len(tokens) != len(calls):
        raise ValueError(f"gold {gold!r}: {len(tokens)} tokens, {len(calls)} calls")
    parts: List[Part] = []
    for index, (token, call) in enumerate(zip(tokens, calls)):
        goal = goals[index] if len(goals) == len(calls) else {}
        if call is None:
            reason = goal.get("reason") or (htype if htype.startswith("unexist") else None)
            parts.append(Part(index, token, False, goal.get("room_id"),
                              goal.get("device_id"), goal.get("device_type"),
                              reason=reason))
            continue
        part = Part(index, token, True, call.room, call.device_id, call.device,
                    call.command)
        try:
            part.action, part.value = tools.mappings.action_for_command(
                call.device, call.command, call.arg)
        except MappingTableError:
            LOGGER.debug("no action for %s", token)
        parts.append(part)
    return parts


# -- goals, plans and the links between them -------------------------------------------


@dataclass
class GoalRef:
    """A structured goal and where it came from."""

    struct_index: int
    goal_id: str
    goal: Dict[str, Any]
    text: str


@dataclass
class DeadStructure:
    """A structuring result that produced no goal, and why."""

    struct_index: int
    code: str
    text: str
    detail: str


def _dead_code(item: Dict[str, Any]) -> Optional[str]:
    """The code of a structure with no goal, None for one that has goals."""
    error = item.get("error") or ""
    question = item.get("question") or item.get("clarification") or ""
    if error:
        if _TIMEOUT.search(error):
            return "infra_llm_timeout"
        if _CONTEXT.search(error):
            return "infra_llm_context"
        if _JSON_INVALID.search(error):
            return "struct_json_invalid"
        return "infra_llm_error"
    if question:
        for pattern, kind in _QUESTION_KINDS:
            if pattern.search(question):
                return f"struct_false_clarify_{kind}"
        return "struct_false_clarify_other"
    structure = item.get("structure")
    if not structure or not (structure.get("goals") or {}):
        return "struct_no_goal"
    return None


def collect_goals(record: Dict[str, Any]) -> Tuple[List[GoalRef], List[DeadStructure]]:
    """Every goal of the record's structures, and the structures that have none."""
    goals: List[GoalRef] = []
    dead: List[DeadStructure] = []
    for index, item in enumerate(record.get("structures") or []):
        code = _dead_code(item)
        text = item.get("text") or ""
        if code:
            dead.append(DeadStructure(index, code, text,
                                      str(item.get("error") or item.get("question")
                                          or "")[:120]))
            continue
        for gid, goal in ((item.get("structure") or {}).get("goals") or {}).items():
            goals.append(GoalRef(index, gid, goal, text))
    return goals, dead


def plan_for(record: Dict[str, Any], text: str, used: set) -> Optional[Dict[str, Any]]:
    """The plan entry made for the structure whose text is ``text`` (first unused one;
    the plan entry's intent text is the structure's text)."""
    for index, plan in enumerate(record.get("plans") or []):
        if index in used:
            continue
        if ((plan.get("intent") or {}).get("text_intent") or "") == text:
            used.add(index)
            return plan
    return None


def _affinity(part: Part, text: str) -> int:
    """How much of the part's room and device the free text ``text`` names (0-2)."""
    low = text.lower().replace("dining", "ding")
    hits = 0
    if part.room and part.room.replace("_", " ") in low.replace("_", " "):
        hits += 1
    if part.device_type and part.device_type.replace("_", " ").rstrip("s") in low.replace(
            "_", " "):
        hits += 1
    return hits


def _link_score(part: Part, ref: GoalRef, tools: Tools) -> int:
    """How well a goal fits a part: device id 8, room 3, device class 3, action 2."""
    goal = ref.goal
    score = 0
    if part.device_id and goal.get("artifact_name") == part.device_id:
        score += 8
    room = tools.vocabulary.room(goal.get("location_class"))
    if part.room not in (None, convert.HOUSEHOLD_ROOM) and room == part.room:
        score += 3
    if part.device_type and tools.vocabulary.device_type(
            goal.get("artifact_class")) == part.device_type:
        score += 3
    if part.action and goal.get("affordance_name") == part.action:
        score += 2
    return score


def link_parts(parts: Sequence[Part], goals: Sequence[GoalRef], tools: Tools
               ) -> Dict[int, Optional[GoalRef]]:
    """``{part index: the goal that stands for it}``, one goal per part at most.

    Parts and goals are paired by descending fit (`_link_score`, at least
    ``MIN_LINK_SCORE``). Parts left over are paired in order with goals left over: such
    a pair shares nothing with the part, and the field comparison will say so.
    """
    scored = sorted(((_link_score(p, g, tools), -p.index, -i, p.index, i)
                     for p in parts for i, g in enumerate(goals)), reverse=True)
    links: Dict[int, Optional[GoalRef]] = {p.index: None for p in parts}
    taken: set = set()
    for score, _, _, part_index, goal_index in scored:
        if score < MIN_LINK_SCORE:
            break
        if links[part_index] is None and goal_index not in taken:
            links[part_index] = goals[goal_index]
            taken.add(goal_index)
    leftover_goals = [g for i, g in enumerate(goals) if i not in taken]
    for part in parts:
        if links[part.index] is None and part.feasible and leftover_goals:
            links[part.index] = leftover_goals.pop(0)
    return links


def bare_class_fields(goal: Dict[str, Any]) -> List[str]:
    """The class fields of a goal written without their ``homeont:`` / ``saref:`` prefix
    (``Foyer`` for ``homeont:Foyer``); a field set to NA / None is not a violation."""
    return [k for k in CLASS_FIELDS
            if str(goal.get(k)) not in ("NA", "None", "", "null")
            and not _PREFIXED.match(str(goal.get(k)))]


def field_diffs(part: Part, ref: GoalRef, tools: Tools, device_ids: set
                ) -> Dict[str, Tuple[Any, Any]]:
    """``{field: (expected, got)}`` for every field of the goal that differs from the
    part's gold: room, device_class, device_name_invalid / device_name_other,
    affordance, value / value_missing. A relative goal (``modify``) is not compared on
    value: its value is computed from the home's state at planning time."""
    goal = ref.goal
    diffs: Dict[str, Tuple[Any, Any]] = {}
    vocab = tools.vocabulary
    location, artifact = goal.get("location_class"), goal.get("artifact_class")
    room = vocab.room(location)
    if any(_ECHO.match(str(goal.get(k) or "")) for k in
           ("location_class", "artifact_class", "artifact_name", "affordance_name")):
        diffs["placeholder_echo"] = (None, "schema field name")
    if part.room not in (None, convert.HOUSEHOLD_ROOM) and room != part.room:
        if room is None and vocab.device_type(location) is not None:
            diffs["slot_confusion"] = (part.room, location)
        elif room is None and str(location) in ("NA", "None", "", "null"):
            diffs["room_missing"] = (part.room, location)
        else:
            diffs["room"] = (part.room, location)
    dtype = vocab.device_type(artifact)
    name = goal.get("artifact_name")
    if dtype is None and (vocab.room(artifact) or str(artifact) in device_ids):
        diffs["slot_confusion"] = (part.device_type, artifact)
    if dtype != part.device_type:
        diffs["device_class"] = (part.device_type, artifact)
    elif name not in _NO_NAME and name != part.device_id:
        kind = "device_name_other" if name in device_ids else "device_name_invalid"
        diffs[kind] = (part.device_id, name)
    if part.action and goal.get("affordance_name") != part.action:
        diffs["affordance"] = (part.action, goal.get("affordance_name"))
    elif part.action and goal.get("goal_effect") != "modify":
        got = norm_value(goal.get("target_value_determined"))
        if got is None and part.value is not None:
            diffs["value_missing"] = (part.value, None)
        elif got != norm_value(part.value):
            diffs["value"] = (part.value, goal.get("target_value_determined"))
    return diffs


# -- one part diagnosed ----------------------------------------------------------------


@dataclass
class PartDiagnosis:
    """The verdict on one gold part: ok, or the stage and code that lost it."""

    index: int
    token: str
    feasible: bool
    reason: Optional[str]
    device_type: Optional[str]
    room: Optional[str]
    command: Optional[str]
    ok: bool
    stage: Optional[str] = None
    code: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    linked: bool = False
    label_suspect: bool = False


def _device_ids(record: Dict[str, Any]) -> set:
    """Ids of the home's devices, from the final state."""
    rooms = (record.get("final_home_state") or {}).get("rooms") or {}
    return {d.get("device_id") for room in rooms.values()
            for d in room.get("devices") or []}


@dataclass
class Context:
    """The things every part's diagnosis reads."""

    record: Dict[str, Any]
    tools: Tools
    parts: List[Part]
    links: Dict[int, Optional[GoalRef]]
    dead: List[DeadStructure]
    device_ids: set
    sent: List[str]
    used_plans: set = field(default_factory=set)
    n_goals: int = 0
    n_intents: int = 0


def _stage(code: str) -> str:
    """The stage a code belongs to."""
    return CODES[code][0]


def token_device(token: str) -> str:
    """``room.device`` of a gold token (``vacuum_robot`` alone for the vacuum): the
    token without its call. A decimal argument (``set_x(2.5)``) is cut off first."""
    return token.split("(", 1)[0].rsplit(".", 1)[0]


def _dead_for(ctx: Context, part: Part) -> Optional[DeadStructure]:
    """The dead structure that best explains an unlinked part, removed from the list
    (each explains one part); the one whose text names the part's room and device
    wins, else the first."""
    if not ctx.dead:
        return None
    best = max(range(len(ctx.dead)),
               key=lambda i: (_affinity(part, ctx.dead[i].text), -i))
    return ctx.dead.pop(best)


def _unlinked(ctx: Context, part: Part) -> Tuple[str, Dict[str, Any]]:
    """The code of a part that no goal stands for."""
    dead = _dead_for(ctx, part)
    if dead is not None:
        return dead.code, {"structure_text": dead.text, "message": dead.detail}
    return "seg_part_lost", {"intents": ctx.n_intents, "goals": ctx.n_goals,
                             "parts": len(ctx.parts)}


def _planning_code(plan: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, Any]]:
    """The planning code of a plan entry that did not produce a plan, None when it did."""
    if plan is None:
        return "plan_missing_entry", {}
    outcome = plan.get("outcome")
    detail = {"source": plan.get("source"), "route": plan.get("route"),
              "outcome": outcome}
    if outcome == "impossible":
        text = plan.get("explanation") or ""
        detail["explanation"] = text[:160]
        if _NO_DEVICE.search(text):
            return "plan_no_device_match", detail
        if _NO_ACTION.search(text):
            return "plan_no_action", detail
        return "plan_false_impossible", detail
    if outcome == "clarify":
        return "plan_false_clarify", detail
    if outcome == "error":
        detail["error"] = plan.get("error")
        return _PLAN_ERRORS.get(plan.get("error"), "plan_error_other"), detail
    return None, detail


def _execution_code(ctx: Context, part: Part, check: Dict[str, Any]
                    ) -> Tuple[str, Dict[str, Any]]:
    """The execution code of a part whose goal and plan are right but whose gold call
    was not met."""
    record = ctx.record
    execution = record.get("execution") or {}
    if check.get("class") == "wrong_value":
        return "exec_wrong_value", {"expected": part.token, "sent": check.get("values")}
    prefix = part.device_prefix or ""
    same_device = [t for t in ctx.sent if token_device(t) == prefix]
    rejected = [a for a in execution.get("actions") or [] if not a.get("success")
                and part.device_id and f"/{part.device_id}/" in (a.get("url") or "")]
    if rejected:
        return "exec_action_rejected", {"url": rejected[0].get("url"),
                                        "status": rejected[0].get("status_code")}
    if same_device:
        return "exec_wrong_command", {"expected": part.token, "sent": same_device}
    failed = [p for p in execution.get("plans") or [] if p.get("status") != "SUCCESS"]
    if failed:
        return "exec_plan_failed", {"status": failed[0].get("status"),
                                    "error": failed[0].get("error")}
    if execution.get("mode") in (None, "none"):
        return "exec_not_run", {"mode": execution.get("mode")}
    return "exec_no_effect", {"expected": part.token}


def diagnose_feasible(ctx: Context, part: Part, check: Dict[str, Any]
                      ) -> PartDiagnosis:
    """Diagnose a feasible part (a gold call): ok when its check is ok under the gold's
    own reading, else the first stage that lost it."""
    base = PartDiagnosis(part.index, part.token, True, None, part.device_type,
                         part.room, part.command, ok=False)
    ref = ctx.links.get(part.index)
    # 1. A check that passed strictly: nothing to blame (the goal is still linked, for
    #    the field statistics).
    if check.get("class") == "ok" and check.get("matched") in ("action", "held"):
        base.ok, base.linked = True, ref is not None
        return base
    if check.get("class") == "ok":
        code, detail = "eval_turn_off_reading", {"matched": check.get("matched")}
    elif ref is None:
        # 2. No goal: lost before or inside structuring.
        code, detail = _unlinked(ctx, part)
    else:
        base.linked = True
        diffs = field_diffs(part, ref, ctx.tools, ctx.device_ids)
        plan = plan_for(ctx.record, ref.text, ctx.used_plans)
        plan_code, plan_detail = _planning_code(plan)
        if diffs:
            # 3. The goal differs from the gold: the first difference in priority order.
            key = next(k for k, _ in _DIFF_CODES if k in diffs)
            code = dict(_DIFF_CODES)[key]
            detail = {"diffs": {k: list(v) for k, v in diffs.items()},
                      "plan_outcome": (plan or {}).get("outcome")}
        elif plan_code:
            # 4. The goal is right; the planner did not produce a plan. A refusal the
            #    goal's own format explains is structuring's, not the planner's.
            code, detail = plan_code, plan_detail
            bare = bare_class_fields(ref.goal)
            if _PARAMETER_NAME.search(plan_detail.get("explanation") or ""):
                code = "struct_parameter_name"
                detail = {**plan_detail, "parameter_name": ref.goal.get("parameter_name")}
            elif plan_code == "plan_unrouted" and ref.goal.get(
                    "artifact_name") in _NO_NAME:
                code = "struct_device_unnamed"
            elif bare and plan_code in _FORMAT_PLAN_CODES:
                code, detail = "struct_class_prefix", {**plan_detail, "bare": bare}
        else:
            # 5. A plan exists; the home did not end up doing the gold call.
            code, detail = _execution_code(ctx, part, check)
    base.code, base.stage, base.detail = code, _stage(code), detail
    return base


def diagnose_infeasible(ctx: Context, part: Part, refused: bool) -> PartDiagnosis:
    """Diagnose an infeasible part (gold ``error_input``): ok when a refusal covers
    it, else what the pipeline did instead of refusing."""
    base = PartDiagnosis(part.index, part.token, False, part.reason, part.device_type,
                         part.room, None, ok=refused)
    ref = ctx.links.get(part.index)
    base.linked = ref is not None
    if refused:
        return base
    gold_tokens = {p.token for p in ctx.parts if p.feasible}
    extra = [t for t in ctx.sent if t not in gold_tokens]
    prefix = part.device_prefix
    on_target = [t for t in extra if prefix and token_device(t) == prefix]
    in_room = [t for t in extra if part.room and t.split(".")[0] == part.room]
    plan = plan_for(ctx.record, ref.text, ctx.used_plans) if ref else None
    plan_code, plan_detail = _planning_code(plan) if ref else (None, {})
    if on_target:
        code, detail = "infeas_accepted_on_target", {"sent": on_target}
        base.label_suspect = part.reason == "unexist_attribute"
    elif in_room:
        code, detail = "infeas_substituted", {"sent": in_room}
    elif ref is None:
        code, detail = _unlinked(ctx, part)
    elif plan_code and plan_code not in ("plan_no_device_match", "plan_no_action",
                                         "plan_false_impossible", "plan_false_clarify"):
        code, detail = plan_code, plan_detail
    elif extra:
        code, detail = "infeas_executed_other", {"sent": extra}
    else:
        code, detail = "infeas_silent", {"plan": plan_detail}
    base.code, base.stage, base.detail = code, _stage(code), detail
    return base


# -- one episode -----------------------------------------------------------------------


@dataclass
class EpisodeAnalysis:
    """Everything the report needs about one episode."""

    name: str
    verdict: str
    case: str
    htype: str
    bucket: Optional[str]
    home: Optional[str]
    query: str
    gold: str
    n_parts: int = 0
    parts: List[PartDiagnosis] = field(default_factory=list)
    primary_stage: Optional[str] = None
    primary_code: Optional[str] = None
    codes: List[str] = field(default_factory=list)
    extra_actions: List[str] = field(default_factory=list)
    missing_gold: List[str] = field(default_factory=list)
    n_intents: int = 0
    n_goals: int = 0
    n_dead: int = 0
    label_corrected: bool = False
    percent_relative: bool = False
    score: Optional[int] = None
    score_extra: Optional[int] = None
    wall_seconds: Optional[float] = None
    note: str = ""


def home_of(meta: Dict[str, Any]) -> Optional[str]:
    """``home74`` of a dataset id like ``home74_one_472``."""
    match = re.match(r"(home\d+)_", meta.get("id") or "")
    return match.group(1) if match else None


def rollup(parts: Sequence[PartDiagnosis]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """The primary stage and code of an episode, and every code of its failing parts.

    The primary stage is the earliest stage any part failed in; the primary code is the
    commonest code within it (ties: the taxonomy's order)."""
    failed = [p for p in parts if not p.ok and p.code]
    if not failed:
        return None, None, []
    codes = [p.code for p in failed]
    stage = min((_stage(c) for c in codes), key=STAGES.index)
    in_stage = Counter(c for c in codes if _stage(c) == stage)
    code = min(in_stage, key=lambda c: (-in_stage[c], CODE_ORDER[c]))
    return stage, code, sorted(set(codes), key=CODE_ORDER.get)


def _infra_code(detail: str) -> str:
    """The infrastructure code of an evaluation's infra detail text."""
    if _CONTEXT.search(detail):
        return "infra_llm_context"
    return "infra_llm_timeout" if "LLM calls failed" in detail or _TIMEOUT.search(
        detail) else "infra_llm_error"


def analyze_episode(record: Dict[str, Any], evaluation: Optional[Dict[str, Any]],
                    source: Optional[Dict[str, Any]], tools: Tools) -> EpisodeAnalysis:
    """Analyze one episode (module docstring).

    ``evaluation`` is the episode's ``.eval.json`` (or a reconstruction of it), None
    when none exists: the verdict is then ``unscored`` and no part is diagnosed.
    ``source`` is the converter's episode (optional): it names the room, device and
    reason of infeasible parts.
    """
    meta = record.get("meta") or {}
    homebench = (evaluation or {}).get("homebench") or {}
    out = EpisodeAnalysis(
        name=record.get("episode") or "", verdict="unscored",
        case=record.get("case") or meta.get("case") or "",
        htype=meta.get("homebench_type") or "", bucket=homebench.get("bucket"),
        home=home_of(meta), query=record.get("query") or "", gold=meta.get("gold") or "",
        label_corrected=(record.get("episode") or "").endswith("_fixed"),
        percent_relative=bool(homebench.get("percent_relative")),
        wall_seconds=record.get("wall_seconds"),
        n_intents=len(record.get("intents") or []))
    if evaluation is None:
        return out
    out.score, out.score_extra = evaluation.get("score"), evaluation.get("score_extra")
    if out.score == -1:
        out.verdict = "infra"
        out.primary_stage = "infrastructure"
        out.primary_code = _infra_code(str(evaluation.get("detail") or ""))
        out.codes = [out.primary_code]
        out.note = str(evaluation.get("detail") or "")
        return out
    out.verdict = "pass" if out.score == 1 else "fail"
    try:
        parts = build_parts(record, source, tools)
    except ValueError as exc:
        out.note = f"gold not parsed: {exc}"
        LOGGER.warning("%s: %s", out.name, out.note)
        return out
    out.n_parts = len(parts)
    goals, dead = collect_goals(record)
    out.n_goals, out.n_dead = len(goals), len(dead)
    sent = gold_log_module.gold_log((record.get("execution") or {}).get("actions")).tokens
    ctx = Context(record, tools, parts, link_parts(parts, goals, tools), list(dead),
                  _device_ids(record), sent, n_goals=len(goals),
                  n_intents=out.n_intents)
    checks = {c.get("goal"): c for c in (evaluation.get("breakdown") or {}).get(
        "checks") or []}
    refusals = homebench.get("refused", 0) + homebench.get("clarified", 0)
    out.parts = _diagnose_all(ctx, checks, refusals)
    if out.verdict == "fail":
        out.primary_stage, out.primary_code, out.codes = rollup(out.parts)
        if out.primary_code is None:
            out.primary_stage, out.primary_code, out.codes = (
                "evaluation", "unattributed", ["unattributed"])
    gold_count = Counter(p.token for p in parts if p.feasible)
    out.extra_actions = sorted((Counter(sent) - gold_count).elements())
    out.missing_gold = sorted((gold_count - Counter(sent)).elements())
    return out


def _diagnose_all(ctx: Context, checks: Dict[int, Dict[str, Any]], refusals: int
                  ) -> List[PartDiagnosis]:
    """Diagnose every part: feasible ones by their stored check; infeasible ones are
    ok in the order of their goal's refusal, as many as there are refusals (the scorer
    pools refusals, they are not tied to parts)."""
    infeasible = [p for p in ctx.parts if not p.feasible]
    # 1. Parts whose own goal was refused come first in the pool of refusals.
    def refused_goal(part: Part) -> bool:
        ref = ctx.links.get(part.index)
        return bool(ref and any(
            (pl.get("intent") or {}).get("text_intent") == ref.text
            and pl.get("outcome") in ("impossible", "clarify")
            for pl in ctx.record.get("plans") or []))
    ordered = sorted(infeasible, key=lambda p: (not refused_goal(p), p.index))
    covered = {p.index for p in ordered[:max(refusals, 0)]}
    diagnoses: List[PartDiagnosis] = []
    for part in ctx.parts:
        if part.feasible:
            diagnoses.append(diagnose_feasible(ctx, part, checks.get(part.index, {})))
        else:
            diagnoses.append(diagnose_infeasible(ctx, part, part.index in covered))
    return diagnoses
