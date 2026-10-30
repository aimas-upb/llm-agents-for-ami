"""The structure extracted from one atomic GOAL_REQUEST.

Two shapes come out of goal structuring, and both are a `GoalStructure`:

    simple       one goal, achievement, no dependency structure -- `G1` alone
    dependency   several goals and the predicates that condition them, each
                 annotated with the role it plays and how it is ordered

Nothing here composes a plan. A predicate says what it is for (`role`) and
which goals it conditions (`applies_to`); a goal says when it starts relative
to time or to other goals (`timing`). Turning that into a behaviour tree is the
planner's job.

Two "missing" values are kept apart, because they mean different things:

    "NA"   the slot does not apply (an action that takes no input parameter)
    None   the slot applies, but its value could not be identified

Only `parameter_name` and `property_field` admit "NA"; anywhere else it is read
as None.

`from_dict` is tolerant -- unknown keys are ignored and values outside a
field's vocabulary become None -- because its input is LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

NA = "NA"

SPECIFICITIES = ("explicit", "incomplete", "ambiguous")
GOAL_KINDS = ("achievement", "maintenance")
GOAL_EFFECTS = ("set", "modify")
STRUCTURES = ("simple", "dependency")
PREDICATE_SUBJECTS = ("device_property", "environment_property")
COMPARISONS = ("==", "!=", ">", ">=", "<", "<=")
QUANTIFIERS = ("any", "all")             # predicates: some / every candidate satisfies it
GOAL_QUANTIFIERS = ("all", "one")        # goals: act on every candidate / on one
DEVICE_REFERENCES = ("name", "description")  # how a request picks out its device
DIRECTIONS = ("high", "low")
CHANGE_DIRECTIONS = ("increase", "decrease")
ROLES = ("guard", "trigger", "scope", "recurrence", "termination", "filter")
CONNECTIVES = ("and", "or")
OFFSET_DIRECTIONS = ("after", "before")
STARTS = ("now", "at_time", "after_delay", "after_goal", "with_goal", "on_trigger")
END_KINDS = ("duration", "at_time")


# --------------------------------------------------------------------------
# Value cleaning
# --------------------------------------------------------------------------

def _text(value: Any) -> Optional[str]:
    """A non-empty string, or None. "NA" and spelled-out nulls count as None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("na", "null", "none"):
        return None
    return text


def _flag(value: Any) -> bool:
    """A boolean; LLM output may spell it as a string. Anything but true is false."""
    if isinstance(value, bool):
        return value
    text = _text(value)
    return text is not None and text.lower() == "true"


def _na_or_text(value: Any) -> Optional[str]:
    """Like `_text`, but "NA" survives: the slot does not apply."""
    if isinstance(value, str) and value.strip().upper() == NA:
        return NA
    return _text(value)


def _choice(value: Any, allowed) -> Optional[str]:
    text = _text(value)
    if text is None:
        return None
    text = text.lower()
    return text if text in allowed else None


def _goal_quantifier(value: Any) -> Optional[str]:
    """A goal's quantifier. "any" -- the predicates' label, which a model may
    still write for a goal -- settles to "one"."""
    text = _choice(value, GOAL_QUANTIFIERS + ("any",))
    return "one" if text == "any" else text


def _comparison(value: Any) -> Optional[str]:
    """Operators are symbols, so `_choice`'s lowercasing does not apply."""
    text = _text(value)
    return text if text in COMPARISONS else None


def _ids(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [i for i in (_text(v) for v in value) if i]


def _drop_empty(out: Dict[str, Any], keep: tuple = ()) -> Dict[str, Any]:
    """Omit None and empty lists, except for the keys a reader always expects."""
    return {k: v for k, v in out.items()
            if k in keep or (v is not None and v != [])}


# --------------------------------------------------------------------------
# Parts
# --------------------------------------------------------------------------

@dataclass
class EnvironmentVar:
    """An environment variable of a space: what a sensor reads and an actuator
    moves. `condition_direction` is set only inside an ambiguous predicate;
    `property_change_direction` only inside an ambiguous goal -- the way the
    request wants the variable to go."""

    property_name: Optional[str] = None
    property_class: Optional[str] = None
    property_quantity: Optional[str] = None
    property_measurement_unit: Optional[str] = None
    space_class: Optional[str] = None
    space_name: Optional[str] = None
    condition_direction: Optional[str] = None
    property_change_direction: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> Optional["EnvironmentVar"]:
        if not isinstance(data, dict):
            return None
        space = data.get("property_sensed_space")
        space = space if isinstance(space, dict) else {}
        return cls(
            property_name=_text(data.get("property_name")),
            property_class=_text(data.get("property_class")),
            property_quantity=_text(data.get("property_quantity")),
            property_measurement_unit=_text(data.get("property_measurement_unit")),
            space_class=_text(space.get("space_class")),
            space_name=_text(space.get("space_name")),
            condition_direction=_choice(data.get("condition_direction"), DIRECTIONS),
            property_change_direction=_choice(data.get("property_change_direction"),
                                              CHANGE_DIRECTIONS),
        )

    def to_wire_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "property_name": self.property_name,
            "property_class": self.property_class,
            "property_quantity": self.property_quantity,
            "property_measurement_unit": self.property_measurement_unit,
            "property_sensed_space": {"space_class": self.space_class,
                                      "space_name": self.space_name},
        }
        if self.condition_direction:
            out["condition_direction"] = self.condition_direction
        if self.property_change_direction:
            out["property_change_direction"] = self.property_change_direction
        return out


def _env_vars(value: Any) -> List[EnvironmentVar]:
    if not isinstance(value, list):
        return []
    return [v for v in (EnvironmentVar.from_dict(item) for item in value) if v]


@dataclass
class GoalTiming:
    """When a goal starts, relative to time or to other goals, and -- if the
    user bounded it by time -- when it ends.

    `completes_with` (a predicate or goal id) is the goal's run -- a cycle it
    starts, which then ends by itself -- asked to complete when that event
    happens or that goal's run completes ("start washer 1 so it and dryer 1
    finish together"). It never stops the goal (`ends` and termination
    predicates do that). With no `starts` the planner chooses the start (when
    the time left before the event equals the run's length); with a `starts`
    the user gave ("now", after another goal), the planner only checks that the
    two agree.

    `starts` / `ends` (or a termination predicate) hold the description the
    plan executes. When the user ALSO stated a clock time for a start or an end
    described relative to an anchor ("at 5:48 PM, which is 23 minutes before
    washer 1 finishes"), that clock time is kept as written in
    `stated_start_time_text` / `stated_end_time_text`, so the planner can check
    that the two name the same moment."""

    starts: Optional[str] = None
    time_text: Optional[str] = None
    offset_text: Optional[str] = None
    ref_goal: Optional[str] = None
    completes_with: Optional[str] = None
    ends_kind: Optional[str] = None
    ends_text: Optional[str] = None
    stated_start_time_text: Optional[str] = None
    stated_end_time_text: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> Optional["GoalTiming"]:
        if not isinstance(data, dict):
            return None
        ends = data.get("ends")
        ends = ends if isinstance(ends, dict) else {}
        return cls(
            starts=_choice(data.get("starts"), STARTS),
            time_text=_text(data.get("time_text")),
            offset_text=_text(data.get("offset_text")),
            ref_goal=_text(data.get("ref_goal")),
            completes_with=_text(data.get("completes_with")),
            ends_kind=_choice(ends.get("kind"), END_KINDS),
            ends_text=_text(ends.get("text")),
            stated_start_time_text=_text(data.get("stated_start_time_text")),
            stated_end_time_text=_text(data.get("stated_end_time_text")),
        )

    def to_wire_dict(self) -> Dict[str, Any]:
        out = _drop_empty({
            "starts": self.starts,
            "time_text": self.time_text,
            "offset_text": self.offset_text,
            "ref_goal": self.ref_goal,
            "completes_with": self.completes_with,
            "stated_start_time_text": self.stated_start_time_text,
            "stated_end_time_text": self.stated_end_time_text,
        }, keep=("starts",))
        if self.ends_kind:
            out["ends"] = {"kind": self.ends_kind, "text": self.ends_text}
        return out

    @property
    def start_chosen(self) -> bool:
        """True when the planner chooses the start: the goal completes with
        something and the user gave no start. Such a goal never starts "now"
        by default."""
        return bool(self.completes_with) and self.starts is None


# --------------------------------------------------------------------------
# Goals and predicates
# --------------------------------------------------------------------------

@dataclass
class GoalSpec:
    """One goal: what to change, on what, and how.

    Explicit and incomplete goals fill the device-side slots (incomplete ones
    leave some None); an ambiguous goal names no device and lists the
    environment variables it implies instead.
    """

    goal_specificity: Optional[str] = None
    goal_kind: Optional[str] = None
    intent_text: Optional[str] = None
    location_class: Optional[str] = None
    artifact_class: Optional[str] = None
    artifact_name: Optional[str] = None
    affordance_class: Optional[str] = None
    parent_class: Optional[str] = None
    affordance_name: Optional[str] = None
    parameter_name: Optional[str] = None
    goal_effect: Optional[str] = None
    # The user's words for the value, and the value the affordance's schema
    # makes of them (an enum member, a number in the schema's unit).
    target_value_text: Optional[str] = None
    target_value_determined: Optional[str] = None
    # Whether target_value_determined is a percentage. Converting it to the
    # input's scale is left to whoever plans the goal.
    target_value_is_percentage: bool = False
    # Incomplete goals only: whether the change is meant for every device that
    # fits ("all") or for one of them ("one"), as the request means it -- set by
    # the structurer alone; null only when the request does not say. Not "any":
    # in English that reads both ways ("any light that is on" is every one), and
    # predicates already use it for "at least one satisfies".
    quantifier: Optional[str] = None
    implied_environment_vars: List[EnvironmentVar] = field(default_factory=list)
    timing: Optional[GoalTiming] = None
    rationale: Optional[str] = None
    # UA-local, never on the wire: the question to ask when resolution found
    # the goal's devices in several rooms and the goal means only one of them,
    # or when the device the request names is not in the home.
    clarification: Optional[str] = None
    # UA-local, never on the wire: how the request picks out its device --
    # "name" (its own name: "kitchen light 2") or "description" (its kind,
    # room or another property: "the fan"). A named device the home does not
    # have is asked about, never completed with another
    # (`goal_resolution.named_device_missing`).
    device_reference: Optional[str] = None
    # UA-local, never on the wire: the request's own words for a device it
    # mentions (by name or by kind) that the home does not have ("air
    # conditioner"). Such a goal is asked about, never completed with another
    # device (`goal_resolution.device_missing`).
    unmatched_device: Optional[str] = None
    # UA-local, never on the wire: the request's words for what it asks a
    # device that IS in the home to change, when none of the device's actions
    # can ("brightness" on an on/off light). Asked about, never moved to
    # another device that has such an action.
    unmatched_action: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "GoalSpec":
        data = data if isinstance(data, dict) else {}
        return cls(
            goal_specificity=_choice(data.get("goal_specificity"), SPECIFICITIES),
            goal_kind=_choice(data.get("goal_kind"), GOAL_KINDS),
            intent_text=_text(data.get("intent_text")),
            location_class=_text(data.get("location_class")),
            artifact_class=_text(data.get("artifact_class")),
            artifact_name=_text(data.get("artifact_name")),
            device_reference=_choice(data.get("device_reference"), DEVICE_REFERENCES),
            unmatched_device=_text(data.get("unmatched_device")),
            unmatched_action=_text(data.get("unmatched_action")),
            affordance_class=_text(data.get("affordance_class")),
            parent_class=_text(data.get("parent_class")),
            affordance_name=_text(data.get("affordance_name")),
            parameter_name=_na_or_text(data.get("parameter_name")),
            goal_effect=_choice(data.get("goal_effect"), GOAL_EFFECTS),
            target_value_text=_text(data.get("target_value_text")),
            target_value_determined=_text(data.get("target_value_determined")),
            target_value_is_percentage=_flag(data.get("target_value_is_percentage")),
            quantifier=_goal_quantifier(data.get("quantifier")),
            implied_environment_vars=_env_vars(data.get("implied_environment_vars")),
            timing=GoalTiming.from_dict(data.get("timing")),
            rationale=_text(data.get("rationale")),
        )

    def to_wire_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "goal_specificity": self.goal_specificity,
            "goal_kind": self.goal_kind,
            "intent_text": self.intent_text,
        }
        if self.goal_specificity == "ambiguous":
            out["implied_environment_vars"] = [
                v.to_wire_dict() for v in self.implied_environment_vars]
        else:
            out.update({
                "location_class": self.location_class,
                "artifact_class": self.artifact_class,
                "artifact_name": self.artifact_name,
                "affordance_class": self.affordance_class,
                "parent_class": self.parent_class,
                "affordance_name": self.affordance_name,
                "parameter_name": self.parameter_name,
                "goal_effect": self.goal_effect,
                "target_value_text": self.target_value_text,
                "target_value_determined": self.target_value_determined,
                "target_value_is_percentage": self.target_value_is_percentage,
                "quantifier": self.quantifier,
            })
        if self.timing:
            out["timing"] = self.timing.to_wire_dict()
        if self.rationale:
            out["rationale"] = self.rationale
        return out


@dataclass
class PredicateSpec:
    """One condition the request depends on, and the role it plays there."""

    predicate_specificity: Optional[str] = None
    predicate_subject: Optional[str] = None
    predicate_text: Optional[str] = None
    # Device property
    location_class: Optional[str] = None
    artifact_class: Optional[str] = None
    artifact_name: Optional[str] = None
    property_class: Optional[str] = None
    parent_class: Optional[str] = None
    property_name: Optional[str] = None
    property_field: Optional[str] = None
    # Both subjects
    comparison: Optional[str] = None
    target_value_text: Optional[str] = None
    target_value_determined: Optional[str] = None
    quantifier: Optional[str] = None
    # Environment property
    environment_var: Optional[EnvironmentVar] = None
    implied_environment_vars: List[EnvironmentVar] = field(default_factory=list)
    qualitative_condition: Optional[str] = None
    # Binding
    role: Optional[str] = None
    applies_to: List[str] = field(default_factory=list)
    connective: Optional[str] = None
    offset_text: Optional[str] = None
    offset_direction: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "PredicateSpec":
        data = data if isinstance(data, dict) else {}
        return cls(
            predicate_specificity=_choice(data.get("predicate_specificity"), SPECIFICITIES),
            predicate_subject=_choice(data.get("predicate_subject"), PREDICATE_SUBJECTS),
            predicate_text=_text(data.get("predicate_text")),
            location_class=_text(data.get("location_class")),
            artifact_class=_text(data.get("artifact_class")),
            artifact_name=_text(data.get("artifact_name")),
            property_class=_text(data.get("property_class")),
            parent_class=_text(data.get("parent_class")),
            property_name=_text(data.get("property_name")),
            property_field=_na_or_text(data.get("property_field")),
            comparison=_comparison(data.get("comparison")),
            target_value_text=_text(data.get("target_value_text")),
            target_value_determined=_text(data.get("target_value_determined")),
            quantifier=_choice(data.get("quantifier"), QUANTIFIERS),
            environment_var=EnvironmentVar.from_dict(data.get("environment_var")),
            implied_environment_vars=_env_vars(data.get("implied_environment_vars")),
            qualitative_condition=_text(data.get("qualitative_condition")),
            role=_choice(data.get("role"), ROLES),
            applies_to=_ids(data.get("applies_to")),
            connective=_choice(data.get("connective"), CONNECTIVES) or "and",
            offset_text=_text(data.get("offset_text")),
            offset_direction=_choice(data.get("offset_direction"), OFFSET_DIRECTIONS),
        )

    def to_wire_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "predicate_specificity": self.predicate_specificity,
            "predicate_subject": self.predicate_subject,
            "predicate_text": self.predicate_text,
        }
        if self.predicate_specificity == "ambiguous":
            out["implied_environment_vars"] = [
                v.to_wire_dict() for v in self.implied_environment_vars]
            out["qualitative_condition"] = self.qualitative_condition
        elif self.predicate_subject == "environment_property":
            out["environment_var"] = (self.environment_var.to_wire_dict()
                                      if self.environment_var else None)
            out["comparison"] = self.comparison
            out["target_value_text"] = self.target_value_text
            out["target_value_determined"] = self.target_value_determined
        else:
            out.update({
                "location_class": self.location_class,
                "artifact_class": self.artifact_class,
                "artifact_name": self.artifact_name,
                "property_class": self.property_class,
                "parent_class": self.parent_class,
                "property_name": self.property_name,
                "property_field": self.property_field,
                "comparison": self.comparison,
                "target_value_text": self.target_value_text,
                "target_value_determined": self.target_value_determined,
            })
        out["quantifier"] = self.quantifier
        out["role"] = self.role
        out["applies_to"] = list(self.applies_to)
        out["connective"] = self.connective
        if self.role == "trigger":
            out["offset_text"] = self.offset_text
            out["offset_direction"] = self.offset_direction
        return out


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------

@dataclass
class GoalStructure:
    """Everything extracted from one atomic GOAL_REQUEST."""

    intent_text: str
    qualifiers: List[str] = field(default_factory=list)
    structure: str = "simple"
    goals: Dict[str, GoalSpec] = field(default_factory=dict)
    predicates: Dict[str, PredicateSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any, intent_text: str = "",
                  qualifiers: Optional[List[str]] = None,
                  structure: Optional[str] = None) -> "GoalStructure":
        """Build from parser output.

        A simple structure may arrive as a bare goal object (the simple prompt
        returns exactly one goal); it is filed as `G1`.
        """
        data = data if isinstance(data, dict) else {}
        structure = _choice(structure or data.get("structure"), STRUCTURES) or (
            "dependency" if "predicates" in data else "simple")

        raw_goals = data.get("goals")
        if not isinstance(raw_goals, dict):
            raw_goals = {"G1": data} if data.get("goal_specificity") else {}
        raw_predicates = data.get("predicates")
        if not isinstance(raw_predicates, dict):
            raw_predicates = {}

        goals = {str(k): GoalSpec.from_dict(v) for k, v in raw_goals.items()}
        for goal in goals.values():
            goal.intent_text = goal.intent_text or intent_text or None
        return cls(
            intent_text=intent_text or _text(data.get("text_intent")) or "",
            qualifiers=list(qualifiers if qualifiers is not None
                            else data.get("qualifiers") or []),
            structure=structure,
            goals=goals,
            predicates={str(k): PredicateSpec.from_dict(v)
                        for k, v in raw_predicates.items()},
        )

    def to_query_string(self) -> str:
        """The user's words, as the planner's fallback key."""
        return self.intent_text

    def to_wire_dict(self) -> Dict[str, Any]:
        # `text_intent` keeps the planner able to fall back on the user's words
        # for any part of this structure it cannot yet consume.
        return {
            "text_intent": self.intent_text,
            "qualifiers": list(self.qualifiers),
            "structure": self.structure,
            "goals": {gid: g.to_wire_dict() for gid, g in self.goals.items()},
            "predicates": {pid: p.to_wire_dict()
                           for pid, p in self.predicates.items()},
        }
