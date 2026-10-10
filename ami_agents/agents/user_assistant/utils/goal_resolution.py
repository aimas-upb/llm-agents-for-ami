"""Resolving incomplete goals and predicates against the environment.

What an incomplete goal leaves empty decides what happens to it here (the
cases are `shared/utils/incomplete_goals.py`'s):

    UNNAMED, NO_ACTION   only the user can say what is meant: the goal is
                         marked for clarification and its request is held back
                         from the InteractionSolver ("turn it off"; "the fan")
    NO_LOCATION          its devices are looked up by class and action:
                           one device         -> promoted to explicit
                           one room class     -> that room is filled in
                           several room classes, goal means "all" -> left as is,
                                                 the solver plans every device
                           several room classes, goal means "one" -> marked for
                                                 clarification (which room?)
    NO_CLASS, NO_NAME    looked up; one device and one action -> promoted to
                         explicit, otherwise left for the solver to choose
    VALUE_ONLY           not looked up: without an action nothing can be
                         promoted; the solver plans it against the candidates

A goal's quantifier ("all" or "one") is the structurer's reading of the
request and is never changed here, except that a promoted goal -- naming its
one device -- drops it.

An environment predicate missing its room ("while the humidity is above 60%")
is looked up by its variable's class: if devices read that variable in one room
only, that room is filled in; several rooms make it quantify over them. A
device predicate naming neither location nor device is forwarded unchanged.

Pure functions: building the query payload and applying its answer. The RPC
itself is the caller's (`GoalStructuringBehaviour`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....shared.models.goal_structure import GoalSpec, GoalStructure, PredicateSpec
from ....shared.utils.incomplete_goals import (
    NO_ACTION,
    NO_CLASS,
    NO_LOCATION,
    NO_NAME,
    UNNAMED,
    incomplete_case,
)


def _names_something(location: Optional[str], device_class: Optional[str],
                     device_name: Optional[str]) -> bool:
    return bool(location or device_class or device_name)


def _words(curie: Optional[str]) -> str:
    """A class as words for a question: "homeont:LivingRoom" -> "living room"."""
    local = (curie or "").split(":")[-1]
    out = ""
    for i, ch in enumerate(local):
        if ch.isupper() and i and not local[i - 1].isupper():
            out += " "
        out += ch.lower()
    return out


# --------------------------------------------------------------------------
# Goals: what only the user can answer
# --------------------------------------------------------------------------

def clarification_question(goal: GoalSpec) -> Optional[str]:
    """The question to ask the user about this goal, or None when the goal can
    go on to planning."""
    # Set by resolution: the devices are in several rooms, and the goal means
    # one of them.
    if goal.clarification:
        return goal.clarification
    case = incomplete_case(goal)
    if case == UNNAMED:
        return "Which device do you mean, and in which room?"
    if case == NO_ACTION:
        device = goal.artifact_name or _words(goal.artifact_class) or "device"
        room = f" in the {_words(goal.location_class)}" if goal.location_class else ""
        return f"What should I do with the {device}{room}?"
    return None


def needs_clarification(goal: GoalSpec) -> bool:
    return clarification_question(goal) is not None


def structure_clarification(structure: GoalStructure) -> Optional[str]:
    """The first question any of the structure's goals raises. A structure is
    planned whole, so one such goal holds all of it back."""
    for goal in structure.goals.values():
        question = clarification_question(goal)
        if question:
            return question
    return None


# --------------------------------------------------------------------------
# Goals: looking the devices up
# --------------------------------------------------------------------------

def goal_query(goal: GoalSpec) -> Optional[Dict[str, Any]]:
    """The ENV_CAPABILITY_QUERY payload for an incomplete goal, or None when it
    is not to be looked up: only goals whose action is known are, since
    without one nothing can be promoted."""
    if incomplete_case(goal) not in (NO_LOCATION, NO_CLASS, NO_NAME):
        return None
    payload: Dict[str, Any] = {
        "text_intent": goal.intent_text,
        "location_class": goal.location_class,
        "device_class": goal.artifact_class,
        "artifact_name": goal.artifact_name,
    }
    # A goal's affordance class is the command class of its action.
    if goal.affordance_class:
        payload["command"] = {"class": goal.affordance_class}
    return payload


def _unique(entries: List[Dict[str, Any]], with_affordance: bool) -> Optional[Dict[str, Any]]:
    """The one entry, if the answer allows only one device (and one affordance)."""
    artifacts = {e.get("artifact_name") for e in entries}
    if len(artifacts) != 1:
        return None
    if with_affordance:
        affordances = {e.get("affordance_name") for e in entries}
        if len(affordances) != 1 or None in affordances:
            return None
    return entries[0]


def _locate(goal: GoalSpec, entries: List[Dict[str, Any]]) -> None:
    """A goal missing its room, found on several devices: one room class fills
    it in; several are fine for "all" and a question for "one"."""
    rooms = sorted({e["workspace_class"] for e in entries if e.get("workspace_class")})
    if len(rooms) == 1:
        goal.location_class = rooms[0]
    elif len(rooms) > 1 and goal.quantifier != "all":
        goal.clarification = ("Which room do you mean: "
                              + " or ".join(f"the {_words(r)}" for r in rooms) + "?")


def apply_goal_resolution(goal: GoalSpec, response: Dict[str, Any]) -> bool:
    """Apply the lookup's answer to the goal. True if promoted to explicit.

    One device (and one action) fills the goal's empty slots from it; with all
    of them filled the goal becomes explicit and drops its quantifier. Several
    devices leave it incomplete -- for a goal missing its room, with the room
    filled in or a question set (`_locate`)."""
    if response.get("outcome") != "found":
        return False
    entries = response.get("entries") or []
    case = incomplete_case(goal)

    # 1. One device: fill the gaps from it, and promote when nothing is left.
    entry = _unique(entries, bool(goal.affordance_class))
    if entry is not None:
        goal.location_class = goal.location_class or entry.get("workspace_class")
        goal.artifact_class = goal.artifact_class or entry.get("artifact_type")
        goal.artifact_name = goal.artifact_name or entry.get("artifact_name")
        goal.affordance_name = goal.affordance_name or entry.get("affordance_name")
        if all((goal.location_class, goal.artifact_class, goal.artifact_name,
                goal.affordance_class, goal.affordance_name)):
            goal.goal_specificity = "explicit"
            goal.quantifier = None
            return True
        return False

    # 2. Several devices: the solver chooses among them -- once the room is
    #    settled, for a goal that did not name it.
    if case == NO_LOCATION:
        _locate(goal, entries)
    return False


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------

def predicate_query(predicate: PredicateSpec) -> Optional[Dict[str, Any]]:
    """The query payload for an incomplete predicate, or None.

    A device predicate is looked up by its location and device. An environment
    predicate missing its room is looked up by its variable's class: the
    rooms where some device reads that variable are the rooms it can mean.
    """
    if predicate.predicate_specificity != "incomplete":
        return None
    if predicate.predicate_subject == "environment_property":
        return _environment_query(predicate)
    if predicate.predicate_subject != "device_property":
        return None
    if not _names_something(predicate.location_class, predicate.artifact_class,
                            predicate.artifact_name):
        return None
    payload: Dict[str, Any] = {
        "text_intent": predicate.predicate_text,
        "location_class": predicate.location_class,
        "device_class": predicate.artifact_class,
        "artifact_name": predicate.artifact_name,
    }
    if predicate.property_class:
        payload["device_property"] = {"class": predicate.property_class}
    return payload


def _environment_query(predicate: PredicateSpec) -> Optional[Dict[str, Any]]:
    var = predicate.environment_var
    if var is None or not var.property_class or var.space_class:
        return None     # nothing to look up by, or the room is already known
    return {
        "text_intent": predicate.predicate_text,
        "environment_variable": {"class": var.property_class},
    }


def _apply_environment_resolution(predicate: PredicateSpec,
                                  entries: List[Dict[str, Any]]) -> bool:
    """One room with a reading of the variable fills the space; several make
    the predicate quantify over rooms."""
    rooms = {(e.get("workspace_class"), e.get("workspace_name")) for e in entries}
    var = predicate.environment_var
    if len(rooms) != 1 or var is None:
        if len(rooms) > 1:
            predicate.quantifier = predicate.quantifier or "any"
        return False
    var.space_class, var.space_name = rooms.pop()
    # Explicit needs the threshold too: variable, room and value all stated.
    if var.space_class and predicate.comparison and predicate.target_value_text:
        predicate.predicate_specificity = "explicit"
        predicate.quantifier = None
        return True
    return False


def apply_predicate_resolution(predicate: PredicateSpec,
                               response: Dict[str, Any]) -> bool:
    """Fill the predicate from a unique answer and mark it explicit; with several
    candidates, make sure it carries a quantifier. True if promoted."""
    entries = (response.get("entries") or []) if response.get("outcome") == "found" else []
    if predicate.predicate_subject == "environment_property":
        return _apply_environment_resolution(predicate, entries)
    entry = _unique(entries, bool(predicate.property_class))
    if entry is None:
        if len({e.get("artifact_name") for e in entries}) > 1:
            # Reading several devices is safe: the condition quantifies over them.
            predicate.quantifier = predicate.quantifier or "any"
        return False
    predicate.location_class = predicate.location_class or entry.get("workspace_class")
    predicate.artifact_class = predicate.artifact_class or entry.get("artifact_type")
    predicate.artifact_name = predicate.artifact_name or entry.get("artifact_name")
    predicate.property_name = predicate.property_name or entry.get("affordance_name")
    if all((predicate.location_class, predicate.artifact_class,
            predicate.artifact_name, predicate.property_class,
            predicate.property_name)):
        predicate.predicate_specificity = "explicit"
        predicate.quantifier = None
        return True
    return False


def default_quantifiers(predicates: Dict[str, PredicateSpec]) -> None:
    """An incomplete or ambiguous predicate left without a quantifier reads as
    "any": at least one candidate satisfies it."""
    for predicate in predicates.values():
        if predicate.predicate_specificity in ("incomplete", "ambiguous") \
                and predicate.quantifier is None:
            predicate.quantifier = "any"
