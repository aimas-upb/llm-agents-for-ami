"""Promoting incomplete goals and predicates when the environment allows only
one reading.

An incomplete goal that names *some* of its location and device -- "turn on
the dishwasher", "set the kitchen fan to 30%" -- is checked against the
environment: if exactly one device and one affordance fit, the missing slots
are filled from it and the goal becomes explicit. Several fits leave it
incomplete (a predicate then quantifies over them); none leaves it as it is.

A goal or device predicate naming *neither* location nor device is forwarded
unchanged: the user would have to be asked, and that dialogue is not built yet.

Pure functions: building the query payload and applying its answer. The RPC
itself is the caller's (`GoalStructuringBehaviour`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from rdflib import URIRef
from rdflib.namespace import RDFS

from ....shared.models.goal_structure import GoalSpec, PredicateSpec
from ....shared.utils.namespaces import SAREF, expand
from ....shared.utils.vocabulary import vocabulary


@lru_cache(maxsize=256)
def is_command_class(curie: str) -> bool:
    """Is this class a command (under `saref:Command`), rather than a property?

    Decides which resolver slot an `affordance_class` goes into: a command is
    looked up among actions, a property among the properties actions change.
    """
    graph = vocabulary()
    return SAREF.Command in set(graph.transitive_objects(URIRef(expand(curie)),
                                                         RDFS.subClassOf))


def _names_something(location: Optional[str], device_class: Optional[str],
                     device_name: Optional[str]) -> bool:
    return bool(location or device_class or device_name)


# --------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------

def goal_query(goal: GoalSpec) -> Optional[Dict[str, Any]]:
    """The ENV_CAPABILITY_QUERY payload for an incomplete goal, or None when it
    is not to be resolved (not incomplete, ambiguous, or naming neither its
    location nor its device)."""
    if goal.goal_specificity != "incomplete":
        return None
    if not _names_something(goal.location_class, goal.artifact_class,
                            goal.artifact_name):
        return None
    payload: Dict[str, Any] = {
        "text_intent": goal.intent_text,
        "location_class": goal.location_class,
        "device_class": goal.artifact_class,
        "artifact_name": goal.artifact_name,
    }
    if goal.affordance_class:
        slot = "command" if is_command_class(goal.affordance_class) else "device_property"
        payload[slot] = {"class": goal.affordance_class}
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


def apply_goal_resolution(goal: GoalSpec, response: Dict[str, Any]) -> bool:
    """Fill the goal from a unique answer and mark it explicit. True if promoted."""
    if response.get("outcome") != "found":
        return False
    entry = _unique(response.get("entries") or [], bool(goal.affordance_class))
    if entry is None:
        return False
    goal.location_class = goal.location_class or entry.get("workspace_class")
    goal.artifact_class = goal.artifact_class or entry.get("artifact_type")
    goal.artifact_name = goal.artifact_name or entry.get("artifact_name")
    goal.affordance_name = goal.affordance_name or entry.get("affordance_name")
    if all((goal.location_class, goal.artifact_class, goal.artifact_name,
            goal.affordance_class, goal.affordance_name)):
        goal.goal_specificity = "explicit"
        return True
    return False


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------

def predicate_query(predicate: PredicateSpec) -> Optional[Dict[str, Any]]:
    """The query payload for an incomplete device predicate, or None.

    Environment predicates are not resolved here: their variable is named by
    title, not class, so they keep their quantifier and the planner reads the
    variable across the spaces that have it.
    """
    if (predicate.predicate_specificity != "incomplete"
            or predicate.predicate_subject != "device_property"):
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


def apply_predicate_resolution(predicate: PredicateSpec,
                               response: Dict[str, Any]) -> bool:
    """Fill the predicate from a unique answer and mark it explicit; with several
    candidates, make sure it carries a quantifier. True if promoted."""
    entries = (response.get("entries") or []) if response.get("outcome") == "found" else []
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
