"""Choosing the devices an incomplete achievement goal is planned on.

An incomplete goal names its action but not exactly one device: "turn on the
kitchen lights" (no device name), "turn on the light in the bedroom" with a
bedroom lamp and a ceiling light of different kinds (no class), "turn off all
the fans" (no room). The goal's own names filter the TD graph, and what comes
back decides the outcome (the cases are `shared/utils/incomplete_goals.py`'s):

    no candidate                       -> impossible, with the reason
    NO_CLASS, devices of several kinds,
      quantifier "one" (or none)       -> back to the user: which one?
    one candidate                      -> planned as an explicit goal
    several, quantifier "all"          -> planned on every one of them
    several, quantifier "one" (or none)-> planned on the first (by title)

Each chosen device becomes an explicit goal (`as_explicit`), so planning it is
the explicit path's business -- deterministic for `set` where it can be. The
per-device plans are then combined into the goal's one plan entry (`combine`).

A goal that names no action at all (VALUE_ONLY) is not chosen here: it is
planned by the small model against every candidate device, see
`specific_capability_context._incomplete_achievement`.

Pure functions over the InteractionSolver's closed TD graph; the behaviour is
`behaviours/incomplete_goal_planning.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph

from ....shared.models.goal_structure import GoalSpec, GoalStructure
from ....shared.utils.incomplete_goals import ISA_CASES, NO_CLASS, incomplete_case
from ....shared.utils.namespaces import TD
from .explicit_goal_plan import find_actions
from .specific_capability_context import device_class, device_location

# Marks plan entries produced here.
SOURCE = "incomplete"

PLAN = "plan"
IMPOSSIBLE = "impossible"
CLARIFY = "clarify"


# --------------------------------------------------------------------------
# Which goals qualify
# --------------------------------------------------------------------------

def incomplete_goal(structure: GoalStructure) -> Optional[GoalSpec]:
    """The goal, if this structure is one incomplete achievement goal the
    InteractionSolver plans itself; None leaves it to the planning workflow
    (dependency structures, maintenance goals)."""
    if structure.structure != "simple" or structure.predicates or len(structure.goals) != 1:
        return None
    (goal,) = structure.goals.values()
    if goal.goal_kind != "achievement" or incomplete_case(goal) not in ISA_CASES:
        return None
    return goal


# --------------------------------------------------------------------------
# Finding and choosing the candidates
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    """One device the goal's names allow, with the action they point to."""

    title: str
    artifact_class: Optional[str]
    location_class: Optional[str]
    affordance_name: Optional[str]


def candidates(graph: Graph, goal: GoalSpec) -> List[Candidate]:
    """Every device (with its action) the goal's names and classes allow,
    sorted by device title -- the order "the first" refers to."""
    out: Dict[Tuple[str, str], Candidate] = {}
    for row in find_actions(graph, goal):
        name = graph.value(row["affordance"], TD.title)
        key = (row["artifact_title"], str(name))
        out.setdefault(key, Candidate(
            title=row["artifact_title"],
            artifact_class=device_class(graph, row["artifact"]),
            location_class=device_location(graph, row["artifact"]),
            affordance_name=str(name) if name is not None else None))
    return sorted(out.values(), key=lambda c: (c.title, c.affordance_name or ""))


@dataclass
class Choice:
    """What to do with the goal: plan it on `chosen`, report it impossible, or
    ask the user to choose among `candidates`."""

    outcome: str
    chosen: List[Candidate] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    reason: str = ""


def choose(goal: GoalSpec, found: List[Candidate]) -> Choice:
    """Apply the goal's case and quantifier to its candidates."""
    # 1. Nothing fits: the request names something the home does not have.
    if not found:
        return Choice(IMPOSSIBLE, reason=f"No {_describe(goal)}.")

    # 2. A device with several actions of the goal's command class (two
    #    setpoints on one AC) cannot be decided by the quantifier: ask.
    devices = {c.title for c in found}
    if len(found) > len(devices):
        return Choice(CLARIFY, candidates=found,
                      reason="Which action do you mean: "
                             + " or ".join(sorted({c.affordance_name or "?" for c in found}))
                             + "?")

    # 3. The goal did not name the kind of device, several kinds fit, and it
    #    means one of them: only the user can say which. Meaning all of them
    #    ("turn off everything in the living room"), the kind does not matter.
    if (incomplete_case(goal) == NO_CLASS and goal.quantifier != "all"
            and len({c.artifact_class for c in found}) > 1):
        return Choice(CLARIFY, candidates=found,
                      reason="Which device do you mean: "
                             + " or ".join(c.title for c in found) + "?")

    # 4. One device, all of them, or the first.
    if len(found) == 1 or goal.quantifier == "all":
        return Choice(PLAN, chosen=found, candidates=found)
    return Choice(PLAN, chosen=found[:1], candidates=found)


def _describe(goal: GoalSpec) -> str:
    """The goal's filter, in words, for an impossible entry."""
    return ((goal.artifact_class or "device")
            + (f" titled {goal.artifact_name!r}" if goal.artifact_name else "")
            + (f" in a {goal.location_class}" if goal.location_class else "")
            + (f" with an action {goal.affordance_name!r}" if goal.affordance_name else "")
            + (f" performing {goal.affordance_class}" if goal.affordance_class else ""))


def as_explicit(goal: GoalSpec, candidate: Candidate) -> GoalSpec:
    """The goal, made explicit for one chosen device."""
    return replace(goal, goal_specificity="explicit", quantifier=None,
                   artifact_name=candidate.title,
                   artifact_class=goal.artifact_class or candidate.artifact_class,
                   location_class=goal.location_class or candidate.location_class,
                   affordance_name=goal.affordance_name or candidate.affordance_name)


# --------------------------------------------------------------------------
# Plan entries
# --------------------------------------------------------------------------

def impossible_entry(reason: str) -> Dict[str, Any]:
    return {"tree": None, "impossible": True, "explanation": reason, "source": SOURCE}


def clarification_entry(choice: Choice) -> Dict[str, Any]:
    """A tree-less entry the UA turns into a question for the user."""
    return {"tree": None, "requires_clarification": True, "explanation": choice.reason,
            "candidates": [c.title for c in choice.candidates], "source": SOURCE}


def combine(results: List[Dict[str, Any]], chosen: List[Candidate]) -> Dict[str, Any]:
    """One plan entry from the per-device entries.

    One device: its entry. Several: their trees in a sequence, each wrapped in
    `ignore_failure` so that a device failing while the plan runs does not stop
    the ones after it; the devices that could not be planned are named in the
    explanation. None planned: impossible (or the first error), with every
    device's reason.
    """
    if len(results) == 1:
        return {**results[0], "source": SOURCE}
    planned = [(c, r) for c, r in zip(chosen, results) if r.get("tree")]
    failed = [(c, r) for c, r in zip(chosen, results) if not r.get("tree")]
    reasons = "; ".join(f"{c.title}: {r.get('explanation', '')}" for c, r in failed)
    if not planned:
        errors = [r for _, r in failed if r.get("error")]
        entry = (dict(errors[0]) if errors
                 else {"tree": None, "impossible": True})
        entry.update(explanation=reasons, source=SOURCE)
        return entry
    explanation = "; ".join(f"{c.title}: {r.get('explanation', '')}" for c, r in planned)
    if failed:
        explanation += f". Not planned -- {reasons}"
    return {"tree": {"type": "sequence", "name": "every matching device",
                     "children": [{"type": "ignore_failure", "name": c.title,
                                   "children": [r["tree"]]} for c, r in planned]},
            "explanation": explanation, "source": SOURCE}

