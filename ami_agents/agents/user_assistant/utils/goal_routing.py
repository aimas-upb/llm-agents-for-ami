"""Which structuring prompt a goal gets, and how much context it needs.

Decided from the segmenter's qualifiers alone:

    simple       achievement, no logical_dependency / temporal_dependency
    dependency   anything else -- a dependency label, or maintenance

and the context follows from what the structure will have to name:

    simple, explicit or incomplete   actions only
    simple, ambiguous                actions + environment variables
    dependency                       actions + device properties + environment
                                     variables (predicates read them)

Missing or contradictory qualifiers route to `dependency` with the full
context: it is the superset, so a mislabelled goal loses nothing but tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

DEPENDENCY_LABELS = frozenset({"logical_dependency", "temporal_dependency"})
SPECIFICITY_LABELS = frozenset({"explicit", "incomplete", "ambiguous"})
KIND_LABELS = frozenset({"achievement", "maintenance"})


@dataclass(frozen=True)
class GoalRoute:
    structure: str            # "simple" | "dependency"
    with_properties: bool     # device property affordances, sensors included
    with_environment: bool    # each space's environment variables


def route_goal(qualifiers: Iterable[str]) -> GoalRoute:
    labels = {str(q).strip().lower() for q in (qualifiers or [])}
    specificity = labels & SPECIFICITY_LABELS
    kind = labels & KIND_LABELS

    well_formed = len(specificity) == 1 and len(kind) == 1
    if well_formed and kind == {"achievement"} and not labels & DEPENDENCY_LABELS:
        return GoalRoute(structure="simple", with_properties=False,
                         with_environment=specificity == {"ambiguous"})
    return GoalRoute(structure="dependency", with_properties=True,
                     with_environment=True)
