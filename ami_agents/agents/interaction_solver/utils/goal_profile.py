"""Profiling a dependency structure: the checklist, and the answers resolved.

A dependency structure says "after G1", "1 minute before the light switches
off", "keep it at 22 degrees". To build a tree, those have to be made concrete
in terms the artifacts can report -- which property says G1's cycle has ended,
which one counts down to the switch-off, whether the thermostat regulates by
itself. A small model answers that, from the Thing Descriptions, for a short
list of questions (GOAL_PROFILE_PROMPT in `prompts.py`):

    build_checklist(structure)        the questions, by fixed rules (no model)
    resolve_profile(checklist, out)   one resolved answer per question: the
                                      model's, or the documented fallback

Checklist rules:

    a goal with goal_kind "maintenance"                maintenance_profile
    a goal starting after_goal an achievement goal     dependency_anchor (ref)
    ... on the same device, with no offset             no question: the next
                                                       instruction to a device
                                                       (`sequential_dependency`)
                                                       follows once the previous
                                                       one has executed
    a goal starting after_goal a maintenance goal      no question; the ref must
                                                       end (ends, scope or
                                                       termination), else
                                                       InconsistentStructure
    a goal starting completes_with                     completion_timing (ref)
    a trigger predicate with offset_direction before   time_to_event
    a trigger / termination predicate on a goal with   time_to_event (for the
      a stated start / end clock time                  stated-time check)

Unresolved means the model could not name, from the TDs, the property a
question needs. It does not stop the plan; the question gets a fallback:

    question             fallback (what the compiler builds)
    dependency_anchor    anchor on goal_success: start once the referenced
                         goal's actuation has executed
    maintenance_profile  mode "unobserved": actuate once, no check-and-repeat
                         loop; the goal still runs until its scope,
                         termination or ends stops it
    completion_timing    start the target goal together with the ref goal
    time_to_event        TIME_TO_EVENT_FALLBACK: "at_event" fires the trigger
                         when the event happens (offset dropped); "skip"
                         leaves the goal out

An answer naming a property that is not in that id's context is treated as
unresolved too, and so is one without the shape its question asks for (a
TimeSource missing its `ref`, say). An "inconsistency" reported by the model raises
InconsistentStructure: the structure cannot be carried out as given, and there
is no fallback.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ....shared.models.goal_structure import GoalSpec, GoalStructure, PredicateSpec
from ..prompts import GOAL_PROFILE_PROMPT

# "at_event" or "skip"; see the fallback table above.
TIME_TO_EVENT_FALLBACK = "at_event"

_FALLBACKS = {
    "dependency_anchor": lambda: {"anchor": "goal_success", "basis": "fallback"},
    "maintenance_profile": lambda: {"mode": "unobserved"},
    "completion_timing": lambda: {"start_with_ref": True},
    "time_to_event": lambda: {"fallback": TIME_TO_EVENT_FALLBACK},
}


class InconsistentStructure(ValueError):
    """The structure can never be carried out as given (e.g. a goal waits for a
    maintenance goal that never ends)."""


class ProfileOutputError(ValueError):
    """The model's output does not match the checklist (unknown or duplicate ids)."""


# --------------------------------------------------------------------------
# The checklist
# --------------------------------------------------------------------------

def _maintenance_goal_ends(goal_id: str, goal: GoalSpec,
                           predicates: Dict[str, PredicateSpec]) -> bool:
    """A maintenance goal ends if it has `ends`, or a scope / termination
    predicate applies to it."""
    if goal.timing is not None and goal.timing.ends_kind:
        return True
    return any(p.role in ("scope", "termination") and goal_id in p.applies_to
               for p in predicates.values())


def _same_device_step(goal, ref_goal) -> bool:
    """The next instruction to the same device, with no wait: it follows once
    the previous one has executed (the compiler's default anchor), so there is
    nothing for the model to profile."""
    timing = goal.timing
    return (not (timing and timing.offset_text)
            and bool(goal.artifact_name) and goal.artifact_name == ref_goal.artifact_name)


def build_checklist(structure: GoalStructure) -> List[Dict[str, Any]]:
    """The questions the profiling model answers for this structure."""
    goals, predicates = structure.goals, structure.predicates
    items: List[Dict[str, Any]] = []

    def add(question: str, target: str, ref: Optional[str] = None) -> None:
        item = {"id": f"Q{len(items) + 1}", "question": question, "target": target}
        if ref is not None:
            item["ref"] = ref
        items.append(item)

    for gid, goal in goals.items():
        timing = goal.timing
        starts = timing.starts if timing else None

        if goal.goal_kind == "maintenance":
            add("maintenance_profile", gid)

        if starts == "after_goal":
            ref_id = timing.ref_goal
            ref_goal = goals.get(ref_id)
            if ref_goal is None:
                raise InconsistentStructure(f"{gid} starts after {ref_id}, which is not a goal")
            if ref_goal.goal_kind == "achievement":
                if not _same_device_step(goal, ref_goal):
                    add("dependency_anchor", gid, ref_id)
            elif not _maintenance_goal_ends(ref_id, ref_goal, predicates):
                raise InconsistentStructure(
                    f"{gid} starts after {ref_id}, a maintenance goal with no scope, "
                    f"termination or ends")

        if starts == "completes_with":
            add("completion_timing", gid, timing.ref)

    # A goal whose start or end the user also stated as a clock time is checked
    # against its anchor before planning (`utils/time_consistency.py`), which
    # needs the anchor's remaining time: whatever the offset's direction, its
    # trigger or termination predicate is asked about.
    stated = {gid for gid, goal in goals.items() if goal.timing and (
        goal.timing.stated_start_time_text or goal.timing.stated_end_time_text)}
    for pid, pred in predicates.items():
        if pred.role == "trigger" and pred.offset_direction == "before":
            add("time_to_event", pid)
        elif pred.role in ("trigger", "termination") and stated & set(pred.applies_to or []):
            add("time_to_event", pid)

    return items


# --------------------------------------------------------------------------
# Resolving the model's answers
# --------------------------------------------------------------------------

def _property_refs(answer: Any) -> Iterable[Dict[str, Any]]:
    """Every PropertyRef in an answer, however deeply it sits."""
    if isinstance(answer, dict):
        if "artifact_name" in answer and "property_name" in answer:
            yield answer
        for value in answer.values():
            yield from _property_refs(value)
    elif isinstance(answer, list):
        for value in answer:
            yield from _property_refs(value)


def _unknown_ref(answer: Any, known: Optional[set]) -> Optional[str]:
    """The first PropertyRef not among `known` (artifact, property) pairs."""
    if known is None:
        return None
    for ref in _property_refs(answer):
        pair = (ref.get("artifact_name"), ref.get("property_name"))
        if pair not in known:
            return f"{pair[0]}.{pair[1]} is not in the context"
    return None


def _time_source(value: Any) -> bool:
    """A TimeSource: {"ref": PropertyRef, "reads_as": "duration" | "timestamp"}."""
    ref = value.get("ref") if isinstance(value, dict) else None
    return (isinstance(ref, dict) and bool(ref.get("artifact_name"))
            and bool(ref.get("property_name"))
            and value.get("reads_as") in ("duration", "timestamp"))


def _malformed(question: str, answer: Any) -> Optional[str]:
    """Why an answer does not have the shape its question asks for (the
    prompt's answer formats), or None. A malformed answer is unresolved: its
    question falls back, rather than the compiler reading a missing field."""
    if not isinstance(answer, dict):
        return "the answer is not an object"
    if question == "time_to_event" and not _time_source(answer.get("time_to_event")):
        return "time_to_event is not a TimeSource {ref, reads_as}"
    if question == "completion_timing":
        if not _time_source(answer.get("target_duration")):
            return "target_duration is not a TimeSource {ref, reads_as}"
        sources = [answer.get(k) for k in ("ref_remaining", "ref_duration")]
        if any(v is not None and not _time_source(v) for v in sources):
            return "ref_remaining / ref_duration is not a TimeSource {ref, reads_as}"
        if all(v is None for v in sources):
            return "neither ref_remaining nor ref_duration is given"
    return None


def resolve_profile(checklist: List[Dict[str, Any]], model_output: Dict[str, Any],
                    known_properties: Optional[Dict[str, set]] = None
                    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """({checklist id: resolved item}, log of fallbacks applied).

    Each resolved item is the checklist item plus "answer" (the model's, or the
    fallback), "source" ("model" / "fallback") and, for a fallback, "reason".
    `known_properties` maps each target / ref id to the (artifact, property)
    pairs of its context; an answer naming another one falls back.
    """
    items = {item["id"]: item for item in checklist}
    resolved: Dict[str, Dict[str, Any]] = {}
    log: List[str] = []
    model_output = model_output if isinstance(model_output, dict) else {}

    for entry in model_output.get("unresolved", []) or []:
        if entry.get("question") == "inconsistency":
            raise InconsistentStructure(f"{entry.get('target')}: {entry.get('reason')}")

    reasons: Dict[str, str] = {}
    for entry in model_output.get("answers", []) or []:
        qid = entry.get("id")
        if qid not in items:
            raise ProfileOutputError(f"answer for unknown checklist id {qid!r}")
        if qid in resolved or qid in reasons:
            raise ProfileOutputError(f"checklist id {qid!r} answered twice")
        item = items[qid]
        known = None
        if known_properties is not None:
            known = set().union(*(known_properties.get(i, set())
                                  for i in (item["target"], item.get("ref")) if i))
        problem = (_malformed(item["question"], entry.get("answer"))
                   or _unknown_ref(entry.get("answer"), known))
        if problem:
            reasons[qid] = problem
            continue
        resolved[qid] = {**item, "answer": entry.get("answer"), "source": "model"}

    for entry in model_output.get("unresolved", []) or []:
        qid = entry.get("id")
        if qid not in items:
            raise ProfileOutputError(f"unresolved entry for unknown checklist id {qid!r}")
        if qid in resolved:
            raise ProfileOutputError(f"checklist id {qid!r} both answered and unresolved")
        reasons[qid] = entry.get("reason", "")

    for qid, item in items.items():
        if qid in resolved:
            continue
        reason = reasons.get(qid, "not returned by the model")
        resolved[qid] = {**item, "answer": _FALLBACKS[item["question"]](),
                         "source": "fallback", "reason": reason}
        log.append(f"{qid} {item['question']} {item['target']}: fallback applied ({reason})")

    return resolved, log


def fallback_profile(checklist: List[Dict[str, Any]], reason: str
                     ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Every question on its fallback (the model failed, or its output was unusable)."""
    return resolve_profile(checklist, {"unresolved": [
        {"id": item["id"], "question": item["question"], "target": item["target"],
         "reason": reason} for item in checklist]})


def render_goal_profile_prompt(goal_structure: str, specific_goal_context: str,
                               checklist: str) -> str:
    """GOAL_PROFILE_PROMPT with its three inputs filled in, by plain replacement
    (the prompt quotes JSON, so str.format would need every brace doubled)."""
    return (GOAL_PROFILE_PROMPT
            .replace("{goal_structure}", goal_structure)
            .replace("{specific_goal_context}", specific_goal_context)
            .replace("{checklist}", checklist))
