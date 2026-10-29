"""HomeBench's scoring rules for the bench (plan 3.7), apart from SimuHome's evaluators.

`tests/simuhome/bench/evaluate.py` scores an episode record; for HomeBench it uses
SimuHome's evaluators where they fit and this module where they do not. Per case:

    feasible            SimuHome's qt3 evaluator, unchanged (equality on the final state)
    infeasible          our execution check, per part (below), and a judge panel
    mixed               both, on each part's own goals; the episode passes only if all parts do

A request is a list of *parts* in gold order (`eval.goals`): a feasible part asks for a
change (`asserts`), an infeasible part (`feasibility: false`) asks for something the home
cannot do. Its `reason` says which:

    unexist_device      the device is not in that room       judge: SimuHome's own (unchanged)
    unexist_attribute   the device exists, lacks the attr    judge: HomeBench's (judge_prompt.py)
    unknown             the converter could not tell          judge: HomeBench's, reads the query

The execution check ("nothing was done on the part that cannot be done, and it was
answered impossible or asked about") needs a scope per part, because a successful action
that a *feasible* part of the same request asks for is never an offence:

    unexist_attribute   an accepted action on that device (plan 3.7), unless a feasible part
                        asks for that action on it (65 episodes share the device)
    unexist_device      an accepted action in that room (SimuHome's stand-in rule, "a device
                        of the missing kind", would match nothing: ids are `<room>_<type>`
                        and the missing device does not exist, so a substitute such as the
                        blinds for the curtain has another id), same exemption
    unresolved part     any accepted action no feasible part asks for (nothing says where)

One extra reading of the benchmark's label (plan 3.7, our criterion, not HomeBench's): a
gold `set_brightness(0)` / `set_intensity(0)` for a request that says "turn off" or "switch
off" is also met by the device being off. The primary score keeps the gold's reading; the
check records `accepted` and `matched`, so the strict rate stays recoverable.

Worked example. "Turn off the light in the garage" has gold `garage.light.set_brightness(0)`.
The agent sends `onOff false`: the final state has `state: off`, `brightness: 40`. Strict
reading: `brightness == 0` fails. `extra_readings` offers `turn_off` (`state == off`), which
holds, so the check is `ok` with `matched: "turn_off"`; `score` stays 0 and `score_extra`
is 1.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ami_agents.environment.integration.HomeBench.mappings import (
    MappingTableError, load_mappings,
)
from tests.homebench import convert, locate
from tests.homebench.judge_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, render_goals
from tests.simuhome.bench import gold_log as gold_log_module
from tests.simuhome.bench import homebench_metrics as metrics
from tests.simuhome.bench.toollog import parse_url

LOGGER = logging.getLogger(__name__)

# -- constants ------------------------------------------------------------------------------

#: Part reasons whose judge is SimuHome's own (the device is missing).
DEVICE_REASON = "unexist_device"
ATTRIBUTE_REASON = "unexist_attribute"
#: The gold values the "turn off" reading applies to: a level of zero.
ZERO_LEVEL_ATTRIBUTES = ("brightness", "intensity")
#: The query says the user asks to turn the device off (plan 3.7).
TURN_OFF_ASKED = re.compile(r"\b(?:turn|switch)\s+off\b", re.IGNORECASE)
#: A relative change in percent ("by 43 percent", "by 26%"): the gold adds those points to
#: the initial value (plan 2.2, point 2), our planner scales the current value by that percent.
PERCENT_RELATIVE = re.compile(r"\bby\s+\d+(?:\.\d+)?\s*(?:%|percent)", re.IGNORECASE)
#: The extra reading's name and what it means on the home's attributes.
TURN_OFF_READING = "turn_off"
TURN_OFF_STATE = {"attribute": "state", "value": "off"}


# -- parts -----------------------------------------------------------------------------------


def is_infeasible(goal: Dict[str, Any]) -> bool:
    """True for a part that asks for something the home cannot do."""
    return goal.get("feasibility") is False


def view(episode: Dict[str, Any], goals: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of ``episode`` whose ``eval.goals`` are ``goals``: what an evaluator written
    for one kind of goal is given (SimuHome's feasible evaluator reads ``goal["asserts"]``
    of every goal and would fail on an infeasible one)."""
    shown = copy.copy(episode)
    shown["eval"] = {**episode["eval"], "goals": list(goals)}
    return shown


def feasible_view(episode: Dict[str, Any]) -> Dict[str, Any]:
    """``episode`` with only its feasible parts."""
    return view(episode, [g for g in episode["eval"]["goals"] if not is_infeasible(g)])


def judge_group(goal: Dict[str, Any]) -> str:
    """``device`` when SimuHome's judge fits the part (its device is missing), else
    ``attribute`` (HomeBench's prompt: a missing attribute, or a part not identified)."""
    return "device" if goal.get("reason") == DEVICE_REASON else "attribute"


def is_resolved(goal: Dict[str, Any]) -> bool:
    """True when the converter knew the part's room and device."""
    return bool(goal.get("resolved", True)) and bool(goal.get("room_id")) \
        and bool(goal.get("device_id"))


# -- the extra reading ---------------------------------------------------------------------------


def asks_turn_off(episode: Dict[str, Any], goal: Dict[str, Any]) -> bool:
    """True when the request asks, for this part, to turn the device off.

    A single-part request: the regex on the whole query. A multi-part request: only the
    clause that names the part's room and device is read, so "turn off the humidifier"
    in one clause does not turn a "brightness 0" of another clause into a turn-off. When
    no clause can be tied to the part, the answer is no (the strict reading stands).
    """
    query = episode.get("query") or ""
    # 1. One part: the whole request is its clause.
    if len(episode["eval"]["goals"]) == 1:
        return bool(TURN_OFF_ASKED.search(query))
    # 2. Several: the clauses that name this room and device.
    for clause in locate.split_clauses(query):
        target = locate.locate(clause)
        if target.device == goal.get("device_type") and (
                goal.get("room_id") == convert.HOUSEHOLD_ROOM
                or target.room == goal.get("room_id")):
            if TURN_OFF_ASKED.search(clause):
                return True
    return False


def extra_readings(episode: Dict[str, Any], goal: Dict[str, Any], attribute: str,
                   expected: Any) -> Dict[str, Dict[str, Any]]:
    """The extra readings a check accepts besides the gold's: ``{name: {"attribute",
    "value"}}``, empty for any check but a zero level the request asked to turn off."""
    zero = attribute in ZERO_LEVEL_ATTRIBUTES and expected == 0 \
        and not isinstance(expected, bool)
    if zero and asks_turn_off(episode, goal):
        return {TURN_OFF_READING: dict(TURN_OFF_STATE)}
    return {}


def describe_reading(spec: Dict[str, Any]) -> str:
    """An extra reading as ``accepted`` shows it: ``state == off``."""
    return f"{spec['attribute']} == {spec['value']}"


def extra_verdict(verdict: Dict[str, Any], extras: Dict[str, Dict[str, Any]],
                  final_attributes: Dict[str, Any]) -> Dict[str, Any]:
    """``verdict`` (a failed check), upgraded to ``ok`` when the final attributes meet an
    extra reading; the check then names it in ``matched`` and ``reading``."""
    for name, spec in extras.items():
        if final_attributes.get(spec["attribute"]) == spec["value"]:
            return {"class": "ok", "matched": name, "reading": name}
    return verdict


# -- the execution check -------------------------------------------------------------------------


def accounted_actions(episode: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """The ``(device id, action name)`` pairs the feasible parts ask for, read from the
    episode's gold string (it covers a part whose assert was superseded by a later one).
    A call no action issues (an unreachable vacuum call) names nothing. Raises
    ``ValueError`` when the episode carries no gold: the converter always writes it."""
    meta = episode.get("meta") or {}
    gold = meta.get("gold")
    if not gold:
        raise ValueError("episode has no meta.gold (not a converter output)")
    mappings = load_mappings()
    multi = convert.multi_parts(meta["homebench_type"]) is not None
    out: Set[Tuple[str, str]] = set()
    for call in filter(None, convert.parse_gold(gold, multi)):
        try:
            action, _ = mappings.action_for_command(call.device, call.command, call.arg)
        except MappingTableError:
            continue
        out.add((call.device_id, action))
    return out


def _scope(goal: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """How far an infeasible part reaches: ``("device", id)``, ``("room", id)`` or
    ``("any", None)`` (module docstring)."""
    if not is_resolved(goal):
        return "any", None
    if goal.get("reason") == DEVICE_REASON:
        return "room", goal["room_id"]
    return "device", goal["device_id"]


def check_execution(record: Dict[str, Any], episode: Dict[str, Any],
                    answered: bool) -> Dict[str, Any]:
    """Infeasible parts: no accepted action in their scope that no feasible part asks for,
    and ``answered`` (the agents answered impossible, or asked). SimuHome's check shape
    (``score``, ``offending_actions``, ``error_type``), with the parts listed."""
    goals = episode["eval"]["goals"]
    accounted = accounted_actions(episode)
    # 1. The accepted actions nobody asked for, by where they happened.
    executed = [(parse_url(a.get("url")) or {}) for a in
                (record.get("execution") or {}).get("actions", []) if a.get("success")]
    stray = [p for p in executed if (p.get("device"), p.get("name")) not in accounted]
    # 2. Each infeasible part against its own scope.
    parts, offending_index = [], set()
    for index, goal in enumerate(goals):
        if not is_infeasible(goal):
            continue
        kind, target = _scope(goal)
        hits = [i for i, p in enumerate(stray)
                if kind == "any" or p.get(kind) == target]
        offending_index.update(hits)
        parts.append({"goal": index, "reason": goal.get("reason"), "scope": kind,
                      "target": target, "offending": len(hits)})
    offending = [stray[i] for i in sorted(offending_index)]
    # 3. The verdict: nothing offending and the agents said so.
    score = int(not offending and answered)
    return {"score": score, "check": "execution",
            "infeasible_part": {"kind": "homebench", "parts": parts},
            "offending_actions": offending, "answered_impossible_or_clarify": answered,
            "unresolved_parts": sum(1 for p in parts if p["scope"] == "any"),
            "error_type": None if score else ("Executed Infeasible Part" if offending
                                              else "Not Reported Infeasible")}


# -- the judge -------------------------------------------------------------------------------------


class HomeBenchJudge:
    """The judge of the infeasible parts of a HomeBench episode, with SimuHome's interface
    (``evaluate(payload, judges)``).

    Parts whose device is missing go to SimuHome's own qt3 infeasible evaluator (``device``
    module, unchanged); the others to HomeBench's prompt (`judge_prompt.py`) through the same
    `run_judge_panel`. ``common`` is SimuHome's `episode_evaluation.common` module, which
    both are built on; they are passed in so this class never imports SimuHome's top-level
    `src` itself (the suite hands them over after `evaluate.simuhome_evaluators()` has put
    the checkout on the path). An episode with parts of both groups is judged by two panels
    (three calls each); the verdict passes only if every panel does.
    """

    def __init__(self, device_module: Any, common: Any) -> None:
        """``device_module``: SimuHome's qt3 infeasible evaluator; ``common``: its
        `episode_evaluation.common` (the helpers and `run_judge_panel`)."""
        self.device_module = device_module
        self.common = common

    def evaluate(self, payload: Dict[str, Any], judges: Any = None) -> Dict[str, Any]:
        """The verdict: ``score`` (1, 0, or -1 when a panel failed), ``error_type``,
        ``judge`` (every panel's letters) and ``panels`` (each panel's own verdict)."""
        episode = payload["episode"]
        goals = [g for g in episode["eval"]["goals"] if is_infeasible(g)]
        # 1. The parts by judge: SimuHome's for a missing device, ours for the rest.
        groups = {name: [g for g in goals if judge_group(g) == name]
                  for name in ("device", "attribute")}
        panels: Dict[str, Dict[str, Any]] = {}
        if groups["device"]:
            panels["device"] = self.device_module.evaluate(
                {**payload, "episode": view(episode, groups["device"])}, judges)
        if groups["attribute"]:
            panels["attribute"] = self._attribute_panel(payload, groups["attribute"], judges)
        # 2. One verdict from the panels: any failure of the judge itself (-1) first, then
        #    any failed panel, else pass.
        letters = [letter for p in panels.values() for letter in p.get("judge") or []]
        failing = [p for p in panels.values() if p.get("score") != 1]
        errors = [p for p in failing if p.get("score") == -1]
        if not panels:
            return {"score": -1, "error_type": "No Infeasible Part", "judge": [], "panels": {}}
        score = -1 if errors else (0 if failing else 1)
        first = (errors or failing or [{}])[0]
        return {"score": score, "error_type": first.get("error_type") if score != 1 else None,
                "judge": letters, "panels": panels}

    def _attribute_panel(self, payload: Dict[str, Any], goals: List[Dict[str, Any]],
                         judges: Any) -> Dict[str, Any]:
        """HomeBench's panel over ``goals``, with SimuHome's preliminaries (the room was
        looked at, the agent answered) and its `run_judge_panel`."""
        common = self.common
        episode = payload["episode"]
        steps = common.extract_steps(payload)
        final_answer = common.extract_final_answer(steps)
        # 1. The same early exits as SimuHome's qt3 evaluator: no judge call is made.
        try:
            required = common.evaluate_required_actions(payload)
        except ValueError as exc:
            return {"score": -1, "error_type": f"Required Actions Schema Error: {exc}",
                    "required_actions": [], "judge": []}
        if not all(action["invoked"] for action in required):
            return {"score": 0, "error_type": "Required Actions Failure",
                    "required_actions": required, "judge": []}
        if not final_answer:
            return {"score": 0, "error_type": "Final Response Failure",
                    "required_actions": required, "judge": []}
        # 2. The panel reads our prompt, with the goals spelled out (room, attribute).
        user = USER_PROMPT_TEMPLATE.format(
            query=episode["query"], goals=render_goals(goals), react_steps=str(steps),
            final_answer=final_answer)
        result = common.run_judge_panel(judges, SYSTEM_PROMPT, user)
        return {"score": result["score"], "error_type": result["error_type"],
                "required_actions": required, "judge": result["judge"]}


class MixedEvaluator:
    """The evaluator of a ``mixed`` episode: SimuHome's feasible module for its feasible
    parts and the `HomeBenchJudge` for the infeasible ones. ``evaluate.py`` runs each on
    the part it is written for and combines them (all parts must pass)."""

    def __init__(self, feasible: Any, infeasible: HomeBenchJudge) -> None:
        """``feasible``: SimuHome's qt3 feasible module; ``infeasible``: the judge."""
        self.feasible = feasible
        self.infeasible = infeasible


# -- what the official metrics and the generic rates need ----------------------------------------


def part_counts(episode: Dict[str, Any], breakdown: Optional[Dict[str, Any]],
                check: Optional[Dict[str, Any]], refusals: int) -> Dict[str, int]:
    """Parts and the parts passed (generic rates, our definitions; `homebench_metrics.parts_rates`).

    A feasible part counts when it has checks (an unreachable or fully superseded part has
    none), and passes when all of its checks are ok under the benchmark's own reading (a
    check met only through an extra reading is not: the rates stay strict, like `score`).
    An infeasible part is clean when no
    accepted action is in its scope; refusals are not attributed to parts, so the number
    passed is the clean parts, at most as many as there were refusals (impossible or asked).
    """
    # 1. Feasible parts: every check of a goal ok, by the benchmark's reading.
    by_goal: Dict[Any, List[bool]] = {}
    for item in (breakdown or {}).get("checks") or []:
        strict = item.get("class") == "ok" and item.get("matched", "simuhome") == "simuhome"
        by_goal.setdefault(item.get("goal"), []).append(strict)
    # 2. Infeasible parts: the clean ones, capped by the refusals.
    parts = ((check or {}).get("infeasible_part") or {}).get("parts") or []
    clean = sum(1 for p in parts if not p["offending"])
    return {"feasible": len(by_goal), "feasible_ok": sum(1 for v in by_goal.values() if all(v)),
            "infeasible": len(parts), "infeasible_ok": min(clean, refusals)}


def episode_block(record: Dict[str, Any], episode: Dict[str, Any],
                  verdict: Dict[str, Any]) -> Dict[str, Any]:
    """The ``homebench`` block of an evaluation: everything the official metrics, the
    generic rates and the caveats of the tables need, so statistics re-run from the
    .eval.json without the episode or the record.

    ``gold_tokens`` and ``predicted_tokens`` are the token lists (`homebench_metrics`);
    ``em``, ``tp``, ``pred``, ``gold`` are `compare`'s counts over them; ``unmapped`` lists
    accepted actions with no gold form (should be empty); ``unreachable`` counts gold parts
    no action can issue; ``unresolved`` the infeasible parts whose target is unknown;
    ``judged`` is whether a judge panel scored the infeasible parts; ``percent_relative``
    whether the request asks for a change "by N percent", which the gold and our planner
    read differently (reported apart, never relaxed).
    """
    meta = episode["meta"]
    # 1. What the agents did, as gold tokens: the accepted actions, then one `error_input`
    #    per goal answered impossible.
    log = gold_log_module.gold_log((record.get("execution") or {}).get("actions"))
    refused = metrics.refused_goals(record)
    predicted = metrics.prediction(log.tokens, record)
    # 2. What the gold says, read the way the official script reads it.
    gold = metrics.gold_tokens(meta["gold"])
    goals = episode["eval"]["goals"]
    check = verdict.get("execution_check")
    # 3. The block: both token lists, their counts, the bucket, the per-part counts.
    return {"id": meta.get("id"), "type": meta["homebench_type"],
            "bucket": metrics.bucket_of(meta["homebench_type"]),
            "gold_tokens": gold, "predicted_tokens": predicted,
            **metrics.compare(predicted, gold),
            "refused": refused[metrics.IMPOSSIBLE], "clarified": refused[metrics.CLARIFY],
            "unmapped": log.unmapped,
            "parts": part_counts(episode, verdict.get("breakdown"), check,
                                 sum(refused.values())),
            "unreachable": sum(1 for g in goals if g.get("unreachable")),
            "unresolved": sum(1 for g in goals if is_infeasible(g) and not is_resolved(g)),
            "judged": "judge" in verdict,
            "percent_relative": bool(PERCENT_RELATIVE.search(episode.get("query") or ""))}
