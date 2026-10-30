"""HomeBench's scoring rule for the bench: the gold tokens, met by what the home did.

`tests/simuhome/bench/evaluate.py` scores an episode record; for HomeBench it hands the
whole verdict to `evaluate_episode` here. No judge is called: our agents do not argue why
a request cannot be done, they report it to the user, and HomeBench expects exactly that,
an `error_input` per part that cannot be done.

The gold of an episode is a list of tokens, one per part of the request
(`'''kitchen.light.set_brightness(20),error_input,'''`). Each is met as follows:

    room.device.op(arg)   an accepted action whose gold token is exactly this one
                          (`gold_log`), or -- no action needed -- the part's assert
                          held both before the run and after it ("held")
    error_input           a goal the agents answered impossible or with a question; k
                          gold `error_input`s need at least k such answers (refusals are
                          not tied to parts)

The episode passes when every gold token is met. Anything else the agents did is not
counted against them, nor how many times they did it: we model the general case (a light is switched on before its
brightness is set, a request is split in more goals than HomeBench has parts), while
HomeBench assumes every device is on and one call per part. A device the request names
but the room lacks is not operated by a substitute without a cost: the substitute's
action is extra, but the `error_input` it should have been is then missing.

One extra reading of the benchmark's label (our criterion, not HomeBench's): a gold
`set_brightness(0)` / `set_intensity(0)` for a request that says "turn off" or "switch
off" is also met by that device's `turn_off()`. It counts in `score_extra` only; `score`
keeps the gold's own reading.

Worked example. "Turn on the light in the garage and set its brightness to 70": the gold
is `garage.light.turn_on(),garage.light.set_brightness(70),`, the light is already on, and
the agents send only `brightness 70`. The second token is met by the action
(`matched: "action"`); the first by the light having been on before and after
(`matched: "held"`). The episode passes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tests.homebench import convert, locate
from tests.simuhome.bench import gold_log as gold_log_module
from tests.simuhome.bench import homebench_metrics as metrics

LOGGER = logging.getLogger(__name__)

# -- constants ------------------------------------------------------------------------------

#: The gold values the "turn off" reading applies to: a level of zero.
ZERO_LEVEL_ATTRIBUTES = ("brightness", "intensity")
#: The query says the user asks to turn the device off (plan 3.7).
TURN_OFF_ASKED = re.compile(r"\b(?:turn|switch)\s+off\b", re.IGNORECASE)
#: A relative change in percent ("by 43 percent", "by 26%"): the gold adds those points to
#: the initial value (plan 2.2, point 2). Reported apart in the statistics.
PERCENT_RELATIVE = re.compile(r"\bby\s+\d+(?:\.\d+)?\s*(?:%|percent)", re.IGNORECASE)
#: How an ok check was met: by an accepted action, by the value holding all along, or
#: (in `score_extra` only) by the device turned off.
MATCHED_ACTION, MATCHED_HELD, TURN_OFF_READING = "action", "held", "turn_off"
#: The readings `score` accepts; `score_extra` accepts the turn-off reading too.
STRICT_READINGS = (MATCHED_ACTION, MATCHED_HELD)


# -- parts -----------------------------------------------------------------------------------


def is_infeasible(goal: Dict[str, Any]) -> bool:
    """True for a part that asks for something the home cannot do."""
    return goal.get("feasibility") is False


def is_resolved(goal: Dict[str, Any]) -> bool:
    """True when the converter knew the part's room and device."""
    return bool(goal.get("resolved", True)) and bool(goal.get("room_id")) \
        and bool(goal.get("device_id"))


def gold_parts(episode: Dict[str, Any]
               ) -> List[Tuple[str, Optional[convert.GoldCall], Dict[str, Any]]]:
    """``(token, call, goal)`` per gold part, in gold order: the token as the official
    metric reads it, the parsed call (``None`` for ``error_input``) and the converter's
    goal for that part (one goal per part, in the same order).

    Raises ``ValueError`` when the episode carries no gold (the converter always writes
    it) or when the three do not line up, which only a hand-made episode can cause.
    """
    # 1. The gold string the converter copied from the dataset row.
    meta = episode.get("meta") or {}
    gold = meta.get("gold")
    if not gold:
        raise ValueError("episode has no meta.gold (not a converter output)")
    # 2. Read three ways: the calls (for the held check and the turn-off reading), the
    #    tokens (what an action must equal) and the goals (the asserts).
    calls = convert.parse_gold(gold, convert.multi_parts(meta["homebench_type"]) is not None)
    tokens = metrics.gold_tokens(gold)
    goals = episode["eval"]["goals"]
    # 3. One of each per part, or the episode was not made by the converter.
    if not len(tokens) == len(calls) == len(goals):
        raise ValueError(f"gold {gold!r}: {len(tokens)} tokens, {len(calls)} calls, "
                         f"{len(goals)} goals")
    return list(zip(tokens, calls, goals))


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


def turn_off_token(episode: Dict[str, Any], token: str, call: convert.GoldCall,
                   goal: Dict[str, Any]) -> Optional[str]:
    """The token that also meets ``token`` under the turn-off reading -- the device's
    ``turn_off()`` -- or ``None`` when the reading does not apply: the gold is not a zero
    level, or the request does not ask for this device to be turned off."""
    zero = call.command in {f"set_{a}" for a in ZERO_LEVEL_ATTRIBUTES} and call.arg == 0
    if not zero or not asks_turn_off(episode, goal):
        return None
    return token.rsplit(".", 1)[0] + ".turn_off()"


# -- the rule --------------------------------------------------------------------------------------


def _attributes(home_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{device id: attributes}`` of a home state (the initial config or the final state)."""
    return {d.get("device_id"): d.get("attributes") or {}
            for room in (home_state.get("rooms") or {}).values()
            for d in room.get("devices") or []}


def _held(goal: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    """True when every assert of the part held before the run and still holds after it,
    so no action was needed. A part with no assert left (a later part overwrites it) has
    nothing to hold."""
    asserts = goal.get("asserts") or []
    return bool(asserts) and all(
        before.get(a["attribute"]) == a["value"] == after.get(a["attribute"])
        for a in asserts)


def _check(episode: Dict[str, Any], index: int, token: str, call: convert.GoldCall,
           goal: Dict[str, Any], sent: Set[str], before: Dict[str, Dict[str, Any]],
           after: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The check of one gold call: ``ok`` with how it was met, else ``wrong_value`` (the
    same call was sent with another argument) or ``missing``. ``sent`` is the set of
    action tokens: how many times an action was sent does not matter, so one action
    meets a gold call the request repeats."""
    out = {"goal": index, "device": call.device_id, "expected": token}
    # 1. An accepted action that is exactly the gold call.
    if token in sent:
        return {**out, "class": "ok", "matched": MATCHED_ACTION}
    # 2. No action needed: the value held before and after.
    device = call.device_id
    if _held(goal, before.get(device, {}), after.get(device, {})):
        return {**out, "class": "ok", "matched": MATCHED_HELD}
    # 3. The turn-off reading (score_extra only).
    extra = turn_off_token(episode, token, call, goal)
    if extra is not None and extra in sent:
        return {**out, "class": "ok", "matched": TURN_OFF_READING}
    # 4. Not met: was the same call sent with another argument?
    stem = token.split("(", 1)[0] + "("
    wrong = sorted(t for t in sent if t.startswith(stem))
    if wrong:
        return {**out, "class": "wrong_value", "values": wrong}
    return {**out, "class": "missing"}


def evaluate_episode(record: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict on one HomeBench record (module docstring): ``score`` (the gold's own
    reading), ``score_extra`` (with the turn-off reading), ``error_type``, ``checks`` (one
    per gold call) and, when the gold has an ``error_input``, ``execution_check`` (the
    refusals against the parts that cannot be done)."""
    parts = gold_parts(episode)
    # 1. What the home did: the gold token of every accepted action, and the refusals.
    sent = set(gold_log_module.gold_log(
        (record.get("execution") or {}).get("actions")).tokens)
    refused = metrics.refused_goals(record)
    refusals = refused[metrics.IMPOSSIBLE] + refused[metrics.CLARIFY]
    # 2. Every gold call, against the actions and the states before and after.
    before = _attributes(episode.get("initial_home_config") or {})
    after = _attributes(record.get("final_home_state") or {})
    checks = [_check(episode, index, token, call, goal, sent, before, after)
              for index, (token, call, goal) in enumerate(parts) if call is not None]
    # 3. The parts that cannot be done: enough refusals.
    expected_refusals = sum(1 for _, call, _ in parts if call is None)
    reported = refusals >= expected_refusals
    # 4. The verdict: all of it, by the gold's reading and with the extra one.
    strict = all(c["class"] == "ok" and c["matched"] in STRICT_READINGS for c in checks)
    extra = all(c["class"] == "ok" for c in checks)
    out: Dict[str, Any] = {
        "score": int(strict and reported), "score_extra": int(extra and reported),
        "check": "gold_tokens", "checks": checks,
        "error_type": None if strict and reported else (
            "Goal Failure" if not strict else "Not Reported Infeasible")}
    if expected_refusals:
        out["execution_check"] = {"score": int(reported), "check": "refusals",
                                  "error_inputs": expected_refusals, "refusals": refusals}
    return out


# -- what the official metrics and the generic rates need ----------------------------------------


def part_counts(checks: Sequence[Dict[str, Any]], expected_refusals: int,
                refusals: int) -> Dict[str, int]:
    """Parts and the parts passed (generic rates, our definitions; `homebench_metrics.parts_rates`).

    A feasible part is a gold call; it passes when its check is ok under the gold's reading
    (an action or a value that held, not the turn-off reading). An infeasible part is a gold
    ``error_input``; refusals are not tied to parts, so the number passed is the refusals,
    at most as many as there are such parts.
    """
    return {"feasible": len(checks),
            "feasible_ok": sum(1 for c in checks if c["class"] == "ok"
                               and c.get("matched") in STRICT_READINGS),
            "infeasible": expected_refusals,
            "infeasible_ok": min(refusals, expected_refusals)}


def episode_block(record: Dict[str, Any], episode: Dict[str, Any],
                  verdict: Dict[str, Any]) -> Dict[str, Any]:
    """The ``homebench`` block of an evaluation: everything the official metrics, the
    generic rates and the caveats of the tables need, so statistics re-run from the
    .eval.json without the episode or the record.

    ``gold_tokens`` and ``predicted_tokens`` are the token lists (`homebench_metrics`);
    ``em``, ``tp``, ``pred``, ``gold`` are `compare`'s counts over them; ``unmapped`` lists
    accepted actions with no gold form (should be empty); ``unresolved`` the infeasible
    parts whose target the converter could not tell; ``percent_relative`` whether the
    request asks for a change "by N percent" (reported apart, never relaxed).
    """
    meta = episode["meta"]
    # 1. What the agents did, as gold tokens: the accepted actions, then one `error_input`
    #    per goal answered impossible or with a question.
    log = gold_log_module.gold_log((record.get("execution") or {}).get("actions"))
    refused = metrics.refused_goals(record)
    predicted = metrics.prediction(log.tokens, record)
    # 2. What the gold says, read the way the official script reads it.
    gold = metrics.gold_tokens(meta["gold"])
    goals = episode["eval"]["goals"]
    checks = (verdict.get("breakdown") or {}).get("checks") or []
    # 3. The block: both token lists, their counts, the bucket, the per-part counts.
    return {"id": meta.get("id"), "type": meta["homebench_type"],
            "bucket": metrics.bucket_of(meta["homebench_type"]),
            "gold_tokens": gold, "predicted_tokens": predicted,
            **metrics.compare(predicted, gold),
            "refused": refused[metrics.IMPOSSIBLE], "clarified": refused[metrics.CLARIFY],
            "unmapped": log.unmapped,
            "parts": part_counts(checks, gold.count(metrics.ERROR_INPUT),
                                 refused[metrics.IMPOSSIBLE] + refused[metrics.CLARIFY]),
            "unresolved": sum(1 for g in goals if is_infeasible(g) and not is_resolved(g)),
            "percent_relative": bool(PERCENT_RELATIVE.search(episode.get("query") or ""))}
