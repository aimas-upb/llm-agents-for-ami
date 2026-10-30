"""Score the bench's episode records, with SimuHome's evaluators where they are
deterministic and our checks elsewhere.

    conda run -n ami-agents python -m tests.simuhome.bench.evaluate \\
        --records tests/simuhome/results/bench/<variant> [--no-judge] [--explain]

A separate process from the run: SimuHome's evaluator and our memory engine
both import as top-level `src`, and scores can be recomputed without re-running.
Each record gets an `evaluation` written next to it (<episode>.eval.json).

    type      feasible                              infeasible
    qt1       tools + the values we read match       tools; nothing read from a stand-in
              `current_value` (ours); a device       for the missing device; room states
              attribute that declares another's
              values (FanModeSequence,
              SupportedModes): a capability answer
              listing exactly those values (ours)
                                                     read correctly (ours) + judge
    qt2       SimuHome qt2.feasible (needs a         nothing executed on the infeasible
              simulator: started here, stopped)      part, answered impossible/clarify
                                                     (ours) + judge
    qt3       SimuHome qt3.feasible                  same as qt2 + judge
    qt4-x     SimuHome qt4.feasible over a replay    same, up to the horizon + judge
              of the lockstep's states

The judge is SimuHome's own evaluator module for the case, unchanged, with a
panel of three instances of the judge model (seeds 42/43/44) reading the UA's
reply and our steps. Infeasible episodes report the execution check, the judge,
and both (`score`: both pass).

Scores: 1 pass, 0 fail, -1 infrastructure (LLM transport / simulator errors,
judge errors), which the report leaves out of the denominator.

The benchmark is the record's own `suite` (SimuHome for a record that names none;
`--suite` overrides): the suite loads the episode, supplies the evaluators and says
which errors are infrastructure. A HomeBench episode, whatever its case, is scored by
`tests/homebench/scoring.py` alone: every gold token met by an accepted action (or by a
value that already held), every `error_input` by a goal answered impossible or with a
question, no judge. Its verdict also carries `score_extra` (the score with the one extra
reading of a label, our criterion) and a `homebench` block (the tokens the official
metrics read); `score` keeps the benchmark's own reading.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIMUHOME = PROJECT_ROOT.parent / "SimuHome"
DEFAULT_BENCHMARK = SIMUHOME / "data" / "benchmark"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

from tests.simuhome.bench.suites import DEFAULT_SUITE, SUITE_NAMES, get_suite  # noqa: E402
from tests.simuhome.bench.suites.simuhome import INFRA_MARKERS  # noqa: E402
from tests.simuhome.bench.toollog import parse_url  # noqa: E402

# The SHTD property each SimuHome room-state key is served as.
ROOM_STATE_NAMES = {"temperature": "air_temperature", "humidity": "relative_humidity",
                    "illuminance": "illuminance", "pm10": "pm10_mass_concentration"}
# A value read later than the episode's start drifts a little (seconds of
# simulated dynamics): the tolerance for "the value we read is the one asked".
VALUE_TOLERANCE_ABS = 0.5
VALUE_TOLERANCE_REL = 0.02
JUDGE_SEEDS = (42, 43, 44)
# The cases with a part that cannot be done, which the judge panel reads (for a suite
# that has a judge: HomeBench has none).
JUDGED_CASES = ("infeasible", "mixed")


# --------------------------------------------------------------------------
# SimuHome's evaluator, importable without its agent stack
# --------------------------------------------------------------------------

def simuhome_evaluators():
    """{(query_type, case): module}, SimuHome's registry, imported with its LLM
    provider types stubbed (the judge is ours, below)."""
    if str(SIMUHOME) not in sys.path:
        sys.path.insert(0, str(SIMUHOME))
    if "src.agents.providers" not in sys.modules:
        package = types.ModuleType("src.agents")
        package.__path__ = []  # type: ignore[attr-defined]
        providers = types.ModuleType("src.agents.providers")
        providers.LLMProvider = object  # type: ignore[attr-defined]
        agent_types = types.ModuleType("src.agents.types")
        agent_types.ChatMessage = lambda role, content: {"role": role, "content": content}
        sys.modules.update({"src.agents": package, "src.agents.providers": providers,
                            "src.agents.types": agent_types})
    from src.pipelines.episode_evaluation.qt1 import feasible as qt1_f, infeasible as qt1_i
    from src.pipelines.episode_evaluation.qt2 import feasible as qt2_f, infeasible as qt2_i
    from src.pipelines.episode_evaluation.qt3 import feasible as qt3_f, infeasible as qt3_i
    from src.pipelines.episode_evaluation.qt4 import (
        feasible as qt4_f,
        infeasible_1,
        infeasible_2,
        infeasible_3,
    )
    return {("qt1", "feasible"): qt1_f, ("qt1", "infeasible"): qt1_i,
            ("qt2", "feasible"): qt2_f, ("qt2", "infeasible"): qt2_i,
            ("qt3", "feasible"): qt3_f, ("qt3", "infeasible"): qt3_i,
            ("qt4-1", "feasible"): qt4_f, ("qt4-2", "feasible"): qt4_f,
            ("qt4-3", "feasible"): qt4_f, ("qt4-1", "infeasible"): infeasible_1,
            ("qt4-2", "infeasible"): infeasible_2, ("qt4-3", "infeasible"): infeasible_3}


def required_actions(payload) -> Tuple[bool, List[Dict[str, Any]]]:
    from src.pipelines.episode_evaluation.common import evaluate_required_actions
    result = evaluate_required_actions(payload)
    return all(a["invoked"] for a in result), result


class Judge:
    """SimuHome's judge interface (`generate(messages) -> str`) over an
    OpenAI-compatible endpoint."""

    def __init__(self, spec: Dict[str, Any], seed: int) -> None:
        import os

        from openai import OpenAI
        self.model = spec.get("model", "gpt-4o-mini")
        self.temperature = float(spec.get("temperature", 0.5))
        self.seed = seed
        base_url = spec.get("base_url")
        self.client = OpenAI(base_url=base_url or None,
                             api_key=os.getenv(spec.get("api_key_env") or "OPENAI_API_KEY")
                             or ("ollama" if base_url else None))

    def generate(self, messages, response_format=None) -> str:
        response = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, seed=self.seed,
            messages=[m if isinstance(m, dict) else {"role": m.role, "content": m.content}
                      for m in messages])
        return response.choices[0].message.content or ""


class ReplayClient:
    """What `qt4.feasible` asks of the simulator, answered from the states the
    lockstep recorded: the state at a tick is the one recorded at it (within a
    couple of ticks) or else the latest before it."""

    def __init__(self, snapshots: List[Dict[str, Any]]) -> None:
        self.snapshots = sorted(snapshots, key=lambda s: s["tick"])

    def state_at(self, tick: int) -> Optional[Dict[str, Any]]:
        near = [s for s in self.snapshots if abs(s["tick"] - tick) <= 2]
        if near:
            return min(near, key=lambda s: abs(s["tick"] - tick))["state"]
        before = [s for s in self.snapshots if s["tick"] <= tick]
        return before[-1]["state"] if before else None

    def fast_forward_to(self, to_tick: int, room_ids=None) -> Dict[str, Any]:
        state = self.state_at(int(to_tick))
        if state is None:
            return {"status": {"code": 500, "message": "no recorded state"}, "data": None,
                    "error": {"type": "ReplayError", "detail": f"nothing at {to_tick}"}}
        return {"status": {"code": 200, "message": "OK"}, "data": state, "error": None}

    def get_home_state(self) -> Dict[str, Any]:
        first = self.snapshots[0]["state"] if self.snapshots else {"current_tick": 0}
        return {"status": {"code": 200}, "data": first, "error": None}


# --------------------------------------------------------------------------
# Our checks
# --------------------------------------------------------------------------

def _reads(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every value our state queries read: {room, device, name, value,
    sensor}; `sensor` when the device is one of SHTD's ambient sensors, whose
    reading is the room's state."""
    sensors = record.get("ambient_sensors") or {}
    out = []
    for query in record.get("queries") or []:
        for affordance in (query.get("response") or {}).get("affordances") or []:
            parts = parse_url(affordance.get("target") or "") or {}
            out.append({"room": parts.get("room") or affordance.get("workspace_name"),
                        "device": parts.get("device") or affordance.get("artifact_name"),
                        "name": parts.get("name") or affordance.get("affordance_name"),
                        "sensor": (parts.get("device") or affordance.get("artifact_name"))
                        in sensors,
                        # The wire format carries `value` once read (no flag).
                        "has_value": affordance.get("value") is not None
                        or bool(affordance.get("has_value")),
                        "value": affordance.get("value")})
    return out


# Matter Fan Control, FanModeSequence (App. Cluster R1.4 §4.4.6.2): the FanMode
# values each sequence allows (Off 0, Low 1, Medium 2, High 3, Auto 5). Kept
# here apart from the TD builder's copy, so the check does not inherit a
# mistake of the code it checks.
FAN_MODE_SEQUENCES = {0: {0, 1, 2, 3}, 1: {0, 1, 3}, 2: {0, 1, 2, 3, 5},
                      3: {0, 1, 3, 5}, 4: {0, 3, 5}, 5: {0, 3}}


def _capabilities(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every capability answer's entry: {device, name, values}, `values` the
    permitted values it listed (an empty set when it listed none)."""
    out = []
    for query in record.get("queries") or []:
        if query.get("kind") != "capabilities":
            continue
        for entry in (query.get("response") or {}).get("entries") or []:
            permitted = entry.get("permitted_values") or {}
            out.append({"device": entry.get("artifact_name"),
                        "name": entry.get("affordance_name"),
                        "values": set(permitted.get("enum") or [])})
    return out


def _declared_values(attribute: str, attributes: Dict[str, Any]) -> Optional[set]:
    """The values a device declares another of its attributes may take, when
    ``attribute`` (e.g. "1.FanControl.FanModeSequence") is such a
    declaration: a fan's FanModeSequence (its FanMode values), a Mode
    cluster's SupportedModes (its CurrentMode values). None for any other
    attribute: a question about it is answered by reading it."""
    parts = attribute.split(".")
    if len(parts) < 2:
        return None
    cluster, name = parts[-2], parts[-1]
    value = attributes.get(attribute)
    if (cluster, name) == ("FanControl", "FanModeSequence"):
        return FAN_MODE_SEQUENCES.get(value)
    if name == "SupportedModes" and isinstance(value, list):
        return {m.get("mode", m.get("Mode")) for m in value if isinstance(m, dict)}
    # Other "Supported..." lists name the values of their setting outright
    # (a dryer's SupportedDrynessLevels [0, 1, 2, 3] for SelectedDrynessLevel).
    if name.startswith("Supported") and isinstance(value, list) \
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return set(value)
    return None


def _close(read: Any, expected: Any) -> bool:
    try:
        read, expected = float(read), float(expected)
    except (TypeError, ValueError):
        return read == expected
    return abs(read - expected) <= max(VALUE_TOLERANCE_ABS, VALUE_TOLERANCE_REL * abs(expected))


def _stated(value: Any, answer: Any) -> bool:
    """Does the answer give this attribute's value in words? Only for a text
    value ("LG Electronics"): a bare number in an answer could be any number."""
    if not isinstance(value, str) or not value.strip() or not isinstance(answer, str):
        return False
    return value.strip().lower() in answer.lower()


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _reported(value: Any, answer: Any) -> bool:
    """Does the answer report this read value -- some number in it within the
    tolerance of it?"""
    if not isinstance(answer, str):
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return any(_close(float(number), value) for number in _NUMBER.findall(answer))


def _room_state_ok(goal: Dict[str, Any], reads: List[Dict[str, Any]], answer: Any) -> bool:
    """A room's state, as SimuHome's qt1 judge scores it ("values match tool
    observations"): the right variable of the right room read, and the value
    read given in the answer. Not the episode's `current_value`: the simulator
    runs in real time, so a room with a purifier on has another PM10 by the
    time any agent reads it -- the episode's value is the state at reset."""
    name = ROOM_STATE_NAMES.get(goal.get("room_state"), goal.get("room_state"))
    return any(r["room"] == goal["room_id"] and (not r["device"] or r["sensor"])
               and r["name"] == name
               and r["has_value"] and _reported(r["value"], answer)
               for r in reads)


def check_qt1(record, episode, payload, feasible: bool) -> Dict[str, Any]:
    """Tools, and the values we read against the episode's goals."""
    # 1. What was read, what capabilities were answered, and the home as the
    #    episode set it up.
    tools_ok, required = required_actions(payload)
    reads = _reads(record)
    capabilities = _capabilities(record)
    devices = [d for data in episode["initial_home_config"]["rooms"].values()
               for d in data.get("devices", [])]
    home = {d["device_id"] for d in devices}
    attributes = {d["device_id"]: d.get("attributes") or {} for d in devices}
    # 2. Each goal against them.
    goals = []
    for goal in episode["eval"]["goals"]:
        read = None
        if goal.get("variant") == "room_state":
            ok = _room_state_ok(goal, reads, record.get("response"))
            name = ROOM_STATE_NAMES.get(goal.get("room_state"), goal.get("room_state"))
            read = any(r["room"] == goal["room_id"] and (not r["device"] or r["sensor"])
                       and r["name"] == name and r["has_value"] for r in reads)
        elif goal.get("device_id") in home:
            device = goal["device_id"]
            declared = _declared_values(goal.get("attribute") or "", attributes[device])
            if declared is not None:
                # "Which fan speeds does it have": answered by the device's own
                # declaration -- exactly those values, not the spec's full enum.
                listed = [c for c in capabilities if c["device"] == device]
                ok = any(c["values"] == declared for c in listed)
                read = bool(listed) or any(r["device"] == device and r["has_value"]
                                           for r in reads)
            elif attributes[device].get(goal.get("attribute")) is None:
                # The device reports nothing for it (a dishwasher's phase while
                # it is not running): reading it and finding no value is right.
                read = any(r["device"] == device for r in reads)
                ok = read
            else:
                # Read from the device, or -- a fact the TD itself states, such
                # as who made it -- given in the answer without a read.
                read = any(r["device"] == device and r["has_value"] for r in reads)
                ok = read or _stated(attributes[device].get(goal.get("attribute")),
                                     record.get("response"))
        else:
            # The device does not exist: nothing read from a device of its kind in
            # its room may stand in for it.
            prefix = f"{goal['room_id']}_{goal.get('device_type', '')}"
            ok = not any(r["device"] and r["device"].startswith(prefix) and r["has_value"]
                         for r in reads)
        goals.append({"goal": goal.get("variant"), "room": goal.get("room_id"),
                      "device": goal.get("device_id"), "ok": ok, "read": read})
    score = int(tools_ok and all(g["ok"] for g in goals))
    return {"score": score, "check": "values", "required_actions": required, "goals": goals,
            "error_type": None if score else ("Required Actions Failure" if not tools_ok
                                              else "Value Mismatch")}


# --------------------------------------------------------------------------
# Partial success: every expected value checked and classified
# --------------------------------------------------------------------------
#
#   ok            the expected value held (in its window for timed goals, at
#                 the end otherwise); `reading: percent_of_range` when it held
#                 as our reading of a level asked in percent (PERCENT_LEVELS)
#   wrong_timing  it was reached, but outside the goal's window (timed goals)
#   wrong_value   the attribute was changed, but not to the expected value
#   missing       the attribute was never changed (`attempted`: an action on
#                 that device was sent and rejected)
#
# One check per (device, attribute, value) of a qt3 / qt4 goal, one per goal
# for qt1 / qt2. The episode's success is the share of `ok` checks.

CLASSES = ("ok", "wrong_value", "wrong_timing", "missing")


# Our criterion, not SimuHome's: a level the user asked for in percent. SimuHome
# writes the raw level into the query as "60 percent" and checks
# CurrentLevel == 60 (24 % of a 0-254 level); the device's intent is 60 % of its
# range, 153. Either is accepted, but only when the query states that number as
# a percent -- "level 60" still means 60.
PERCENT_LEVELS = {"1.LevelControl.CurrentLevel": "1.LevelControl.MaxLevel"}
DEFAULT_MAX_LEVEL = 254


def _readings(episode, device: str, attribute: str, expected: Any) -> Dict[str, Any]:
    """{reading: value} the check accepts: SimuHome's value, plus -- for a
    level the query states in percent -- that percent of the device's range."""
    readings: Dict[str, Any] = {"simuhome": expected}
    max_attribute = PERCENT_LEVELS.get(attribute)
    if max_attribute and isinstance(expected, (int, float)) and not isinstance(expected, bool) \
            and re.search(rf"(?<![\d.]){re.escape(str(expected))}\s*(%|percent)",
                          episode.get("query") or "", re.IGNORECASE):
        top = _initial(episode, device, max_attribute) or DEFAULT_MAX_LEVEL
        readings["percent_of_range"] = round(float(expected) / 100.0 * float(top))
    return readings


def _matches(value: Any, readings: Dict[str, Any]) -> Optional[str]:
    """The reading ``value`` matches (a percent reading within 1 for rounding)."""
    for name, accepted in readings.items():
        if value == accepted:
            return name
        if name == "percent_of_range" and isinstance(value, (int, float)) \
                and not isinstance(value, bool) and abs(value - accepted) <= 1:
            return name
    return None


def _attempted(record, device: Optional[str] = None, room: Optional[str] = None) -> bool:
    for action in (record.get("execution") or {}).get("actions", []):
        parts = parse_url(action.get("url")) or {}
        if (device and parts.get("device") == device) or (room and parts.get("room") == room):
            return True
    return False


def _initial(episode, device: str, attribute: str) -> Any:
    for data in episode["initial_home_config"]["rooms"].values():
        for d in data.get("devices", []):
            if d["device_id"] == device:
                return (d.get("attributes") or {}).get(attribute)
    return None


def _series(record, device: str, attribute: str) -> List[Tuple[int, Any]]:
    """(tick, value) each time the attribute changed, from the lockstep's
    timeline (or, for older records, its state snapshots)."""
    key = f"{device}|{attribute}"
    points = [(e["tick"], e["values"].get(key)) for e in record.get("timeline") or []
              if key in e["values"]]
    if not points:
        for snap in record.get("snapshots") or []:
            for room in (snap["state"].get("rooms") or {}).values():
                for d in room.get("devices") or []:
                    if d.get("device_id") == device:
                        points.append((snap["tick"], (d.get("attributes") or {}).get(attribute)))
    return sorted(points, key=lambda p: p[0])


def _timed_check(record, initial, device, attribute, expected, lo, hi,
                 readings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify one assert of a timed goal from the attribute's history."""
    readings = readings or {"simuhome": expected}
    points = _series(record, device, attribute)
    if not points:
        return {"class": "missing", "attempted": _attempted(record, device)}
    spans = [(t, points[i + 1][0] if i + 1 < len(points) else float("inf"), v)
             for i, (t, v) in enumerate(points)]
    held = [(start, end, _matches(v, readings)) for start, end, v in spans
            if _matches(v, readings)]
    in_window = [h for h in held if h[0] <= hi and h[1] > lo]
    if in_window:
        return _ok(in_window[0][2])
    if held:
        start = held[0][0]
        return {"class": "wrong_timing", "reached_at": start,
                "early_by_ticks" if start < lo else "late_by_ticks":
                    lo - start if start < lo else start - hi}
    seen = sorted({json.dumps(v) for _, _, v in spans} - {json.dumps(initial)})
    if seen:
        return {"class": "wrong_value", "values": [json.loads(v) for v in seen]}
    return {"class": "missing", "attempted": _attempted(record, device)}


def _ok(reading: Optional[str]) -> Dict[str, Any]:
    """`ok`, with the reading that matched in `matched` ("simuhome" -- the benchmark's own
    value -- or the name of an extra reading), and in `reading` when it is ours rather
    than the benchmark's. The strict rate is the checks matched by "simuhome"."""
    return {"class": "ok", "matched": reading or "simuhome",
            **({} if reading in (None, "simuhome") else {"reading": reading})}


def _final_check(record, initial, final, expected, device,
                 readings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify one assert checked on the final state."""
    reading = _matches(final, readings or {"simuhome": expected})
    if reading:
        return _ok(reading)
    if final != initial:
        return {"class": "wrong_value", "values": [final]}
    return {"class": "missing", "attempted": _attempted(record, device)}


def breakdown(record, episode, simuhome_result: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
    """Every expected value of a feasible episode, classified."""
    qt = record["query_type"]
    goals = episode["eval"]["goals"]
    checks: List[Dict[str, Any]] = []

    if qt.startswith("qt4") or qt == "qt3":
        final_devices = {d.get("device_id"): d.get("attributes") or {}
                         for room in ((record.get("final_home_state") or {}).get("rooms")
                                      or {}).values()
                         for d in room.get("devices") or []}
        for index, goal in enumerate(goals):
            when = goal.get("when") or {}
            for target in goal.get("targets") or [goal]:
                device = target.get("device_id")
                for check in target.get("asserts") or []:
                    attribute, expected = check["attribute"], check["value"]
                    initial = _initial(episode, device, attribute)
                    readings = _readings(episode, device, attribute, expected)
                    if qt.startswith("qt4"):
                        at, tol = int(when["at_tick"]), int(when.get("tolerance_ticks", 0))
                        verdict = _timed_check(record, initial, device, attribute,
                                               expected, at - tol, at + tol, readings)
                    else:
                        verdict = _final_check(record, initial,
                                               final_devices.get(device, {}).get(attribute),
                                               expected, device, readings)
                    checks.append({"goal": index, "device": device, "attribute": attribute,
                                   "expected": expected,
                                   **({"accepted": readings} if len(readings) > 1 else {}),
                                   "description": check.get("description"), **verdict})
    elif qt == "qt2":
        directions = (simuhome_result or {}).get("direction_checks") or []
        for index, goal in enumerate(goals):
            found = next((d for d in directions if d.get("room_id") == goal.get("room_id")
                          and d.get("room_state") == goal.get("room_state")), None)
            if found and found.get("ok"):
                verdict = {"class": "ok"}
            elif any(a.get("success") and (parse_url(a["url"]) or {}).get("room")
                     == goal.get("room_id")
                     for a in (record.get("execution") or {}).get("actions", [])):
                verdict = {"class": "wrong_value",
                           "values": [found.get("to")] if found else []}
            else:
                verdict = {"class": "missing",
                           "attempted": _attempted(record, room=goal.get("room_id"))}
            checks.append({"goal": index, "room": goal.get("room_id"),
                           "expected": f"{goal.get('room_state')} {goal.get('direction')}",
                           **verdict})
    elif qt == "qt1":
        for index, goal in enumerate((simuhome_result or {}).get("goals") or []):
            verdict = {"class": "ok"} if goal["ok"] else \
                {"class": "wrong_value" if goal.get("read") else "missing"}
            checks.append({"goal": index, "room": goal.get("room"), "device": goal.get("device"),
                           "expected": goal.get("goal"), **verdict})

    return summarise_checks(checks)


def summarise_checks(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A breakdown from its checks: the checks, how many of each class, and the share
    that are ok (None for an episode with no check). Shared with HomeBench's scorer,
    whose checks are one per gold call."""
    counts = {c: sum(1 for check in checks if check["class"] == c) for c in CLASSES}
    return {"checks": checks, "total": len(checks), **counts,
            "success": round(counts["ok"] / len(checks), 3) if checks else None}


def infeasible_part(episode) -> Dict[str, Any]:
    """What cannot be done: devices to keep away from, or a room's state."""
    meta, goals = episode["meta"], episode["eval"]["goals"]
    home = {d["device_id"] for data in episode["initial_home_config"]["rooms"].values()
            for d in data.get("devices", [])}
    qt = meta["query_type"]
    if qt.startswith("qt4"):
        return {"kind": "timed", "devices": sorted({t["device_id"] for g in goals
                                                    for t in g.get("targets", [])})}
    if meta.get("infeasible_family") == "contextual_saturation":
        # Only the goals marked infeasible; a request may pair one with a
        # feasible goal in another room, which may run.
        saturated = [g for g in goals if g.get("feasibility") is False] or goals
        return {"kind": "saturation", "rooms": sorted({g["room_id"] for g in saturated}),
                "states": [(g["room_id"], g.get("room_state"), g.get("direction"))
                           for g in saturated]}
    missing = [g for g in goals if g.get("device_id") and g["device_id"] not in home]
    return {"kind": "missing", "devices": [g["device_id"] for g in missing],
            "stand_ins": sorted({f"{g['room_id']}_{g.get('device_type', '')}"
                                 for g in missing})}


def answered_impossible(record) -> bool:
    """True when the agents answered some goal "impossible", or asked the user about it
    (a plan entry, a clarification at structuring, or the episode's own outcome)."""
    outcomes = [p.get("outcome") for p in record.get("plans") or []]
    return (any(o in ("impossible", "clarify") for o in outcomes)
            or any(s.get("clarification") for s in record.get("structures") or [])
            or record.get("outcome") in ("impossible", "clarify"))


def check_execution(record, episode) -> Dict[str, Any]:
    """Infeasible: nothing executed on the part that cannot be done, and that
    part answered as impossible or with a question."""
    part = infeasible_part(episode)
    executed = [a for a in (record.get("execution") or {}).get("actions", [])
                if a.get("success")]
    devices = [(parse_url(a["url"]) or {}) for a in executed]
    if part["kind"] == "timed":
        offending = [d for d in devices if d.get("device") in part["devices"]]
    elif part["kind"] == "saturation":
        offending = [d for d in devices if d.get("room") in part["rooms"]]
    else:
        offending = [d for d in devices if d.get("device") and
                     any(d["device"].startswith(p) for p in part["stand_ins"])]
    answered = answered_impossible(record)
    score = int(not offending and answered)
    return {"score": score, "check": "execution", "infeasible_part": part,
            "offending_actions": offending, "answered_impossible_or_clarify": answered,
            "error_type": None if score else ("Executed Infeasible Part" if offending
                                              else "Not Reported Infeasible")}


def infra_error(record, markers: Tuple[str, ...] = INFRA_MARKERS) -> Optional[str]:
    """An infrastructure failure, not the model's: a step whose every LLM call
    failed, or an exception from the simulator / the transport -- an error text
    holding one of ``markers`` (the suite's; SimuHome's by default)."""
    error = record.get("error") or ""
    if any(k in error for k in markers):
        return error
    calls = (record.get("meter") or {}).get("calls") or []
    by_step: Dict[str, List[bool]] = {}
    for call in calls:
        by_step.setdefault(call["step"], []).append(bool(call["ok"]))
    failed = [step for step, oks in by_step.items() if oks and not any(oks)]
    return f"LLM calls failed in {failed}" if failed else None


# --------------------------------------------------------------------------
# One record
# --------------------------------------------------------------------------

def evaluate_record(record, episode, evaluators, judges, sim_client,
                    markers: Tuple[str, ...] = INFRA_MARKERS) -> Dict[str, Any]:
    """The verdict on one record: infrastructure failures (``markers``) and
    stopped runs first, then the evaluator of its (type, case)."""
    qt, case = record["query_type"], record["case"]
    infra = infra_error(record, markers)
    if infra:
        return {"score": -1, "error_type": "Infrastructure", "detail": infra}
    if record.get("outcome") == "stopped":
        return {"score": -1, "error_type": "Not Run", "detail": "stopped early"}

    # 1. HomeBench: its own rule for every case (no judge), its checks as the breakdown,
    #    and the tokens and counts the official metrics read.
    if qt.startswith("hb"):
        from tests.homebench import scoring
        evaluators[(qt, case)]                  # an unknown type or case is a KeyError
        result = scoring.evaluate_episode(record, episode)
        result["breakdown"] = summarise_checks(result.pop("checks"))
        describe = getattr(evaluators, "describe", None)
        if describe is not None:
            result["homebench"] = describe(record, episode, result)
        return result

    payload = {"episode": episode, "tools_invoked": record.get("tools_invoked") or [],
               "steps": record.get("steps") or [], "client": sim_client,
               "final_home_state": record.get("final_home_state") or {}}
    module = evaluators[(qt, case)]

    # 2. Feasible: SimuHome's evaluator, then every expected value classified.
    if case == "feasible":
        result = _feasible(record, episode, payload, module, qt, sim_client)
        if result.get("score", -1) >= 0:
            result["breakdown"] = breakdown(record, episode, result)
    # 3. Infeasible: our check, and SimuHome's judge on the reply.
    else:
        result = _infeasible(record, episode, payload, module, qt, judges)
    return result


def _feasible(record, episode, payload, module, qt, sim_client) -> Dict[str, Any]:
    """SimuHome's evaluator (our value check for qt1)."""
    if qt == "qt1":
        return check_qt1(record, episode, payload, True)
    if qt.startswith("qt4"):
        snapshots = record.get("snapshots") or []
        if not snapshots:
            return {"score": 0, "error_type": "No Execution Trace"}
        replay = ReplayClient(snapshots)
        payload["client"] = replay
        payload["final_home_state"] = replay.snapshots[0]["state"]
    if qt == "qt2" and sim_client is None:
        return {"score": -1, "error_type": "No Simulator",
                "detail": "qt2 feasible needs a simulator for its baseline"}
    if not payload["final_home_state"]:
        return {"score": 0, "error_type": "No Final State"}
    result = module.evaluate(payload, [])
    result.setdefault("error_type", None if result.get("score") == 1 else "Goal Failure")
    result["check"] = "simuhome"
    return result


def _infeasible(record, episode, payload, module, qt, judges) -> Dict[str, Any]:
    """Our execution check, and SimuHome's judge on the reply."""
    if qt == "qt1":
        check = check_qt1(record, episode, payload, False)
    else:
        check = check_execution(record, episode)
    out = {"execution_check": check}
    if judges:
        judged = module.evaluate(payload, judges)
        out["judge"] = judged
        if judged.get("score") == -1:
            out.update(score=-1, error_type=judged.get("error_type") or "Judge Error")
            return out
        out["score"] = int(check["score"] == 1 and judged.get("score") == 1)
    else:
        out["score"] = check["score"]
    out["error_type"] = None if out["score"] else (check.get("error_type") or
                                                   (out.get("judge") or {}).get("error_type"))
    return out


def _readable(value: Any) -> Any:
    """The verdict as printed: SimuHome's per-tick lists (`tested_ticks`, one
    entry per tick of a goal's window) shortened to first..last (count). The
    saved .eval.json keeps them whole."""
    if isinstance(value, dict):
        return {k: (f"{v[0]}..{v[-1]} ({len(v)} ticks)" if k == "tested_ticks" and
                    isinstance(v, list) and len(v) > 3 else _readable(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_readable(v) for v in value]
    return value


def _scored_since(path: Path) -> bool:
    """True when ``path`` has an .eval.json written after it: its score is
    current, so `--skip-scored` leaves it (a re-run record is scored again)."""
    scored = path.with_name(path.stem + ".eval.json")
    return scored.exists() and scored.stat().st_mtime >= path.stat().st_mtime


def main() -> int:
    """The CLI: score each record, write its .eval.json, print the rates."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True,
                        help="a variant's record directory, or one record file")
    parser.add_argument("--suite", choices=SUITE_NAMES,
                        help="the benchmark (default: each record's own, else simuhome)")
    parser.add_argument("--benchmark", help="the benchmark directory (default: the suite's)")
    parser.add_argument("--no-judge", action="store_true",
                        help="infeasible: the execution check only (no LLM calls)")
    parser.add_argument("--judge-model", help="override the variant's judge model")
    parser.add_argument("--explain", action="store_true", help="print each verdict in full")
    parser.add_argument("--episodes", help="only the records matching this pattern, "
                                           "e.g. 'qt1_*' (default: all)")
    parser.add_argument("--skip-scored", action="store_true",
                        help="leave records whose .eval.json is newer than the record "
                             "(a resumed run: no judge calls paid twice)")
    args = parser.parse_args()

    # 1. The records to score: one file, or a variant's directory -- without
    #    those already scored since they were written, when asked.
    target = Path(args.records)
    files = [target] if target.is_file() else sorted(
        p for p in target.glob("*.json") if not p.name.endswith(".eval.json")
        and p.name not in ("run.json", "summary.json"))
    if args.episodes:
        files = [p for p in files if p.match(args.episodes + ".json")
                 or p.match(args.episodes)]
    if args.skip_scored:
        files = [p for p in files if not _scored_since(p)]
        if not files:
            print("Every record is scored already.")
            return 0
    if not files:
        print("No records.")
        return 1

    # 2. Shared across records: each suite's evaluators (built once), a
    #    simulator started only if a qt2 feasible record needs one, one judge
    #    panel per model.
    evaluators_by_suite: Dict[str, Any] = {}
    worker = None
    judges_by_spec: Dict[str, List[Judge]] = {}
    totals: Dict[Tuple[str, str], List[int]] = {}
    try:
        # 3. Each record: its episode, the tools it needs, the verdict, written
        #    next to it and printed in one line.
        for path in files:
            record = json.loads(path.read_text(encoding="utf-8"))
            suite = get_suite(args.suite or record.get("suite") or DEFAULT_SUITE)
            if suite.name not in evaluators_by_suite:
                evaluators_by_suite[suite.name] = suite.evaluators()
            evaluators = evaluators_by_suite[suite.name]
            benchmark = Path(args.benchmark) if args.benchmark else suite.default_benchmark
            episode = suite.load_episode(benchmark, record["episode"])
            sim_client = None
            if record["query_type"] == "qt2" and record["case"] == "feasible":
                if worker is None:
                    from ami_agents.environment.integration.SimuHome.worker import (
                        SimuHomeWorker,
                    )
                    worker = SimuHomeWorker(benchmark=benchmark).start()
                sim_client = worker.client
            judges: List[Judge] = []
            if record["case"] in JUDGED_CASES and suite.judge and not args.no_judge:
                spec = dict(((record.get("variant") or {}).get("steps") or {}).get("judge")
                            or {"model": "gpt-4o-mini", "temperature": 0.5})
                if args.judge_model:
                    spec["model"] = args.judge_model
                key = json.dumps(spec, sort_keys=True)
                judges = judges_by_spec.setdefault(key, [Judge(spec, s) for s in JUDGE_SEEDS])
            try:
                verdict = evaluate_record(record, episode, evaluators, judges, sim_client,
                                          suite.infra_markers)
            except Exception as exc:  # noqa: BLE001 - recorded as an evaluation error
                verdict = {"score": -1, "error_type": "Evaluation Runtime Error",
                           "detail": f"{type(exc).__name__}: {exc}"}
            verdict.update(episode=record["episode"], query_type=record["query_type"],
                           case=record["case"], outcome=record.get("outcome"))
            path.with_name(path.stem + ".eval.json").write_text(
                json.dumps(verdict, default=str, indent=1), encoding="utf-8")
            totals.setdefault((record["query_type"], record["case"]), []).append(
                verdict["score"])
            line = f"{record['episode']}: score={verdict['score']}" + (
                f" ({verdict.get('error_type')})" if verdict.get("error_type") else "")
            parts = verdict.get("breakdown")
            if parts and parts["total"]:
                line += (f"  checks {parts['ok']}/{parts['total']} ok ({parts['success']:.0%})"
                         + "".join(f", {parts[c]} {c.replace('_', ' ')}"
                                   for c in CLASSES[1:] if parts[c]))
            print(line)
            if args.explain:
                print(json.dumps(_readable({k: v for k, v in verdict.items()
                                            if k not in ("episode", "query_type", "case")}),
                                 indent=1, default=str))
    finally:
        if worker is not None:
            worker.stop()

    # 4. The rates per (type, case); infrastructure failures left out.
    print()
    for (qt, case), scores in sorted(totals.items()):
        counted = [s for s in scores if s >= 0]
        rate = f"{sum(counted) / len(counted):.0%}" if counted else "-"
        print(f"{qt:6s} {case:10s} {sum(counted)}/{len(counted)} = {rate}"
              + (f"  ({len(scores) - len(counted)} infra)" if len(counted) < len(scores)
                 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
