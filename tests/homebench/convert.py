#!/usr/bin/env python3
"""HomeBench -> SimuHome-shaped benchmark files (the P1 data layer).

Reads ``test_data.jsonl`` (17,366 rows) and ``home_status_method.jsonl`` (100 homes)
from a HomeBench checkout and writes, under ``--out`` (default
``tests/homebench/data``)::

    homes/home<id>.json                    initial_home_config, once per home
    episodes/<type>_<case>_seed_<n>.json   one small file per row
    manifest.json                          RNG seed, counts, source hashes, diagnostics

  python tests/homebench/convert.py --homebench ../HomeBench --out tests/homebench/data

Naming keeps the SimuHome bench's selection machinery working: ``type`` is ``hb1``
for single requests and ``hb<N>`` for a multi request of N parts; ``case`` is
``feasible`` (every gold part is a call), ``infeasible`` (every part is
``error_input``) or ``mixed``; ``seed`` is the 1-based index inside ``(type, case)``
after a deterministic shuffle, so ``seed_1..L`` is a fixed random sample across the
100 homes. ``meta.id`` and ``meta.homebench_type`` keep the official id and label.

Worked example. The row::

    {"id": "home86_multi_329", "type": "multi8_mix", "home_id": 86,
     "input": "Set the humidifier intensity to 90 in the guest bedroom, ...",
     "output": "'''error_input,balcony.light.set_brightness(40),...,'''"}

becomes ``episodes/hb8_mixed_seed_<n>.json`` with ``eval.goals`` in gold order: an
infeasible part ``{room_id: guest_bedroom, device_id: guest_bedroom_humidifier,
feasibility: false, reason: "unexist_device"}``, then a feasible part
``{room_id: balcony, device_id: balcony_light, asserts: [{attribute: "brightness",
value: 40}]}``, and so on. ``load_episode(benchmark_dir, name)`` merges
``homes/home86.json`` back in as ``initial_home_config``.

Gold semantics are those of ``ami_agents/environment/integration/HomeBench/engine.py``
(no preconditions, corrected heating range); ``verify_gold`` replays every gold call
on its home through that engine and reports the violations.

A few golds contradict HomeBench's own home files; ``label_corrections.yaml`` replaces
them by row id, after naming. A corrected episode is renamed for its new case with a
``_fixed`` suffix, keeping its seed (``hb1_infeasible_seed_2631`` becomes
``hb1_feasible_seed_2631_fixed``, which cannot collide with the feasible group's own
seed 2631), and records the original (``meta.label_correction``).
``--apply-label-corrections`` applies the table to an already converted directory,
without the HomeBench checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:  # allow `python tests/homebench/convert.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ami_agents.environment.integration.HomeBench import engine
from ami_agents.environment.integration.HomeBench.engine import (
    CLEANING_AREAS, HOUSEHOLD_ROOM, HomeBenchEngine, make_device_id,
)
from tests.homebench import locate, mapping_tables

LOGGER = logging.getLogger(__name__)

# -- constants ----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DEFAULT_HOMEBENCH = HERE.parents[1].parent / "HomeBench"
DEFAULT_OUT = HERE / "data"
#: Ground-truth corrections by HomeBench row id (see the module docstring).
LABEL_CORRECTIONS = HERE / "label_corrections.yaml"
#: Appended to the name of an episode whose gold was corrected.
FIXED_SUFFIX = "_fixed"
TEST_DATA = "dataset/test_data.jsonl"
HOME_DATA = "dataset/home_status_method.jsonl"

#: Fixed seed of the per-(type, case) shuffle; stored in the manifest.
DEFAULT_RNG_SEED = 20261024
MANIFEST_FORMAT = 1

ERROR_INPUT = "error_input"
CASE_FEASIBLE, CASE_INFEASIBLE, CASE_MIXED = "feasible", "infeasible", "mixed"
VACUUM = "vacuum_robot"

#: The five official HomeBench buckets (``code/eval.py``), by ``homebench_type``.
BUCKETS = ("normal_single", "unexist_single", "normal_multi", "mix_multi",
           "error_multi")

_MULTI_TYPE = re.compile(
    r"^multi(10|[2-9])_(mix|normal|unexist_device|unexist_attribute)$")
_CALL = re.compile(
    r"^(?:(?P<room>[a-z_]+)\.)?(?P<device>[a-z_]+)\.(?P<command>[a-z_]+)"
    r"\((?P<arg>-?\d+|[A-Za-z_]+)?\)$"
)


# -- errors and parsed gold --------------------------------------------------------------


class ConversionError(ValueError):
    """A row or home that violates what the converter (or the plan) expects."""


@dataclass(frozen=True)
class GoldCall:
    """One gold call ``room.device.command(arg)``; ``room`` is ``None`` for the
    vacuum robot, ``arg`` an ``int``, a bare word, or ``None``."""

    room: Optional[str]
    device: str
    command: str
    arg: Any

    @property
    def device_id(self) -> str:
        """The id of the device this call targets (``household_`` for the vacuum)."""
        return make_device_id(self.room, self.device)


# -- the gold grammar ----------------------------------------------------------------------


def parse_gold(output: str, multi: bool) -> List[Optional[GoldCall]]:
    """Parse one gold ``output`` into parts; ``None`` stands for ``error_input``.

    Grammar (zero violations over the 17,366 test rows): ``'''`` body ``'''``; a
    single row is one call or ``error_input`` with no comma; a multi row is the
    parts separated by commas *with a trailing comma*. A call is
    ``room.device.command(arg?)`` or ``vacuum_robot.command(arg?)``; ``arg`` is an
    int or a bare word. Normal singles carry one leading space after ``'''``.
    Raises ``ConversionError`` on anything else, so a grammar change cannot slip by.
    """
    # 1. The triple-quote frame.
    if not (output.startswith("'''") and output.endswith("'''") and len(output) > 6):
        raise ConversionError(f"gold is not framed by ''': {output!r}")
    body = output[3:-3].strip()
    # 2. Comma discipline: multi rows end with a comma, single rows have none.
    if multi:
        if not body.endswith(","):
            raise ConversionError(f"multi gold lacks the trailing comma: {output!r}")
        tokens = body[:-1].split(",")
    else:
        if "," in body:
            raise ConversionError(f"single gold holds a comma: {output!r}")
        tokens = [body]
    # 3. Each token is error_input or a well-formed call.
    return [_parse_token(token, output) for token in tokens]


def _parse_token(token: str, output: str) -> Optional[GoldCall]:
    """Parse one gold token (``error_input`` -> ``None``) or raise."""
    if token == ERROR_INPUT:
        return None
    found = _CALL.match(token)
    if found is None:
        raise ConversionError(f"bad gold call {token!r} in {output!r}")
    room, device = found["room"], found["device"]
    # The vacuum is the only roomless device; any other device needs a room.
    if (room is None) != (device == VACUUM):
        raise ConversionError(f"room/device mismatch in {token!r}")
    arg = found["arg"]
    if arg is not None and re.fullmatch(r"-?\d+", arg):
        arg = int(arg)
    return GoldCall(room, device, found["command"], arg)


# -- type, case and bucket ------------------------------------------------------------------


def multi_parts(homebench_type: str) -> Optional[int]:
    """Return N for ``multiN_*`` types, ``None`` for the three single types."""
    found = _MULTI_TYPE.match(homebench_type)
    if found:
        return int(found[1])
    if homebench_type in {"normal", "unexist_device", "unexist_attribute"}:
        return None
    raise ConversionError(f"unknown homebench type {homebench_type!r}")


def episode_type(homebench_type: str) -> str:
    """Return ``hb1`` for singles and ``hb<N>`` for a multi of N parts."""
    parts = multi_parts(homebench_type)
    return f"hb{parts or 1}"


def official_bucket(homebench_type: str) -> str:
    """Return the official ``code/eval.py`` bucket of a ``homebench_type``."""
    if multi_parts(homebench_type) is None:
        return "normal_single" if homebench_type == "normal" else "unexist_single"
    kind = _MULTI_TYPE.match(homebench_type)[2]
    return {"normal": "normal_multi", "mix": "mix_multi"}.get(kind, "error_multi")


def case_of(parts: Sequence[Optional[GoldCall]]) -> str:
    """``feasible`` if every part is a call, ``infeasible`` if none is, else ``mixed``."""
    errors = sum(part is None for part in parts)
    if errors == 0:
        return CASE_FEASIBLE
    return CASE_INFEASIBLE if errors == len(parts) else CASE_MIXED


def _label_reason(homebench_type: str) -> Optional[str]:
    """The failure kind the row's own label states (``None`` for normal and mix)."""
    for kind in ("unexist_device", "unexist_attribute"):
        if homebench_type == kind or homebench_type.endswith(f"_{kind}"):
            return kind
    return None


# -- homes -----------------------------------------------------------------------------------


def convert_home(record: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one ``home_status_method`` record into an ``initial_home_config``.

    Rooms keep HomeBench's keys (``ding_room`` stays ``ding_room``). Every device
    becomes ``{device_id, device_type, attributes, constraints}``: ``attributes`` is
    flat and includes ``state``; ``constraints`` carries the (corrected) ranges and
    options the engine validates. The attribute key ``" degree"`` (leading space in
    the data) is normalised to ``degree``. The vacuum robot, which HomeBench gives no
    room, goes into the synthetic ``household`` room and gains a ``cleaning_area``
    attribute (initially ``None``) whose options are the 12 area words the gold uses.
    """
    rooms: Dict[str, Any] = {}
    for room, status in record["home_status"].items():
        # 1. The vacuum is a device dict directly under a pseudo-room key.
        if room == "VacuumRobot":
            rooms[HOUSEHOLD_ROOM] = {
                "state": {},
                "devices": [_convert_device(HOUSEHOLD_ROOM, VACUUM, status)],
            }
            continue
        # 2. Ordinary rooms: every key but ``room_name`` is a device.
        devices = [
            _convert_device(room, name, block)
            for name, block in status.items() if name != "room_name"
        ]
        rooms[room] = {"state": {}, "devices": devices}
    return {"tick_interval": engine.TICK_INTERVAL, "base_time": engine.BASE_TIME,
            "rooms": rooms}


def _convert_device(room: str, name: str, block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one HomeBench device block (``state`` + ``attributes``)."""
    # 1. Normalise attribute keys (" degree") and split value from constraint.
    declared = {key.strip(): spec for key, spec in block["attributes"].items()}
    attributes: Dict[str, Any] = {"state": block["state"]}
    attributes.update({key: spec["value"] for key, spec in declared.items()})
    constraints = engine.corrected_constraints(name, declared)
    # 2. The vacuum's ``set_cleaning_area`` has no attribute in the data: add one.
    if name == VACUUM:
        attributes["cleaning_area"] = None
        constraints["cleaning_area"] = {"options": list(CLEANING_AREAS)}
    return {"device_id": make_device_id(room, name), "device_type": name,
            "attributes": attributes, "constraints": constraints}


# -- one row -> goals ---------------------------------------------------------------------------


def _value_type(value: Any) -> str:
    """The bench's ``value_type`` tag for an assert value."""
    return "int" if isinstance(value, int) else "enum"


def _feasible_goal(call: GoldCall, devices: Dict[str, Any]) -> Dict[str, Any]:
    """Goal for a gold call: asserts only the attribute the call sets."""
    device = devices.get(call.device_id)
    if device is None:
        raise ConversionError(f"gold device {call.device_id} not in its home")
    goal: Dict[str, Any] = {
        "room_id": call.room or HOUSEHOLD_ROOM, "device_id": call.device_id,
        "device_type": call.device, "asserts": [],
    }
    # One assert: the attribute the call writes, with the call's value (the vacuum's
    # start / stop / pause / charge write its `state`, as every state command does).
    attribute, value = engine.command_effect(call.command, call.arg)
    arg_text = "" if call.arg is None else str(call.arg)
    goal["asserts"].append({
        "attribute": attribute, "value": value, "value_type": _value_type(value),
        "description": f"{call.command}({arg_text})",
    })
    return goal


def _infeasible_goal(
    target: Optional[locate.Target], label: Optional[str], devices: Dict[str, Any],
    stats: Counter,
) -> Dict[str, Any]:
    """Goal for an ``error_input`` part: ``feasibility: false`` plus a reason.

    The reason is the row label's when it states one (``unexist_*`` rows), else it
    is inferred from the home: the located device is absent -> ``unexist_device``;
    present but lacking the attribute -> ``unexist_attribute``. When both exist and
    disagree the label wins and the disagreement is counted in ``stats``.
    """
    goal: Dict[str, Any] = {
        "room_id": None, "device_id": None, "device_type": None,
        "feasibility": False, "reason": label or "unknown", "resolved": False,
    }
    # 1. Nothing located: keep the label (or "unknown") and count it.
    if target is None or target.device is None or (
            target.room is None and target.device != VACUUM):
        stats["parts_unresolved"] += 1
        return goal
    device_id = make_device_id(target.room, target.device)
    goal.update(room_id=target.room or HOUSEHOLD_ROOM, device_id=device_id,
                device_type=target.device, resolved=True)
    stats["parts_resolved"] += 1
    # 2. What the home says about the located (room, device, attribute).
    device = devices.get(device_id)
    inferred = None
    if device is None:
        inferred = "unexist_device"
    elif target.attribute and target.attribute not in device["attributes"]:
        inferred = "unexist_attribute"
    # 3. Label wins; a disagreement is a diagnostic, not an error.
    if label and inferred and label != inferred:
        stats["reason_label_disagrees"] += 1
    if label is None:
        goal["reason"] = inferred or "unknown"
    if goal["reason"] == "unexist_attribute" and target.attribute:
        goal["attribute"] = target.attribute
    return goal


def _mark_superseded(goals: List[Dict[str, Any]]) -> int:
    """Drop an earlier assert that a later part overwrites with another value.

    The gold is applied in order, so when two parts set the same
    ``(device, attribute)`` differently, only the last value is in the final state.
    The earlier goal keeps no assert for it and records ``superseded_by`` (the goal
    index). Identical repeats are kept. Returns how many asserts were dropped.
    """
    last: Dict[Tuple[str, str], int] = {}
    for index, goal in enumerate(goals):
        for check in goal.get("asserts", []):
            last[(goal["device_id"], check["attribute"])] = index
    dropped = 0
    for index, goal in enumerate(goals):
        if "asserts" not in goal:          # an infeasible part asserts nothing
            continue
        kept = []
        for check in goal["asserts"]:
            later = last[(goal["device_id"], check["attribute"])]
            same = goals[later]["asserts"] and any(
                c["attribute"] == check["attribute"] and c["value"] == check["value"]
                for c in goals[later]["asserts"])
            if later == index or same:
                kept.append(check)
            else:
                goal.setdefault("superseded_by", later)
                dropped += 1
        goal["asserts"] = kept
    return dropped


@dataclass
class _Built:
    """A converted row before its seed number is known."""

    row: Dict[str, Any]
    type: str
    case: str
    episode: Dict[str, Any] = field(default_factory=dict)


def build_episode(
    row: Dict[str, Any], home: Dict[str, Any], stats: Counter
) -> _Built:
    """Convert one test row to its episode (without ``meta.seed``).

    ``home`` is the row's ``initial_home_config``. ``stats`` accumulates the
    diagnostics that end up in the manifest (alignment method, unresolved parts,
    superseded asserts).
    """
    homebench_type = row["type"]
    expected = multi_parts(homebench_type)
    parts = parse_gold(row["output"], multi=expected is not None)
    # 1. The gold must have exactly as many parts as the type says.
    if len(parts) != (expected or 1):
        raise ConversionError(f"{row['id']}: {len(parts)} parts, type says {expected}")
    devices = {d["device_id"]: d for r in home["rooms"].values() for d in r["devices"]}
    # 2. Locate the target of each infeasible part from the request text.
    gold_keys = [None if p is None else (p.room, p.device) for p in parts]
    targets: Optional[List[locate.Target]] = [None] * len(parts)
    if any(key is None for key in gold_keys):
        clauses = locate.split_clauses(row["input"]) if expected else [row["input"]]
        targets, method = locate.align(clauses, gold_keys)
        stats[f"align_{method}"] += 1
        if targets is None:
            targets = [None] * len(parts)
    # 3. One goal per gold part, in gold order.
    label = _label_reason(homebench_type)
    goals = [
        _feasible_goal(part, devices) if part is not None
        else _infeasible_goal(targets[i], label, devices, stats)
        for i, part in enumerate(parts)
    ]
    stats["asserts_superseded"] += _mark_superseded(goals)
    # 4. Rooms the request names: one get_room_devices each, first-seen order.
    rooms = list(dict.fromkeys(
        g["room_id"] for g in goals
        if g["room_id"] and g["room_id"] != HOUSEHOLD_ROOM))
    meta: Dict[str, Any] = {
        "seed": None, "query_type": episode_type(homebench_type),
        "case": case_of(parts), "num_rooms": len(home["rooms"]),
        "num_devices": len(devices),
        "devices_count": len({g["device_id"] for g in goals if g["device_id"]}),
        "id": row["id"], "homebench_type": homebench_type, "gold": row["output"],
    }
    episode = {
        "meta": meta, "query": row["input"], "user_location": None,
        "eval": {"required_actions": [
            {"tool": "get_room_devices", "params": {"room_id": r}} for r in rooms],
            "goals": goals},
        "home_id": row["home_id"],
    }
    return _Built(row, meta["query_type"], meta["case"], episode)


# -- the whole dataset -------------------------------------------------------------------------


@dataclass
class Conversion:
    """Result of converting the dataset: nothing is written until asked."""

    rng_seed: int
    homes: Dict[int, Dict[str, Any]]
    episodes: Dict[str, Dict[str, Any]]
    manifest: Dict[str, Any]

    @property
    def names_by_id(self) -> Dict[str, str]:
        """Official row id -> episode name."""
        return {e["meta"]["id"]: name for name, e in self.episodes.items()}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON-lines file into a list of dicts."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a source file, so a manifest says which data it came from."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_dataset(
    homebench_dir: Any, rng_seed: int = DEFAULT_RNG_SEED
) -> Conversion:
    """Convert the HomeBench test set; deterministic for a given ``rng_seed``.

    Rows are grouped by ``(type, case)``; each group is sorted by official id, then
    shuffled with an RNG seeded from ``rng_seed`` and the group name, so a group's
    order does not depend on file order or on any other group. ``seed_n`` is the
    1-based index in that order.
    """
    root = Path(homebench_dir)
    rows = read_jsonl(root / TEST_DATA)
    home_records = read_jsonl(root / HOME_DATA)
    homes = {rec["home_id"]: convert_home(rec) for rec in home_records}
    stats: Counter = Counter()
    # 1. Convert every row against its home.
    built = [build_episode(row, homes[row["home_id"]], stats) for row in rows]
    # 2. Per-(type, case) deterministic shuffle, then number the episodes.
    groups: Dict[Tuple[str, str], List[_Built]] = {}
    for item in built:
        groups.setdefault((item.type, item.case), []).append(item)
    episodes: Dict[str, Dict[str, Any]] = {}
    for (qtype, case), items in sorted(groups.items()):
        items.sort(key=lambda b: b.row["id"])
        random.Random(f"{rng_seed}:{qtype}:{case}").shuffle(items)
        for number, item in enumerate(items, start=1):
            item.episode["meta"]["seed"] = number
            name = f"{qtype}_{case}_seed_{number}"
            if name in episodes:
                raise ConversionError(f"duplicate episode name {name}")
            episodes[name] = item.episode
    # 3. The golds HomeBench's own homes contradict, corrected and renamed.
    corrected = apply_label_corrections(episodes, homes)
    # 4. Manifest: provenance and counts.
    manifest = _manifest(root, rng_seed, rows, episodes, stats)
    manifest["label_corrections"] = corrected
    return Conversion(rng_seed, homes, episodes, manifest)


# -- ground-truth corrections ------------------------------------------------------------------


def load_label_corrections(path: Any = LABEL_CORRECTIONS) -> Dict[str, Dict[str, Any]]:
    """``{row id: {"gold", "reason", "attribute", "part_reason"}}`` from the corrections
    table (empty without one); for a part corrected to ``error_input``, ``attribute``
    names what it lacks and ``part_reason`` why it cannot be done (default
    ``unexist_attribute``)."""
    table_path = Path(path)
    if not table_path.is_file():
        return {}
    table = yaml.safe_load(table_path.read_text(encoding="utf-8")) or {}
    reason = table.get("reason")
    return {row["id"]: {"gold": row["gold"], "reason": row.get("reason", reason),
                        "attribute": row.get("attribute"),
                        "part_reason": row.get("part_reason")}
            for row in table.get("rows") or []}


def _corrected_type(homebench_type: str, case: str) -> str:
    """HomeBench's label for a corrected answer: a single row is ``normal`` (the only
    corrections are calls); a multi row keeps its part count and takes the suffix of
    its new case (``multi3_normal`` with one part made impossible is ``multi3_mix``)."""
    parts = multi_parts(homebench_type)
    if parts is None:
        return "normal"
    suffix = {"feasible": "normal", "mixed": "mix"}.get(case)
    return f"multi{parts}_{suffix}" if suffix else homebench_type


def correct_episode(episode: Dict[str, Any], home: Dict[str, Any],
                    correction: Dict[str, Any]) -> None:
    """Replace one episode's gold with the corrected one, part by part, in place.

    A part whose call is unchanged keeps its goal; a part corrected to a call gets the
    goal the converter builds for any call (`_feasible_goal`); a part corrected to
    ``error_input`` becomes an infeasible part on the device the old call named, the
    attribute it lacks taken from the correction (``unexist_attribute``). The case and
    the label follow the new gold; the seed and home stay (the name is the caller's,
    `fixed_name`). ``meta.label_correction`` keeps what was there.
    """
    meta = episode["meta"]
    # 1. Both golds, parsed the way the row's type says, part for part.
    multi = multi_parts(meta["homebench_type"]) is not None
    old_calls = parse_gold(meta["gold"], multi)
    calls = parse_gold(correction["gold"], multi)
    if len(calls) != len(old_calls):
        raise ConversionError(f"{meta['id']}: the correction has {len(calls)} parts, "
                              f"the gold {len(old_calls)}")
    devices = {d["device_id"]: d for r in home["rooms"].values() for d in r["devices"]}
    # 2. One goal per part: kept, rebuilt for a new call, or made infeasible.
    goals = []
    for old_goal, old, new in zip(episode["eval"]["goals"], old_calls, calls):
        if new == old:
            goals.append(old_goal)
        elif new is not None:
            goals.append(_feasible_goal(new, devices))
        else:
            goals.append({"room_id": old.room or HOUSEHOLD_ROOM, "device_id": old.device_id,
                          "device_type": old.device, "feasibility": False,
                          "reason": correction.get("part_reason") or "unexist_attribute",
                          "resolved": True, "attribute": correction.get("attribute")})
    _mark_superseded(goals)
    # 3. What was there, then the new gold, case, label and goals.
    meta["label_correction"] = {"original_gold": meta["gold"],
                                "original_type": meta["homebench_type"],
                                "reason": correction.get("reason")}
    case = case_of(calls)
    meta.update(gold=correction["gold"], case=case,
                homebench_type=_corrected_type(meta["homebench_type"], case))
    episode["eval"]["goals"] = goals


def fixed_name(name: str, case: str) -> str:
    """The name of a corrected episode: its new case, its seed, and ``_fixed``
    (``hb1_infeasible_seed_2631`` -> ``hb1_feasible_seed_2631_fixed``)."""
    qtype, _, rest = name.partition("_")
    seed = rest.rsplit("_seed_", 1)[1].removesuffix(FIXED_SUFFIX)
    return f"{qtype}_{case}_seed_{seed}{FIXED_SUFFIX}"


def apply_label_corrections(episodes: Dict[str, Dict[str, Any]],
                            homes: Dict[Any, Dict[str, Any]],
                            corrections: Optional[Dict[str, Dict[str, Any]]] = None
                            ) -> Dict[str, str]:
    """Correct every episode the table names (by ``meta.id``) and rename it in
    ``episodes`` (`fixed_name`); ``{old name: new name}`` of the episodes renamed. An
    episode already corrected to the table's gold is left as it is; one corrected to
    an earlier gold is corrected again, keeping the dataset's original as the
    original."""
    corrections = load_label_corrections() if corrections is None else corrections
    renamed: Dict[str, str] = {}
    for name in sorted(episodes):
        episode = episodes[name]
        correction = corrections.get(episode["meta"].get("id"))
        if correction is None:
            continue
        # 1. The gold, corrected to the table's; a second correction keeps the
        #    dataset's original as the original.
        meta = episode["meta"]
        if meta["gold"] != correction["gold"]:
            earlier = meta.get("label_correction")
            correct_episode(episode, homes[episode["home_id"]], correction)
            if earlier:
                meta["label_correction"].update(original_gold=earlier["original_gold"],
                                                original_type=earlier["original_type"])
        # 2. The name, for its (possibly new) case.
        new = fixed_name(name, meta["case"])
        if new != name:
            renamed[name] = new
    # The new names, once every correction is made.
    for old, new in renamed.items():
        if new in episodes:
            raise ConversionError(f"corrected episode {new} already exists")
        episodes[new] = episodes.pop(old)
    return renamed


def correct_converted(benchmark_dir: Any) -> Dict[str, str]:
    """Apply the corrections table to an already converted directory, in place: the
    episodes it names are written under their new names and the old files removed,
    the rest untouched; ``{old name: new name}``."""
    root = Path(benchmark_dir)
    corrections = load_label_corrections()
    ids = set(corrections)
    # 1. Only the episodes the table names are read and rewritten.
    episodes, homes = {}, {}
    for path in sorted((root / "episodes").glob("*.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        if episode["meta"].get("id") in ids:
            episodes[path.stem] = episode
            homes.setdefault(episode["home_id"], load_home(root, episode["home_id"]))
    # 2. Every episode the table names is written (a re-correction keeps its name);
    #    a renamed one's old file goes.
    renamed = apply_label_corrections(episodes, homes, corrections)
    for name, episode in episodes.items():
        _dump(root / "episodes" / f"{name}.json", episode)
    for old in renamed:
        (root / "episodes" / f"{old}.json").unlink()
    return renamed


def _manifest(
    root: Path, rng_seed: int, rows: List[Dict[str, Any]],
    episodes: Dict[str, Dict[str, Any]], stats: Counter,
) -> Dict[str, Any]:
    """Build the conversion manifest (RNG seed, source hashes, counts, diagnostics)."""
    # 1. Counts the unit tests and reports compare against (plan 2.2).
    types = Counter(r["type"] for r in rows)
    buckets = Counter(official_bucket(r["type"]) for r in rows)
    groups = Counter(f"{e['meta']['query_type']}_{e['meta']['case']}"
                     for e in episodes.values())
    # 2. Provenance (RNG seed, source hashes) plus the converter's own diagnostics.
    return {
        "format": MANIFEST_FORMAT,
        "rng_seed": rng_seed,
        "shuffle": "per (type, case): sort by id, random.Random(f'{seed}:{type}:{case}')",
        "source": {name: {"path": rel, "sha256": _sha256(root / rel)}
                   for name, rel in (("test_data", TEST_DATA), ("homes", HOME_DATA))},
        "rows": len(rows), "episodes": len(episodes),
        "by_homebench_type": dict(sorted(types.items())),
        "by_bucket": {b: buckets[b] for b in BUCKETS},
        "by_type_case": dict(sorted(groups.items())),
        "diagnostics": dict(sorted(stats.items())),
    }


# -- writing and loading ----------------------------------------------------------------------------


def _dump(path: Path, payload: Any) -> None:
    """Write ``payload`` as UTF-8 JSON (indent 1: readable, still a few KB)."""
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_conversion(conversion: Conversion, out_dir: Any) -> None:
    """Write homes, episodes and the manifest under ``out_dir``.

    Stale ``*.json`` files in ``homes/`` and ``episodes/`` are removed first so a
    conversion with another RNG seed cannot leave two sets side by side.
    """
    out = Path(out_dir)
    # 1. Clear only this tool's own outputs: stale *.json in homes/ and episodes/.
    for sub in ("homes", "episodes"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for stale in (out / sub).glob("*.json"):
            stale.unlink()
    # 2. Homes once each, then one small file per episode, then the manifest.
    for home_id, config in conversion.homes.items():
        _dump(out / "homes" / f"home{home_id}.json", config)
    for name, episode in conversion.episodes.items():
        _dump(out / "episodes" / f"{name}.json", episode)
    _dump(out / "manifest.json", conversion.manifest)


def load_home(benchmark_dir: Any, home_id: int) -> Dict[str, Any]:
    """Return ``homes/home<id>.json`` (an ``initial_home_config``)."""
    return engine.load_home_file(Path(benchmark_dir) / "homes" / f"home{home_id}.json")


def load_episode(benchmark_dir: Any, name: str) -> Dict[str, Any]:
    """Load ``episodes/<name>.json`` and merge its home in as ``initial_home_config``.

    The result has the shape of a SimuHome benchmark episode (``meta``, ``query``,
    ``user_location``, ``eval``, ``initial_home_config``) plus ``home_id``. This is
    the one seam where HomeBench differs from SimuHome's plain JSON read.
    """
    path = Path(benchmark_dir) / "episodes" / f"{name}.json"
    episode = json.loads(path.read_text(encoding="utf-8"))
    episode["initial_home_config"] = load_home(benchmark_dir, episode["home_id"])
    return episode


# -- verification --------------------------------------------------------------------------------


def verify_gold(conversion: Conversion, homebench_dir: Any) -> Dict[str, Any]:
    """Replay every gold call on its home through the engine; report violations.

    Returns ``{"rows", "calls", "applied", "errors"}``; ``errors``
    lists ``(row id, call, message)`` for every call the engine refused (an
    unknown device, a missing command, or an enum/range violation).
    """
    rows = read_jsonl(Path(homebench_dir) / TEST_DATA)
    machines: Dict[int, HomeBenchEngine] = {}
    report: Dict[str, Any] = {"rows": len(rows), "calls": 0, "applied": 0,
                              "errors": []}
    for row in rows:
        # 1. One engine per home, reset to the home file before every row.
        machine = machines.setdefault(
            row["home_id"], HomeBenchEngine(conversion.homes[row["home_id"]]))
        machine.reset()
        parts = parse_gold(row["output"], multi=multi_parts(row["type"]) is not None)
        # 2. Apply the calls in gold order; classify each outcome.
        for call in filter(None, parts):
            report["calls"] += 1
            try:
                machine.apply(call.device_id, call.command, call.arg)
            except engine.EngineError as exc:
                report["errors"].append((row["id"], call, str(exc)))
                continue
            report["applied"] += 1
    return report


# -- mapping tables ---------------------------------------------------------------------------


def write_mappings(homebench_dir: Any, out_dir: Any) -> Dict[str, int]:
    """Generate the TD mapping tables from the homes file; return row counts.

    Reads only ``home_status_method.jsonl`` (not the 17k test rows), converts each
    home as ``convert_home`` does, and hands them to ``mapping_tables``, which fails
    on a device type, attribute or room it has no specification for and keeps any
    hand-approved row already on disk.
    """
    # 1. The homes, in the shape the TD layer will see them.
    records = read_jsonl(Path(homebench_dir) / HOME_DATA)
    homes = [convert_home(record) for record in records]
    # 2. The tables, written beside the previous ones (approved rows preserved).
    counts = mapping_tables.write_tables(mapping_tables.generate_tables(homes), out_dir)
    LOGGER.info("wrote mapping tables to %s: %s", out_dir, counts)
    return counts


# -- command line ----------------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    """Convert the dataset, verify the gold on the engine, write the files."""
    parser = argparse.ArgumentParser(description="Convert HomeBench to bench episodes")
    parser.add_argument("--homebench", default=str(DEFAULT_HOMEBENCH),
                        help="HomeBench checkout (holds dataset/)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_RNG_SEED,
                        help="RNG seed of the per-group shuffle (stored in the manifest)")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the gold replay on the engine")
    parser.add_argument("--mappings-out", default=str(mapping_tables.MAPPINGS_DIR),
                        help="directory of the mapping tables (TD layer)")
    parser.add_argument("--mappings-only", action="store_true",
                        help="write only the mapping tables (reads the homes file "
                             "alone, a few seconds), then stop")
    parser.add_argument("--apply-label-corrections", action="store_true",
                        help="apply label_corrections.yaml to the converted --out "
                             "directory in place (no HomeBench checkout needed), then stop")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # 0. Corrections only, on what is already converted.
    if args.apply_label_corrections:
        renamed = correct_converted(args.out)
        LOGGER.info("corrected %d episodes: %s", len(renamed),
                    ", ".join(f"{old} -> {new}" for old, new in renamed.items()) or "none")
        return 0
    # 0. The mapping tables need only the homes; --mappings-only stops after them.
    if args.mappings_only:
        write_mappings(args.homebench, args.mappings_out)
        return 0
    # 1. Convert, 2. verify, 3. write.
    conversion = convert_dataset(args.homebench, args.seed)
    if not args.no_verify:
        report = verify_gold(conversion, args.homebench)
        LOGGER.info("gold replay: %d calls, %d applied, %d errors",
                    report["calls"], report["applied"], len(report["errors"]))
        conversion.manifest["gold_replay"] = {k: v for k, v in report.items()
                                              if k != "errors"} | {
            "errors": len(report["errors"])}
    write_conversion(conversion, args.out)
    LOGGER.info("wrote %d episodes and %d homes to %s", len(conversion.episodes),
                len(conversion.homes), args.out)
    # 4. The mapping tables, from the homes just converted.
    write_mappings(args.homebench, args.mappings_out)
    LOGGER.info("diagnostics: %s", conversion.manifest["diagnostics"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
