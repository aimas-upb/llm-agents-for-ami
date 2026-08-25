#!/usr/bin/env python3
"""
Verify that every actuatable attribute the CORPUS can produce is dispatchable.

This is a static check -- it reads the benchmark episodes and the dispatch
tables, and needs no running server. That matters, because the gap it exists to
catch is invisible to a live test: an episode only exercises the devices it
happens to contain, so a missing mapping stays hidden until some other seed
loads. `TemperatureControl.TemperatureSetpoint` was unmapped for exactly this
reason -- 2,156 instances across freezers, refrigerators and laundry washers,
advertised in every one of their Thing Descriptions and returning HTTP 400 on
invocation.

Three things are checked:

  1. Every (cluster, attribute) the classifier calls actuatable, corpus-wide,
     resolves to a dispatch path -- an attribute write, or a command.
  2. Every command SHTD may send names its arguments the way SimuHome's own
     cluster handler does. The two disagree with the Matter spec, and with each
     other: three Mode Base derivatives take `new_mode`, two take `mode`.
  3. No dispatch entry is dead -- a mapping for something the corpus never
     produces is either a typo or a leftover.

    python ami_agents/environment/integration/SimuHome/verify_dispatch.py
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from classify import classify  # noqa: E402
from shtd import (  # noqa: E402
    _COMMAND_ARGS, _COMMAND_ARGS_BY_CLUSTER, _COMMAND_FOR,
)

DEFAULT_BENCHMARK = HERE.parents[3].parent / "SimuHome" / "data" / "benchmark"
DEFAULT_SIMUHOME = HERE.parents[3].parent / "SimuHome"

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def corpus_actuatable(benchmark: Path) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Every actuatable (cluster, attribute) in the benchmark, with counts."""
    found: Dict[Tuple[str, str], Dict[str, int]] = {}
    for path in sorted(glob.glob(str(benchmark / "*.json"))):
        config = json.loads(Path(path).read_text(encoding="utf-8"))["initial_home_config"]
        for room in (config.get("rooms") or {}).values():
            for device in room.get("devices") or []:
                for attribute_path in (device.get("attributes") or {}):
                    parts = str(attribute_path).split(".", 2)
                    if len(parts) != 3:
                        continue
                    _endpoint, cluster, attribute = parts
                    record = classify(cluster, attribute)
                    if (record["role"] != "affordance"
                            or record["affordance"] != "actuatable"):
                        continue
                    entry = found.setdefault(
                        (cluster, attribute),
                        {"instances": 0, "mechanism": record["mechanism"]})
                    entry["instances"] += 1
    return found


def simulator_signatures(simuhome: Path) -> Dict[str, Set[str]]:
    """Argument names SimuHome's cluster handlers accept, read from source.

    The simulator is the authority: its handlers disagree with the Matter spec
    on casing, and with each other on `mode` vs `new_mode`.
    """
    signatures: Dict[str, Set[str]] = {}
    # `self` may sit on its own line and parameters may span several, so the
    # parameter list is matched laxly -- but it must NOT contain a `def` or a
    # `)`, or a handler with no return annotation lets the match run on and
    # swallow the handlers after it. (`_process_lift_motion(self):` did exactly
    # that, hiding `_go_to_lift_percentage` and `_set_temperature`.)
    pattern = re.compile(
        r"def (_[a-z_0-9]+)\(\s*self\s*,?\s*([^)]*?)\)\s*(?:->|:)", re.S)
    for source in sorted(glob.glob(str(simuhome / "src/simulator/domain/clusters/*.py"))):
        text = Path(source).read_text(encoding="utf-8")
        for handler, params in pattern.findall(text):
            names = set()
            for param in params.split(","):
                param = param.strip()
                if not param or param.startswith("*"):
                    continue
                names.add(param.split(":")[0].split("=")[0].strip())
            if names:
                # Keyed by FILE and handler, not handler alone: `_change_to_mode`
                # takes `new_mode` in three cluster files and `mode` in two, and
                # unioning them would hide exactly the mismatch this checks for.
                module = Path(source).stem
                signatures.setdefault(f"{module}:{handler}", set()).update(names)
                signatures.setdefault(handler, set()).update(names)
    return signatures


# Which source file implements each cluster, so a per-cluster signature can be
# preferred over the union across files.
_CLUSTER_MODULE = {
    "OnOff": "on_off",
    "LevelControl": "level_control",
    "WindowCovering": "window_covering",
    "Thermostat": "thermostat",
    "TemperatureControl": "temperature_control",
    "FanControl": "fan_control",
    "DishwasherMode": "dishwasher_mode",
    "LaundryWasherMode": "laundry_washer_mode",
    "RVCRunMode": "rvc_run_mode",
    "RVCCleanMode": "rvc_clean_mode",
    "RTCCMode": "rtcc_mode",
}


def handler_name(command: str) -> str:
    """`GoToLiftPercentage` -> `_go_to_lift_percentage`."""
    return "_" + re.sub(r"(?<!^)(?=[A-Z])", "_", command).lower()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check every actuatable attribute is dispatchable.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--simuhome", type=Path, default=DEFAULT_SIMUHOME)
    args = parser.parse_args()

    if not args.benchmark.is_dir():
        print(f"benchmark not found: {args.benchmark}", file=sys.stderr)
        return 2

    actuatable = corpus_actuatable(args.benchmark)
    mechanisms = Counter(v["mechanism"] for v in actuatable.values())
    print(f"Corpus: {len(actuatable)} distinct actuatable attributes "
          f"({dict(mechanisms)})\n")

    print("1. Every actuatable attribute has a dispatch path")
    unmapped = []
    for (cluster, attribute), entry in sorted(actuatable.items()):
        if entry["mechanism"] == "attribute_write":
            continue  # written directly; no command mapping needed
        if (cluster, attribute) not in _COMMAND_FOR:
            unmapped.append(f"{cluster}.{attribute} ({entry['instances']} instances)")
    check("no command-driven attribute is unmapped", not unmapped,
          "; ".join(unmapped) if unmapped
          else f"{mechanisms.get('command', 0)} command-driven, all mapped")

    print("\n2. Command arguments match SimuHome's own handler signatures")
    if not args.simuhome.is_dir():
        check("SimuHome source available", False, str(args.simuhome))
    else:
        signatures = simulator_signatures(args.simuhome)
        wrong = []
        # Every command any mapping may emit.
        commands: Set[Tuple[str, str]] = set()
        for (cluster, attribute), chooser in _COMMAND_FOR.items():
            for probe in (True, False, 1, 0):
                try:
                    command = chooser(probe)
                except Exception:
                    continue
                if command:
                    commands.add((cluster, command))
        for cluster, command in sorted(commands):
            builder = (_COMMAND_ARGS_BY_CLUSTER.get((cluster, command))
                       or _COMMAND_ARGS.get(command))
            if builder is None:
                continue  # no arguments; nothing to check
            try:
                emitted = set(builder(1).keys())
            except Exception as exc:
                wrong.append(f"{cluster}.{command}: builder raised {exc}")
                continue
            # Prefer the handler in this cluster's own module.
            module = _CLUSTER_MODULE.get(cluster)
            accepted = None
            if module:
                accepted = signatures.get(f"{module}:{handler_name(command)}")
            if accepted is None:
                accepted = signatures.get(handler_name(command))
            if accepted is None:
                wrong.append(f"{cluster}.{command}: no handler "
                             f"{handler_name(command)} found in the simulator")
                continue
            unknown = emitted - accepted
            if unknown:
                wrong.append(f"{cluster}.{command}: sends {sorted(unknown)}, "
                             f"handler takes {sorted(accepted)}")
        check("every emitted argument name is accepted", not wrong,
              "; ".join(wrong[:3]) if wrong
              else f"{len(commands)} command paths checked")

    print("\n3. No dead dispatch entries")
    dead = [f"{cluster}.{attribute}"
            for (cluster, attribute) in _COMMAND_FOR
            if (cluster, attribute) not in actuatable]
    check("every mapping corresponds to a real attribute", not dead,
          "; ".join(dead) if dead else f"{len(_COMMAND_FOR)} mappings, all used")

    failures = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 60}")
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    for _, name, detail in failures:
        print(f"  FAIL {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
