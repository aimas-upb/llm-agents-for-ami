"""Replay the deterministic structure repairs over recorded harness episodes.

For every record under the given roots, the dependency structures the
structurer produced are rebuilt, `repair_structure` is applied, and the
structure-level gates of the dependency planner are re-run without a model or
a home: the unused-trigger check of the compiler, the checklist builder's
consistency check, and the compiler's "some goal starts on its own" check.
The report says, per configuration, how many recorded plan failures the
repairs would have let through to planning, and what still stops the rest.

    python -m tests.simuhome.bench.replay_repairs --roots results/1 results/2 results/3

This is a dry gate, not a re-run: a structure that passes here still has to
plan, profile and execute. Standard library plus the agent's own models only.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ami_agents.agents.interaction_solver.utils.goal_profile import (
    InconsistentStructure,
    build_checklist,
)
from ami_agents.agents.interaction_solver.utils.structure_repair import repair_structure
from ami_agents.shared.models.goal_structure import GoalStructure

HEAD_STARTS = (None, "now", "at_time", "after_delay", "on_trigger")


def structure_gate(structure: GoalStructure) -> Optional[str]:
    """The first structure-level reason the dependency planner would stop on,
    or None (mirrors `_Compiler._unused_triggers` / `compile` step 2 and
    `build_checklist`)."""
    for pid, pred in structure.predicates.items():
        if pred.role != "trigger":
            continue
        goals = [structure.goals[g] for g in pred.applies_to or [] if g in structure.goals]
        if goals and not any(g.timing and (g.timing.starts == "on_trigger"
                                           or g.timing.completes_with == pid) for g in goals):
            return "unused trigger"
    try:
        build_checklist(structure)
    except InconsistentStructure as exc:
        return f"inconsistent: {exc}"
    heads = [gid for gid, goal in structure.goals.items()
             if ((goal.timing.starts if goal.timing else None) in HEAD_STARTS
                 and not (goal.timing and goal.timing.start_chosen))
             or (goal.timing and goal.timing.start_chosen
                 and goal.timing.completes_with in structure.predicates)]
    if not heads:
        return "no goal starts on its own"
    return None


def records(roots: Iterable[Path]) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    for root in roots:
        for path in sorted(root.glob("*/qt*.json")):
            if path.name.endswith(".eval.json"):
                continue
            try:
                yield path.parent.name, path, json.loads(path.read_text())
            except (OSError, ValueError):
                continue


def dependency_structures(record: Dict[str, Any]) -> List[GoalStructure]:
    out = []
    for item in record.get("structures") or []:
        raw = (item or {}).get("structure") if isinstance(item, dict) else None
        if isinstance(raw, dict) and raw.get("structure") == "dependency":
            out.append(GoalStructure.from_dict(raw, intent_text=raw.get("text_intent") or ""))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roots", nargs="+", type=Path, required=True,
                    help="attempt directories holding <config>/<episode>.json records")
    ap.add_argument("--types", default="qt2,qt4-1,qt4-2,qt4-3",
                    help="episode types to look at (comma list)")
    args = ap.parse_args(argv)
    types = tuple(t.strip() for t in args.types.split(","))

    # per config: counters
    seen = collections.Counter()           # dependency structures
    failed_before = collections.Counter()  # failed the gate as recorded
    pass_after = collections.Counter()     # ... and pass it after the repairs
    repaired = collections.Counter()       # structures changed by a repair
    residual: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_type: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    examples: Dict[str, List[str]] = collections.defaultdict(list)

    for config, path, record in records(args.roots):
        if not path.name.startswith(types):
            continue
        for structure in dependency_structures(record):
            seen[config] += 1
            before = structure_gate(structure)
            notes = repair_structure(structure)
            after = structure_gate(structure)
            if notes:
                repaired[config] += 1
            if before:
                failed_before[config] += 1
                key = f"{record.get('query_type')} {record.get('case')}"
                by_type[config][key + " gate-failed"] += 1
                if after is None:
                    pass_after[config] += 1
                    by_type[config][key + " now-pass"] += 1
                    if len(examples[config]) < 3:
                        examples[config].append(f"{path.name}: {notes[0]}")
                else:
                    residual[config][after.split(":")[0]] += 1

    print(f"{'config':36s} {'dep. structures':>15s} {'gate failed':>11s} {'repaired':>8s} "
          f"{'now pass':>8s}")
    for config in sorted(seen):
        print(f"{config:36s} {seen[config]:15d} {failed_before[config]:11d} "
              f"{repaired[config]:8d} {pass_after[config]:8d}")
        for key in sorted(by_type[config]):
            print(f"    {key:34s} {by_type[config][key]}")
        if residual[config]:
            print("    still stopped by: " + ", ".join(f"{k} ({v})"
                                                       for k, v in residual[config].most_common()))
        for ex in examples[config]:
            print(f"    e.g. {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
