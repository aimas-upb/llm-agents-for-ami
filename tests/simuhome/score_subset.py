"""Score a subset re-run against the baseline it was drawn from.

A prompt change that fixes over-segmentation can also break episodes that were
already right. One aggregate number hides that trade, so this reports the two
groups separately: of the episodes that were over-split, how many are now
correct; of the episodes that were correct, how many still are.

    python tests/simuhome/score_subset.py \
        --baseline tests/simuhome/results/atomic_seg_gpt5mini_medium/rows.jsonl \
        --trial    tests/simuhome/results/seg_mixed_trial/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict


def load(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        raise SystemExit(f"rows file not found: {path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["file"]] = row
    return rows


def delta(row: dict) -> int:
    return row["n_intents"] - row["n_goals"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--show", type=int, default=6,
                        help="example episodes to print per group")
    args = parser.parse_args()

    base, trial = load(args.baseline), load(args.trial)
    shared = sorted(set(base) & set(trial))
    if not shared:
        raise SystemExit("no episodes in common between the two runs")

    groups = {"was over-split": [], "was correct": [], "was under-split": []}
    for name in shared:
        d = delta(base[name])
        key = ("was over-split" if d > 0
               else "was correct" if d == 0 else "was under-split")
        groups[key].append(name)

    print(f"episodes compared: {len(shared)}\n")
    net = 0
    for key, names in groups.items():
        if not names:
            continue
        now = Counter()
        changed = []
        for name in names:
            d_new = delta(trial[name])
            now["correct" if d_new == 0 else
                "over-split" if d_new > 0 else "under-split"] += 1
            if (delta(base[name]) == 0) != (d_new == 0):
                changed.append((name, delta(base[name]), d_new))
        fixed = now["correct"] if key != "was correct" else 0
        broken = len(names) - now["correct"] if key == "was correct" else 0
        net += fixed - broken
        print(f"{key} ({len(names)}):")
        for state, n in now.most_common():
            print(f"    now {state:12s} {n:3d}")
        if key == "was over-split":
            print(f"  -> FIXED {now['correct']}/{len(names)}")
        elif key == "was correct":
            print(f"  -> KEPT  {now['correct']}/{len(names)}"
                  f"   BROKEN {len(names) - now['correct']}")
        for name, d_old, d_new in changed[:args.show]:
            print(f"       {name}  {d_old:+d} -> {d_new:+d}")
        print()

    print(f"NET correct episodes: {net:+d}")
    if net <= 0:
        print("  (not an improvement -- do not adopt this wording)")


if __name__ == "__main__":
    main()
