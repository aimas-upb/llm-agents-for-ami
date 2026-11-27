#!/usr/bin/env python3
"""Re-score saved BEIA plans against the cases' ground truth -- no agents, no LLM.

`run_cases.py` writes one JSON line per case (the plan the agents proposed, and
the row it scored). After a ground truth or the scorer changes, this re-scores
those plans without asking the agents again:

    conda run -n ami-agents python tests/BEIA/score_plans.py tests/BEIA/results/beia-<stamp>.jsonl

It prints one line per case (`static=pass|fail` and what is missing) and, with
`--write`, writes the rows back with the new `static` column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from tests.BEIA.ground_truth import score  # noqa: E402


def load_cases(case_dir: Path) -> dict:
    """Case id -> case, from the case files."""
    cases = {}
    for path in sorted(case_dir.glob("*.json")):
        case = json.loads(path.read_text())
        cases[case.get("id") or path.stem] = case
    return cases


def main(argv=None) -> int:
    """Re-score every row of a results file; exit 1 when any fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="a run_cases.py JSONL results file")
    parser.add_argument("--case-dir", default=str(HERE / "cases"))
    parser.add_argument("--write", action="store_true", help="write the new rows back")
    args = parser.parse_args(argv)

    # 1. The cases and the saved rows.
    cases = load_cases(Path(args.case_dir))
    path = Path(args.results)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # 2. Each saved plan, scored again.
    failed = 0
    for row in rows:
        case = cases.get(row.get("case"))
        if case is None:
            print(f"?    {row.get('case')}: no such case")
            continue
        row["static"] = score(row.get("plan"), case["ground_truth"])
        ok = row["static"]["static_pass"]
        failed += not ok
        print(f"{'pass' if ok else 'FAIL'} {row['case']:<4} {case['query']}")
        for item in row["static"]["missing"]:
            print(f"       - {item}")

    # 3. Optionally, back to the file.
    if args.write:
        path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
