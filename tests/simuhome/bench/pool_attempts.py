"""Pool several attempts of one harness config into one statistics table.

The grid protocol repeats the implicit and timed request types (qt2, qt4-x)
three times and the others once. Each attempt is a separate harness run with
its own records directory (slurm_harness.sbatch: ATTEMPT=1, 2, 3 under one
OUT_ROOT). `stats.py` reads every record in one directory and takes the
episode's type and case from the record itself, not from the file name, so
pooling is: link every attempt's records (and their .eval.json) into one
directory under distinct names, then run the statistics there.

    python -m tests.simuhome.bench.pool_attempts \\
        --root ~/experiments/semantic/harness --name fep-M1-qwen3-8b_coder-7b \\
        [--attempts 1,2,3] [--out ~/experiments/semantic/harness/pooled]

An episode present in several attempts counts once per attempt; the rates are
then over attempts x episodes, which is what "three attempts" means. Records
not yet scored are pooled as they are (stats.py counts them as unscored).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.simuhome.bench.stats import compute  # noqa: E402


def pool(root: Path, name: str, attempts: list[str], out: Path) -> Path:
    """Link the attempts' records into out/<name>/ and return that directory."""
    target = out / name
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.json"):
        if old.is_symlink():
            old.unlink()
    linked = 0
    for attempt in attempts:
        source = root / attempt / name
        if not source.is_dir():
            print(f"attempt {attempt}: no records at {source}", file=sys.stderr)
            continue
        for record in sorted(source.glob("*.json")):
            if record.name.endswith(".eval.json") or record.name == "run.json":
                continue
            stem = record.stem
            (target / f"{stem}.a{attempt}.json").symlink_to(record)
            scored = record.with_name(f"{stem}.eval.json")
            if scored.exists():
                (target / f"{stem}.a{attempt}.eval.json").symlink_to(scored)
            linked += 1
    print(f"{linked} records linked into {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="OUT_ROOT of the sbatch runs")
    parser.add_argument("--name", required=True, help="the variant name (records subdirectory)")
    parser.add_argument("--attempts", default="1,2,3", help="comma-separated attempt directories")
    parser.add_argument("--out", help="pooled root (default: <root>/pooled)")
    args = parser.parse_args()
    root = Path(os.path.expanduser(args.root)).resolve()
    out = Path(os.path.expanduser(args.out)).resolve() if args.out else root / "pooled"
    target = pool(root, args.name, [a.strip() for a in args.attempts.split(",") if a.strip()], out)
    return 0 if compute(target, f"SimuHome harness (pooled attempts): {args.name}") else 1


if __name__ == "__main__":
    sys.exit(main())
