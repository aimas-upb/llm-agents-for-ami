"""Run and score a grid of model variants, then report across them.

    conda run -n ami-agents python -m tests.simuhome.bench.sweep \\
        --grid tests/simuhome/bench/variants/sweep.example.yaml [--dry-run]

A grid file:

    name: planning-models                 # results under <out>/<name>/
    base: {segmentation: {model: gpt-5-mini, reasoning_effort: low}}   # or a variant file
    vary:                                 # per step, the alternatives
      planning.explicit:
        - {model: gpt-5-nano, reasoning_effort: low}
        - {model: qwen3:8b, base_url: http://localhost:11434/v1}
      structuring:
        - {model: gpt-5-mini, reasoning_effort: medium}
    combine: product                      # product (every combination) | one-at-a-time
    run:                                  # run.py options
      episodes: qt3_*,qt4-1_*
      limit: 10
      workers: 2
      context: none
      memory: off
    evaluate: {judge: true}

Every variant runs with the same episodes and a shared step cache: variants
whose segmentation and structuring models are the same reuse those results
(and their measured times, marked `cached`), so a planning-only sweep pays for
them once. `--dry-run` lists the variants and the commands without running.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = PROJECT_ROOT / "tests" / "simuhome" / "results" / "sweeps"


def _label(spec: Dict[str, Any]) -> str:
    model = str(spec.get("model", "?")).replace(":", "-").replace("/", "-")
    effort = spec.get("reasoning_effort")
    return model + (f"@{effort}" if effort else "")


def _set(raw: Dict[str, Any], step: str, spec: Dict[str, Any]) -> None:
    if step.startswith("planning."):
        raw.setdefault("planning", {})[step.split(".", 1)[1]] = spec
    else:
        raw[step] = spec


def variants(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The variant mappings the grid asks for, each named after its choices."""
    base = grid.get("base") or {}
    if isinstance(base, str):
        base = yaml.safe_load(Path(base).read_text(encoding="utf-8")) or {}
    vary: Dict[str, List[Dict[str, Any]]] = grid.get("vary") or {}
    steps = list(vary)
    out = []
    if grid.get("combine", "product") == "one-at-a-time":
        combos = [{}] + [{step: spec} for step in steps for spec in vary[step]]
    else:
        combos = [dict(zip(steps, choice))
                  for choice in itertools.product(*[vary[s] for s in steps])]
    seen = set()
    for combo in combos:
        raw = yaml.safe_load(yaml.safe_dump(base))  # deep copy
        for step, spec in combo.items():
            _set(raw, step, spec if isinstance(spec, dict) else {"model": spec})
        name = "__".join(f"{step.split('.')[-1]}-{_label(spec if isinstance(spec, dict) else {'model': spec})}"
                         for step, spec in combo.items()) or "base"
        if name in seen:
            continue
        seen.add(name)
        raw["name"] = name
        out.append(raw)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8")) or {}
    root = Path(args.out) / str(grid.get("name") or Path(args.grid).stem)
    root.mkdir(parents=True, exist_ok=True)
    run_options = grid.get("run") or {}
    judge = bool((grid.get("evaluate") or {}).get("judge", True))

    plans = []
    for raw in variants(grid):
        path = root / "variants" / f"{raw['name']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        run = [sys.executable, "-m", "tests.simuhome.bench.run", "--variant", str(path),
               "--out", str(root), "--cache", str(root / "cache"), "--skip-existing"]
        for key in ("episodes", "seeds", "limit", "workers", "context", "memory",
                    "step_seconds", "exec_timeout"):
            if run_options.get(key) is not None:
                run += [f"--{key.replace('_', '-')}", str(run_options[key])]
        evaluate = [sys.executable, "-m", "tests.simuhome.bench.evaluate",
                    "--records", str(root / raw["name"])] + ([] if judge else ["--no-judge"])
        plans.append((raw["name"], run, evaluate))

    print(f"{len(plans)} variant(s) -> {root}")
    for name, run, evaluate in plans:
        print(f"\n# {name}\n  " + " ".join(run[1:]) + "\n  " + " ".join(evaluate[1:]))
    if args.dry_run:
        return 0

    for name, run, evaluate in plans:
        print(f"\n=== {name}", flush=True)
        subprocess.run(run, cwd=PROJECT_ROOT, check=False)
        subprocess.run(evaluate, cwd=PROJECT_ROOT, check=False)
    subprocess.run([sys.executable, "-m", "tests.simuhome.bench.report", "--root", str(root)],
                   cwd=PROJECT_ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
