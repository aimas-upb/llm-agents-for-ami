"""Run SimuHome episodes with the plans executed as scheduled workflows.

The same bench as `run.py` -- same steps, same options, same records -- with
execution done SimuHome's way: the trees are dry-run in memory and their
actions submitted as `schedule_workflow` calls (`scheduling.py`), instead of
ticked against the simulator in lockstep. Results go to their own directory, so
the two runners' records never mix.

    conda run -n ami-agents python -m tests.simuhome.bench.run_scheduled \\
        --episode qt4-3_feasible_seed_45

`run.py` stays the tick-based runner (`--execution ticks`, its default).
"""

from __future__ import annotations

import sys

from .run import PROJECT_ROOT, main

# Where scheduled runs write their records.
DEFAULT_OUT = PROJECT_ROOT / "tests" / "simuhome" / "results" / "bench_scheduled"


if __name__ == "__main__":
    sys.exit(main(execution="scheduled", out=DEFAULT_OUT))
