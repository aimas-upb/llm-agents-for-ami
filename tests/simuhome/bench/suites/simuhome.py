"""The SimuHome suite: what the bench did before suites existed, behind the same object.

Every value here is what `run.py`, `pipeline.py`, `stubs.py`, `evaluate.py` and
`harness.py` used to hard-code, so a run with `--suite simuhome` (the default)
behaves as it did: episodes are the plain JSON files `qt*_seed_*.json` of SimuHome's
benchmark directory, the worker is `SimuHomeWorker`, the TDs come from `SimuHomeTD`,
and the scoring registry is SimuHome's own (`evaluate.simuhome_evaluators`).

Worked example::

    suite = get_suite("simuhome")
    suite.names(suite.default_benchmark)[:1]        # ['qt1_feasible_seed_1']
    suite.timed("qt4-2"), suite.timed("qt3")        # (True, False)
    suite.estimate("qt3_infeasible_seed_1")         # (3, 6, 3): LLM calls low, high, judge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from . import PROJECT_ROOT, Suite

BENCHMARK = PROJECT_ROOT.parent / "SimuHome" / "data" / "benchmark"
#: Rough LLM calls per episode for the agents' steps (low, high), and per judged episode.
CALLS_PER_EPISODE = (3, 6)
JUDGE_CALLS = 3
#: Error text that marks an infrastructure failure (the transport or the simulator), not the model's.
INFRA_MARKERS = ("SimuHomeError", "APIConnectionError", "APITimeoutError",
                 "RateLimitError", "InternalServerError", "ConnectError")


def names(benchmark: Path) -> List[str]:
    """The episode names in SimuHome's benchmark directory, sorted."""
    return sorted(p.stem for p in Path(benchmark).glob("qt*_seed_*.json"))


def load_episode(benchmark: Path, name: str) -> Dict[str, Any]:
    """One SimuHome episode: a plain JSON read (the file already has its home)."""
    return json.loads((Path(benchmark) / f"{name.removesuffix('.json')}.json")
                      .read_text(encoding="utf-8"))


def evaluators() -> Mapping[Tuple[str, str], Any]:
    """SimuHome's registry, {(query_type, case): evaluator module}."""
    # Imported here: evaluate.py imports the suites, and importing it loads .env.
    from tests.simuhome.bench.evaluate import simuhome_evaluators
    return simuhome_evaluators()


def timed(query_type: str) -> bool:
    """True for the timed types (qt4-x): always run in lockstep."""
    return query_type.startswith("qt4")


def estimate(name: str) -> Tuple[int, int, int]:
    """(low, high, judge) LLM calls for the episode ``name``; the judge only
    reads infeasible episodes."""
    return (*CALLS_PER_EPISODE, JUDGE_CALLS if "_infeasible_" in name else 0)


SUITE = Suite(
    name="simuhome",
    title="SimuHome",
    default_benchmark=BENCHMARK,
    default_episodes="qt*",
    results_root=PROJECT_ROOT / "tests" / "simuhome" / "results",
    worker_path="ami_agents.environment.integration.SimuHome.worker:SimuHomeWorker",
    client_path="ami_agents.environment.integration.SimuHome.sim_client:SimuHomeClient",
    td_path="ami_agents.environment.integration.SimuHome.td_builder:SimuHomeTD",
    names=names,
    load_episode=load_episode,
    evaluators=evaluators,
    timed=timed,
    estimate=estimate,
    judged_label="infeasible",
    scheduled=True,
    infra_markers=INFRA_MARKERS,
    # Only the agents' own file: SimuHome has no benchmark-specific conditions.
    guidelines={"default": None},
    default_guidelines="default",
    home_key=None,
)
