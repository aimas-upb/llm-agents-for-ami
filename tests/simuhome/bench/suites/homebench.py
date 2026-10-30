"""The HomeBench suite: HomeBench rows as SimuHome-shaped episodes (plan section 3).

`tests/homebench/convert.py` writes one small file per HomeBench row,
`<benchmark>/episodes/hb<N>_<case>_seed_<n>.json` (N parts, case feasible /
infeasible / mixed), and one `homes/home<id>.json` per home; `load_episode` merges the
home in, so the pipeline sees the shape it reads for SimuHome. The worker is
`HomeBenchWorker` (an engine and an HBTD per worker), the TDs come from
`HomeBenchTD`, no episode is timed, and scheduled execution does not apply.

Scoring is one rule for every case, with no judge (`tests/homebench/scoring.py`): every
gold token met by an accepted action (or by a value that already held), every
`error_input` by a goal answered impossible or with a question; anything else the agents
did is not counted. `HomeBenchEvaluators.describe` adds the `homebench` block (gold and
predicted tokens, the official bucket, per-part counts) that `homebench_metrics.py` and
`stats.py` read.

Worked example::

    suite = get_suite("homebench")
    suite.estimate("hb3_mixed_seed_2")        # (4, 4, 0): 1 + 3 LLM calls, no judge
    suite.evaluators()[("hb3", "mixed")]      # the scoring module
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from . import PROJECT_ROOT, Suite
from .simuhome import INFRA_MARKERS

BENCHMARK = PROJECT_ROOT / "tests" / "homebench" / "data"
#: The HomeBench planning conditions: none (plan 3.10); see the file for why.
GUIDELINES = PROJECT_ROOT / "tests" / "homebench" / "guidelines.yaml"
#: `hb<N>_<case>_seed_<n>`: N parts in the request, then the case and the seed; an
#: episode whose gold was corrected ends in `_fixed` (`convert.fixed_name`).
NAME = re.compile(r"^hb(?P<parts>\d+)_(?P<case>feasible|infeasible|mixed)_seed_(?P<seed>\d+)"
                  r"(?:_fixed)?$")
#: A request is one segmentation call plus one structuring call per part (plan 5.1).
SEGMENTATION_CALLS = 1


# -- episodes ---------------------------------------------------------------------


def names(benchmark: Path) -> List[str]:
    """The episode names under ``<benchmark>/episodes``, sorted."""
    return sorted(p.stem for p in (Path(benchmark) / "episodes").glob("hb*_seed_*.json"))


def load_episode(benchmark: Path, name: str) -> Dict[str, Any]:
    """One episode with its home merged in (`convert.load_episode`: the one seam
    where HomeBench differs from SimuHome's plain JSON read)."""
    from tests.homebench.convert import load_episode as convert_load_episode
    return convert_load_episode(benchmark, name.removesuffix(".json"))


def home_key(benchmark: Path, name: str) -> str:
    """The home an episode runs in (``home<id>``), read from its file: episodes of
    one home are kept together, so a worker's engine is reset to the same home in a row."""
    path = Path(benchmark) / "episodes" / f"{name.removesuffix('.json')}.json"
    return f"home{json.loads(path.read_text(encoding='utf-8'))['home_id']}"


# -- scoring ------------------------------------------------------------------------


class HomeBenchEvaluators(dict):
    """{(query_type, case): evaluator} for every `hb<N>` type and case.

    A dict subclass so `evaluators[(qt, case)]` (what `evaluate_record` checks) works for
    all ten types without listing them. Every key gives the same evaluator, the scoring
    module: HomeBench has one rule for every case, and nothing of SimuHome's is imported.
    """

    def __missing__(self, key: Tuple[str, str]) -> Any:
        """The scoring module for ``key``, remembered; KeyError for a type or case
        HomeBench lacks."""
        query_type, case = key
        if not re.fullmatch(r"hb\d+", query_type) or case not in ("feasible", "infeasible", "mixed"):
            raise KeyError(key)
        from tests.homebench import scoring
        self[key] = scoring
        return scoring

    @staticmethod
    def describe(record: Dict[str, Any], episode: Dict[str, Any],
                 verdict: Dict[str, Any]) -> Dict[str, Any]:
        """The `homebench` block `evaluate_record` stores with the verdict (see
        `scoring.episode_block`): the tokens and counts the official metrics read."""
        from tests.homebench.scoring import episode_block
        return episode_block(record, episode, verdict)


def evaluators() -> Mapping[Tuple[str, str], Any]:
    """The HomeBench scoring registry (see the module docstring)."""
    return HomeBenchEvaluators()


# -- cost ---------------------------------------------------------------------------


def estimate(name: str) -> Tuple[int, int, int]:
    """(low, high, judge) LLM calls for the episode ``name``: one segmentation call
    and one structuring call per part, exactly (plan 5.1); no judge calls (HomeBench is
    scored without one)."""
    match = NAME.match(name)
    if match is None:
        raise ValueError(f"not a HomeBench episode name: {name!r}")
    calls = SEGMENTATION_CALLS + int(match["parts"])
    return calls, calls, 0


SUITE = Suite(
    name="homebench",
    title="HomeBench",
    default_benchmark=BENCHMARK,
    default_episodes="hb*",
    results_root=PROJECT_ROOT / "tests" / "homebench" / "results",
    worker_path="ami_agents.environment.integration.HomeBench.worker:HomeBenchWorker",
    client_path="ami_agents.environment.integration.HomeBench.sim_client:HomeBenchClient",
    td_path="ami_agents.environment.integration.HomeBench.td_builder:HomeBenchTD",
    names=names,
    load_episode=load_episode,
    evaluators=evaluators,
    timed=lambda query_type: False,
    estimate=estimate,
    judged_label="with an infeasible part",
    # Refusals are scored as `error_input` tokens: no judge reads the replies.
    judge=False,
    scheduled=False,
    infra_markers=INFRA_MARKERS,
    # "homebench": no device conditions (the default); "default": as deployed.
    guidelines={"homebench": GUIDELINES, "default": None},
    default_guidelines="homebench",
    home_key=home_key,
    # The room environments and action effects can be switched off (`--environment`).
    environment=True,
)
