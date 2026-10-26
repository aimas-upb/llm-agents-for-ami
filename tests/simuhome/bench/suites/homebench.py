"""The HomeBench suite: HomeBench rows as SimuHome-shaped episodes (plan section 3).

`tests/homebench/convert.py` writes one small file per HomeBench row,
`<benchmark>/episodes/hb<N>_<case>_seed_<n>.json` (N parts, case feasible /
infeasible / mixed), and one `homes/home<id>.json` per home; `load_episode` merges the
home in, so the pipeline sees the shape it reads for SimuHome. The worker is
`HomeBenchWorker` (an engine and an HBTD per worker), the TDs come from
`HomeBenchTD`, no episode is timed, and scheduled execution does not apply.

The evaluators (plan 3.7; the rules are in `tests/homebench/scoring.py`):

    feasible      SimuHome's qt3 evaluator, unchanged: equality on the final-state
                  attributes the gold call sets
    infeasible    our execution check per part, and a judge panel: SimuHome's own for a part
                  whose device is missing, HomeBench's prompt for a missing attribute
    mixed         both, each on its own parts; the episode passes only if every part does

`HomeBenchEvaluators.describe` adds the `homebench` block (gold and predicted tokens, the
official bucket, per-part counts) that `homebench_metrics.py` and `stats.py` read.

Worked example::

    suite = get_suite("homebench")
    suite.estimate("hb3_mixed_seed_2")        # (4, 4, 3): 1 + 3 LLM calls, 3 judge calls
    suite.evaluators()[("hb3", "feasible")]   # SimuHome's qt3 feasible module
    suite.evaluators()[("hb3", "mixed")]      # a scoring.MixedEvaluator
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import PROJECT_ROOT, Suite
from .simuhome import INFRA_MARKERS, JUDGE_CALLS

BENCHMARK = PROJECT_ROOT / "tests" / "homebench" / "data"
#: The HomeBench planning conditions: none (plan 3.10); see the file for why.
GUIDELINES = PROJECT_ROOT / "tests" / "homebench" / "guidelines.yaml"
#: `hb<N>_<case>_seed_<n>`: N parts in the request, then the case and the seed.
NAME = re.compile(r"^hb(?P<parts>\d+)_(?P<case>feasible|infeasible|mixed)_seed_(?P<seed>\d+)$")
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
    """{(query_type, case): evaluator} for every `hb<N>` type, built on first lookup.

    A dict subclass so `evaluators[(qt, case)]` (what `evaluate_record` does) works
    for all ten types without listing them. SimuHome's evaluator modules are imported
    once, on the first lookup (they need the checkout on the path).
    """

    def __init__(self) -> None:
        """An empty registry; nothing from SimuHome is imported yet."""
        super().__init__()
        self._simuhome: Optional[Mapping[Tuple[str, str], Any]] = None
        self._judge: Optional[Any] = None

    def _modules(self) -> Mapping[Tuple[str, str], Any]:
        """SimuHome's evaluator modules, imported on first use."""
        if self._simuhome is None:
            from tests.simuhome.bench.evaluate import simuhome_evaluators
            self._simuhome = simuhome_evaluators()
        return self._simuhome

    def _judge_for_infeasible_parts(self) -> Any:
        """The `HomeBenchJudge` (built once): SimuHome's qt3 infeasible evaluator for a
        missing device, HomeBench's prompt for the rest, over SimuHome's common helpers."""
        if self._judge is None:
            from src.pipelines.episode_evaluation import common  # SimuHome's, on the path
            from tests.homebench.scoring import HomeBenchJudge
            self._judge = HomeBenchJudge(self._modules()[("qt3", "infeasible")], common)
        return self._judge

    def __missing__(self, key: Tuple[str, str]) -> Any:
        """The evaluator for ``key``, remembered; KeyError for a type or case HomeBench lacks."""
        query_type, case = key
        # 1. Only HomeBench's own types and cases.
        if not re.fullmatch(r"hb\d+", query_type) or case not in ("feasible", "infeasible", "mixed"):
            raise KeyError(key)
        # 2. Feasible: SimuHome's qt3 evaluator (equality on final-state attributes).
        feasible = self._modules()[("qt3", "feasible")]
        if case == "feasible":
            evaluator = feasible
        # 3. Infeasible: the judge of its parts; mixed: that and the feasible evaluator.
        elif case == "infeasible":
            evaluator = self._judge_for_infeasible_parts()
        else:
            from tests.homebench.scoring import MixedEvaluator
            evaluator = MixedEvaluator(feasible, self._judge_for_infeasible_parts())
        self[key] = evaluator
        return evaluator

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
    and one structuring call per part, exactly (plan 5.1); judge calls (3) only for
    episodes with a part that cannot be done. The judge figure is a floor: an episode with
    a missing-device part and a missing-attribute (or unidentified) part is judged by two
    panels, 6 calls (2,266 of the 10,934 episodes with an infeasible part, 21%), which the
    name alone does not reveal."""
    match = NAME.match(name)
    if match is None:
        raise ValueError(f"not a HomeBench episode name: {name!r}")
    calls = SEGMENTATION_CALLS + int(match["parts"])
    return calls, calls, JUDGE_CALLS if match["case"] != "feasible" else 0


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
    scheduled=False,
    infra_markers=INFRA_MARKERS,
    # "homebench": no device conditions (the default); "default": as deployed.
    guidelines={"homebench": GUIDELINES, "default": None},
    default_guidelines="homebench",
    home_key=home_key,
)
