"""The benchmark a bench run is over: a small `Suite` object per benchmark.

The bench (`run.py`, `pipeline.py`, `stubs.py`, `evaluate.py`, `harness.py`, `sweep.py`)
was written for SimuHome. A suite carries what differs between benchmarks, so the same
code runs both, and SimuHome stays the default:

    names            the episode names a benchmark directory holds
    load_episode     one episode, in the shape the pipeline reads
                     (`meta`, `query`, `eval`, `initial_home_config`)
    worker_cls       the simulator + TD server pair that serves an episode
    client_cls       the simulator client (`reset`, `home_state`, ...)
    td_cls           the TD builder `stubs.Home` discovers the home with
    evaluators       {(query_type, case): object with `evaluate(payload, judges)`}
    plus the defaults a run needs: benchmark directory, episode pattern, results
    directory, which types are timed, which errors are infrastructure, the planning
    guidelines files, whether scheduled execution applies, whether the suite has the
    `environment` option (HomeBench: the room environments and action effects on or
    off), and the cost estimate.

The classes are named by import path and imported on first use, so reading a suite
(for `--dry-run`, say) does not import a TD builder or start anything.

Worked example::

    suite = get_suite("homebench")
    names = suite.names(suite.default_benchmark)     # ['hb10_feasible_seed_1', ...]
    episode = suite.load_episode(suite.default_benchmark, names[0])
    worker = suite.worker_cls(benchmark=suite.default_benchmark).start()
    # ... the pipeline runs against worker.shtd_base and worker.client
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]

#: The suite a run uses when none is named: SimuHome, as before suites existed.
DEFAULT_SUITE = "simuhome"
#: Every suite name (the CLI's `--suite` choices).
SUITE_NAMES = ("simuhome", "homebench")
#: The values of the `environment` option of a suite that has it, and its default. The
#: same as HomeBench's `td_builder.ENVIRONMENT_MODES` (a test pins them together): they
#: are repeated here so reading the options does not import the HomeBench module.
ENVIRONMENT_MODES = ("on", "off")
DEFAULT_ENVIRONMENT = "on"


def _import(path: str) -> Any:
    """The object at ``"package.module:Name"``."""
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class Suite:
    """One benchmark's hooks for the bench (see the module docstring)."""

    name: str
    title: str
    default_benchmark: Path
    default_episodes: str
    results_root: Path
    worker_path: str
    client_path: str
    td_path: str
    names: Callable[[Path], List[str]]
    load_episode: Callable[[Path, str], Dict[str, Any]]
    evaluators: Callable[[], Mapping[Tuple[str, str], Any]]
    timed: Callable[[str], bool]
    estimate: Callable[[str], Tuple[int, int, int]]
    judged_label: str
    scheduled: bool
    infra_markers: Tuple[str, ...]
    guidelines: Mapping[str, Optional[Path]] = field(default_factory=dict)
    default_guidelines: str = "default"
    home_key: Optional[Callable[[Path, str], str]] = None
    environment: bool = False

    @property
    def worker_cls(self) -> Any:
        """The worker class (a simulator + TD server pair)."""
        return _import(self.worker_path)

    @property
    def client_cls(self) -> Any:
        """The simulator client class."""
        return _import(self.client_path)

    @property
    def td_cls(self) -> Any:
        """The TD builder class `stubs.Home` is built with."""
        return _import(self.td_path)

    def guidelines_path(self, choice: Optional[str]) -> Optional[Path]:
        """The planning-guidelines file for ``choice`` (None: the suite's default
        choice); None when the choice means the agents' own file ("default").

        Raises ValueError for a choice this suite does not offer, so a typo never
        silently runs with the wrong conditions.
        """
        # 1. The suite's own default when nothing is chosen.
        choice = choice or self.default_guidelines
        # 2. Only the choices the suite lists.
        if choice not in self.guidelines:
            raise ValueError(f"suite {self.name!r} has no guidelines {choice!r}; "
                             f"choose from {', '.join(sorted(self.guidelines))}")
        return self.guidelines[choice]


def get_suite(name: Optional[str] = None) -> Suite:
    """The suite called ``name`` (SimuHome when None). Raises ValueError for another."""
    name = name or DEFAULT_SUITE
    # 1. Imported on demand: a SimuHome run never imports the HomeBench module.
    if name == "simuhome":
        from .simuhome import SUITE
    elif name == "homebench":
        from .homebench import SUITE
    else:
        raise ValueError(f"unknown suite {name!r}; choose from {', '.join(SUITE_NAMES)}")
    return SUITE
