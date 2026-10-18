"""Behaviour: the condition-guard check on plans outside dependency structures.

Spawned by ``GoalRequestBehaviour._plan_mixed`` once every entry is planned,
with the entries that did not come from the dependency route, and before the
power-on guard (which then also covers the guards' own actions). Per entry with
a tree (`utils/condition_check.py` and `utils/condition_guard.py` hold the rules
and a worked example):

    1. steps      the tree's actions on devices with a condition
                  (`planning_conditions.conditioned_steps`); none -> untouched
    2. decide     memory first, then one small-model call for the misses
    3. wrap       each guarded step behind its guard
    4. drop       the guards the plan already satisfies, earlier in execution
                  order, removed again (`drop_satisfied`)

Each entry is changed in place: `tree` becomes the guarded tree; `guards`
lists every decision (what the UA records after a successful run);
`condition_call` keeps the model call, as `profile_call` does; notes go into
`explanation`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.condition_check import guard_trees
from ..utils.condition_guard import drop_satisfied, guards_wire
from ..utils.environment_graph import fetch_environment_graph
from ..utils.planning_conditions import load_conditions
from ..utils.power_on_guard import guidelines_path


class ConditionGuardBehaviour(OneShotBehaviour):
    """Guard the conditioned steps of ``entries`` (plan entries), in place."""

    def __init__(self, entries: List[Dict[str, Any]], logger=None,
                 graph: Optional[Graph] = None) -> None:
        """``graph`` is the environment's TD graph when the caller has it
        already; otherwise it is fetched when the behaviour runs."""
        super().__init__()
        self.entries = entries
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.graph = graph

    async def run(self) -> None:
        """Check every entry that has a tree; entries without one (an error,
        a clarification) are left alone. Never raises past an entry: a failed
        check leaves that entry as it was planned."""
        # 1. The conditions: none (or no file) -> nothing to check.
        conditions = load_conditions(guidelines_path(getattr(self.agent, "config", None)))
        planned = [e for e in self.entries if isinstance(e.get("tree"), dict)]
        if not conditions or not planned:
            return

        # 2. The TD graph, once for all entries.
        if self.graph is None:
            self.graph = await fetch_environment_graph(self.agent, self.logger)

        # 3. Each entry on its own: its own request, its own model call.
        for entry in planned:
            try:
                await self._guard(entry, conditions)
            except Exception as exc:  # noqa: BLE001 - the plan stands unguarded
                self.logger.warning(f"Condition check skipped for an entry: {exc}")

    async def _guard(self, entry: Dict[str, Any], conditions) -> None:
        """One entry through the check (see the module docstring)."""
        # 1-3. Steps, decisions, wrapped tree.
        trace: Dict[str, Any] = {}
        (tree,), decisions, notes = await guard_trees(
            self.agent, self.logger, self.graph, [(None, entry["tree"])],
            entry.get("intent") or entry.get("explanation"), conditions, trace)
        if not decisions:
            return

        # 4. The guards this plan already satisfies, removed again.
        tree, dropped = drop_satisfied(self.graph, tree)

        # 5. The entry: the tree, every decision, the call, the notes.
        entry["tree"] = tree
        entry["guards"] = guards_wire(decisions, tree)
        if trace:
            entry["condition_call"] = trace
        notes = notes + dropped
        if notes:
            entry["explanation"] = "; ".join(p for p in (entry.get("explanation"), *notes) if p)
            for note in notes:
                self.logger.info(demo(f"[CONDITION] {note}"))
