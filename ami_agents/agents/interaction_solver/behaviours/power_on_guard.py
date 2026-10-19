"""Behaviour: the cycle checks and the "power on first" guard on plans outside dependency structures.

Spawned by ``GoalRequestBehaviour._plan_mixed`` once every entry is planned, with
the entries that did not come from the dependency route. Fetches the TD graph
once, then, on each entry's tree:

    1. cycle checks     every start / pause / resume / stop sent only from a
                        state that allows it (`utils/cycle_guard.py`)
    2. power on first   the power-on guard in front of each guarded action
                        (`utils/power_on_guard.py`, where the rules and an
                        example are)

in that order, so a stopped device is not switched on only to be refused.

Each entry is changed in place: `tree` becomes the guarded tree, and actions
left unguarded (no on/off command found, ...) are noted in `explanation`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.cycle_guard import guard_cycles
from ..utils.environment_graph import fetch_environment_graph
from ..utils.power_on_guard import guard_tree, guidelines_path, load_applies_to


class PowerOnGuardBehaviour(OneShotBehaviour):
    """Guard the actions of ``entries`` (plan entries), in place."""

    def __init__(self, entries: List[Dict[str, Any]], logger=None,
                 graph: Optional[Graph] = None) -> None:
        """``graph`` is the environment's TD graph when the caller has it
        already; otherwise it is fetched when the behaviour runs."""
        super().__init__()
        self.entries = entries
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.graph = graph

    async def run(self) -> None:
        """Guard every entry that has a tree; entries without one (an error, a
        clarification) are left alone."""
        # 1. The trees, and the power-on condition's devices (none listed, or
        #    no file: no power-on guard; the cycle checks apply regardless).
        applies_to = load_applies_to(guidelines_path(getattr(self.agent, "config", None)))
        planned = [e for e in self.entries if isinstance(e.get("tree"), dict)]
        if not planned:
            return

        # 2. The TD graph, once for all entries.
        graph = self.graph if self.graph is not None else \
            await fetch_environment_graph(self.agent, self.logger)

        # 3. Each tree checked, then guarded; why an action stayed unguarded
        #    goes on its entry.
        for entry in planned:
            entry["tree"], notes = guard_cycles(graph, entry["tree"])
            if applies_to is not None:
                entry["tree"], power_notes = guard_tree(graph, entry["tree"], applies_to)
                notes = notes + power_notes
            if notes:
                entry["explanation"] = "; ".join(
                    [p for p in (entry.get("explanation"), *notes) if p])
                for note in notes:
                    self.logger.info(demo(f"[GUARD] {note}"))
