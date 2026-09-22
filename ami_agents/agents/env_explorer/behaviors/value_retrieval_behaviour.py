"""Behaviour: read the values of already-resolved property affordances.

The second half of answering a state request. Resolution says *what can answer
the question and where*; this says *what the answer is*. They are separate
because the first is worth asking on its own -- a planner wants to know what
senses temperature in a room without reading every thermometer.

One behaviour per retrieval, per CLAUDE.md 3.2: the reads of one request are
independent of another's, and each request's reads run concurrently.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from spade.behaviour import OneShotBehaviour

from ....shared.utils.logger import LoggerFactory
from ..utils.state_resolution import ResolvedAffordance

# A property read is a single GET against a device that is meant to be local.
# Long enough to absorb a slow device, short enough that one unreachable
# artifact does not hold up the whole answer.
READ_TIMEOUT_SECONDS = 5.0


class ValueRetrievalBehaviour(OneShotBehaviour):
    """Read every affordance in `affordances`, concurrently."""

    def __init__(self, affordances: List[ResolvedAffordance], logger=None) -> None:
        super().__init__()
        self.affordances = affordances
        self.logger = logger or LoggerFactory.get_logger("EnvExplorer")
        # Populated by ``run``: the same affordances, each carrying its value.
        self.result: List[ResolvedAffordance] = []
        self.error: Optional[str] = None

    async def run(self) -> None:
        if not self.affordances:
            self.result = []
            return

        try:
            timeout = aiohttp.ClientTimeout(total=READ_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                await asyncio.gather(*[
                    self._read(session, affordance)
                    for affordance in self.affordances
                ])
        except Exception as exc:
            # The session itself failed, so nothing was read. The resolution
            # still stands and is returned unread rather than discarded.
            self.error = str(exc)
            self.logger.warning("Value retrieval failed: %s", exc)

        self.result = self.affordances

    async def _read(self, session, affordance: ResolvedAffordance) -> None:
        """Dereference one affordance's read target.

        The target is the `hctl:hasTarget` of a form whose operation type is
        `td:readProperty` -- an ordinary GET returning the value and nothing
        else, already in the unit the Thing Description advertises. A property
        whose output schema is an object returns a dictionary; it is stored as
        it arrives, not flattened.
        """
        try:
            async with session.get(affordance.target) as response:
                response.raise_for_status()
                affordance.value = await response.json(content_type=None)
                affordance.has_value = True
        except Exception as exc:
            # The resolution still stands: the caller knows what would answer
            # the question and where, even when this read did not land.
            affordance.detail = f"read failed: {exc}"
            self.logger.warning(
                "Failed to read %s: %s", affordance.target, exc)
