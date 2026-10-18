"""The experience engine's MemoryStore and GuardStore, used directly (no EnvExplorer).

Recall goes through the ISA's own `signifier_memory.recall`, whose
SIGNIFIER_MATCH query `stubs.Explorer.query` answers from this store.
Recording follows the UA's rules (`user_assistant/signifiers.py`): the context
is captured before the plan's first tick, and after a success
`ExecutionContext.should_record` decides and `build_signifier` builds the record.

Condition-guard decisions go the same way: recall through the ISA's
`guard_memory.recall_guard` (GUARD_MATCH, answered by `stubs.Explorer.query`
from the GuardStore), recording after a success by the UA's rule
(`user_assistant/guards.guards_to_record`).

The engine imports as top-level `src`, like SimuHome's evaluator; it is only
imported when memory is on, and the evaluator runs in another process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class Memory:
    """One store for a run (or a worker), on disk under ``directory``."""

    def __init__(self, directory: Path) -> None:
        from ami_agents.shared.memory import ensure_engine_on_path
        ensure_engine_on_path()
        from src.storage.guard_store import GuardStore  # noqa: E402 - engine path
        from src.storage.memory_store import MemoryStore  # noqa: E402 - engine path
        directory.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(str(directory))
        self.guards = GuardStore(str(directory))
        self.hits = 0
        self.recorded = 0
        self.guard_hits = 0
        self.guards_recorded = 0

    def match(self, kind: Optional[str], key: Optional[Dict[str, Any]]):
        if not isinstance(kind, str) or not isinstance(key, dict):
            return None
        record = self.store.match(kind, key)
        if record is None:
            return None
        self.hits += 1
        return record.to_json_doc()

    def record(self, signifier: Dict[str, Any]) -> str:
        stored = self.store.record(signifier)
        self.recorded += 1
        return stored.signifier_id

    def match_guard(self, key: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The guard decision under exactly ``key``, or None."""
        if not isinstance(key, dict):
            return None
        record = self.guards.match(key)
        if record is not None:
            self.guard_hits += 1
        return record

    def record_guard(self, guard: Dict[str, Any]) -> str:
        """Store one guard decision; its id."""
        stored = self.guards.record(guard)
        self.guards_recorded += 1
        return stored["guard_id"]

