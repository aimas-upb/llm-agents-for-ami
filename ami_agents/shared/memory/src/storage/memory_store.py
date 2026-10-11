"""The memory store: remembered goal executions, found again by exact key.

One JSON file per record under `<storage_dir>/signifiers/`. An in-memory index
maps each key digest to its latest record, built from the files when the store
opens; a newer record for the same key supersedes the older one (the older file
stays on disk, the index points past it).

    record(signifier)      store it, and make it the key's answer
    match(kind, key)       the key's latest record, or None -- exact, no scoring
    get(id), list(), clear()
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.models.signifier import GoalSignifier, key_digest

logger = logging.getLogger(__name__)


class MemoryStore:
    """Remembered goal executions, indexed by the digest of their key."""

    def __init__(self, storage_dir: Union[str, Path]):
        self.dir = Path(storage_dir) / "signifiers"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, GoalSignifier] = {}
        self._by_key: Dict[str, str] = {}          # key digest -> latest record id
        self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> None:
        """Read every record on disk; an unreadable file is skipped, not fatal."""
        for path in sorted(self.dir.glob("*.json")):
            try:
                record = GoalSignifier(**json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
                logger.warning("Skipping unreadable memory record %s: %s", path.name, exc)
                continue
            self._index(record)
        logger.info("MemoryStore at %s: %d record(s)", self.dir, len(self._records))

    def _index(self, record: GoalSignifier) -> None:
        self._records[record.signifier_id] = record
        current = self._records.get(self._by_key.get(record.key_digest, ""))
        if current is None or record.created_at >= current.created_at:
            self._by_key[record.key_digest] = record.signifier_id

    # -- writing -----------------------------------------------------------

    def record(self, signifier: Union[GoalSignifier, Dict[str, Any]]) -> GoalSignifier:
        """Store a record and make it its key's answer. A dict without an id
        gets one."""
        if isinstance(signifier, dict):
            data = dict(signifier)
            data.setdefault("signifier_id", f"sig-{uuid.uuid4().hex[:12]}")
            data.pop("key_digest", None)            # always recomputed from the key
            signifier = GoalSignifier(**data)
        path = self.dir / f"{signifier.signifier_id}.json"
        path.write_text(json.dumps(signifier.to_json_doc(), indent=2), encoding="utf-8")
        self._index(signifier)
        return signifier

    def clear(self) -> int:
        """Forget everything. Returns how many records were removed."""
        removed = 0
        for path in self.dir.glob("*.json"):
            path.unlink()
            removed += 1
        self._records.clear()
        self._by_key.clear()
        return removed

    # -- reading -----------------------------------------------------------

    def match(self, kind: str, key: Dict[str, Any]) -> Optional[GoalSignifier]:
        """The latest record stored under exactly this kind and key."""
        record_id = self._by_key.get(key_digest(kind, key))
        return self._records.get(record_id) if record_id else None

    def get(self, signifier_id: str) -> Optional[GoalSignifier]:
        return self._records.get(signifier_id)

    def list(self, limit: Optional[int] = None) -> List[GoalSignifier]:
        """Every record, newest first."""
        records = sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
        return records[:limit] if limit else records

    def __len__(self) -> int:
        return len(self._records)
