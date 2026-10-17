"""The guard store: remembered protection decisions for device actions.

Some device actions are refused unless the device is in a certain state: a
dryer refuses a new dryness level while its cycle runs, an appliance in "dead
front" (switched off) refuses most commands. A *protection guard* sits in front
of such an action in a plan: check a property, and when it does not hold, run a
protecting action first. Whether an action needs one is decided once -- by the
profiling model, resolved against the TD graph -- and remembered here, for that
exact action of that exact device, so later plans apply it without asking.

Separate from the goal signifiers (`memory_store.py`): its own records, its own
key, its own directory. Nothing is shared with signifier recall.

Worked example -- the dryness level of the utility-room dryer:

    key     {"affordance_name": "selectedDrynessLevel",
             "affordance_class": "saref:SetAbsoluteLevelCommand",
             "device_name": "utility_room_laundry_dryer_1",
             "device_class": "homeont:LaundryDryer",
             "room_class": "homeont:UtilityRoom"}

    record  {"guard_id": "grd-3f2a9c01b7de", "key": {...as above...},
             "key_digest": "<sha256 of the key>",
             "protection": true,
             "check":  {"title": "operationalState", "class": "...", "target": "<read URL>",
                        "value": 0},           # proceed only while Stopped ...
             "action": {"title": "operationalState", "class": "...", "target": "<invoke URL>",
                        "value": 0},           # ... else stop the cycle first
             "created_at": "2026-10-16T19:20:00+00:00"}

A "no protection needed" decision is the same record with `"protection": false`
and no check / action: it is remembered too, so the question is not asked again.

Flow:
    1. The UserAssistant, after a plan ran successfully, sends each decision the
       model made; the EnvExplorer calls `record(guard)`.
    2. On a later request, the InteractionSolver sends the key of each action
       that may need a guard; the EnvExplorer answers with `match(key)`.
    3. A hit is applied as stored; a miss becomes a profiling question.

Storage: one JSON file per record under `<storage_dir>/guards/`. An in-memory
index maps each key digest to its latest record, built from the files when the
store opens; a newer record for the same key supersedes the older one (the
older file stays on disk, the index points past it).

    record(guard)   store it, and make it the key's answer
    match(key)      the key's latest record, or None -- exact, no scoring
    get(id), list(), clear()
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# The fields a decision is remembered under. The same action name on another
# device, or the same device moved to another kind of room, is another key: a
# decision is only reused where it was made.
KEY_FIELDS = ("affordance_name", "affordance_class", "device_name", "device_class",
              "room_class")


def guard_key_digest(key: Dict[str, Any]) -> str:
    """The digest a key is indexed under.

    Only the five KEY_FIELDS count, in a fixed order; extra fields are ignored
    and a missing field counts as None. Two keys with the same five values give
    the same digest, whatever else they carry."""
    canonical = {field: (key or {}).get(field) for field in KEY_FIELDS}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


class GuardStore:
    """Protection decisions, indexed by the digest of their key."""

    def __init__(self, storage_dir: Union[str, Path]):
        """Open (or create) the store under `<storage_dir>/guards/` and load
        every record already there."""
        self.dir = Path(storage_dir) / "guards"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, Dict[str, Any]] = {}   # record id -> record
        self._by_key: Dict[str, str] = {}               # key digest -> latest record id
        self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> None:
        """Read every record on disk; an unreadable file is skipped, not fatal."""
        for path in sorted(self.dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                self._index(record)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
                logger.warning("Skipping unreadable guard record %s: %s", path.name, exc)
        logger.info("GuardStore at %s: %d record(s)", self.dir, len(self._records))

    def _index(self, record: Dict[str, Any]) -> None:
        """Add a record to the index; it becomes its key's answer unless the
        key already points to a newer record (files load in name order, not
        in time order)."""
        self._records[record["guard_id"]] = record
        current = self._records.get(self._by_key.get(record["key_digest"], ""))
        if current is None or record["created_at"] >= current["created_at"]:
            self._by_key[record["key_digest"]] = record["guard_id"]

    # -- writing -----------------------------------------------------------

    def record(self, guard: Dict[str, Any]) -> Dict[str, Any]:
        """Store a decision and make it its key's answer; returns the stored
        record (with its id, digest and timestamp).

        `guard` needs `key` and `protection`; a guard (protection true) also its
        `check` and `action`. Raises ValueError when one is missing, so a
        half-formed decision is never remembered."""
        # 1. Validate: a decision without its key, or a guard without what it
        #    checks and does, would be applied wrongly on every later hit.
        if not isinstance(guard, dict) or not isinstance(guard.get("key"), dict):
            raise ValueError("a guard record needs a 'key' object")
        if not isinstance(guard.get("protection"), bool):
            raise ValueError("a guard record needs 'protection': true | false")
        if guard["protection"] and not (isinstance(guard.get("check"), dict)
                                        and isinstance(guard.get("action"), dict)):
            raise ValueError("a protecting guard needs its 'check' and 'action'")

        # 2. Build the record: only the key's five fields are kept, and check /
        #    action only for a guard. An id and a timestamp given by the caller
        #    are kept (a record copied from elsewhere); otherwise new ones.
        record = {
            "guard_id": guard.get("guard_id") or f"grd-{uuid.uuid4().hex[:12]}",
            "key": {field: guard["key"].get(field) for field in KEY_FIELDS},
            "key_digest": guard_key_digest(guard["key"]),
            "protection": guard["protection"],
            **({"check": guard["check"], "action": guard["action"]}
               if guard["protection"] else {}),
            "created_at": guard.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }

        # 3. Write it to disk, then index it: it now answers its key.
        (self.dir / f"{record['guard_id']}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        self._index(record)
        return record

    def clear(self) -> int:
        """Delete every record, on disk and in the index; returns how many
        files were removed."""
        removed = 0
        for path in self.dir.glob("*.json"):
            path.unlink()
            removed += 1
        self._records.clear()
        self._by_key.clear()
        return removed

    # -- reading -----------------------------------------------------------

    def match(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The latest decision stored under exactly this key, or None.

        Exact: all five key fields must be equal; there is no partial or
        similarity match."""
        record_id = self._by_key.get(guard_key_digest(key))
        return self._records.get(record_id) if record_id else None

    def get(self, guard_id: str) -> Optional[Dict[str, Any]]:
        """The record with this id, or None (superseded records included)."""
        return self._records.get(guard_id)

    def list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Every record, superseded ones included, newest first; at most
        `limit` of them when given."""
        records = sorted(self._records.values(), key=lambda r: r["created_at"], reverse=True)
        return records[:limit] if limit else records

    def __len__(self) -> int:
        """How many records are stored, superseded ones included."""
        return len(self._records)
