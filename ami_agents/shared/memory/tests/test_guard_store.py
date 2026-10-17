"""The guard store: remembered protection decisions, found again by exact key."""

import pytest

from src.storage.guard_store import GuardStore, guard_key_digest

KEY = {"affordance_name": "selectedDrynessLevel",
       "affordance_class": "saref:SetAbsoluteLevelCommand",
       "device_name": "dryer_1", "device_class": "homeont:LaundryDryer",
       "room_class": "homeont:UtilityRoom"}
CHECK = {"title": "operationalState", "class": "homeont:OperationalState",
         "target": "http://h/dryer_1/properties/operationalState", "value": 0}
ACTION = {"title": "operationalState", "class": "homeont:SetOperationalStateCommand",
          "target": "http://h/dryer_1/actions/operationalState", "value": 0}


def _guard(**fields):
    """A protecting decision for KEY; ``fields`` override or add to it."""
    return {"key": KEY, "protection": True, "check": CHECK, "action": ACTION, **fields}


class TestGuardStore:
    def test_a_decision_is_found_by_its_exact_key(self, tmp_path):
        store = GuardStore(tmp_path)
        record = store.record(_guard())
        assert record["guard_id"].startswith("grd-")
        assert store.match(dict(KEY))["check"] == CHECK

    def test_any_difference_in_the_key_misses(self, tmp_path):
        """Each of the five key fields counts: another device, another room
        class, another action is another key."""
        store = GuardStore(tmp_path)
        store.record(_guard())
        for field in KEY:
            assert store.match({**KEY, field: "other"}) is None, field

    def test_extra_key_fields_are_ignored(self, tmp_path):
        store = GuardStore(tmp_path)
        store.record(_guard(key={**KEY, "note": "x"}))
        assert store.match(KEY) is not None
        assert guard_key_digest({**KEY, "note": "x"}) == guard_key_digest(KEY)

    def test_no_protection_is_remembered_without_check_or_action(self, tmp_path):
        store = GuardStore(tmp_path)
        store.record({"key": KEY, "protection": False})
        record = store.match(KEY)
        assert record["protection"] is False and "check" not in record

    def test_a_newer_decision_supersedes_the_older(self, tmp_path):
        store = GuardStore(tmp_path)
        store.record(_guard())
        newer = store.record({"key": KEY, "protection": False})
        assert store.match(KEY)["guard_id"] == newer["guard_id"]
        assert len(store) == 2                               # the older stays on disk

    def test_records_survive_reopening(self, tmp_path):
        """A new store on the same directory loads what the first wrote."""
        GuardStore(tmp_path).record(_guard())
        assert GuardStore(tmp_path).match(KEY)["action"] == ACTION

    @pytest.mark.parametrize("bad", [
        {"protection": True, "check": CHECK, "action": ACTION},   # no key
        {"key": KEY},                                             # no protection
        {"key": KEY, "protection": "yes"},                        # not a boolean
        {"key": KEY, "protection": True, "check": CHECK},         # guard without action
    ])
    def test_a_half_formed_decision_is_refused(self, tmp_path, bad):
        with pytest.raises(ValueError):
            GuardStore(tmp_path).record(bad)

    def test_clear_removes_everything(self, tmp_path):
        store = GuardStore(tmp_path)
        store.record(_guard())
        assert store.clear() == 1
        assert store.match(KEY) is None and len(GuardStore(tmp_path)) == 0
