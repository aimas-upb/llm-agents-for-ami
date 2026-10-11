"""The memory store and its API: exact-key records of successful goal executions."""

import pytest
from fastapi.testclient import TestClient

from src.models.signifier import GoalSignifier, key_digest
from src.storage.memory_store import MemoryStore

TREE = {"type": "action", "action_url": "http://h/ac/hvacMode", "parameters": {"value": 3}}
KEY = {"goal_specificity": "explicit", "artifact_name": "ac", "affordance_name": "hvacMode",
       "target_value_text": "cooling", "target_value_is_percentage": False}


def _explicit(**fields):
    return {"kind": "explicit", "key": KEY, "intent_text": "cool it", "tree": TREE, **fields}


class TestMemoryStore:
    def test_a_record_is_found_by_its_exact_key(self, tmp_path):
        store = MemoryStore(tmp_path)
        record = store.record(_explicit())
        assert record.signifier_id.startswith("sig-")
        assert store.match("explicit", dict(KEY)).tree == TREE

    def test_any_difference_in_the_key_misses(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.record(_explicit())
        assert store.match("explicit", {**KEY, "target_value_text": "cold"}) is None
        assert store.match("incomplete", KEY) is None          # the kind is part of it

    def test_key_order_does_not_matter(self):
        assert key_digest("explicit", {"a": 1, "b": 2}) == key_digest("explicit", {"b": 2, "a": 1})

    def test_a_newer_record_supersedes_the_older(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.record(_explicit())
        newer = store.record(_explicit(tree={"type": "action", "action_url": "http://h/b"}))
        assert store.match("explicit", KEY).signifier_id == newer.signifier_id
        assert len(store) == 2

    def test_records_survive_reopening(self, tmp_path):
        record = MemoryStore(tmp_path).record(_explicit())
        reopened = MemoryStore(tmp_path)
        assert reopened.match("explicit", KEY).signifier_id == record.signifier_id
        assert reopened.get(record.signifier_id).key_digest == record.key_digest

    def test_an_unreadable_file_is_skipped(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.record(_explicit())
        (store.dir / "broken.json").write_text("{not json")
        assert len(MemoryStore(tmp_path)) == 1

    def test_clear(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.record(_explicit())
        assert store.clear() == 1
        assert store.match("explicit", KEY) is None and len(store) == 0

    def test_an_ambiguous_record_keeps_context_and_used(self, tmp_path):
        record = MemoryStore(tmp_path).record({
            "kind": "ambiguous", "key": {"variables": [{"property_class": "x"}]},
            "context": {"variables": [{"properties": [{"url": "http://h/p", "value": 61}]}]},
            "used": {"actions": [{"url": "http://h/a", "parameters": {"value": True}}],
                     "properties": []}})
        assert isinstance(record, GoalSignifier)
        assert record.context["variables"][0]["properties"][0]["value"] == 61


class TestAPI:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from src.api import main
        from src.api.routes import signifiers
        monkeypatch.setattr(signifiers, "_store", MemoryStore(tmp_path))
        return TestClient(main.app)

    def test_record_then_match(self, client):
        assert client.post("/signifiers", json=_explicit()).status_code == 201
        found = client.post("/matching/match", json={"kind": "explicit", "key": KEY}).json()
        assert found["match"]["tree"] == TREE
        missed = client.post("/matching/match",
                             json={"kind": "explicit", "key": {**KEY, "artifact_name": "x"}})
        assert missed.json()["match"] is None

    def test_list_and_delete(self, client):
        client.post("/signifiers", json=_explicit())
        assert client.get("/signifiers").json()["total_count"] == 1
        assert client.delete("/signifiers").json()["deleted_count"] == 1

    def test_a_malformed_record_or_kind_is_refused(self, client):
        assert client.post("/signifiers", json={"kind": "vague", "key": {}}).status_code == 400
        assert client.post("/matching/match", json={"kind": "vague", "key": {}}).status_code == 400
