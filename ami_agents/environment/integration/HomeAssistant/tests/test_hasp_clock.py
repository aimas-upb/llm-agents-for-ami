"""HASP's GET /_clock: the home's time, for the agents' plan clock.

Covered: an ISO timestamp state is served as `now` in the agents' clock format
(and kept as `iso`); no clock entity configured, or one with no time now, is a
404 -- the agents' clock then keeps its last time and logs once. The route is
called directly over a fake state cache: no HA, no app startup.
"""

import asyncio
from datetime import datetime

import pytest
from fastapi import HTTPException

import hasp as appmod

# The format the agents' clock reads (environment.yaml > clock.http.format,
# `shared/utils/clock.py` DEFAULT_FORMAT). HASP's tests run with HASP's folder
# as the import root, so the agents' package is not imported here.
AGENTS_CLOCK_FORMAT = "%Y-%m-%d %H:%M:%S"


class FakeCache:
    """HASP's state cache, holding only the clock entity's state."""

    def __init__(self, state):
        self.state = state
        # What `_ensure_graph_cache` compares, so it keeps this cache.
        self.ws_client, self.rest_client = appmod.ha_client, appmod.ha_rest
        self.base_uri = appmod.BASE_WS_URI.rstrip("/") + "/"
        self.allowed_workspaces = set(appmod.AREAS)

    def entity_state(self, entity_id):
        return dict(self.state) if entity_id == "sensor.sim_time" else {}


@pytest.fixture()
def read_clock(monkeypatch):
    """GET /_clock over a cache whose `sensor.sim_time` holds the given state."""
    def read(state, entity="sensor.sim_time"):
        monkeypatch.setattr(appmod, "CLOCK_ENTITY", entity)
        monkeypatch.setattr(appmod.app.state, "graph_cache", FakeCache(state), raising=False)
        return asyncio.run(appmod.get_clock())
    return read


def test_an_iso_timestamp_is_the_homes_now(read_clock):
    assert read_clock({"state": "2026-11-27T13:05:30+02:00"}) == {
        "now": "2026-11-27 13:05:30", "iso": "2026-11-27T13:05:30+02:00",
        "entity_id": "sensor.sim_time"}


def test_no_clock_entity_serves_no_clock(read_clock):
    with pytest.raises(HTTPException) as raised:
        read_clock({"state": "2026-11-27T13:05:30+02:00"}, entity="")
    assert raised.value.status_code == 404
    assert raised.value.detail["error"] == "no_clock"


@pytest.mark.parametrize("state", ["unavailable", "unknown", "", None])
def test_a_clock_with_no_time_now_is_a_404(read_clock, state):
    with pytest.raises(HTTPException) as raised:
        read_clock({"state": state})
    assert raised.value.status_code == 404
    assert raised.value.detail["error"] == "clock_unavailable"


def test_the_agents_clock_reads_it(read_clock):
    """The plan clock's HttpClock format parses what /_clock serves."""
    body = read_clock({"state": "2026-11-27T13:05:30+02:00"})
    assert datetime.strptime(body["now"], AGENTS_CLOCK_FORMAT) == datetime(2026, 11, 27, 13, 5, 30)
