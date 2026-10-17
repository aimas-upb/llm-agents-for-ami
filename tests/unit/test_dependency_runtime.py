"""The pieces dependency plans run on: time words, the clock, the new IR nodes,
and the runtime that keeps their blackboard keys current."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import py_trees
import pytest
from py_trees.common import Status

from ami_agents.agents.interaction_solver.utils.time_text import (
    TimeTextError,
    span_seconds,
    time_of_day,
)
from ami_agents.bt_planning.nodes import registry
from ami_agents.bt_planning.nodes.dependency_nodes import CLOCK_KEY
from ami_agents.bt_planning.nodes.http_client import HTTPResponse
from ami_agents.bt_planning.runtime import DERIVED_OPS, PlanRuntime
from ami_agents.bt_planning.serialization import validate_tree
from ami_agents.shared.utils.clock import (
    ClockError,
    HttpClock,
    SystemClock,
    from_seconds,
    make_clock,
    parse_moment,
    to_seconds,
)

T0 = datetime(2026, 10, 14, 13, 0, 0)


@pytest.fixture(autouse=True)
def clean_blackboard():
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, now=T0):
        self.moment = now

    def now(self):
        return self.moment

    def advance(self, seconds):
        from datetime import timedelta
        self.moment = self.moment + timedelta(seconds=seconds)


def _tick(tree, runtime, times=1):
    for _ in range(times):
        runtime.before_tick()
        tree.tick_once()
    return tree.status


def _compile(spec):
    assert validate_tree(spec) == [], validate_tree(spec)
    tree = registry.compile_node(spec, lambda child: _compile_child(child))
    tree.setup_with_descendants()
    return tree


def _compile_child(spec):
    return registry.compile_node(spec, _compile_child)


class TestTimeText:
    @pytest.mark.parametrize("text, seconds", [
        ("13 minutes", 780), ("1 hour and 30 minutes", 5400), ("half an hour", 1800),
        ("90s", 90), ("for the next 5 hours", 18000), ("two minutes", 120)])
    def test_spans(self, text, seconds):
        assert span_seconds(text) == seconds

    @pytest.mark.parametrize("text, clock", [
        ("1:39 PM", "13:39"), ("13:39", "13:39"), ("7 am", "07:00"), ("noon", "12:00"),
        ("12:15 am", "00:15"), ("at 9:05:30", "09:05:30")])
    def test_times_of_day(self, text, clock):
        assert time_of_day(text) == clock

    def test_words_without_a_time_are_refused(self):
        with pytest.raises(TimeTextError):
            span_seconds("a while")
        with pytest.raises(TimeTextError):
            time_of_day("in 5")


class TestClock:
    def test_the_system_clock_is_the_default(self):
        assert isinstance(make_clock({}), SystemClock)
        assert isinstance(make_clock({"clock": {"source": "system"}}), SystemClock)

    def test_an_http_clock_reads_the_simulators_time(self):
        clock = make_clock({"clock": {"source": "http",
                                      "http": {"url": "http://shtd/_shtd/time"}}})
        body = json.dumps({"now": "2026-10-14 13:00:05"}).encode()
        response = MagicMock(read=MagicMock(return_value=body))
        response.__enter__ = lambda s: s
        response.__exit__ = lambda *a: False
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            assert clock.now() == datetime(2026, 10, 14, 13, 0, 5)
        assert urlopen.call_args[0][0] == "http://shtd/_shtd/time"

    def test_an_unreadable_clock_raises(self):
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            with pytest.raises(ClockError):
                HttpClock("http://shtd/_shtd/time").now()

    def test_times_as_numbers_round_trip(self):
        assert from_seconds(to_seconds(T0)) == T0
        assert parse_moment("2026-10-14 13:00:00") == to_seconds(T0)
        assert parse_moment(12.5) == 12.5 and parse_moment("soon") is None


class TestClockNodes:
    def test_a_timer_runs_on_the_plan_clock_not_the_wall(self):
        clock = FakeClock()
        runtime = PlanRuntime({"clock": True}, clock=clock, synchronous=True)
        tree = _compile({"type": "timer", "name": "t", "seconds": 120})
        assert _tick(tree, runtime) == Status.RUNNING
        clock.advance(119)
        assert _tick(tree, runtime) == Status.RUNNING
        clock.advance(1)
        assert _tick(tree, runtime) == Status.SUCCESS

    def test_wait_until_resolves_today_or_tomorrow(self):
        clock = FakeClock()
        runtime = PlanRuntime({"clock": True}, clock=clock, synchronous=True)
        later = _compile({"type": "wait_until", "name": "w", "time_of_day": "13:02"})
        assert _tick(later, runtime) == Status.RUNNING
        clock.advance(120)
        assert _tick(later, runtime) == Status.SUCCESS
        passed = _compile({"type": "wait_until", "name": "w2", "time_of_day": "12:00"})
        assert _tick(passed, runtime) == Status.RUNNING     # tomorrow noon

    def test_without_a_clock_time_nodes_hold(self):
        tree = _compile({"type": "timer", "name": "t", "seconds": 0})
        runtime = PlanRuntime({"clock": True}, clock=MagicMock(now=MagicMock(
            side_effect=ClockError("down"))), synchronous=True)
        assert _tick(tree, runtime, 3) == Status.RUNNING

    def test_stamp_records_the_clock(self):
        runtime = PlanRuntime({"clock": True}, clock=FakeClock(), synchronous=True)
        tree = _compile({"type": "stamp", "name": "s", "key": "/goals/x/G1/started_at"})
        assert _tick(tree, runtime) == Status.SUCCESS
        reader = py_trees.blackboard.Client(name="r")
        reader.register_key(key="/goals/x/G1/started_at", access=py_trees.common.Access.READ)
        assert reader.get("/goals/x/G1/started_at") == to_seconds(T0)


class TestIRTypes:
    def _writer(self, **values):
        writer = py_trees.blackboard.Client(name="w")
        for key, value in values.items():
            writer.register_key(key=key, access=py_trees.common.Access.WRITE)
            writer.set(key, value)

    def test_check_and_combinations(self):
        self._writer(**{"/a/x": 1, "/a/y": 0})
        one = _compile({"type": "check", "name": "c", "checks": [
            {"key": "/a/x", "op": "eq", "value": 1}]})
        anyof = _compile({"type": "check", "name": "c2", "combine": "or_", "checks": [
            {"key": "/a/x", "op": "eq", "value": 1}, {"key": "/a/y", "op": "eq", "value": 1}]})
        allof = _compile({"type": "check", "name": "c3", "combine": "and_", "checks": [
            {"key": "/a/x", "op": "eq", "value": 1}, {"key": "/a/y", "op": "eq", "value": 1}]})
        for tree, status in ((one, Status.SUCCESS), (anyof, Status.SUCCESS),
                             (allof, Status.FAILURE)):
            tree.tick_once()
            assert tree.status == status

    def test_wait_for_holds_until_true(self):
        tree = _compile({"type": "wait_for", "name": "w", "checks": [
            {"key": "/a/z", "op": "ge", "value": 5}]})
        tree.tick_once()
        assert tree.status == Status.RUNNING              # unset: still waiting
        self._writer(**{"/a/z": 5})
        tree.tick_once()
        assert tree.status == Status.SUCCESS

    def test_compare_two_keys(self):
        self._writer(**{"/a/rem": 100, "/a/dur": 120})
        tree = _compile({"type": "compare", "name": "c", "key1": "/a/rem", "key2": "/a/dur",
                         "op": "le"})
        tree.tick_once()
        assert tree.status == Status.SUCCESS

    def test_decorators_wrap_one_child_in_children(self):
        spec = {"type": "inverter", "name": "i", "children": [
            {"type": "running", "name": "r"}]}
        assert validate_tree(spec) == []
        assert validate_tree({"type": "repeat", "name": "r", "children": []})

    def test_malformed_specs_are_refused(self):
        assert validate_tree({"type": "check", "name": "c", "checks": [
            {"key": "/a", "op": "approx", "value": 1}]})
        assert validate_tree({"type": "wait_until", "name": "w", "time_of_day": "25:00"})
        assert validate_tree({"type": "timer", "name": "t", "seconds": -1})


class TestRuntime:
    def test_reads_refresh_their_keys(self):
        http = MagicMock()
        http.get.side_effect = [HTTPResponse(200, 7, {}, "u", 0.0),
                                HTTPResponse(200, {"remaining": 3}, {}, "u2", 0.0)]
        runtime = PlanRuntime({"reads": [
            {"key": "/artifacts/w/countdown", "url": "u"},
            {"key": "/artifacts/w/state/remaining", "url": "u2", "field": "remaining"}]},
            http_client=http, synchronous=True)
        runtime.before_tick()
        assert runtime._get("/artifacts/w/countdown") == 7
        assert runtime._get("/artifacts/w/state/remaining") == 3

    def test_a_failed_read_keeps_the_last_value(self):
        http = MagicMock()
        http.get.side_effect = [HTTPResponse(200, 7, {}, "u", 0.0),
                                HTTPResponse(500, None, {}, "u", 0.0)]
        runtime = PlanRuntime({"reads": [{"key": "/k", "url": "u"}]}, http_client=http,
                              synchronous=True)
        runtime.before_tick()
        runtime.before_tick()
        assert runtime._get("/k") == 7

    def test_reads_do_not_block_the_tick(self):
        # Asynchronous: the first refresh only submits; a later one writes.
        http = MagicMock()
        http.get.return_value = HTTPResponse(200, 9, {}, "u", 0.0)
        runtime = PlanRuntime({"reads": [{"key": "/k", "url": "u"}]}, http_client=http)
        runtime.before_tick()
        import time
        for _ in range(50):
            time.sleep(0.01)
            runtime.before_tick()
            if runtime._get("/k") == 9:
                break
        assert runtime._get("/k") == 9

    def test_derived_values(self):
        now = to_seconds(T0)
        assert DERIVED_OPS["to_seconds"]([2], {"factor": 60}, None) == 120
        assert DERIVED_OPS["timestamp_minus"](["2026-10-14 13:30:00"], {"seconds": 60},
                                              None) == to_seconds(T0) + 1740
        assert DERIVED_OPS["remaining_from_timestamp"](["2026-10-14 13:01:00"], {}, now) == 60
        assert DERIVED_OPS["remaining_from_duration"]([10, now - 300], {"factor": 60},
                                                      now) == 300
        assert DERIVED_OPS["abs_deviation"]([21.2], {"target": 22}, None) == pytest.approx(0.8)

    def test_derived_keys_wait_for_their_inputs(self):
        runtime = PlanRuntime({"clock": True, "derived": [
            {"key": "/goals/x/G1/remaining", "op": "remaining_from_timestamp",
             "inputs": ["/artifacts/w/endTime"]}]}, clock=FakeClock(), synchronous=True)
        runtime.before_tick()
        assert runtime._get("/goals/x/G1/remaining") is None
        writer = py_trees.blackboard.Client(name="w")
        writer.register_key(key="/artifacts/w/endTime", access=py_trees.common.Access.WRITE)
        writer.set("/artifacts/w/endTime", "2026-10-14 13:05:00")
        runtime.before_tick()
        assert runtime._get("/goals/x/G1/remaining") == 300
