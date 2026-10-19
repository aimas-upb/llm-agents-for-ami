"""The scheduled runner: a dry run of the trees, submitted as SimuHome workflows.

On a small TD home, with a fake simulator and fake SHTD (no server, no LLM):

    washer_1   operational state, countdown, on/off; actions acting upon them
    dryer_1    operational state, countdown
    living     a room temperature reading

    the home model   cycles end when their countdown runs out, a pause freezes
                     them, our actions set what they act upon, anything else is
                     read from the simulator at that moment and reused
    run_scheduled    start-moment actions sent, later ones scheduled as one
                     workflow per (moment, plan); the simulator jumps only to
                     the eval marks, a tick past our actions and cycle ends, and
                     the horizon; plan statuses from the dry run
    SHTD dispatch    the Matter call an action makes, for a step (a benchmark
                     home, built offline)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from rdflib import Graph

from ami_agents.bt_planning.nodes.http_client import HTTPClient, HTTPResponse
from ami_agents.shared.utils.vocabulary import close_types
from tests.simuhome.bench.execution import TIME_FORMAT
from tests.simuhome.bench.scheduling import (
    PAUSED,
    RUNNING,
    STOPPED,
    Cycle,
    DryRunHome,
    VirtualClock,
    _action_names,
    run_scheduled,
)
from tests.simuhome.bench.toollog import ToolLog
from tests.unit.test_dependency_compiler import PREFIXES, _prop, _room
from tests.unit.test_planning_conditions import _action, _device

HOME = PREFIXES + "".join([
    _room("utility", "Utility Room", "UtilityRoom", ["washer_1", "dryer_1"]),
    _room("living", "Living Room", "LivingRoom", ["thermo"]),
    _device("washer_1", "LaundryWasher"),
    _prop("washer_1", "operationalState", "LaundryWasherOperationalState", "js:IntegerSchema"),
    _prop("washer_1", "countdownTime", "RemainingTime", "js:IntegerSchema"),
    _prop("washer_1", "onOff", "LaundryWasherOnOff", "js:BooleanSchema"),
    _action("washer_1", "operationalState", "homeont:SetOperationalStateCommand",
            acts_upon="operationalState"),
    _action("washer_1", "onOff", "homeont:SetOnOffCommand", acts_upon="onOff",
            schema="js:BooleanSchema"),
    _device("dryer_1", "LaundryDryer"),
    _prop("dryer_1", "operationalState", "LaundryDryerOperationalState", "js:IntegerSchema"),
    _prop("dryer_1", "countdownTime", "RemainingTime", "js:IntegerSchema"),
    _device("thermo", "TemperatureSensor"),
    _prop("thermo", "temperature", "AirTemperature", "js:NumberSchema"),
])

P = "http://h/{}/p/{}".format
A = "http://h/{}/a/{}".format
START = datetime(2025, 8, 23, 13, 26, 31)


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


class Reader:
    """The simulator's values for the home model: a function of (url, seconds)."""

    def __init__(self, values):
        self.values, self.calls = values, []

    def __call__(self, url, seconds):
        self.calls.append((url, seconds))
        value = self.values.get(url)
        return value(seconds) if callable(value) else value


def _home(graph, values, fresh=10.0):
    clock = VirtualClock(START)
    reader = Reader(values)
    return DryRunHome(graph, clock, reader, fresh), clock, reader


# --------------------------------------------------------------------------
# The home model
# --------------------------------------------------------------------------

class TestCycles:
    def test_a_running_cycle_ends_when_its_countdown_runs_out(self):
        cycle = Cycle(state_url="s", countdown_url="c", state=RUNNING, remaining=600.0,
                      as_of=0.0, known=True)
        cycle.at(599)
        assert cycle.state == RUNNING and cycle.remaining == 1
        cycle.at(600)
        assert cycle.state == STOPPED and cycle.remaining is None and cycle.ends == [600]

    def test_the_dryer_is_read_once_then_runs_on_its_countdown(self, graph):
        home, clock, reader = _home(graph, {P("dryer_1", "operationalState"): 1,
                                            P("dryer_1", "countdownTime"): 600})
        assert home.read(P("dryer_1", "operationalState")) == RUNNING
        clock.seconds = 599
        assert home.read(P("dryer_1", "operationalState")) == RUNNING
        assert home.read(P("dryer_1", "countdownTime")) == 1
        clock.seconds = 600
        assert home.read(P("dryer_1", "operationalState")) == STOPPED
        assert len(reader.calls) == 2 and home.cycle_ends() == [600]

    def test_a_pause_freezes_the_countdown_and_a_resume_continues_it(self, graph):
        home, clock, _ = _home(graph, {P("washer_1", "operationalState"): 1,
                                       P("washer_1", "countdownTime"): 600})
        home.read(P("washer_1", "operationalState"))
        clock.seconds = 100
        home.write(A("washer_1", "operationalState"), {"value": PAUSED})
        clock.seconds = 300
        assert home.read(P("washer_1", "operationalState")) == PAUSED
        home.write(A("washer_1", "operationalState"), {"value": RUNNING})
        clock.seconds = 799
        assert home.read(P("washer_1", "operationalState")) == RUNNING
        clock.seconds = 800                       # 100 run + 200 paused + 500 run
        assert home.read(P("washer_1", "operationalState")) == STOPPED

    def test_a_cycle_we_start_is_read_from_the_simulator_once(self, graph):
        """Its length is the device's: read once it has started, then counted."""
        home, clock, reader = _home(graph, {
            P("washer_1", "operationalState"): lambda s: 0 if s < 10 else 1,
            P("washer_1", "countdownTime"): lambda s: None if s < 10 else 300})
        home.read(P("washer_1", "operationalState"))
        clock.seconds = 10
        home.write(A("washer_1", "operationalState"), {"value": RUNNING})
        clock.seconds = 20
        assert home.read(P("washer_1", "countdownTime")) == 300
        clock.seconds = 320
        assert home.read(P("washer_1", "operationalState")) == STOPPED
        assert len(reader.calls) == 4               # twice at 0, twice at 20


class TestOtherValues:
    def test_an_action_sets_what_it_acts_upon(self, graph):
        home, _, reader = _home(graph, {})
        home.write(A("washer_1", "onOff"), {"value": True})
        assert home.read(P("washer_1", "onOff")) is True and reader.calls == []

    def test_a_reading_comes_from_the_simulator_and_is_reused_while_fresh(self, graph):
        home, clock, reader = _home(graph, {P("thermo", "temperature"): lambda s: 20 + s / 100})
        assert home.read(P("thermo", "temperature")) == 20
        clock.seconds = 9
        assert home.read(P("thermo", "temperature")) == 20        # reused
        clock.seconds = 10
        assert home.read(P("thermo", "temperature")) == 20.1      # read again
        assert [s for _, s in reader.calls] == [0, 10]


# --------------------------------------------------------------------------
# The whole run
# --------------------------------------------------------------------------

class FakeSim:
    """The simulator: a clock that jumps, and the workflows scheduled on it."""

    def __init__(self):
        self.tick, self.jumps, self.workflows = 0, [], []

    def home_state(self):
        moment = START + timedelta(seconds=self.tick * 0.1)
        return {"current_tick": self.tick, "current_time": moment.strftime(TIME_FORMAT),
                "rooms": {}}

    def fast_forward_to(self, tick):
        self.tick = int(tick)
        self.jumps.append(self.tick)
        return {"data": self.home_state()}

    def schedule_workflow(self, start_time, steps, description=None):
        self.workflows.append({"start_time": start_time, "steps": steps})
        return {"workflow_id": f"wf{len(self.workflows)}"}

    def workflow_list(self):
        return [{"workflow_id": f"wf{i + 1}", "status": "completed"}
                for i in range(len(self.workflows))]


def _seq(*children):
    return {"type": "sequence", "name": "s", "memory": True, "children": list(children)}


def _act(device, name, value):
    return {"type": "action", "name": f"{device}.{name}", "action_url": A(device, name),
            "parameters": {"value": value}}


DRYER_KEY = "/artifacts/dryer_1/operationalState"
PLANS = [
    # 1. Now: switch the washer on and start it.
    {"tree": _seq(_act("washer_1", "onOff", True), _act("washer_1", "operationalState", 1))},
    # 2. When the dryer finishes: pause the washer.
    {"tree": _seq({"type": "wait_for", "name": "P1",
                   "checks": [{"key": DRYER_KEY, "op": "eq", "value": 0}]},
                  _act("washer_1", "operationalState", 2)),
     "runtime": {"reads": [{"key": DRYER_KEY, "url": P("dryer_1", "operationalState"),
                            "field": None}], "derived": [], "clock": True},
     "execution_mode": "dependency"},
    # 3. In two minutes: switch the washer off.
    {"tree": _seq({"type": "timer", "name": "2 min", "seconds": 120.0},
                  _act("washer_1", "onOff", False)),
     "runtime": {"reads": [], "derived": [], "clock": True}, "execution_mode": "dependency"},
]
SIM_VALUES = {P("dryer_1", "operationalState"): 1, P("dryer_1", "countdownTime"): 600,
              P("washer_1", "operationalState"): 0, P("washer_1", "countdownTime"): None,
              P("washer_1", "onOff"): False}
GOALS = [{"when": {"at_tick": 6000, "tolerance_ticks": 300}, "targets": []}]


@pytest.fixture
def run(monkeypatch):
    """run_scheduled on PLANS, with SHTD faked behind HTTPClient: reads from
    SIM_VALUES (the washer running after its start), actions acknowledged,
    /dispatch echoing what it was asked."""
    posts = []

    def get(client, url, headers=None):
        value = SIM_VALUES.get(url)
        if url == P("washer_1", "operationalState") and posts:
            value = 1
        if url == P("washer_1", "countdownTime") and posts:
            value = 1800
        return HTTPResponse(200, value, {}, url, 0.0)

    def post(client, url, payload=None, headers=None):
        posts.append((url, payload))
        body = ({"tool": "execute_command", "args": {"url": url, **(payload or {})}}
                if url.endswith("/dispatch") else {"ok": True})
        return HTTPResponse(200, body, {}, url, 0.0)

    monkeypatch.setattr(HTTPClient, "get", get)
    monkeypatch.setattr(HTTPClient, "post", post)
    sim = FakeSim()
    execution = run_scheduled(PLANS, ToolLog(), sim, turtle=HOME, tick_interval=0.1,
                              eval_goals=GOALS)
    return execution, sim, posts


class TestRunScheduled:
    def test_every_plan_concludes_and_its_actions_land_at_their_moments(self, run):
        execution, _, _ = run
        assert [p["status"] for p in execution.plans] == ["SUCCESS"] * 3
        when = [(a["plan"], a["name"], a["tick"], a["how"]) for a in execution.actions]
        assert when == [(0, "washer_1.onOff", 0, "sent"),
                        (0, "washer_1.operationalState", 0, "sent"),
                        (2, "washer_1.onOff", 1200, "scheduled"),
                        (1, "washer_1.operationalState", 6000, "scheduled")]
        assert all(a["success"] for a in execution.actions)

    def test_later_actions_become_workflows_for_the_state_predicted_then(self, run):
        """The pause is dispatched for a running washer (Pause, not Resume)."""
        execution, sim, _ = run
        assert [w["start_time"] for w in sim.workflows] == ["2025-08-23 13:28:31",
                                                            "2025-08-23 13:36:31"]
        pause = sim.workflows[1]["steps"][0]["args"]
        assert pause["value"] == 2 and pause["current"] == {"operational_state": RUNNING}
        assert len(execution.details["workflows"]) == 2

    def test_the_simulator_jumps_only_where_something_is_looked_at(self, run):
        """Eval marks, a tick past each action and cycle end, the horizon --
        not every 10 seconds of the 44-minute wait."""
        execution, sim, _ = run
        assert sim.jumps == [1, 1201, 5700, 6000, 6001, 6300, 6900]
        assert execution.end_tick == 6900 and execution.snapshots[0]["tick"] == 0


def test_two_actions_on_one_url_keep_their_own_names():
    """A guard's pause and a goal's start of one cycle share a URL: each is
    named by what it sends (seed 46's start was logged as the pause)."""
    import py_trees
    from types import SimpleNamespace

    def action(name, value):
        node = py_trees.behaviours.Success(name=name)
        node.action_url, node.parameters = "http://h/dryer/a/operationalState", {"value": value}
        return node

    root = py_trees.composites.Sequence("s", memory=True,
                                        children=[action("guard pause", 2), action("start", 1)])
    names = _action_names([SimpleNamespace(index=0, tree=root)])
    url = "http://h/dryer/a/operationalState"
    assert names[(0, url, '{"value": 1}')] == "start"
    assert names[(0, url, '{"value": 2}')] == "guard pause"
    assert names[(0, url)] == "guard pause"           # parameters filled in at run time


# --------------------------------------------------------------------------
# SHTD's dispatch
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
EPISODE = REPO.parent / "SimuHome" / "data" / "benchmark" / "qt4-3_feasible_seed_45.json"


@pytest.mark.skipif(not EPISODE.is_file(), reason="SimuHome benchmark corpus not available")
class TestDispatch:
    @pytest.fixture(scope="class")
    def td(self):
        from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
        config = json.loads(EPISODE.read_text())["initial_home_config"]
        return SimuHomeTD("http://localhost:8097", "qt4-3_feasible_seed_45", config)

    def test_a_mode_is_a_change_to_mode_command(self, td):
        from ami_agents.environment.integration.SimuHome.shtd import _dispatch_call
        call = _dispatch_call(td, "utility_room", "utility_room_laundry_washer_1",
                              "currentMode", 2)
        assert call == {"tool": "execute_command", "args": {
            "device_id": "utility_room_laundry_washer_1", "endpoint_id": 1,
            "cluster_id": "LaundryWasherMode", "command_id": "ChangeToMode",
            "args": {"new_mode": 2}}}

    def test_running_is_start_or_resume_by_the_state_it_is_sent_from(self, td):
        from ami_agents.environment.integration.SimuHome.shtd import _dispatch_call
        washer = "utility_room_laundry_washer_1"
        start = _dispatch_call(td, "utility_room", washer, "operationalState", 1, STOPPED)
        resume = _dispatch_call(td, "utility_room", washer, "operationalState", 1, PAUSED)
        assert start["args"]["command_id"] == "Start"
        assert resume["args"]["command_id"] == "Resume"


class DriftingSim(FakeSim):
    """A simulator whose clock ran on by itself to tick 1500 before the first
    submission (it advances in real time between our calls)."""

    def home_state(self):
        if self.tick == 0 and getattr(self, "_drifted", False):
            self.tick = 1500
        self._drifted = True
        return super().home_state()


def test_an_action_the_simulator_has_passed_is_sent_not_scheduled(monkeypatch):
    """A workflow must start after the simulator's moment: the switch-off due at
    tick 1200 is sent once the simulator is found at 1500."""
    def get(client, url, headers=None):
        return HTTPResponse(200, SIM_VALUES.get(url), {}, url, 0.0)

    def post(client, url, payload=None, headers=None):
        return HTTPResponse(200, {"tool": "execute_command", "args": {}}
                            if url.endswith("/dispatch") else {"ok": True}, {}, url, 0.0)

    monkeypatch.setattr(HTTPClient, "get", get)
    monkeypatch.setattr(HTTPClient, "post", post)
    sim = DriftingSim()
    execution = run_scheduled(PLANS[2:], ToolLog(), sim, turtle=HOME, tick_interval=0.1,
                              eval_goals=[])
    (switch_off,) = execution.actions
    assert switch_off["how"] == "sent" and sim.workflows == []
