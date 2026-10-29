"""The SimuHome bench's own parts (`tests/simuhome/bench`), without LLMs or a
simulator: the tool log, the meter, the nested-behaviour shim, the variant
files, lockstep execution on a fake simulator, and the evaluation checks (run
in a subprocess, as the bench runs them: SimuHome's evaluator imports as `src`)."""

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import py_trees
import pytest

from ami_agents.bt_planning.nodes.http_client import HTTPClient, HTTPResponse
from tests.simuhome.bench import execution
from tests.simuhome.bench.meter import Meter
from tests.simuhome.bench.models import load_variant, variant_from_dict
from tests.simuhome.bench.stubs import add_behaviour
from tests.simuhome.bench.toollog import ToolLog, parse_url

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT.parent / "SimuHome" / "data" / "benchmark"
SHTD = "http://127.0.0.1:9/workspaces/home"


@pytest.fixture(autouse=True)
def clean_blackboard():
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


class TestToolLog:
    def test_urls_map_to_simuhome_tools(self):
        log = ToolLog()
        log.room_devices("kitchen", "goal_context")
        log.read(f"{SHTD}/kitchen/properties/illuminance", None, "state_query")
        log.read(f"{SHTD}/kitchen/artifacts/kitchen_fan_1/properties/onOff", None, "plan")
        log.action(f"{SHTD}/kitchen/artifacts/kitchen_fan_1/actions/onOff", True)
        log.room_devices("kitchen", "candidates")            # repeated: kept once
        assert [(e["tool"], e["params"]) for e in log.tools_invoked()] == [
            ("get_room_devices", {"room_id": "kitchen"}),
            ("get_room_states", {"room_id": "kitchen"}),
            ("get_attribute", {"device_id": "kitchen_fan_1"}),
            ("execute_command", {"device_id": "kitchen_fan_1"})]
        assert "source" not in log.tools_invoked()[0]

    def test_an_ambient_sensor_reading_is_the_rooms_state(self):
        log = ToolLog()
        log.sensors = {"kitchen_light_sensor_1": "kitchen"}
        log.read(f"{SHTD}/kitchen/artifacts/kitchen_light_sensor_1/properties/illuminance",
                 None, "state_query")
        assert [(e["tool"], e["params"]) for e in log.tools_invoked()] == [
            ("get_room_states", {"room_id": "kitchen"})]

    def test_the_bench_discovers_each_rooms_ambient_sensors(self):
        import logging
        from tests.simuhome.bench.stubs import Home
        if not BENCHMARK.is_dir():
            pytest.skip("SimuHome benchmark absent")
        config = json.loads((BENCHMARK / "qt1_feasible_seed_1.json").read_text())
        home = Home("http://127.0.0.1:9", "h", {"rooms": config["initial_home_config"]["rooms"]},
                    {}, logging.getLogger("t"))
        assert "utility_room_light_sensor_1" in home.rooms["utility_room"]
        assert home.sensors["utility_room_light_sensor_1"] == "utility_room"
        assert "utility_room_light_sensor_1" in home.agent.artifacts

    def test_a_capability_lookup_that_finds_nothing_still_looked_in_its_room(self):
        """qt1_infeasible_seed_25: "fan speeds of humidifier 1 in the bathroom",
        with no humidifier there. Answering "there is none" is an answer about
        the bathroom's devices: get_room_devices {bathroom}."""
        import logging
        from tests.simuhome.bench.stubs import Explorer, Home
        if not (BENCHMARK / "qt1_infeasible_seed_25.json").is_file():
            pytest.skip("SimuHome benchmark absent")
        config = json.loads((BENCHMARK / "qt1_infeasible_seed_25.json").read_text())
        home = Home("http://127.0.0.1:9", "h", {"rooms": config["initial_home_config"]["rooms"]},
                    {}, logging.getLogger("t"))
        log = ToolLog()
        explorer = Explorer(home, log)
        response = asyncio.run(explorer.resolve_candidates(None, None, {
            "location_class": "homeont:Bathroom", "device_class": "homeont:Humidifier",
            "device_property": {"class": "homeont:FanControlFanModeSequence"}}))
        assert response["entries"] == []
        assert ("get_room_devices", {"room_id": "bathroom"}) in [
            (e["tool"], e["params"]) for e in log.tools_invoked()]

    def test_parse_url(self):
        assert parse_url(f"{SHTD}/office/properties/air_temperature") == {
            "home": "home", "room": "office", "device": None, "kind": "properties",
            "name": "air_temperature"}
        assert parse_url("http://elsewhere/x") is None


class TestMeter:
    def test_calls_are_booked_to_the_innermost_step(self):
        meter = Meter()

        async def fake_call(step_name):
            # What the wrapper records, booked by the context variable.
            from tests.simuhome.bench import meter as module
            meter.calls.append({"step": module._NAME.get(), "model": "m", "ok": True,
                                "seconds": 0.1, "tokens_in": 10, "tokens_out": 2})

        async def scenario():
            with meter.installed():
                async with meter.step("structuring"):
                    async with meter.step("structuring.intent"):
                        await fake_call("x")
                    await asyncio.gather(fake_call("y"), fake_call("z"))
                with meter.tag("planning.dependency"):
                    await fake_call("w")

        asyncio.run(scenario())
        assert [c["step"] for c in meter.calls] == [
            "structuring.intent", "structuring", "structuring", "planning.dependency"]
        summary = meter.summary()
        assert summary["structuring"]["calls"] == 2 and summary["structuring"]["count"] == 1
        assert summary["structuring.intent"]["tokens_in"] == 10

    def test_the_openai_methods_are_restored(self):
        """Both APIs' `create` are wrapped while a meter is installed, and
        restored after: Chat Completions, and the Responses API the BT planner
        uses on OpenAI's endpoint."""
        from openai.resources.chat.completions import AsyncCompletions
        from openai.resources.responses import AsyncResponses
        # 1. The originals, then wrapped while installed.
        originals = (AsyncCompletions.create, AsyncResponses.create)
        with Meter().installed():
            assert AsyncCompletions.create is not originals[0]
            assert AsyncResponses.create is not originals[1]
        # 2. Restored on exit.
        assert (AsyncCompletions.create, AsyncResponses.create) == originals

    def test_responses_calls_are_booked_with_their_tokens(self):
        """A Responses API call is booked to the step open when it was made,
        with its tokens read under that API's names (input/output_tokens)."""
        from openai.resources.responses import AsyncResponses
        meter = Meter()
        usage = SimpleNamespace(input_tokens=120, output_tokens=30)

        # 1. A fake `create` under the meter's wrapper, called inside a step.
        async def fake_create(self, **kwargs):
            return SimpleNamespace(usage=usage)

        async def scenario():
            original = AsyncResponses.create
            AsyncResponses.create = fake_create
            try:
                with meter.installed():
                    async with meter.step("planning"):
                        await AsyncResponses.create(None, model="gpt-6-luna")
            finally:
                AsyncResponses.create = original

        asyncio.run(scenario())
        # 2. One call, in planning, with both token counts.
        assert meter.calls[0]["step"] == "planning"
        assert (meter.calls[0]["tokens_in"], meter.calls[0]["tokens_out"]) == (120, 30)


class TestShim:
    def test_nested_behaviours_can_be_joined(self):
        from spade.behaviour import OneShotBehaviour

        class Inner(OneShotBehaviour):
            async def run(self):
                await asyncio.sleep(0.01)
                self.done = True

        async def scenario():
            agent = SimpleNamespace()
            inner = Inner()
            add_behaviour(agent, inner)
            await inner.join()
            return inner

        inner = asyncio.run(scenario())
        assert inner.done and inner.agent is not None


class TestVariants:
    def test_defaults_fill_unnamed_steps(self, tmp_path):
        variant = variant_from_dict({"planning": {"explicit": {
            "model": "qwen3:8b", "base_url": "http://localhost:11434/v1"}}}, "local")
        assert variant["planning.explicit"].local
        assert variant["planning.explicit"].call_kwargs() == {"temperature": 0.0}
        assert variant["segmentation"].call_kwargs() == {"reasoning_effort": "low"}
        assert variant["judge"].model == "gpt-4o-mini"

    def test_the_cache_key_follows_only_the_named_steps(self):
        a = variant_from_dict({"planning": {"explicit": {"model": "a"}}})
        b = variant_from_dict({"planning": {"explicit": {"model": "b"}}})
        assert a.key("segmentation", "structuring") == b.key("segmentation", "structuring")
        assert a.key("planning.explicit") != b.key("planning.explicit")

    def test_the_shipped_variant_files_load(self):
        for path in (ROOT / "tests/simuhome/bench/variants").glob("*.yaml"):
            if "sweep" not in path.name:
                assert load_variant(str(path)).name

    def test_profiling_and_guarding_follow_planning_explicit_unless_named(self):
        """As the agents do: both run on the small model (`explicit_llm`)."""
        small = {"model": "qwen3:8b", "base_url": "http://localhost:11434/v1"}
        variant = variant_from_dict({"planning": {"explicit": small}})
        assert variant["profiling"] == variant["guarding"] == variant["planning.explicit"]
        named = variant_from_dict({"planning": {"explicit": small},
                                   "guarding": {"model": "gpt-4o-mini"}})
        assert named["guarding"].model == "gpt-4o-mini"
        assert named["profiling"] == named["planning.explicit"]

    def test_a_provider_names_the_endpoint(self):
        """`provider:` fills base_url / api_key_env; the step's own fields win;
        the built-in "openai" is the client's default (no base_url)."""
        # 1. Built-in and configured providers, and the steps under `models:`.
        variant = variant_from_dict({
            "providers": {"gpu": {"base_url": "http://gpu:8000/v1", "api_key_env": "GPU_KEY"}},
            "models": {"segmentation": {"provider": "local", "model": "qwen3:8b"},
                       "structuring": {"provider": "gpu", "model": "llama"},
                       "response": {"provider": "gpu", "model": "llama",
                                    "base_url": "http://other:9000/v1"},
                       "judge": {"provider": "openai", "model": "gpt-4o-mini"}},
            "run": {"workers": 4}})
        assert variant["segmentation"].base_url == "http://localhost:11434/v1"
        assert variant["segmentation"].local
        assert (variant["structuring"].base_url, variant["structuring"].api_key_env) == \
            ("http://gpu:8000/v1", "GPU_KEY")
        assert variant["response"].base_url == "http://other:9000/v1"
        assert variant["judge"].base_url is None and not variant["judge"].local
        # 2. Naming OpenAI hashes as naming nothing (step caches stay valid).
        assert variant_from_dict({"segmentation": {"provider": "openai", "model": "m"}}).key(
            "segmentation") == variant_from_dict({"segmentation": {"model": "m"}}).key(
            "segmentation")

    def test_an_unknown_provider_is_an_error(self):
        with pytest.raises(ValueError, match="unknown provider 'olama'"):
            variant_from_dict({"segmentation": {"provider": "olama", "model": "m"}})


class FakeSim:
    """A simulator whose clock moves only when told to."""

    def __init__(self, base="2025-08-23 11:00:00", interval=0.1):
        self.base = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
        self.interval = interval
        self.tick = 0
        self.jumps = []

    def home_state(self):
        now = self.base + timedelta(seconds=self.tick * self.interval)
        return {"current_tick": self.tick, "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "rooms": {}}

    def fast_forward_to(self, to_tick, room_ids=None):
        self.tick = max(self.tick, int(to_tick))
        self.jumps.append(self.tick)
        return {"status": {"code": 200}, "data": self.home_state(), "error": None}


DELAYED = {"tree": {"type": "sequence", "name": "G1", "children": [
    {"type": "timer", "name": "wait", "seconds": 120},
    {"type": "action", "name": "fan.onOff",
     "action_url": f"{SHTD}/kitchen/artifacts/kitchen_fan_1/actions/onOff",
     "parameters": {"value": True}}]},
    "runtime": {"reads": [], "derived": [], "clock": True}, "execution_mode": "dependency"}


class TestLockstep:
    def test_an_action_fires_when_due_in_virtual_time(self):
        sim, log = FakeSim(), ToolLog()
        goals = [{"when": {"at_tick": 1200, "tolerance_ticks": 300}}]
        with patch.object(HTTPClient, "post",
                          return_value=HTTPResponse(200, {"ok": True}, {}, "u", 0.01)):
            result = execution.run_lockstep([DELAYED], log, sim, step_seconds=10,
                                            tick_interval=0.1, eval_goals=goals)
        (action,) = result.actions
        assert action["tick"] == 1200 and action["success"]       # due at 120 s, not before
        assert result.plans[0]["status"] == "SUCCESS"
        # Steps of 100 ticks, landing on the window's edges and at_tick.
        assert {900, 1200, 1500} <= set(sim.jumps)
        assert all(b - a <= 100 for a, b in zip([0] + sim.jumps, sim.jumps))
        assert result.horizon_tick == 1500 + 600
        snapshot_ticks = [s["tick"] for s in result.snapshots]
        assert 0 in snapshot_ticks and all(900 <= t <= 1500 for t in snapshot_ticks[1:-1])
        assert [e["tool"] for e in log.tools_invoked()] == ["execute_command"]

    def test_an_infeasible_episode_runs_to_its_horizon(self):
        sim = FakeSim()
        result = execution.run_lockstep([], ToolLog(), sim, step_seconds=10,
                                        tick_interval=0.1,
                                        eval_goals=[{"when": {"at_tick": 600,
                                                              "tolerance_ticks": 0}}])
        assert result.end_tick >= 600 and result.actions == []

    def test_no_jump_while_an_action_is_in_flight(self):
        sim, order = FakeSim(), []

        def slow_post(self, url, payload=None, headers=None):
            import time
            time.sleep(0.2)
            order.append(("post", sim.tick))
            return HTTPResponse(200, {}, {}, url, 0.2)

        immediate = {"tree": {"type": "action", "name": "fan.onOff",
                              "action_url": DELAYED["tree"]["children"][1]["action_url"],
                              "parameters": {"value": True}}}
        original = sim.fast_forward_to

        def jump(to_tick, room_ids=None):
            order.append(("jump", to_tick))
            return original(to_tick, room_ids)

        sim.fast_forward_to = jump
        with patch.object(HTTPClient, "post", slow_post):
            execution.run_lockstep([immediate], ToolLog(), sim, step_seconds=10,
                                   tick_interval=0.1,
                                   eval_goals=[{"when": {"at_tick": 300,
                                                         "tolerance_ticks": 0}}])
        assert order[0] == ("post", 0)                 # answered before the first jump


def _evaluate(tmp_path, record, judge=False):
    path = tmp_path / f"{record['episode']}.json"
    path.write_text(json.dumps(record))
    command = [sys.executable, "-m", "tests.simuhome.bench.evaluate", "--records", str(path)]
    if not judge:
        command.append("--no-judge")
    subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    return json.loads(path.with_name(path.stem + ".eval.json").read_text())


needs_benchmark = pytest.mark.skipif(not BENCHMARK.is_dir(), reason="SimuHome benchmark absent")


@needs_benchmark
class TestEvaluation:
    def _record(self, name, **extra):
        episode = json.loads((BENCHMARK / f"{name}.json").read_text())
        return {"episode": name, "query_type": episode["meta"]["query_type"],
                "case": episode["meta"]["case"], "outcome": "executed", "plans": [],
                "structures": [], "queries": [], "tools_invoked": [], "steps": [],
                "meter": {"calls": []}, **extra}, episode

    def test_qt4_feasible_replays_the_lockstep_states(self, tmp_path):
        record, episode = self._record("qt4-1_feasible_seed_1")
        goals = episode["eval"]["goals"]
        tools = [{"tool": a["tool"], "params": a["params"],
                  "outcome": {"ok": True, "status_code": 200, "error_type": None}}
                 for a in episode["eval"]["required_actions"]]
        start = {"current_tick": 0, "rooms": episode["initial_home_config"]["rooms"]}

        def satisfied(goal):
            rooms = json.loads(json.dumps(episode["initial_home_config"]["rooms"]))
            for target in goal["targets"]:
                for device in rooms[target["room_id"]]["devices"]:
                    if device["device_id"] == target["device_id"]:
                        for check in target["asserts"]:
                            device["attributes"][check["attribute"]] = check["value"]
            return {"current_tick": goal["when"]["at_tick"], "rooms": rooms}

        # Every goal's state recorded at its at_tick: all satisfied.
        snapshots = [{"tick": 0, "state": start}] + [
            {"tick": g["when"]["at_tick"], "state": satisfied(g)} for g in goals]
        verdict = _evaluate(tmp_path, {**record, "tools_invoked": tools,
                                       "snapshots": snapshots})
        assert verdict["score"] == 1, verdict
        # Nothing ever happened: fails.
        verdict = _evaluate(tmp_path, {**record, "tools_invoked": tools,
                                       "snapshots": [{"tick": 0, "state": start}]})
        assert verdict["score"] == 0

    def test_an_infeasible_command_on_a_stand_in_device_fails(self, tmp_path):
        record, episode = self._record("qt3_infeasible_seed_1", outcome="impossible",
                                       plans=[{"outcome": "impossible"}])
        rooms = episode["initial_home_config"]["rooms"]
        home = {d["device_id"] for r in rooms.values() for d in r["devices"]}
        missing = next(g for g in episode["eval"]["goals"] if g["device_id"] not in home)
        prefix = f"{missing['room_id']}_{missing['device_type']}"
        stand_in = next((d for d in home if d.startswith(prefix)), None)
        ok = _evaluate(tmp_path, {**record, "execution": {"actions": []}})
        assert ok["score"] == 1 and ok["execution_check"]["answered_impossible_or_clarify"]
        if stand_in:
            url = f"{SHTD}/{missing['room_id']}/artifacts/{stand_in}/actions/onOff"
            bad = _evaluate(tmp_path, {**record, "execution": {
                "actions": [{"url": url, "success": True}]}})
            assert bad["score"] == 0
            assert bad["error_type"] == "Executed Infeasible Part"

    def test_failed_llm_calls_are_infrastructure(self, tmp_path):
        record, _ = self._record("qt3_infeasible_seed_1", outcome="nothing", meter={
            "calls": [{"step": "segmentation", "ok": False, "seconds": 1.0}]})
        assert _evaluate(tmp_path, record)["score"] == -1


class TestImmediate:
    def test_an_action_whose_guard_then_holds_is_still_recorded(self):
        # selector(already off? else turn off): once the POST has switched the
        # device off, the guard holds and the action is never ticked again --
        # its answer is never consumed, but the home did receive it.
        url = "http://h/workspaces/x/living_room/artifacts/dh_1/"
        state = {"on": True}

        def get(self, target, *args, **kwargs):
            return HTTPResponse(200, state["on"], {}, target, 0.0)

        def post(self, target, payload=None, headers=None):
            if target.endswith("onOff"):
                state["on"] = payload["value"]
                return HTTPResponse(200, {}, {}, target, 0.0)
            return HTTPResponse(422, {}, {}, target, 0.0)

        tree = {"name": "t", "type": "sequence", "children": [
            {"name": "sel", "type": "selector", "children": [
                {"name": "is_off", "type": "condition", "expected_value": False,
                 "operator": "==", "property_url": url + "properties/onOff"},
                {"name": "turn_off", "type": "action", "parameters": {"value": False},
                 "action_url": url + "actions/onOff"}]},
            {"name": "ig", "type": "ignore_failure", "children": [
                {"name": "fan", "type": "action", "parameters": {"value": 0},
                 "action_url": url + "actions/fanMode"}]}]}
        log = ToolLog()
        with patch.object(HTTPClient, "get", get), patch.object(HTTPClient, "post", post):
            result = execution.run_immediate([{"tree": tree}], log, tick_period=0.01,
                                             tick_budget=100, timeout=10)
        assert [(a["name"], a["status_code"], a["success"]) for a in result.actions] == [
            ("turn_off", 200, True), ("fan", 422, False)]
        assert [(e["params"]["device_id"], e["outcome"]["ok"]) for e in log.tools_invoked()
                if e["tool"] == "execute_command"] == [("dh_1", True), ("dh_1", False)]



class TestBreakdown:
    """Each expected value classified: ok, wrong_timing, wrong_value, missing."""

    @staticmethod
    def _record(changes, actions=()):
        key = "fan_1|1.FanControl.PercentSetting"
        return {"timeline": [{"tick": t, "values": {key: v}} for t, v in changes],
                "execution": {"actions": list(actions)}}

    def _classify(self, changes, actions=()):
        from tests.simuhome.bench.evaluate import _timed_check
        # Window 1000..1600 for value 30; the attribute starts at 0.
        return _timed_check(self._record(changes, actions), 0, "fan_1",
                            "1.FanControl.PercentSetting", 30, 1000, 1600)

    def test_reached_in_the_window(self):
        assert self._classify([(0, 0), (1200, 30)])["class"] == "ok"

    def test_held_from_before_the_window(self):
        assert self._classify([(0, 0), (500, 30)])["class"] == "ok"

    def test_reached_after_the_window(self):
        verdict = self._classify([(0, 0), (1700, 30)])
        assert verdict == {"class": "wrong_timing", "reached_at": 1700, "late_by_ticks": 100}

    def test_reached_and_left_before_the_window(self):
        verdict = self._classify([(0, 0), (400, 30), (800, 50)])
        assert verdict["class"] == "wrong_timing" and verdict["early_by_ticks"] == 600

    def test_set_to_something_else(self):
        assert self._classify([(0, 0), (1200, 50)]) == {"class": "wrong_value", "values": [50]}

    def test_never_changed_but_tried(self):
        rejected = {"url": f"{SHTD}/kitchen/artifacts/fan_1/actions/fanSpeed", "success": False}
        assert self._classify([(0, 0)], [rejected]) == {"class": "missing", "attempted": True}

    def test_final_state_checks(self):
        from tests.simuhome.bench.evaluate import _final_check
        record = {"execution": {"actions": []}}
        assert _final_check(record, 0, 1, 1, "d")["class"] == "ok"
        assert _final_check(record, 0, 2, 1, "d")["class"] == "wrong_value"
        assert _final_check(record, 0, 0, 1, "d") == {"class": "missing", "attempted": False}

    def test_a_level_asked_in_percent_accepts_either_reading(self):
        from tests.simuhome.bench.evaluate import _matches, _readings
        episode = {"query": "turn on dimmer light 1 to 60 percent",
                   "initial_home_config": {"rooms": {"r": {"devices": [
                       {"device_id": "d", "attributes": {"1.LevelControl.MaxLevel": 254}}]}}}}
        readings = _readings(episode, "d", "1.LevelControl.CurrentLevel", 60)
        assert readings == {"simuhome": 60, "percent_of_range": 152}
        assert _matches(60, readings) == "simuhome"
        assert _matches(153, readings) == "percent_of_range"      # rounding within 1
        assert _matches(100, readings) is None
        # Stated as a level, it means the level.
        episode["query"] = "set dimmer light 1 to level 60"
        assert _readings(episode, "d", "1.LevelControl.CurrentLevel", 60) == {"simuhome": 60}


class TestQt1DeclaredValues:
    """qt1_feasible_seed_25: "what fan speeds are available on humidifier 1".
    The goal names the fan's FanModeSequence (1 = Off, Low, High): a capability
    answer counts when it lists exactly those values, not the spec's full
    FanMode enum."""

    EPISODE = {"initial_home_config": {"rooms": {"bathroom": {"devices": [
        {"device_id": "bathroom_humidifier_1",
         "attributes": {"1.FanControl.FanModeSequence": 1}}]}}},
        "eval": {"goals": [{"variant": "device_attribute", "room_id": "bathroom",
                            "device_id": "bathroom_humidifier_1",
                            "attribute": "1.FanControl.FanModeSequence"}]}}

    def _check(self, values):
        """The qt1 check on a record whose capability answer lists ``values``."""
        from tests.simuhome.bench import evaluate
        record = {"queries": [{"kind": "capabilities", "response": {"entries": [
            {"artifact_name": "bathroom_humidifier_1", "affordance_name": "fanMode",
             "permitted_values": {"enum": values}}]}}]}
        with patch.object(evaluate, "required_actions", return_value=(True, [])):
            return evaluate.check_qt1(record, self.EPISODE, {}, feasible=True)

    def test_the_devices_own_modes_pass(self):
        assert self._check([0, 1, 3])["score"] == 1

    def test_the_specs_full_enum_is_a_wrong_value(self):
        result = self._check([0, 1, 2, 3, 4, 5, 6])
        assert result["score"] == 0 and result["goals"][0]["read"] is True


class TestQt1StatedFacts:
    """qt1_feasible_seed_34: "what brand is dehumidifier 1?". The vendor is
    stated on the TD, so the answer gives it without a read; it counts when it
    gives the device's own value. A number is never matched in an answer."""

    def _check(self, attribute, value, answer):
        """The qt1 check on a record that read nothing and answered ``answer``."""
        from tests.simuhome.bench import evaluate
        episode = {"initial_home_config": {"rooms": {"utility_room": {"devices": [
            {"device_id": "dehumidifier_1", "attributes": {attribute: value}}]}}},
            "eval": {"goals": [{"variant": "device_attribute", "room_id": "utility_room",
                                "device_id": "dehumidifier_1", "attribute": attribute}]}}
        record = {"queries": [], "response": answer}
        with patch.object(evaluate, "required_actions", return_value=(True, [])):
            return evaluate.check_qt1(record, episode, {}, feasible=True)

    def test_the_vendor_given_in_the_answer_passes(self):
        result = self._check("0.BasicInformation.VendorName", "LG Electronics",
                             "The Dehumidifier in the Utility Room is made by LG Electronics.")
        assert result["score"] == 1 and result["goals"][0]["read"] is False

    def test_another_vendor_fails(self):
        assert self._check("0.BasicInformation.VendorName", "LG Electronics",
                           "It is made by Samsung.")["score"] == 0

    def test_a_number_in_the_answer_is_not_matched(self):
        """"its product ID is 4" could be any 4: numbers need a read."""
        assert self._check("0.BasicInformation.ProductID", 4, "Its ID is 4.")["score"] == 0


class TestQt1RoomState:
    """A room's state is scored as SimuHome's judge scores it: against what the
    agent read, not the episode's value at reset (the simulator runs in real
    time; a purifier takes PM10 from 20 to 0 before any agent reads it)."""

    EPISODE = {"initial_home_config": {"rooms": {"kitchen": {"devices": []}}},
               "eval": {"goals": [{"variant": "room_state", "room_id": "kitchen",
                                   "room_state": "pm10", "current_value": 20.0}]}}

    def _check(self, read, answer):
        """The qt1 check on a record that read ``read`` and answered ``answer``."""
        from tests.simuhome.bench import evaluate
        record = {"ambient_sensors": {"kitchen_air_quality_sensor_1": "kitchen"},
                  "response": answer,
                  "queries": [{"kind": "state", "response": {"affordances": [{
                      "target": "http://h/workspaces/home/kitchen/artifacts/"
                                "kitchen_air_quality_sensor_1/properties/pm10",
                      "value": read}]}}]}
        with patch.object(evaluate, "required_actions", return_value=(True, [])), \
                patch.dict(evaluate.ROOM_STATE_NAMES, {"pm10": "pm10"}):
            return evaluate.check_qt1(record, self.EPISODE, {}, feasible=True)

    def test_the_value_read_and_reported_passes_though_the_room_moved(self):
        assert self._check(0.0, "The PM10 in the Kitchen is 0.0.")["score"] == 1

    def test_a_number_other_than_the_one_read_fails(self):
        assert self._check(0.0, "The PM10 in the Kitchen is 20.0.")["score"] == 0


class TestQt1DeviceFacts:
    """Two device answers that are right without a value read: a dryer's
    levels listed from its SupportedDrynessLevels (seed 9), and a phase the
    dishwasher does not report while stopped, read and found empty (seed 50)."""

    def _check(self, attribute, value, record):
        """The qt1 check of one device attribute against ``record``."""
        from tests.simuhome.bench import evaluate
        episode = {"initial_home_config": {"rooms": {"kitchen": {"devices": [
            {"device_id": "d_1", "attributes": {attribute: value}}]}}},
            "eval": {"goals": [{"variant": "device_attribute", "room_id": "kitchen",
                                "device_id": "d_1", "attribute": attribute}]}}
        with patch.object(evaluate, "required_actions", return_value=(True, [])):
            return evaluate.check_qt1(record, episode, {}, feasible=True)

    def test_supported_levels_listed_exactly_pass(self):
        record = {"queries": [{"kind": "capabilities", "response": {"entries": [
            {"artifact_name": "d_1", "affordance_name": "selectedDrynessLevel",
             "permitted_values": {"enum": [0, 1, 2, 3]}}]}}]}
        result = self._check("1.LaundryDryerControls.SupportedDrynessLevels", [0, 1, 2, 3],
                             record)
        assert result["score"] == 1

    def test_an_empty_attribute_read_and_found_empty_passes(self):
        record = {"queries": [{"kind": "state", "response": {"affordances": [
            {"target": "http://h/workspaces/home/kitchen/artifacts/d_1/properties/currentPhase",
             "value": None}]}}]}
        assert self._check("1.OperationalState.CurrentPhase", None, record)["score"] == 1
        # Not read at all: still a miss.
        assert self._check("1.OperationalState.CurrentPhase", None, {"queries": []})["score"] == 0
