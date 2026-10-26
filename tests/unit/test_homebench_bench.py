"""HomeBench bench wiring (P3): suites, the generalised run / harness / sweep /
stats, the guidelines choice and the worker lifecycle.

No LLM is called. Most tests build a tiny converter output in ``tmp_path``; the
ones marked "real data" use ``tests/homebench/data`` (the converter's output, skipped
when it has not been generated) and the lifecycle tests start a real engine and HBTD
on free ports. The groups, and what each pins:

* suites: the registry, SimuHome's values unchanged, HomeBench's names / estimate /
  home grouping, the evaluators (feasible reuses qt3);
* ``run.py`` helpers: selection per suite, episodes kept together by home, the
  guidelines choice written into the agents' config, the CLI's refusals;
* the gate "``PLANNING_GUIDELINES_PATH`` reaches the ISA's guard": by the environment
  variable, by ``run.guidelines``, and the guard itself switching on a ``DimmableLight``
  with the repo's file and staying off with HomeBench's;
* ``stubs.Home`` with HomeBench's TD builder;
* ``harness.py`` / ``sweep.py``: commands, estimates, the three configs;
* ``stats.py`` / ``report.py``: the ``mixed`` case and ``hbN`` types;
* the worker lifecycle gate: two episodes in one worker.
"""

import asyncio
import http.server
import json
import logging
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ami_agents.agents.interaction_solver.behaviours.condition_guard import (
    ConditionGuardBehaviour,
)
from ami_agents.agents.interaction_solver.behaviours.power_on_guard import (
    PowerOnGuardBehaviour,
)
from ami_agents.agents.interaction_solver.utils.planning_conditions import load_conditions
from ami_agents.agents.interaction_solver.utils.power_on_guard import (
    guidelines_path, load_applies_to,
)
from ami_agents.agents.interaction_solver.utils.specific_capability_context import (
    DIRECT_VALUE_KEY,
)
from ami_agents.environment.integration.HomeBench.engine import HomeBenchEngine
from ami_agents.environment.integration.HomeBench.td_builder import HomeBenchTD
from ami_agents.environment.integration.HomeBench.worker import HomeBenchWorker
from ami_agents.shared.utils.config_loader import ConfigLoader
from tests.homebench import convert
from tests.simuhome.bench import evaluate, harness, report, run, stats, sweep
from tests.simuhome.bench.models import load_variant
from tests.simuhome.bench.stubs import Home, isa_agent
from tests.simuhome.bench.suites import SUITE_NAMES, get_suite
from tests.simuhome.bench.suites.homebench import GUIDELINES, HomeBenchEvaluators

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "homebench" / "data"
HAVE_DATA = (DATA / "manifest.json").is_file()
SIMUHOME_CHECKOUT = ROOT.parent / "SimuHome"
VARIANTS = ROOT / "tests" / "homebench" / "variants"
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run tests/homebench/convert.py first")

HOMEBENCH = get_suite("homebench")
SIMUHOME = get_suite("simuhome")


# -- fixtures ----------------------------------------------------------------------------------


def _num(value, low=0, high=100):
    """A HomeBench numeric attribute block."""
    return {"value": value, "lowest": str(low), "highest": str(high)}


def raw_home():
    """A small HomeBench home: a colour light (a DimmableLight) and an air conditioner
    in the living room, a light in the kitchen, and the room-less vacuum robot."""
    return {
        "living_room": {"room_name": "living_room",
                        "light": {"state": "off", "attributes": {
                            "brightness": _num(40), "color": {"value": [1, 2, 3]}}},
                        "air_conditioner": {"state": "off", "attributes": {
                            "temperature": _num(24, 16, 30)}}},
        "kitchen": {"room_name": "kitchen",
                    "light": {"state": "on", "attributes": {"brightness": _num(10)}}},
        "VacuumRobot": {"state": "on", "attributes": {"battery": _num(80)}},
    }


@pytest.fixture()
def home_config():
    """The small home as an ``initial_home_config``."""
    return convert.convert_home({"home_id": 7, "home_status": raw_home()})


@pytest.fixture()
def bench_dir(tmp_path):
    """A tiny converter output: two homes, six episodes (homes 1, 2, 1, 2, 1, 2)."""
    (tmp_path / "homes").mkdir()
    (tmp_path / "episodes").mkdir()
    for home_id in (1, 2):
        (tmp_path / "homes" / f"home{home_id}.json").write_text(
            json.dumps({"rooms": {}, "tick_interval": 0.1, "base_time": "t"}), encoding="utf-8")
    placed = [("hb1_feasible_seed_1", 1), ("hb1_feasible_seed_2", 2),
              ("hb1_feasible_seed_3", 1), ("hb2_mixed_seed_1", 2),
              ("hb2_infeasible_seed_1", 1), ("hb10_feasible_seed_1", 2)]
    for name, home_id in placed:
        (tmp_path / "episodes" / f"{name}.json").write_text(
            json.dumps({"meta": {"query_type": name.split("_")[0]}, "query": "q",
                        "home_id": home_id}), encoding="utf-8")
    return tmp_path


def home_of(config, logger=None, builder_cls=HomeBenchTD):
    """A ``stubs.Home`` over the engine's state for ``config``."""
    state = HomeBenchEngine(config).home_state()
    return Home("http://hb.test", "home7", state, {}, logger or logging.getLogger("t"),
                builder_cls=builder_cls)


# -- suites ------------------------------------------------------------------------------------


class TestSuites:
    """The registry, and what each suite carries."""

    def test_registry_names_and_default(self):
        """Both suites are known; no name means SimuHome."""
        assert SUITE_NAMES == ("simuhome", "homebench")
        assert get_suite().name == "simuhome" == get_suite(None).name
        assert get_suite("homebench").name == "homebench"

    def test_unknown_suite_is_an_error(self):
        """A typo never silently runs SimuHome."""
        with pytest.raises(ValueError, match="unknown suite 'nope'"):
            get_suite("nope")

    def test_classes_resolve_by_import_path(self):
        """The worker, client and TD builder classes are the suite's own."""
        assert [SIMUHOME.worker_cls.__name__, SIMUHOME.client_cls.__name__,
                SIMUHOME.td_cls.__name__] == ["SimuHomeWorker", "SimuHomeClient", "SimuHomeTD"]
        assert [HOMEBENCH.worker_cls.__name__, HOMEBENCH.client_cls.__name__,
                HOMEBENCH.td_cls.__name__] == ["HomeBenchWorker", "HomeBenchClient",
                                               "HomeBenchTD"]

    def test_simuhome_values_are_what_the_bench_hard_coded(self):
        """The SimuHome suite is the old behaviour: qt4 timed, 3-6 calls, scheduled."""
        assert SIMUHOME.timed("qt4-1") and SIMUHOME.timed("qt4-3")
        assert not SIMUHOME.timed("qt3") and not SIMUHOME.timed("qt1")
        assert SIMUHOME.estimate("qt3_feasible_seed_1") == (3, 6, 0)
        assert SIMUHOME.estimate("qt3_infeasible_seed_1") == (3, 6, 3)
        assert SIMUHOME.scheduled and SIMUHOME.default_episodes == "qt*"
        assert "SimuHomeError" in SIMUHOME.infra_markers
        assert SIMUHOME.guidelines_path(None) is None
        assert SIMUHOME.default_benchmark == ROOT.parent / "SimuHome" / "data" / "benchmark"

    def test_simuhome_names_and_load(self, tmp_path):
        """SimuHome's episodes are the qt*_seed_*.json files, read as plain JSON."""
        (tmp_path / "qt3_feasible_seed_2.json").write_text('{"a": 1}', encoding="utf-8")
        (tmp_path / "qt1_feasible_seed_1.json").write_text('{"a": 2}', encoding="utf-8")
        (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
        assert SIMUHOME.names(tmp_path) == ["qt1_feasible_seed_1", "qt3_feasible_seed_2"]
        assert SIMUHOME.load_episode(tmp_path, "qt3_feasible_seed_2") == {"a": 1}
        assert SIMUHOME.load_episode(tmp_path, "qt3_feasible_seed_2.json") == {"a": 1}

    @pytest.mark.parametrize("name,expected", [
        ("hb1_feasible_seed_1", (2, 2, 0)),
        ("hb1_infeasible_seed_4", (2, 2, 3)),
        ("hb3_mixed_seed_2", (4, 4, 3)),
        ("hb10_feasible_seed_1", (11, 11, 0)),
    ])
    def test_homebench_estimate_is_one_segmentation_plus_one_call_per_part(self, name,
                                                                          expected):
        """(1 + N) calls, exactly; judge calls only when a part cannot be done."""
        assert HOMEBENCH.estimate(name) == expected

    def test_homebench_estimate_rejects_a_foreign_name(self):
        """A SimuHome name is not a HomeBench episode."""
        with pytest.raises(ValueError, match="not a HomeBench episode name"):
            HOMEBENCH.estimate("qt3_feasible_seed_1")

    def test_homebench_names_home_key_and_load(self, bench_dir):
        """Episodes come from episodes/, with the home id read from the file."""
        names = HOMEBENCH.names(bench_dir)
        assert names == sorted(names) and len(names) == 6
        assert HOMEBENCH.home_key(bench_dir, "hb1_feasible_seed_2") == "home2"
        assert HOMEBENCH.home_key(bench_dir, "hb1_feasible_seed_1.json") == "home1"
        episode = HOMEBENCH.load_episode(bench_dir, "hb2_mixed_seed_1")
        assert episode["home_id"] == 2
        assert episode["initial_home_config"]["base_time"] == "t"     # the home merged in

    def test_homebench_is_untimed_and_not_scheduled(self):
        """No HomeBench type is timed, and scheduled execution is SimuHome's."""
        assert not any(HOMEBENCH.timed(f"hb{n}") for n in range(1, 11))
        assert not HOMEBENCH.scheduled and HOMEBENCH.default_episodes == "hb*"
        assert HOMEBENCH.judged_label != SIMUHOME.judged_label

    def test_guidelines_choices(self):
        """HomeBench defaults to its own file; "default" is the agents' file; a
        choice the suite does not offer is an error."""
        assert HOMEBENCH.guidelines_path(None) == GUIDELINES == HOMEBENCH.guidelines_path(
            "homebench")
        assert GUIDELINES.is_file()
        assert HOMEBENCH.guidelines_path("default") is None
        with pytest.raises(ValueError, match="no guidelines 'bogus'"):
            HOMEBENCH.guidelines_path("bogus")
        with pytest.raises(ValueError, match="no guidelines 'homebench'"):
            SIMUHOME.guidelines_path("homebench")

    def test_homebench_guidelines_file_has_no_conditions(self):
        """The file parses and lists nothing, so neither guard has anything to apply."""
        data = yaml.safe_load(GUIDELINES.read_text(encoding="utf-8"))
        assert data == {"conditions": []}


class TestEvaluators:
    """The HomeBench scoring registry (the rules themselves are tested in
    test_homebench_scoring.py)."""

    def test_foreign_keys_are_refused(self):
        """A SimuHome type, or a case HomeBench lacks, is a KeyError, not a guess."""
        registry = HomeBenchEvaluators()
        for key in (("qt3", "feasible"), ("hb3", "bogus"), ("hbx", "feasible")):
            with pytest.raises(KeyError):
                registry[key]

    def test_feasible_is_simuhomes_qt3_evaluator(self):
        """hbN feasible reuses SimuHome's qt3 module (plan 3.7). In a subprocess:
        SimuHome's evaluator imports as the top-level `src`."""
        if not SIMUHOME_CHECKOUT.is_dir():
            pytest.skip("SimuHome checkout not found")
        code = ("from tests.simuhome.bench.suites import get_suite\n"
                "from tests.simuhome.bench.evaluate import simuhome_evaluators\n"
                "h = get_suite('homebench').evaluators()\n"
                "s = simuhome_evaluators()\n"
                "print(all(h[(f'hb{n}', 'feasible')] is s[('qt3', 'feasible')]"
                " for n in (1, 2, 10)))\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        assert result.stdout.strip().splitlines()[-1] == "True", result.stderr

    def test_infra_markers_are_a_parameter(self):
        """An error text is infrastructure only when it holds one of the markers given;
        SimuHome's list is the default."""
        record = {"error": "SimuHomeError: 502", "meter": {"calls": []},
                  "query_type": "hb1", "case": "feasible"}
        assert evaluate.infra_error(record) == "SimuHomeError: 502"
        assert evaluate.infra_error(record, ("ConnectError",)) is None
        assert evaluate.infra_error({"error": "x ConnectError y", "meter": {}},
                                    ("ConnectError",))
        verdict = evaluate.evaluate_record(record, {}, {}, [], None, ("SimuHomeError",))
        assert verdict["error_type"] == "Infrastructure"

    def test_breakdown_checks_the_final_state_for_homebench_types(self):
        """An hbN feasible goal is checked on the final state, as qt3's are."""
        def device(value):
            return {"rooms": {"kitchen": {"devices": [
                {"device_id": "kitchen_light", "attributes": {"brightness": value}}]}}}
        episode = {"query": "set the kitchen light to 20", "initial_home_config": device(10),
                   "eval": {"goals": [{"room_id": "kitchen", "device_id": "kitchen_light",
                                       "device_type": "light",
                                       "asserts": [{"attribute": "brightness", "value": 20,
                                                    "description": "d"}]}]}}
        classes = {}
        for final in (20, 15, 10):
            record = {"query_type": "hb1", "final_home_state": device(final),
                      "execution": {"actions": []}}
            checks = evaluate.breakdown(record, episode)["checks"]
            classes[final] = checks[0]["class"]
        assert classes == {20: "ok", 15: "wrong_value", 10: "missing"}


# -- run.py helpers ----------------------------------------------------------------------------


class TestRunHelpers:
    """Selection per suite, grouping by home, the guidelines choice, the CLI's refusals."""

    def test_select_episodes_for_homebench(self, bench_dir):
        """Patterns, seeds and the per-(type, case) limit work on HomeBench names, and
        hb2_* does not catch hb10."""
        everything = run.select_episodes(bench_dir, "hb*", None, None, HOMEBENCH)
        assert len(everything) == 6
        assert run.select_episodes(bench_dir, "hb2_*", None, None, HOMEBENCH) == [
            "hb2_infeasible_seed_1", "hb2_mixed_seed_1"]
        assert run.select_episodes(bench_dir, "hb1_feasible", None, 2, HOMEBENCH) == [
            "hb1_feasible_seed_1", "hb1_feasible_seed_2"]
        assert run.select_episodes(bench_dir, "hb1_*", "3", None, HOMEBENCH) == [
            "hb1_feasible_seed_3"]
        assert run.select_episodes(bench_dir, "qt*", None, None, HOMEBENCH) == []

    def test_select_episodes_defaults_to_simuhome(self, tmp_path):
        """No suite given: SimuHome's qt names, as before."""
        for name in ("qt3_feasible_seed_1", "qt3_feasible_seed_2", "qt1_infeasible_seed_1"):
            (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
        assert run.select_episodes(tmp_path, "qt3_feasible", None, 1) == ["qt3_feasible_seed_1"]
        assert len(run.select_episodes(tmp_path, "qt*", None, None)) == 3

    def test_group_by_home_keeps_a_homes_episodes_together(self, bench_dir):
        """Homes in order of first appearance, episodes in their order within a home."""
        names = HOMEBENCH.names(bench_dir)
        grouped = run.group_by_home(names, HOMEBENCH, bench_dir)
        homes = [HOMEBENCH.home_key(bench_dir, n) for n in grouped]
        first = HOMEBENCH.home_key(bench_dir, names[0])          # homes by first appearance
        other = "home2" if first == "home1" else "home1"
        assert sorted(grouped) == sorted(names)
        assert homes == sorted(homes, key=homes.index)           # no home reappears
        assert homes == [first] * 3 + [other] * 3
        assert [n for n in grouped if HOMEBENCH.home_key(bench_dir, n) == "home1"] == [
            n for n in names if HOMEBENCH.home_key(bench_dir, n) == "home1"]

    def test_group_by_home_is_a_no_op_for_simuhome(self, tmp_path):
        """SimuHome has one home per episode: the order is left alone."""
        names = ["qt3_feasible_seed_1", "qt1_feasible_seed_1"]
        assert run.group_by_home(names, SIMUHOME, tmp_path) == names

    def test_split_for_workers_gives_each_home_to_one_worker(self, bench_dir):
        """Whole homes per worker; no empty worker is started."""
        names = HOMEBENCH.names(bench_dir)
        chunks = run.split_for_workers(names, 2, HOMEBENCH, bench_dir)
        assert len(chunks) == 2 and sorted(n for c in chunks for n in c) == sorted(names)
        for chunk in chunks:
            assert len({HOMEBENCH.home_key(bench_dir, n) for n in chunk}) == 1
        assert len(run.split_for_workers(names, 5, HOMEBENCH, bench_dir)) == 2   # two homes

    def test_split_for_workers_is_round_robin_for_simuhome(self, tmp_path):
        """SimuHome's split is the old one, `names[i::workers]`."""
        names = [f"qt3_feasible_seed_{i}" for i in range(1, 6)]
        assert run.split_for_workers(names, 2, SIMUHOME, tmp_path) == [names[0::2], names[1::2]]

    def test_apply_guidelines_sets_the_path_the_guard_reads(self):
        """The choice lands at planning.guidelines.path of the ISA's section, and
        None leaves the config alone."""
        config = {"interaction_solver": {"planning": {"guidelines": {"path": ""}}}}
        run.apply_guidelines(config, None)
        assert config["interaction_solver"]["planning"]["guidelines"]["path"] == ""
        run.apply_guidelines(config, GUIDELINES)
        assert guidelines_path(config["interaction_solver"]) == str(GUIDELINES)
        bare = {}
        run.apply_guidelines(bare, GUIDELINES)                   # sections created on demand
        assert guidelines_path(bare["interaction_solver"]) == str(GUIDELINES)

    def test_scheduled_execution_is_refused_for_homebench(self, monkeypatch, capsys):
        """`--execution scheduled` is SimuHome's: a parser error, before anything runs."""
        monkeypatch.setattr(sys, "argv", ["run", "--suite", "homebench", "--execution",
                                          "scheduled"])
        with pytest.raises(SystemExit) as stop:
            run.main()
        assert stop.value.code == 2
        assert "SimuHome's" in capsys.readouterr().err

    def test_unknown_guidelines_choice_is_refused(self, monkeypatch, capsys):
        """A guidelines choice the suite lacks is a parser error, not a silent default."""
        monkeypatch.setattr(sys, "argv", ["run", "--guidelines", "homebench"])
        with pytest.raises(SystemExit) as stop:
            run.main()
        assert stop.value.code == 2
        assert "no guidelines 'homebench'" in capsys.readouterr().err

    @needs_data
    def test_real_data_quick_selection(self):
        """Real data: the quick config's selection is 44 episodes over 11 groups, kept
        together by home."""
        names = run.select_episodes(DATA, "hb1_*,hb2_*,hb3_*,hb4_*", None, 4, HOMEBENCH)
        assert len(names) == 44
        assert len({n.rsplit("_seed_", 1)[0] for n in names}) == 11
        grouped = run.group_by_home(names, HOMEBENCH, DATA)
        homes = [HOMEBENCH.home_key(DATA, n) for n in grouped]
        assert homes == sorted(homes, key=homes.index)


# -- the gate: PLANNING_GUIDELINES_PATH reaches the ISA's guard ---------------------------------


def _light_tree(home):
    """A plan tree that sets the living-room light's brightness, and that action's URL."""
    url = f"{home.builder.artifact_path('living_room', 'living_room_light')}/actions/brightness"
    return {"type": "action", "name": "set brightness", "action_url": url,
            "parameters": {DIRECT_VALUE_KEY: 40}}


class TestGuidelinesReachTheGuard:
    """The ISA's guards read `getattr(agent, "config")`; the bench's `isa_agent` stub
    hands them the ISA's section of agents.yaml, so the choice has to be in it."""

    def _isa(self, config):
        """The bench's ISA stub over ``config`` (every step on a local model)."""
        variant = load_variant(str(VARIANTS / "homebench.quick.yaml"))
        explorer = SimpleNamespace(query=None)
        return isa_agent(config, variant, explorer, logging.getLogger("t"))

    def test_the_environment_variable_reaches_the_isa_stub(self, monkeypatch):
        """PLANNING_GUIDELINES_PATH, read by agents.yaml, is in `agent.config` and is
        what `guidelines_path` returns: nothing in the stub drops it."""
        monkeypatch.setenv("PLANNING_GUIDELINES_PATH", str(GUIDELINES))
        config = ConfigLoader.load_with_env_vars(str(run.AGENTS_YAML))
        agent = self._isa(config)
        assert guidelines_path(agent.config) == str(GUIDELINES)

    def test_run_guidelines_reaches_the_isa_stub(self, monkeypatch):
        """`run.guidelines` (apply_guidelines) lands in the same place, whatever the
        environment variable says."""
        monkeypatch.setenv("PLANNING_GUIDELINES_PATH", "")
        config = ConfigLoader.load_with_env_vars(str(run.AGENTS_YAML))
        assert guidelines_path(self._isa(config).config) is None        # nothing chosen
        run.apply_guidelines(config, HOMEBENCH.guidelines_path(None))
        assert guidelines_path(self._isa(config).config) == str(GUIDELINES)

    def test_homebench_file_turns_both_guards_off(self):
        """No power_on_first entry and no condition: the power-on guard and the
        condition-guard check have nothing to apply. The repo's file has both."""
        assert load_applies_to(str(GUIDELINES)) is None
        assert load_conditions(str(GUIDELINES)) == ()
        assert load_applies_to(None) is not None and load_conditions(None)

    def test_the_power_on_guard_switches_with_the_repo_file_and_not_with_homebenchs(
            self, home_config):
        """End to end on a HomeBench light (a DimmableLight): with the repo's file the
        brightness action gets "switch on first" in front; with HomeBench's it is left
        as planned (the gold has no turn_on, plan 3.10)."""
        home = home_of(home_config)
        graph = home.graph()

        def guarded(path):
            """The tree the power-on guard leaves, for the guidelines file ``path``."""
            entry = {"tree": _light_tree(home)}
            behaviour = PowerOnGuardBehaviour([entry], logger=logging.getLogger("t"),
                                              graph=graph)
            config = {"planning": {"guidelines": {"path": path}}}
            behaviour.agent = SimpleNamespace(config=config)
            asyncio.run(behaviour.run())
            return entry["tree"]

        default_tree = guarded("")
        assert default_tree["type"] == "sequence" and "power on first" in default_tree["name"]
        homebench_tree = guarded(str(GUIDELINES))
        assert homebench_tree == _light_tree(home)

    def test_the_condition_guard_does_nothing_with_homebenchs_file(self, home_config):
        """With no conditions it returns before any model call or graph fetch."""
        home = home_of(home_config)
        entry = {"tree": _light_tree(home)}
        behaviour = ConditionGuardBehaviour([entry], logger=logging.getLogger("t"))
        behaviour.agent = SimpleNamespace(
            config={"planning": {"guidelines": {"path": str(GUIDELINES)}}})
        asyncio.run(behaviour.run())                 # no graph, no LLM client: must not need them
        assert entry["tree"] == _light_tree(home)


# -- stubs.Home --------------------------------------------------------------------------------


class TestHomeStub:
    """`stubs.Home` is built with the suite's TD builder."""

    def test_homebench_builder(self, home_config):
        """The home is discovered through HomeBenchTD: no ambient sensors, the
        room-less vacuum in the synthetic household room, every device an artifact."""
        home = home_of(home_config)
        assert isinstance(home.builder, HomeBenchTD)
        assert home.sensors == {}
        assert set(home.rooms) == {"living_room", "kitchen", "household"}
        assert home.rooms["household"] == ["household_vacuum_robot"]
        assert set(home.agent.artifacts) == set(home.device_room)
        assert home.device_room["kitchen_light"] == "kitchen"

    def test_a_builder_without_ambient_sensors_is_skipped(self, home_config):
        """A builder that has no `ambient_sensors` at all (no room physics) yields none
        (a wrapper hides the method from `Home`; the TD code behind it still works)."""
        class NoSensors:
            """A HomeBenchTD that does not offer `ambient_sensors`."""

            def __init__(self, base, home, state):
                """Wrap a HomeBenchTD."""
                self._td = HomeBenchTD(base, home, state)

            def __getattr__(self, name):
                """Everything but `ambient_sensors`, from the wrapped builder."""
                if name == "ambient_sensors":
                    raise AttributeError(name)
                return getattr(self._td, name)

        home = home_of(home_config, builder_cls=NoSensors)
        assert not hasattr(home.builder, "ambient_sensors")
        assert home.sensors == {} and "kitchen_light" in home.device_room

    def test_default_builder_is_simuhomes(self):
        """The old signature keeps working: the builder defaults to SimuHomeTD."""
        import inspect
        default = inspect.signature(Home.__init__).parameters["builder_cls"].default
        # By name, not identity: test_ambient_sensors.py reloads SimuHome's TD builder
        # module (importlib.reload), which gives the class a second object in a full run.
        assert default.__name__ == "SimuHomeTD"
        assert default.__module__.endswith("SimuHome.td_builder")


# -- the environment option --------------------------------------------------------------------


class FakeWorker:
    """A worker that records how it was built (the suite's ``worker_cls``)."""

    built = []

    def __init__(self, **kwargs):
        FakeWorker.built.append(kwargs)
        self.client = object()

    def start(self):
        return self

    def stop(self):
        pass


def _settings(tmp_path, **extra):
    """The settings dict ``run_worker`` reads, with no episodes to run."""
    return {"suite": "homebench", "guidelines": None, "variant": None,
            "out": str(tmp_path / "out"), "benchmark": str(tmp_path), "memory": "off",
            "context": "none", "step_seconds": 10.0, "exec_timeout": 60.0,
            "stop_after": None, "debug": False, "sim": None, "skip_existing": False,
            **extra}


class TestEnvironmentOption:
    """``--environment on|off``: the suite's option, the worker, the TD builder the
    pipeline discovers with, ``run.json`` and the statistics header."""

    def test_only_homebench_has_the_option(self):
        from tests.simuhome.bench import suites
        assert HOMEBENCH.environment is True and SIMUHOME.environment is False
        assert suites.ENVIRONMENT_MODES == ("on", "off")

    def test_run_refuses_the_option_for_simuhome(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["run", "--environment", "off"])
        with pytest.raises(SystemExit) as stop:
            run.main()
        assert stop.value.code == 2
        assert "has none to switch" in capsys.readouterr().err

    def test_run_refuses_another_mode(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["run", "--suite", "homebench", "--environment",
                                          "maybe"])
        with pytest.raises(SystemExit) as stop:
            run.main()
        assert stop.value.code == 2

    @pytest.mark.parametrize("argv, expected", [
        ([], "on"), (["--environment", "on"], "on"), (["--environment", "off"], "off")])
    def test_run_json_records_the_mode(self, bench_dir, tmp_path, monkeypatch, argv,
                                       expected):
        """The mode is a setting of the run: on by default, as given otherwise."""
        monkeypatch.setattr(run, "run_worker", lambda *a, **k: [])
        out = tmp_path / "records"
        monkeypatch.setattr(sys, "argv", [
            "run", "--suite", "homebench", "--benchmark", str(bench_dir), "--out", str(out),
            "--episodes", "hb1_feasible_seed_1", *argv])
        assert run.main() == 0
        settings = json.loads(next(out.glob("*/run.json")).read_text(encoding="utf-8"))
        assert settings["settings"]["environment"] == expected

    def test_run_json_of_simuhome_has_no_environment(self, tmp_path, monkeypatch):
        """SimuHome's settings are what they were: no new key."""
        bench = tmp_path / "bench"
        bench.mkdir()
        (bench / "qt1_feasible_seed_1.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(run, "run_worker", lambda *a, **k: [])
        out = tmp_path / "records"
        monkeypatch.setattr(sys, "argv", ["run", "--benchmark", str(bench), "--out", str(out),
                                          "--episodes", "qt1_feasible"])
        assert run.main() == 0
        settings = json.loads(next(out.glob("*/run.json")).read_text(encoding="utf-8"))
        assert "environment" not in settings["settings"]

    @pytest.mark.parametrize("mode", ["on", "off"])
    def test_a_homebench_worker_gets_the_mode_and_simuhomes_does_not(self, tmp_path,
                                                                    monkeypatch, mode):
        """``run_worker`` builds the worker with ``environment`` only when the suite has
        the option (``SimuHomeWorker`` has no such argument)."""
        import dataclasses
        FakeWorker.built.clear()
        fake = "tests.unit.test_homebench_bench:FakeWorker"
        monkeypatch.setattr(run, "get_suite", lambda name=None: dataclasses.replace(
            HOMEBENCH if name == "homebench" else SIMUHOME, worker_path=fake))
        run.run_worker(0, [], _settings(tmp_path, environment=mode))
        assert FakeWorker.built[-1]["environment"] == mode
        run.run_worker(0, [], _settings(tmp_path, suite="simuhome"))
        assert "environment" not in FakeWorker.built[-1]

    def test_the_pipeline_discovers_with_the_mode(self, home_config):
        """``Home`` builds the suite's TD builder with the keyword arguments given:
        the environment on (the default) or off."""
        state = HomeBenchEngine(home_config).home_state()
        on = Home("http://hb.test", "home7", state, {}, logging.getLogger("t"),
                  builder_cls=HomeBenchTD)
        off = Home("http://hb.test", "home7", state, {}, logging.getLogger("t"),
                   builder_cls=HomeBenchTD, builder_kwargs={"environment": False})
        assert on.builder.environment is True and off.builder.environment is False

    def test_options_default_to_no_environment(self):
        from tests.simuhome.bench.pipeline import Options
        assert Options().environment is None

    def test_harness_passes_the_mode_on_and_checks_it(self, tmp_path):
        run_cmd, _eval = harness.commands(Path("c.yaml"), tmp_path / "r", tmp_path,
                                          {"episodes": "hb1_*", "environment": "off"},
                                          True, tmp_path, HOMEBENCH)
        assert run_cmd[run_cmd.index("--environment") + 1] == "off"
        harness.check_environment({"environment": "on"}, HOMEBENCH)
        harness.check_environment({}, HOMEBENCH)
        harness.check_environment({}, SIMUHOME)
        with pytest.raises(ValueError, match="quote it"):
            harness.check_environment({"environment": False}, HOMEBENCH)   # bare YAML off
        with pytest.raises(ValueError, match="no environment option"):
            harness.check_environment({"environment": "on"}, SIMUHOME)

    def test_a_config_with_an_unquoted_off_is_refused(self, monkeypatch, capsys, tmp_path):
        config = yaml.safe_load((VARIANTS / "homebench.quick.yaml").read_text(encoding="utf-8"))
        config["run"]["environment"] = False
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["harness", "--config", str(path), "--dry-run"])
        assert harness.main() == 2
        assert "environment must be one of" in capsys.readouterr().out

    def test_the_three_configs_state_the_flat_baseline(self):
        """The existing configs keep their flat graph: ``environment: "off"`` (a string:
        YAML reads a bare off as false)."""
        for name in ("quick", "spread", "full"):
            config = yaml.safe_load((VARIANTS / f"homebench.{name}.yaml")
                                    .read_text(encoding="utf-8"))
            assert config["run"]["environment"] == "off"

    def test_sweep_passes_the_mode(self, monkeypatch, capsys, tmp_path):
        grid = {"suite": "homebench", "name": "g", "base": {},
                "vary": {"segmentation": [{"model": "a"}]},
                "run": {"episodes": "hb1_*", "environment": "off"}}
        path = tmp_path / "grid.yaml"
        path.write_text(yaml.safe_dump(grid), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["sweep", "--grid", str(path), "--out",
                                          str(tmp_path / "out"), "--dry-run"])
        assert sweep.main() == 0
        assert "--environment off" in capsys.readouterr().out

    def test_stats_and_report_name_the_mode(self, tmp_path):
        """The mode of ``run.json`` heads ``summary.md`` and the report; None when the
        run does not say."""
        variant = tmp_path / "v"
        variant.mkdir()
        assert stats.run_environment(variant) is None
        (variant / "run.json").write_text(
            json.dumps({"settings": {"environment": "off"}}), encoding="utf-8")
        assert stats.run_environment(variant) == "off"
        assert report.environments(tmp_path, ["v", "absent"]) == {"v": "off"}
        rows = [_row("a", "hb1", "feasible", 1)]
        out = stats.write(variant, rows, stats.aggregate(rows), "t", "off")
        assert "Environment: **off**" in (out / "summary.md").read_text(encoding="utf-8")
        out = stats.write(tmp_path / "w", rows, stats.aggregate(rows), "t")
        assert "Environment:" not in (out / "summary.md").read_text(encoding="utf-8")


# -- harness.py and sweep.py -------------------------------------------------------------------


class TestHarness:
    """Commands, estimates, dry-runs and the three configs."""

    def test_simuhome_commands_are_unchanged(self, tmp_path):
        """The scheduled runner, no --suite, no --guidelines: the old command lines."""
        run_cmd, eval_cmd = harness.commands(Path("c.yaml"), tmp_path / "r", tmp_path,
                                             {"episodes": "qt*"}, True, tmp_path)
        assert run_cmd[2] == "tests.simuhome.bench.run_scheduled"
        assert "--suite" not in run_cmd and "--suite" not in eval_cmd
        assert "--guidelines" not in run_cmd and "--no-judge" not in eval_cmd

    def test_homebench_commands_tick_and_name_the_suite(self, tmp_path):
        """run.py (never run_scheduled) with --suite homebench and the guidelines choice;
        the scoring step is told the suite too."""
        options = {"episodes": "hb1_*", "limit": 4, "guidelines": "default"}
        run_cmd, eval_cmd = harness.commands(Path("c.yaml"), tmp_path / "r", tmp_path,
                                             options, False, tmp_path, HOMEBENCH)
        assert run_cmd[2] == "tests.simuhome.bench.run"
        assert run_cmd[run_cmd.index("--suite") + 1] == "homebench"
        assert run_cmd[run_cmd.index("--guidelines") + 1] == "default"
        assert run_cmd[run_cmd.index("--limit") + 1] == "4"
        assert eval_cmd[eval_cmd.index("--suite") + 1] == "homebench"
        assert "--no-judge" in eval_cmd

    def test_estimates(self, tmp_path):
        """SimuHome: a range, unchanged wording. HomeBench: exact calls per part, the
        judge only for episodes with a part that cannot be done, recorded ones skipped."""
        sim = harness.estimate(["qt3_feasible_seed_1", "qt3_infeasible_seed_1"], tmp_path, True)
        assert sim == ("2 episode(s) selected, 0 already recorded, 2 to run: ~6-12 LLM calls "
                       "for the steps + ~3 judge calls (1 infeasible) "
                       "(none for steps on a local model)")
        names = ["hb1_feasible_seed_1", "hb3_mixed_seed_1", "hb2_infeasible_seed_1"]
        (tmp_path / "hb1_feasible_seed_1.json").write_text("{}", encoding="utf-8")
        text = harness.estimate(names, tmp_path, True, HOMEBENCH)
        assert "3 episode(s) selected, 1 already recorded, 2 to run: ~7 LLM calls" in text
        assert "~6 judge calls (2 with an infeasible part)" in text
        assert "judge" not in harness.estimate(names, tmp_path, False, HOMEBENCH)

    @needs_data
    @pytest.mark.parametrize("config,episodes,calls", [
        ("homebench.quick.yaml", 44, 160), ("homebench.spread.yaml", 27, 172),
        ("homebench.full.yaml", 17366, 56557)])
    def test_dry_run_prints_the_episode_count_and_call_estimate(
            self, config, episodes, calls, monkeypatch, capsys, tmp_path):
        """Real data: `harness --dry-run` on each config prints the episodes selected
        and the calls, runs nothing, and writes nothing."""
        monkeypatch.setattr(sys, "argv", ["harness", "--config", str(VARIANTS / config),
                                          "--out", str(tmp_path), "--dry-run"])
        assert harness.main() == 0
        out = capsys.readouterr().out
        assert f"{episodes} episode(s) selected, 0 already recorded, {episodes} to run" in out
        assert f"~{calls} LLM calls for the steps" in out
        assert "tests.simuhome.bench.run --variant" in out and "run_scheduled" not in out
        assert "--suite homebench" in out and "--guidelines homebench" in out
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_data_directory_says_how_to_make_it(self, monkeypatch, capsys, tmp_path):
        """An unconverted benchmark directory is named, with the converter."""
        monkeypatch.setattr(sys, "argv", ["harness", "--config",
                                          str(VARIANTS / "homebench.quick.yaml"),
                                          "--benchmark", str(tmp_path), "--dry-run"])
        assert harness.main() == 1
        assert "convert.py" in capsys.readouterr().out

    @pytest.mark.parametrize("key,value,message", [
        ("suite", "nope", "unknown suite 'nope'"),
        ("guidelines", "nope", "no guidelines 'nope'")])
    def test_a_bad_config_is_refused(self, key, value, message, monkeypatch, capsys, tmp_path):
        """An unknown suite or guidelines choice in the config ends the run with 2."""
        config = yaml.safe_load((VARIANTS / "homebench.quick.yaml").read_text(encoding="utf-8"))
        if key == "suite":
            config["suite"] = value
        else:
            config["run"]["guidelines"] = value
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["harness", "--config", str(path), "--dry-run"])
        assert harness.main() == 2
        assert message in capsys.readouterr().out

    def test_the_three_configs_differ_only_in_run(self):
        """Same suite, providers, models and evaluate; no name (the file names the
        records); run differs in the episodes and the limit."""
        configs = {name: yaml.safe_load((VARIANTS / f"homebench.{name}.yaml")
                                        .read_text(encoding="utf-8"))
                   for name in ("quick", "spread", "full")}
        stripped = [{k: v for k, v in c.items() if k != "run"} for c in configs.values()]
        assert stripped[0] == stripped[1] == stripped[2]
        assert stripped[0]["suite"] == "homebench" and "name" not in stripped[0]
        runs = {name: c["run"] for name, c in configs.items()}
        assert runs["quick"]["limit"] == 4 and runs["spread"]["limit"] == 1
        assert runs["full"]["limit"] is None
        # everything else in run: is shared
        shared = [{k: v for k, v in r.items() if k not in ("episodes", "limit")}
                  for r in runs.values()]
        assert shared[0] == shared[1] == shared[2]

    def test_every_config_key_is_read_by_python(self):
        """No YAML key nobody reads: top-level keys are the harness's and the variant
        loader's, every run: key is one the harness passes on."""
        for name in ("quick", "spread", "full"):
            config = yaml.safe_load((VARIANTS / f"homebench.{name}.yaml")
                                    .read_text(encoding="utf-8"))
            assert set(config) <= {"suite", "providers", "models", "run", "evaluate"}
            assert set(config["run"]) <= set(harness.RUN_OPTIONS)
            assert set(config["evaluate"]) <= {"judge"}
            variant = load_variant(str(VARIANTS / f"homebench.{name}.yaml"))
            assert variant.name == f"homebench.{name}"           # the file names the records

    def test_sweep_names_the_suite_in_its_commands(self, monkeypatch, capsys, tmp_path):
        """A grid with `suite: homebench` runs the tick-based run.py with the suite and
        the guidelines choice, and scores with the suite."""
        grid = {"suite": "homebench", "name": "g", "base": {},
                "vary": {"segmentation": [{"model": "a"}]},
                "run": {"episodes": "hb1_*", "limit": 1, "guidelines": "default"}}
        path = tmp_path / "grid.yaml"
        path.write_text(yaml.safe_dump(grid), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["sweep", "--grid", str(path), "--out",
                                          str(tmp_path / "out"), "--dry-run"])
        assert sweep.main() == 0
        out = capsys.readouterr().out
        assert out.count("--suite homebench") == 2 and "--guidelines default" in out
        assert "run_scheduled" not in out

    def test_sweep_refuses_an_unknown_suite(self, monkeypatch, capsys, tmp_path):
        """An unknown suite in a grid ends with 2 instead of running SimuHome."""
        path = tmp_path / "grid.yaml"
        path.write_text(yaml.safe_dump({"suite": "nope", "vary": {}}), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["sweep", "--grid", str(path), "--dry-run"])
        assert sweep.main() == 2
        assert "unknown suite" in capsys.readouterr().out


# -- stats.py and report.py --------------------------------------------------------------------


def _row(name, query_type, case, score, **evaluation):
    """A stats row for a record of (type, case) scored ``score``."""
    record = {"episode": name, "query_type": query_type, "case": case, "outcome": "executed"}
    return stats.episode_metrics(record, {"score": score, **evaluation})


class TestStatsAndReport:
    """The `mixed` case and the hbN types, which the SimuHome-only lists dropped or misordered."""

    ROWS = [_row("a", "hb1", "feasible", 1), _row("b", "hb2", "feasible", 0),
            _row("c", "hb10", "feasible", 1), _row("d", "hb2", "mixed", 1),
            _row("e", "hb2", "mixed", 0), _row("f", "hb2", "infeasible", 1)]

    def test_a_mixed_case_gets_rows_of_its_own(self):
        """Every episode is in a (type, case) row and an (all, case) row; none vanishes."""
        groups = {key: len(members) for key, members in stats._groups(self.ROWS)}
        assert groups[("hb2", "mixed")] == 2 and groups[("all", "mixed")] == 2
        assert sum(n for (t, c), n in groups.items() if "all" not in (t, c)) == 6
        assert groups[("all", "all")] == 6

    def test_homebench_types_sort_naturally(self):
        """hb1, hb2, hb10 -- not hb1, hb10, hb2; qt types keep SimuHome's order."""
        keys = [key for key, _ in stats._groups(self.ROWS)]
        types = [t for t, c in keys if c == "all" and t != "all"]
        assert types == ["hb1", "hb2", "hb10"]
        qt = [_row(f"x{t}", t, "feasible", 1) for t in ("qt4-2", "qt1", "qt3", "qt4-1")]
        order = [t for (t, c), _ in stats._groups(qt) if c == "all" and t != "all"]
        assert order == ["qt1", "qt3", "qt4-1", "qt4-2"]

    def test_a_case_nobody_listed_is_not_dropped(self):
        """A case outside CASES still gets its rows, after the listed ones."""
        rows = [_row("a", "hb1", "feasible", 1), _row("b", "hb1", "weird", 0)]
        keys = [key for key, _ in stats._groups(rows)]
        assert ("hb1", "weird") in keys and keys.index(("hb1", "feasible")) < keys.index(
            ("hb1", "weird"))

    def test_mixed_rows_carry_the_execution_check_rate(self):
        """Episodes with a part that cannot be done report the exec-check rate."""
        rows = [_row("a", "hb2", "mixed", 1, execution_check={"score": 1}),
                _row("b", "hb2", "mixed", 0, execution_check={"score": 0})]
        summary = stats.summarise("hb2", "mixed", rows)
        assert summary["execution_check_rate"] == 0.5
        assert stats.summarise("hb2", "feasible", [_row("c", "hb2", "feasible", 1)]
                               )["execution_check_rate"] is None

    def test_the_legend_explains_homebench_only_for_homebench_records(self, tmp_path):
        """The HomeBench note is appended when a record is HomeBench's, and only then."""
        for rows, expected in ((self.ROWS, True), ([_row("q", "qt3", "feasible", 1)], False)):
            out = stats.write(tmp_path / str(expected), rows, stats.aggregate(rows), "t")
            text = (out / "summary.md").read_text(encoding="utf-8")
            assert ("HomeBench: `type` is hbN" in text) is expected

    def test_report_orders_types_naturally_and_scores_mixed_checks(self):
        """report.rates: hb1, hb2, hb10; a mixed group reports the exec-check rate."""
        def item(query_type, case, **evaluation):
            return {"record": {"query_type": query_type, "case": case},
                    "evaluation": {"score": 1, **evaluation}}
        data = {"v": [item("hb10", "feasible"), item("hb2", "mixed",
                                                      execution_check={"score": 1}),
                      item("hb1", "feasible")]}
        rows = report.rates(data)
        assert [r["type"] for r in rows] == ["hb1", "hb2", "hb10"]
        assert rows[1]["execution_check_rate"] == 1.0
        assert report.type_key("hb2") < report.type_key("hb10") < report.type_key("qt1")


# -- the pipeline ------------------------------------------------------------------------------


class TestPipeline:
    """Loading the pipeline module must not load what only SimuHome's scheduled runs use."""

    def test_scheduling_is_imported_lazily(self):
        """`scheduling` is not in sys.modules after importing the pipeline."""
        code = ("import sys, tests.simuhome.bench.pipeline\n"
                "print('tests.simuhome.bench.scheduling' in sys.modules)\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        assert result.stdout.strip().splitlines()[-1] == "False", result.stderr

    def test_run_episode_takes_the_suite_and_defaults_to_simuhome(self):
        """The new parameter is last and optional, so existing callers are unchanged."""
        import inspect
        from tests.simuhome.bench.pipeline import run_episode
        parameter = inspect.signature(run_episode).parameters["suite"]
        assert parameter.default is None
        assert list(inspect.signature(run_episode).parameters)[-1] == "suite"


# -- the worker lifecycle gate -----------------------------------------------------------------


class _Challenge(http.server.BaseHTTPRequestHandler):
    """A WebSub callback that echoes the hub's challenge, so the hub accepts it."""

    def do_GET(self):
        """Answer the intent verification with the challenge."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        body = query.get("hub.challenge", [""])[0].encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Silence the request log."""


@pytest.fixture()
def callback():
    """The URL of a running WebSub callback server; stopped afterwards."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Challenge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/callback"
    server.shutdown()
    server.server_close()


def _json(url):
    """GET ``url`` as JSON."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def _subscribe(worker, callback_url):
    """Subscribe ``callback_url`` to the home workspace through the worker's hub."""
    topic = f"{worker.shtd_base}/workspaces/{worker.home_id}"
    form = urllib.parse.urlencode({"hub.mode": "subscribe", "hub.topic": topic,
                                   "hub.callback": callback_url}).encode()
    request = urllib.request.Request(f"{worker.shtd_base}/hub/", data=form, method="POST",
                                     headers={"Content-Type":
                                              "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(request, timeout=20) as response:
        assert response.status == 202


def _attributes(state):
    """{device_id: attributes} over a home state."""
    return {d["device_id"]: d["attributes"] for room in state["rooms"].values()
            for d in room["devices"]}


def _dirty(worker):
    """Flip the first switchable device of the loaded home; the (device, new state)."""
    state = worker.client.home_state()
    for device_id, attributes in _attributes(state).items():
        command = "turn_off" if attributes.get("state") == "on" else "turn_on"
        if attributes.get("state") in ("on", "off"):
            worker.client.execute_command(device_id, command)
            assert _attributes(worker.client.home_state())[device_id]["state"] != attributes["state"]
            return device_id
    raise AssertionError("no switchable device")


def _initial(episode):
    """{device_id: attributes} of an episode's home as the home file has them."""
    return {d["device_id"]: d["attributes"]
            for room in episode["initial_home_config"]["rooms"].values()
            for d in room["devices"]}


def _two_names(same_home):
    """Two real episodes of the same home, or of different homes."""
    homes = {}
    for name in HOMEBENCH.names(DATA)[:400]:
        homes.setdefault(HOMEBENCH.home_key(DATA, name), []).append(name)
    if same_home:
        return next(names[:2] for names in homes.values() if len(names) >= 2)
    first, second = list(homes.values())[:2]
    return [first[0], second[0]]


@needs_data
class TestWorkerLifecycle:
    """Real engine and HBTD processes: the gate's two-episode lifecycle test."""

    def test_two_episodes_in_one_worker(self, callback):
        """After ``end_episode`` the hub holds no subscription, HBTD is restarted on the
        new home, and the engine state equals the new home's initial state."""
        first, second = _two_names(same_home=False)
        with HomeBenchWorker(benchmark=DATA) as worker:
            # 1. Episode one: HBTD on its home, a subscription held, the engine dirtied.
            episode_one = worker.load(first)
            assert worker.home_id == f"home{episode_one['home_id']}"
            assert _json(f"{worker.shtd_base}/_shtd/status")["home"] == worker.home_id
            assert _attributes(worker.client.home_state()) == _initial(episode_one)
            _subscribe(worker, callback)
            assert worker.subscription_count() == 1
            _dirty(worker)
            old_hbtd = worker._shtd

            # 2. Episode two: the first ends (its HBTD stopped), a new one serves the new home.
            episode_two = worker.load(second)
            assert old_hbtd.poll() is not None                       # the old process is gone
            assert worker._shtd is not old_hbtd and worker._shtd.poll() is None
            assert worker.subscription_count() == 0                  # a fresh hub: empty
            assert worker.home_id == f"home{episode_two['home_id']}" != f"home{episode_one['home_id']}"
            assert _json(f"{worker.shtd_base}/_shtd/status")["home"] == worker.home_id
            assert worker.episode_name == second

            # 3. The engine is the new home's initial state, nothing of episode one left.
            assert _attributes(worker.client.home_state()) == _initial(episode_two)

    def test_the_same_home_twice_starts_from_its_initial_state(self):
        """Home ids repeat across episodes: the second episode of a home sees the
        initial state again, through a restarted HBTD on the same id."""
        first, second = _two_names(same_home=True)
        with HomeBenchWorker(benchmark=DATA) as worker:
            episode = worker.load(first)
            _dirty(worker)
            assert _attributes(worker.client.home_state()) != _initial(episode)
            old_hbtd, home = worker._shtd, worker.home_id
            episode = worker.load(second)
            assert worker.home_id == home and old_hbtd.poll() is not None
            assert _attributes(worker.client.home_state()) == _initial(episode)

    def test_end_episode_stops_hbtd_and_keeps_the_engine(self):
        """`end_episode` leaves no TD server and no episode; the engine stays up."""
        with HomeBenchWorker(benchmark=DATA) as worker:
            worker.load(HOMEBENCH.names(DATA)[0])
            worker.end_episode()
            assert worker.episode is None and worker.episode_name is None
            assert worker.home_id is None and worker._shtd is None
            with pytest.raises(urllib.error.URLError):
                _json(f"{worker.shtd_base}/_shtd/status")
            assert worker.client.health()
            assert worker.subscription_count() == 0

    def test_stop_reaps_both_processes(self):
        """Leaving the context stops the engine and HBTD."""
        with HomeBenchWorker(benchmark=DATA) as worker:
            worker.load(HOMEBENCH.names(DATA)[0])
            processes = [worker._sim, worker._shtd]
        assert all(p.poll() is not None for p in processes)

    def test_load_of_a_missing_episode_fails_clearly(self):
        """An unknown episode name is a FileNotFoundError naming the path."""
        with HomeBenchWorker(benchmark=DATA) as worker:
            with pytest.raises(FileNotFoundError, match="episode not found"):
                worker.load("hb1_feasible_seed_999999")

    def test_each_worker_has_its_own_ports(self):
        """Two workers never share a port (they are not started here)."""
        a, b = HomeBenchWorker(benchmark=DATA), HomeBenchWorker(benchmark=DATA)
        assert len({a.sim_port, a.shtd_port, b.sim_port, b.shtd_port}) == 4
