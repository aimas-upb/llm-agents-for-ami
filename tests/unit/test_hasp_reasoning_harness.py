"""HASP backend of the shared reasoning harness; no live HA or LLM requests."""
import asyncio
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient

from tests.hasp.hasp_lab308e_benchmark import HA_DIR
from tests.hasp.bench.backend import HASPHome, Lab308eBackend, execution_succeeded
from tests.hasp.bench.evaluate import evaluate, main as evaluate_main
from tests.hasp.bench.run import select_episodes, DEFAULT_BENCHMARK
from tests.hasp.bench import harness
from tests.simuhome.bench import pipeline
from tests.simuhome.bench.execution import Execution
from tests.simuhome.bench.models import load_variant
from tests.simuhome.bench.stats import compute
from tests.simuhome.bench.stubs import Explorer
from tests.simuhome.bench.toollog import ToolLog, parse_url
from ami_agents.bt_planning.nodes.http_client import HTTPClient, HTTPResponse
from ami_agents.shared.models.goal_structure import GoalStructure


def fixture_home():
    """Build actual HASP TDs against the recorded HA fixture, entirely offline."""
    sys.path.insert(0, str(HA_DIR))
    with patch.dict(os.environ, HA_TOKEN="offline-test", SEMANTIC_CONFIG=str(HA_DIR / "lab308e.yaml")):
        import hasp
        snapshot = json.loads((HA_DIR / "tests/fixtures/lab308e_ha.json").read_text())
        class WS:
            async def get_areas(self): return snapshot["areas"]
            async def get_devices(self, area_id=None):
                return [d for d in snapshot["devices"] if area_id is None or d.get("area_id") == area_id]
            async def get_entities(self): return snapshot["entities"]
        class REST:
            async def get_states(self): return snapshot["states"]
            async def get_services(self): return snapshot["services"]
        # A TestClient used without a lifespan context lazily builds HASP's cache
        # using these fixtures; it does not start the HA WebSocket sync loop.
        with patch.multiple(hasp, AREAS={"lab308e"}, BASE_WS_URI="http://testserver/",
                            ha_client=WS(), ha_rest=REST()):
            hasp.reload_semantics()
            hasp.app.state.graph_cache = None
            client = TestClient(hasp.app)
            try:
                workspace = client.get("/workspaces/lab308e")
                if workspace.status_code != 200:
                    raise AssertionError(workspace.text)
                documents = {}
                for device in snapshot["devices"]:
                    response = client.get(f"/workspaces/lab308e/artifacts/{device['name']}")
                    if response.status_code != 200:
                        raise AssertionError(response.text)
                    documents[device["name"]] = response.text
                return HASPHome("http://testserver", workspace.text, documents, {}, logging.getLogger("test"))
            finally:
                client.close()
                hasp.app.state.graph_cache = None


class HarnessTests(unittest.TestCase):
    def test_report_preserves_saved_models_when_config_has_changed(self):
        current = load_variant(None)
        saved = current.as_dict()
        saved["steps"]["segmentation"] = {"model": "gpt-5-nano", "reasoning_effort": "low"}
        saved["steps"]["planning.general"] = {"model": "qwen3:8b", "base_url": "http://gpu-box:8000/v1"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run.json").write_text(json.dumps({"variant": saved}))
            legend = harness.report_legend(out, current)
        self.assertIn("| segmentation | gpt-5-nano@low | OpenAI | chat-completions |", legend)
        self.assertIn("| planning.general | qwen3:8b | local http://gpu-box:8000/v1 |", legend)
        self.assertNotIn("| judge |", legend)

    def test_same_three_stages_and_cli_overrides(self):
        args = SimpleNamespace(runs=2, episodes="make_*", workers=None, memory="off")
        options = harness.run_options({"run": {"runs": 7, "workers": 1}}, args)
        self.assertEqual(options["runs"], 2)
        commands = harness.commands(Path("config.yaml"), Path("out/v"), Path("out"), options, False, DEFAULT_BENCHMARK)
        self.assertEqual([cmd[2] for cmd in commands], ["tests.hasp.bench.run", "tests.hasp.bench.evaluate"])
        self.assertIn("--skip-existing", commands[0])
        self.assertIn("--skip-scored", commands[1])
        self.assertTrue(any(cmd[i:i+2] == ["--runs", "2"] for cmd in commands for i in range(len(cmd)-1)))

    def test_five_scenarios_times_n_and_pattern_selection(self):
        names = select_episodes(DEFAULT_BENCHMARK, "*", 3)
        self.assertEqual(len(names), 15)
        self.assertEqual(len(set(names)), 15)
        self.assertEqual(len(select_episodes(DEFAULT_BENCHMARK, "make_*", 2)), 4)
        with self.assertRaises(ValueError):
            select_episodes(DEFAULT_BENCHMARK, "*", 0)

    def test_hasp_urls_are_logged_with_the_right_room_and_device(self):
        url = "http://h/workspaces/lab308e/artifacts/desk_lamp_308e/actions/onOff"
        self.assertEqual(parse_url(url)["room"], "lab308e")
        tools = ToolLog()
        tools.action(url, True)
        self.assertEqual(tools.entries[0]["params"], {"device_id": "desk_lamp_308e"})

    def test_exported_scores_feed_the_original_statistics(self):
        record = {"episode": "make_it_brighter_run_001", "query_type": "make_it_brighter", "case": "feasible",
                  "outcome": "executed", "meter": {"stages": {"planning": {"seconds": 2, "calls": 1}}},
                  "evaluation": {"score": 1, "assertions": [{"type": "property_gte", "passed": True}]}}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)
            (out / (record["episode"] + ".json")).write_text(json.dumps(record))
            self.assertEqual(evaluate_main(["--records", str(out)]), 0)
            self.assertIsNotNone(compute(out, "HASP", legend=harness.report_legend(out, load_variant(None))))
            rows = json.loads((out / "stats/summary.json").read_text())
            overall = next(row for row in rows if row["type"] == row["case"] == "all")
            self.assertEqual(overall["success_rate"], 1)
            self.assertEqual(overall["planning_s"], 2)
            self.assertIn("lab308e", (out / "stats/summary.md").read_text())
            self.assertIn("## Model configuration", (out / "stats/summary.md").read_text())

    def test_any_of_remains_one_logical_check(self):
        scored = evaluate({"episode": "x", "query_type": "x", "case": "feasible",
                           "evaluation": {"score": 1, "assertions": [{"type": "any_of", "passed": True,
                               "results": [{"passed": False}, {"passed": True}]}]}})
        self.assertEqual(scored["breakdown"]["success"], 1)


class BackendTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = fixture_home()

    async def test_actual_hasp_tds_have_a_resolvable_room_and_goal_context(self):
        home = self.home
        self.assertEqual(home.rooms_of_location("homeont:StudyRoom"), ["lab308e"])
        self.assertIn("ambient_lights_308e", home.agent.artifacts)
        explorer = Explorer(home, ToolLog(), logger=logging.getLogger("test"))
        text = await explorer.fetch_goal_context(None, None, "make it brighter", with_environment=True)
        self.assertIn("lab308e", text)
        self.assertIn("Illuminance", text)

    async def test_discovery_strips_artifact_fragments(self):
        """Discover actual artifact URLs without retaining RDF subject fragments."""
        # Isolate discovery from home setup and the managed dynamics process.
        backend = object.__new__(Lab308eBackend)
        backend.hasp = "http://testserver"
        backend.case = {"pre_query_settle_seconds": 0}
        backend.reset = AsyncMock()
        backend.setup_steps = AsyncMock()
        backend.start_dynamics = AsyncMock()
        backend.simulator = SimpleNamespace(poll=lambda: None)
        device = "ambient_lights_308e"
        artifact = f"/workspaces/lab308e/artifacts/{device}"
        documents = {
            "/_graph/refresh-states": "{}",
            "/workspaces/lab308e": self.home.agent.environment_map["lab308e"].rdf,
            "/workspaces/lab308e/artifacts":
                f"<{backend.hasp}{artifact}#artifact> a <https://purl.org/hmas/Artifact> .",
            artifact: self.home.agent.artifacts[device].thing_description.rdf,
        }
        # Serve captured TDs entirely offline and inspect the requested URLs.
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, text=documents[request.url.path])
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch("tests.hasp.bench.backend.httpx.AsyncClient", return_value=client):
            home = await backend.load({}, logging.getLogger("test"))
        self.assertEqual(list(home.agent.artifacts), [device])
        self.assertEqual(home.rooms_of_location("homeont:StudyRoom"), ["lab308e"])
        self.assertFalse(any(request.url.fragment for request in requests))

    async def test_impossible_plan_returns_without_execution_or_confirmation(self):
        from unittest.mock import Mock
        structure = GoalStructure.from_dict({"goal_specificity": "explicit", "goal_kind": "achievement",
                    "goal_effect": "set", "location_class": "homeont:StudyRoom", "artifact_class": "homeont:Fan",
                    "artifact_name": "ceiling_fan_308e", "affordance_name": "onOff",
                    "target_value_determined": "true"}, intent_text="turn on the fan", structure="simple")
        front = {"intents": [{"text": "turn on the fan"}], "queries": [],
                 "goals": [{"text": "turn on the fan", "structure": structure.to_wire_dict()}]}
        class Planning:
            def __init__(self, *args, **kwargs):
                self.results = [{"tree": None, "impossible": True, "explanation": "No matching action"}]
            async def run(self): pass
        class Guard:
            def __init__(self, *args, **kwargs): pass
            async def run(self): pass
        backend = SimpleNamespace(load=AsyncMock(return_value=self.home), clock={"source": "system"},
                                  execute=Mock(side_effect=AssertionError("Must not execute an impossible plan")),
                                  home_state=lambda: {})
        episode = {"query": "turn on the fan", "meta": {"query_type": "test", "case": "feasible"},
                   "initial_home_config": {}}
        with patch.dict(os.environ, OPENAI_API_KEY="offline-test"), \
             patch.object(pipeline, "_understand", AsyncMock(return_value=front)), \
             patch.object(pipeline, "route", return_value={"explicit": [(0, structure)]}), \
             patch.object(pipeline, "route_behaviour", return_value=Planning), \
             patch.object(pipeline, "ConditionGuardBehaviour", Guard), \
             patch.object(pipeline, "PowerOnGuardBehaviour", Guard):
            record = await asyncio.wait_for(pipeline.run_episode("test", episode, load_variant(None), {},
                          pipeline.Target(backend, "http://testserver", "lab308e", backend), pipeline.Options(),
                          logging.getLogger("test")), timeout=5)
        self.assertIsNone(record["error"], record.get("traceback"))
        self.assertEqual(record["outcome"], "impossible")
        self.assertEqual(record["execution"]["mode"], "none")
        backend.execute.assert_not_called()
        self.assertIn("meter", record)

    async def test_failed_plan_is_scored_without_polling_to_timeout(self):
        backend = object.__new__(Lab308eBackend)
        backend.case = {"pass_criteria": [{"type": "final_reply_contains", "value": "success"}]}
        record = {"outcome": "impossible", "execution": {"plans": []}}
        result = await asyncio.wait_for(backend.check(record), 1)
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["assertions"][0]["passed"])

    async def test_partial_execution_does_not_claim_success(self):
        record = {"outcome": "executed", "execution": {"plans": [{"status": "FAILURE"}]}}
        self.assertFalse(execution_succeeded(record))

    async def test_same_executor_posts_actions_to_hasp(self):
        backend = object.__new__(Lab308eBackend)
        url = "http://hasp/workspaces/lab308e/artifacts/ceiling_fan_308e/actions/onOff"
        tree = {"type": "action", "name": "fan", "action_url": url, "parameters": {}}
        tools = ToolLog()
        with patch.object(HTTPClient, "post", return_value=HTTPResponse(200, {}, {}, url, 0)) as post:
            result = await asyncio.to_thread(backend.execute, [{"tree": tree}], tools, {}, pipeline.Options(exec_timeout=2))
        self.assertEqual(result.plans[0]["status"], "SUCCESS")
        self.assertEqual(result.actions[0]["url"], url)
        post.assert_called_once()
        self.assertEqual(tools.entries[0]["params"]["device_id"], "ceiling_fan_308e")


if __name__ == "__main__":
    unittest.main()
