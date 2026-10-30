"""Exercise API switching against fake HTTP endpoints, including a real planner retry.

No LLM or HA is contacted: MockTransport checks the wire format and returns
SDK-parsed responses so parameter, tool and metering regressions are visible.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from openai.resources.responses import AsyncResponses

from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from tests.hasp.bench import harness
from tests.simuhome.bench.meter import Meter
from tests.simuhome.bench.models import ModelSpec, load_variant, variant_from_dict
from tests.simuhome.bench.responses import use_responses


def payload(output=None, **extra):
    """Return the minimal valid Responses wire payload with known token usage."""
    return {"id": "resp_test", "object": "response", "created_at": 1,
            "model": "gpt-6-luna", "status": "completed",
            "output": output if output is not None else [
                {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": '{"ok":true}', "annotations": []}]}],
            "usage": {"input_tokens": 17, "output_tokens": 5, "total_tokens": 22,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 3}}, **extra}


class ResponsesTests(unittest.IsolatedAsyncioTestCase):
    """Cover actual SDK serialization for the harness's supported request shapes."""

    def client(self, handler):
        """Make a Responses-enabled client with an HTTP-only fake endpoint."""
        client = use_responses(AsyncOpenAI(api_key="offline-test", max_retries=0,
                              base_url="http://offline.test/v1",
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
        self.addAsyncCleanup(client.close)
        return client

    async def test_text_schema_reasoning_budget_and_usage(self):
        """Translate structured text requests and count a single Responses call."""
        # 1. Capture the actual SDK request, including its endpoint path.
        requests = []
        def handle(request):
            """Capture the request and return an ordinary text response."""
            requests.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, json=payload())
        client = self.client(handle)
        meter = Meter()
        with meter.installed():
            async with meter.step("structuring"):
                result = await client.chat.completions.create(
                    model="gpt-6-luna", messages=[{"role": "system", "content": "Return JSON"},
                                                {"role": "user", "content": "make it brighter"}],
                    reasoning_effort="low", max_completion_tokens=900,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "result", "strict": True, "schema": {"type": "object"}}})
        # 2. Assert both the wire format and the unchanged consumer/meter interface.
        path, body = requests[0]
        self.assertEqual(path, "/v1/responses")
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["max_output_tokens"], 900)
        self.assertEqual(body["text"]["format"]["name"], "result")
        self.assertFalse(body["store"])
        self.assertNotIn("messages", body)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(result.choices[0].message.content, '{"ok":true}')
        self.assertEqual(result.usage.prompt_tokens, 17)
        self.assertEqual(len(meter.calls), 1)
        self.assertEqual(meter.calls[0]["step"], "structuring")
        self.assertEqual(meter.calls[0]["tokens_out"], 5)

    async def test_real_planner_retries_with_reasoning_and_tool_output(self):
        """A validation failure retains encrypted reasoning and the original call_id."""
        # 1. Reject the first empty tree, then let the planner report impossibility.
        requests = []
        reasoning = {"id": "rs_test", "type": "reasoning", "summary": [],
                     "encrypted_content": "encrypted-test"}
        def handle(request):
            """Return real function-call items to exercise the planner's retry loop."""
            body = json.loads(request.content)
            requests.append(body)
            args = {"tree": {}, "explanation": "no matching action", "impossible": len(requests) > 1}
            return httpx.Response(200, json=payload([
                reasoning, {"id": "fc_test", "type": "function_call", "call_id": "call_test",
                            "name": "generate_behavior_tree", "arguments": json.dumps(args),
                            "status": "completed"}]))
        meter = Meter()
        with meter.installed():
            async with meter.step("planning"):
                result = await AsyncBTPlanner(max_attempts=2).generate_bt(
                    intents=["cool down the room"], affordances=[], client=self.client(handle),
                    model="gpt-6-luna", reasoning_effort="low")
        # 2. Preserve non-strict schemas and exact reasoning/tool linkage on retry.
        self.assertTrue(result["impossible"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["tool_choice"], {"type": "function", "name": "generate_behavior_tree"})
        self.assertFalse(requests[0]["tools"][0]["strict"])
        self.assertNotIn("function", requests[0]["tools"][0])
        items = requests[1]["input"]
        self.assertEqual(items[2], reasoning)
        self.assertEqual(items[3]["call_id"], "call_test")
        self.assertEqual(items[4]["type"], "function_call_output")
        self.assertEqual(items[4]["call_id"], "call_test")
        self.assertIn("previous response was invalid", items[4]["output"])
        self.assertEqual(len(meter.calls), 2)
        self.assertEqual(meter.stages()["planning"]["tokens_in"], 34)

    async def test_failed_requests_are_metered_and_methods_restored(self):
        """HTTP failures count as failed calls, and SDK patches do not leak."""
        def handle(request):
            """Return a non-retryable API error without making a network call."""
            return httpx.Response(400, json={"error": {"message": "bad schema", "type": "invalid_request_error"}})
        original = AsyncResponses.create
        meter = Meter()
        with meter.installed(), self.assertRaises(Exception):
            await self.client(handle).chat.completions.create(
                model="gpt-6-luna", messages=[{"role": "user", "content": "test"}])
        self.assertIs(AsyncResponses.create, original)
        self.assertEqual(len(meter.calls), 1)
        self.assertFalse(meter.calls[0]["ok"])

    async def test_chat_default_still_uses_chat_endpoint(self):
        """The default spec never opts into Responses, including when metered."""
        requests = []
        def handle(request):
            """Return a standard Chat Completion for the legacy transport."""
            requests.append(request.url.path)
            return httpx.Response(200, json={"id": "chat_test", "object": "chat.completion",
                "created": 1, "model": "gpt-5-nano", "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}})
        client = AsyncOpenAI(api_key="offline-test", http_client=httpx.AsyncClient(
                            transport=httpx.MockTransport(handle)))
        self.addAsyncCleanup(client.close)
        with patch("tests.simuhome.bench.models.AsyncOpenAI", return_value=client):
            selected = ModelSpec("gpt-5-nano").client()
        meter = Meter()
        with meter.installed():
            await selected.chat.completions.create(model="gpt-5-nano", messages=[])
        self.assertEqual(requests, ["/v1/chat/completions"])
        self.assertEqual(len(meter.calls), 1)
        self.assertEqual(meter.calls[0]["tokens_in"], 8)

    async def test_model_spec_selects_responses(self):
        """The model factory used by every harness phase activates the adapter."""
        def handle(request):
            """Assert the chosen endpoint without allowing a real SDK connection."""
            self.assertEqual(request.url.path, "/v1/responses")
            return httpx.Response(200, json=payload())
        client = AsyncOpenAI(api_key="offline-test", http_client=httpx.AsyncClient(
                            transport=httpx.MockTransport(handle)))
        self.addAsyncCleanup(client.close)
        with patch("tests.simuhome.bench.models.AsyncOpenAI", return_value=client):
            selected = ModelSpec("gpt-6-luna", api="responses").client()
        result = await selected.chat.completions.create(model="gpt-6-luna", messages=[])
        self.assertEqual(result.choices[0].finish_reason, "stop")


class ConfigTests(unittest.TestCase):
    """Keep YAML choices, CLI overrides and historical reports reproducible."""

    def test_provider_api_and_cli_override(self):
        """Provider defaults allow mixed transports; CLI overrides every resolved spec."""
        # 1. Per-model transport overrides a provider's Responses default.
        variant = variant_from_dict({"providers": {"openai": {"api": "responses"}}, "models": {
            "segmentation": {"provider": "openai", "model": "gpt-6-luna"},
            "structuring": {"provider": "openai", "model": "gpt-5-nano", "api": "chat-completions"}}})
        self.assertEqual(variant["segmentation"].api, "responses")
        self.assertEqual(variant["structuring"].api, "chat-completions")
        # 2. CLI-level selection survives serialization into each episode's metadata.
        selected = load_variant(None, api="responses")
        self.assertTrue(all(spec.api == "responses" for spec in selected.specs.values()))
        self.assertEqual(selected.as_dict()["steps"]["segmentation"]["api"], "responses")
        with self.assertRaises(ValueError):
            load_variant(None, api="typo")

    def test_cli_override_is_forwarded_and_saved_api_is_reported(self):
        """The wrapper passes the flag to its subprocess and reports historical APIs."""
        from types import SimpleNamespace
        options = harness.run_options({"run": {"llm_api": "chat-completions"}},
                                      SimpleNamespace(llm_api="responses"))
        commands = harness.commands(Path("config.yaml"), Path("out/v"), Path("out"),
                                    options, False, Path("cases"))
        self.assertIn("--llm-api", commands[0])
        self.assertEqual(commands[0][commands[0].index("--llm-api") + 1], "responses")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run.json").write_text(json.dumps({"variant": load_variant(None, api="responses").as_dict()}))
            legend = harness.report_legend(out, load_variant(None))
        self.assertIn("| OpenAI | responses |", legend)

    def test_native_tool_session_respects_explicit_transport_and_auto(self):
        """The upstream planner's native tool path must respect the harness CLI too."""
        from types import SimpleNamespace
        from ami_agents.shared.utils.llm_tool_call import api_for
        # Defaults follow upstream: OpenAI tools use Responses; local tools use chat.
        self.assertEqual(api_for(SimpleNamespace(base_url="https://api.openai.com/v1")), "responses")
        self.assertEqual(api_for(SimpleNamespace(base_url="http://localhost:11434/v1")), "chat")
        for api, expected in [("auto", "responses"), ("responses", "responses"), ("chat-completions", "chat")]:
            client = SimpleNamespace(base_url="https://api.openai.com/v1",
                                     chat=SimpleNamespace(completions=SimpleNamespace()))
            with self.subTest(api=api), patch("tests.simuhome.bench.models.AsyncOpenAI", return_value=client):
                chosen = ModelSpec("gpt-6-luna", api=api).client()
                self.assertEqual(api_for(chosen), expected)


if __name__ == "__main__":
    unittest.main()
