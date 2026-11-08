"""The prompt report: how LLM calls are caught, and how a prompt is taken apart.

No LLM, simulator or network: the recorder is exercised against fake HTTP endpoints
(MockTransport), through the real planner so the call site it names is the real one; the
decomposition is run on prompts built the way the agents build them, from the live
constants, so a test fails when a prompt is no longer explained by its parts.
"""
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions
from openai.resources.responses import AsyncResponses

from ami_agents.agents.user_assistant.prompts.goal_structuring_prompts import render_goal_prompt
from ami_agents.agents.user_assistant.prompts.intent_response_prompts import (
    PLAN_SUMMARY_SYSTEM_PROMPT,
)
from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.prompts import BT_PLANNING_SYSTEM_PROMPT
from tests.homebench import prompt_parts, prompt_report
from tests.simuhome.bench.meter import Meter
from tests.simuhome.bench.prompt_capture import PromptRecorder
from tests.simuhome.bench.responses import use_responses

IMPOSSIBLE = {"tree": {}, "impossible": True, "explanation": "no such device"}


def chat_payload(arguments: dict) -> dict:
    """A Chat Completions reply in which the model calls the planner's tool."""
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function", "function": {
                        "name": "generate_behavior_tree",
                        "arguments": json.dumps(arguments)}}]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}


class RecorderTests(unittest.IsolatedAsyncioTestCase):
    """The recorder keeps a call in full, labelled with its step and its call site."""

    def client(self, handler, responses: bool = False) -> AsyncOpenAI:
        """A client whose endpoint is a fake; the Responses transport when asked."""
        client = AsyncOpenAI(api_key="offline-test", max_retries=0,
                             base_url="http://offline.test/v1",
                             http_client=httpx.AsyncClient(
                                 transport=httpx.MockTransport(handler)))
        self.addAsyncCleanup(client.close)
        return use_responses(client) if responses else client

    async def test_chat_planner_call_is_kept_with_step_site_and_reply(self):
        """A planner call over Chat Completions: request, step, caller, reply, tokens."""
        # 1. Plan once under a step, the meter installed inside the recorder as the bench does.
        recorder = PromptRecorder()
        original = AsyncCompletions.create
        with recorder.installed():
            meter = Meter()
            with meter.installed():
                async with meter.step("planning"):
                    await AsyncBTPlanner(max_attempts=1).generate_bt(
                        intents=["turn on the light"], affordances=[],
                        client=self.client(lambda r: httpx.Response(
                            200, json=chat_payload(IMPOSSIBLE))), model="local-model")
        # 2. The call as kept.
        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        self.assertEqual((call["step"], call["api"], call["model"], call["ok"]),
                         ("planning", "chat", "local-model", True))
        self.assertEqual(call["site"]["function"], "AsyncBTPlanner.generate_bt")
        self.assertTrue(call["site"]["file"].endswith("bt_planner.py"))
        messages = call["request"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "turn on the light")
        self.assertIn("tools", call["request"]["params"])
        self.assertEqual(call["response"]["tool_calls"][0]["name"], "generate_behavior_tree")
        self.assertEqual(call["response"]["usage"], {"tokens_in": 11, "tokens_out": 3})
        # 3. The SDK method is put back.
        self.assertIs(AsyncCompletions.create, original)

    async def test_responses_call_keeps_instructions_as_the_system_message(self):
        """The Responses API's instructions are shown as the system message."""
        def handle(request):
            """A function call to the planner's tool, in the Responses wire shape."""
            return httpx.Response(200, json={
                "id": "r1", "object": "response", "created_at": 1, "model": "gpt-6-luna",
                "status": "completed", "output": [{
                    "id": "fc", "type": "function_call", "call_id": "call_1",
                    "name": "generate_behavior_tree", "status": "completed",
                    "arguments": json.dumps(IMPOSSIBLE)}],
                "usage": {"input_tokens": 17, "output_tokens": 5, "total_tokens": 22,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}})

        recorder = PromptRecorder()
        original = AsyncResponses.create
        with recorder.installed():
            await AsyncBTPlanner(max_attempts=1).generate_bt(
                intents=["open the blinds"], affordances=[],
                client=self.client(handle, responses=True), model="gpt-6-luna",
                reasoning_effort="low")
        call = recorder.calls[0]
        self.assertEqual(call["api"], "responses")
        self.assertEqual([m["role"] for m in call["request"]["messages"]], ["system", "user"])
        self.assertEqual(call["response"]["tool_calls"][0]["name"], "generate_behavior_tree")
        self.assertEqual(call["response"]["usage"]["tokens_in"], 17)
        self.assertIs(AsyncResponses.create, original)

    async def test_a_failed_call_is_kept_with_its_error(self):
        """The error is recorded and still raised to the agent, which handles it."""
        recorder = PromptRecorder()
        with recorder.installed():
            result = await AsyncBTPlanner(max_attempts=1).generate_bt(
                intents=["x"], affordances=[], model="m", client=self.client(
                    lambda r: httpx.Response(400, json={"error": {
                        "message": "bad schema", "type": "invalid_request_error"}})))
        self.assertIn("LLM call failed", result["explanation"])
        self.assertFalse(recorder.calls[0]["ok"])
        self.assertIn("bad schema", recorder.calls[0]["error"])


class DecompositionTests(unittest.TestCase):
    """A message is explained by the constants of the code and the fields filled in."""

    def test_format_template_gives_its_fields_back(self):
        """The planner's system prompt (str.format, escaped braces) -> template + 3 fields."""
        text = BT_PLANNING_SYSTEM_PROMPT.format(
            capability_context="CAP", signifier_hints="SIG", observable_property_hints="OBS")
        part = prompt_parts.decompose_message(text)
        self.assertEqual((part.kind, part.name), ("template", "BT_PLANNING_SYSTEM_PROMPT"))
        fields = {c.name: c.text for c in part.children if c.kind == "field"}
        self.assertEqual(fields, {"{capability_context}": "CAP", "{signifier_hints}": "SIG",
                                  "{observable_property_hints}": "OBS"})
        self.assertEqual(prompt_parts.static_dynamic(part), (len(text) - 9, 9))
        # The long inline body is outlined by its markdown sections.
        inline = [c for c in part.children if c.kind == "inline" and c.children]
        self.assertTrue(inline and inline[0].children[0].kind == "section")

    def test_composed_constants_are_nested(self):
        """The goal prompt is built from constants that are built from constants."""
        part = prompt_parts.decompose_message(render_goal_prompt("simple", "THE CONTEXT"))
        self.assertEqual(part.name, "GOAL_STRUCTURING_SIMPLE_PROMPT")
        names = {c.name: c for c in part.children}
        self.assertIn("GOAL_SPEC_RULES", names)
        self.assertIn("CONTEXT_GUIDE", names)
        self.assertEqual(names["{goal_context}"].text, "THE CONTEXT")
        self.assertIn("GOAL_SPEC_INTRO", [c.name for c in names["GOAL_SPEC_RULES"].children])

    def test_a_constant_sent_as_it_is(self):
        """A prompt with no fields is that constant, whole."""
        part = prompt_parts.decompose_message(PLAN_SUMMARY_SYSTEM_PROMPT)
        self.assertEqual((part.kind, part.name), ("constant", "PLAN_SUMMARY_SYSTEM_PROMPT"))

    def test_user_patterns_most_specific_first(self):
        """The goal-structuring request, and the modify-recipe request that also looks
        like the planner's `<intent> + Structured goal` message."""
        goal = prompt_parts.decompose_message(
            "Request: dim the lamp\nSegmenter qualifiers: explicit, achievement")
        self.assertEqual(goal.name, "goal-structuring request")
        self.assertEqual([c.text for c in goal.children if c.kind == "field"],
                         ["dim the lamp", "explicit, achievement"])
        modify = prompt_parts.decompose_message(
            "Request: dim\n\nStructured goal:\nG\n\nAffordances:\n[]")
        self.assertEqual(modify.name, "modify-recipe request")

    def test_a_json_message_adds_up_whatever_its_style(self):
        """Fields of a JSON message, sized as the message wrote them (compact here)."""
        for text in (json.dumps({"a": "é", "b": [1, 2]}),
                     json.dumps({"a": "é", "b": [1, 2]}, separators=(",", ":"),
                                ensure_ascii=False)):
            part = prompt_parts.decompose_message(text)
            self.assertEqual(part.kind, "json")
            self.assertEqual(sum(c.chars for c in part.children), len(text))
            static, dynamic = prompt_parts.static_dynamic(part)
            self.assertGreaterEqual(static, 0)
            self.assertEqual(static + dynamic, len(text))

    def test_what_nothing_explains_is_shown_whole(self):
        """A changed prompt is reported as unexplained, not as the constant it was."""
        text = BT_PLANNING_SYSTEM_PROMPT.format(
            capability_context="", signifier_hints="", observable_property_hints="") + "!"
        self.assertEqual(prompt_parts.decompose_message(text).kind, "dynamic")
        self.assertEqual(prompt_parts.decompose_message("turn on the light").kind, "dynamic")

    def test_catalog_lists_top_level_prompts_only(self):
        """A constant inside another is shown within it, not as its own entry."""
        catalog = prompt_parts.catalog_markdown()
        self.assertIn("## GOAL_STRUCTURING_SIMPLE_PROMPT", catalog)
        self.assertNotIn("## CONTEXT_GUIDE", catalog)


class ReportTests(unittest.TestCase):
    """Choosing the episode, the run command, and the files written."""

    NAMES = ["hb10_mixed_seed_132", "hb1_feasible_seed_1085", "hb1_feasible_seed_69_fixed",
             "hb2_feasible_seed_14"]

    def test_simplest_episode_is_the_default(self):
        """Fewest parts first, then the lowest seed; a pattern picks within its matches."""
        self.assertEqual(prompt_report.pick_episode(self.NAMES, None),
                         "hb1_feasible_seed_69_fixed")
        self.assertEqual(prompt_report.pick_episode(self.NAMES, "hb2_*"), "hb2_feasible_seed_14")
        self.assertEqual(prompt_report.pick_episode(self.NAMES, "hb10_mixed_seed_132.json"),
                         "hb10_mixed_seed_132")
        with self.assertRaises(ValueError):
            prompt_report.pick_episode(self.NAMES, "hb99_*")

    def test_default_episodes_are_the_three_that_cover_the_prompts(self):
        """No --episode runs these in turn, each by exact name, with its own stop; the
        index lists them in that order and says so for one that did not run."""
        names = [episode for episode, _, _ in prompt_report.PROMPT_EPISODES]
        self.assertEqual(names, ["hb2_feasible_seed_14", "hb8_mixed_seed_224_fixed",
                                 "hb3_feasible_seed_2_fixed"])
        for name in names:
            self.assertEqual(prompt_report.pick_episode(names, name), name)
        self.assertEqual([stop for _, stop, _ in prompt_report.PROMPT_EPISODES],
                         [None, "planning", "planning"])
        rows = [(episode, stop, purpose, 0, Path("x") / episode)
                for episode, stop, purpose in prompt_report.PROMPT_EPISODES]
        rows[2] = (*rows[2][:3], 2, None)
        index = prompt_report.index_markdown("M3", rows)
        self.assertLess(index.index(names[0]), index.index(names[1]))
        self.assertLess(index.index(names[1]), index.index(names[2]))
        self.assertIn(f"({names[0]}/README.md)", index)
        self.assertIn("not run", index)

    def test_run_argv_is_the_harness_command_for_one_episode(self):
        """The resume flag and the episode patterns go; the episode, unscored, comes."""
        command = [["py", "-m", "tests.simuhome.bench.run", "--variant", "v.yaml", "--out",
                    "o", "--skip-existing", "--suite", "homebench", "--episodes", "hb*",
                    "--environment", "on"]]
        argv = prompt_report.run_argv(command, "hb1_x_seed_1", "planning")
        self.assertEqual(argv, ["--variant", "v.yaml", "--out", "o", "--suite", "homebench",
                               "--environment", "on", "--episode", "hb1_x_seed_1",
                               "--no-evaluate", "--stop-after", "planning"])

    def test_report_files_and_render_from_the_capture(self):
        """Every file is written, and --render rebuilds them from calls.json alone."""
        call = {"seq": 1, "step": "segmentation", "api": "chat", "model": "m",
                "base_url": "http://x/v1/", "started": 0.5, "seconds": 1.0, "ok": True,
                "error": None, "site": {"file": "ami_agents/a.py", "line": 3,
                                        "function": "segment"},
                "request": {"messages": [
                    {"role": "system", "content": PLAN_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": "```fenced``` request"}],
                    "params": {"temperature": 0}},
                "response": {"content": "{}", "reasoning": None, "tool_calls": [],
                             "finish_reason": "stop",
                             "usage": {"tokens_in": 100, "tokens_out": 2}}}
        meta = {"episode": "hb1_x_seed_1", "config": "c.yaml", "config_name": "c",
                "started": "now", "commit": "abc", "input": "request",
                "outcome": "executed", "models": {"segmentation": "m"}}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            prompt_report.write_report(folder, meta, [call])
            stem = "01_segmentation__segment"
            for name in ("README.md", "abstract.md", "catalog.md", "calls.json",
                         f"calls/{stem}.md", f"calls/{stem}.json",
                         f"calls/{stem}.system.txt", f"calls/{stem}.user.txt"):
                self.assertTrue((folder / name).exists(), name)
            # The text files hold the message exactly, nothing added.
            self.assertEqual((folder / f"calls/{stem}.user.txt").read_text(encoding="utf-8"),
                             "```fenced``` request")
            # A fence longer than the text's own backticks keeps the page well formed.
            self.assertIn("````text\n```fenced``` request\n````",
                          (folder / f"calls/{stem}.md").read_text(encoding="utf-8"))
            (folder / "README.md").unlink()
            prompt_report.render_only(folder)
            self.assertIn("hb1_x_seed_1", (folder / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
