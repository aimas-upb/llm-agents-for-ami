"""Tests for the decoding report (`tests/homebench/decoding_report.py`).

The trace is made by the real decoding loop against the fake Ollama of
`test_iri_trie.py`, put in a record the way the pipeline does
(`structures[].decoding`), and drawn; so a change to the trace's events that the page
does not follow fails here.
"""

import json

import pytest

from ami_agents.agents.user_assistant.utils.iri_trie import IriTrie, decode_with_iri_trie
from tests.homebench import decoding_report as dr
from tests.unit.test_iri_trie import MESSAGES, FakeOllama, _text, _top
from tests.unit.test_prompt_variants import _home_data


async def _record(first, script):
    """An episode record holding the decoding of ``first`` with the scripted calls."""
    result = await decode_with_iri_trie(FakeOllama(first, script).create, "m", MESSAGES,
                                        {}, _home_data())
    return {"episode": "hb1_feasible_seed_1", "query": "Dim the light", "outcome": "ok",
            "variant": {"name": "arm", "prompt_variants": {}},
            "structures": [{"text": "Dim the light", "decoding": result.as_dict()}]}


# -- the pieces ---------------------------------------------------------------

class TestPieces:
    def test_the_prefix_tree_merges_one_branch_chains_and_marks_the_value(self):
        tree = dr.prefix_tree(["homeont:SetModeCommand", "homeont:SetOnOffCommand",
                               "saref:SetAbsoluteLevelCommand"], "homeont:SetModeCommand")
        assert tree.splitlines() == [
            "|-- homeont:Set",
            "|   |-- ModeCommand   <== decoded",
            "|   `-- OnOffCommand   (a value)",
            "`-- saref:SetAbsoluteLevelCommand   (a value)"]

    def test_a_value_that_is_also_a_prefix_is_marked_where_it_ends(self):
        tree = dr.prefix_tree(["NA", "NAME"], "NA")
        assert tree.splitlines() == ["`-- NA   <== decoded", "    `-- ME   (a value)"]

    def test_the_verdict_says_why_a_token_is_refused(self):
        trie = IriTrie(["homeont:MasterBedroom"])
        assert dr.verdict(trie, "homeont:", "Master") == "allowed"
        assert dr.verdict(trie, "homeont:", "Bed") == "no: leaves the trie"
        assert dr.verdict(trie, "homeont:Master", '",') == "no: closes an incomplete value"

    def test_the_overruling_gap(self):
        step = {"top": [["home", -0.1], ["saref", -2.6]], "rank": 1}
        assert "a gap of 2.50 nats" in dr.overrule_line(step)
        assert "no overruling" in dr.overrule_line({"top": [["x", 0.0]], "rank": 0})
        assert "nearest" in dr.overrule_line({"top": [["x", 0.0]], "rank": None})

    def test_tokens_show_quotes_and_escape_pipes(self):
        assert dr.tok('",') == '`"\\","`'
        assert dr.tok("a|b") == '`"a\\|b"`'
        assert dr.tok(None) == "*(none)*"


# -- the page -----------------------------------------------------------------

class TestPage:
    @pytest.mark.asyncio
    async def test_a_step_is_drawn_with_its_tokens_and_the_one_taken(self):
        record = await _record(_text('{"location_class": "Laundry"}'),
                               [_top("Master", "La"), _text(""), _text("}")])
        page = dr.decoding_markdown(record, {}, "rec.json")
        # The timeline, then the value in detail: its tree, the step, the close.
        assert "| - | **outside the trie** | `location_class` = `\"Laundry\"`" in page
        assert "| - | forced | `\"homeont:\"` (one branch, no call) |" in page
        assert "| 2 | step after `\"homeont:\"` | took `\"Master\"`, rank 1 of 2 |" in page
        assert "|   |-- MasterBedroom   <== decoded" not in page   # one level, no nesting
        assert "|-- MasterBedroom   <== decoded" in page
        assert "| 1 | `\"Master\"` | 0.000 | 100.0% | allowed | **taken** |" in page
        assert "| 2 | `\"La\"` | -1.000 | 36.8% | allowed |  |" in page
        assert "```diff" in page and '+{"location_class": "homeont:MasterBedroom"}' in page

    @pytest.mark.asyncio
    async def test_the_evidence_is_shown(self):
        first = _text('{"artifact_name": "master_bedroom_light", '
                      '"affordance_class": "Dim", "affordance_name": "brightness"}')
        record = await _record(first, [_top('",'), _text(' "affordance_name": '
                                                         '"brightness"}')])
        page = dr.decoding_markdown(record, {}, "rec.json")
        assert ("written after it by the model, used as evidence: `affordance_name` = "
                "`\"brightness\"`") in page
        assert "`-- saref:SetAbsoluteLevelCommand   <== decoded" in page

    @pytest.mark.asyncio
    async def test_calls_link_to_the_prompt_report_when_certain(self, tmp_path):
        record = await _record(_text('{"location_class": "Laundry"}'),
                               [_top("Master"), _text(""), _text("}")])
        calls = [{"seq": 1, "step": "segmentation", "site": {"function": "seg"}}] + [
            {"seq": n, "step": "structuring.intent", "site": {"function": "f"}}
            for n in range(2, 6)]
        (tmp_path / "calls.json").write_text(json.dumps({"calls": calls}))
        pages = dr.call_pages(tmp_path, record)
        assert pages == {1: "calls/02_structuring.intent__f.md",
                         2: "calls/03_structuring.intent__f.md",
                         3: "calls/04_structuring.intent__f.md",
                         4: "calls/05_structuring.intent__f.md"}
        # A count that does not match (a retry, say): no links rather than wrong ones.
        (tmp_path / "calls.json").write_text(json.dumps({"calls": calls[:-1]}))
        assert dr.call_pages(tmp_path, record) == {}

    def test_a_record_without_a_trace_lists_its_interventions(self):
        record = {"episode": "e", "variant": {"name": "arm"}, "structures": [{
            "text": "t", "decoding": {"calls": 3, "interventions": [
                {"slot": "location_class", "free": "Foyer", "decoded": "homeont:Foyer",
                 "how": "trie", "steps": 0, "conditioned": True, "candidates": 1}]}}]}
        page = dr.decoding_markdown(record, {}, "rec.json")
        assert "`\"Foyer\"` | `\"homeont:Foyer\"` | trie" in page
        assert "This record has no trace" in page

    def test_a_record_of_another_arm_says_so(self):
        page = dr.decoding_markdown({"episode": "e", "structures": [{"text": "t"}]}, {},
                                    "rec.json")
        assert "No structuring reply of this record was decoded with the trie" in page
