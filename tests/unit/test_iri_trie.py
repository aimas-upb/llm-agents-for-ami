"""Tests for the GENRE-style trie decoding of structuring replies (`decoding: iri_trie`).

The decoding loop runs against a fake `create` that answers like Ollama: a first free
reply, then for a prefilled assistant message either one token with its top logprobs
or a free continuation.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ami_agents.agents.user_assistant.utils import goal_structuring_variants as gsv
from ami_agents.agents.user_assistant.utils.goal_context import GoalContext, vocabulary
from ami_agents.agents.user_assistant.utils.iri_trie import (
    IriTrie,
    decode_with_iri_trie,
    first_invalid_slot,
    iri_paths,
    nearest,
    slot_values,
    written_slots,
)
from ami_agents.shared.utils import prompt_variants as pv
from tests.unit.test_prompt_variants import BARE, _home_data, _run

ROOMS = ["homeont:MasterBedroom", "homeont:LaundryRoom"]


# -- the trie -----------------------------------------------------------------

class TestIriTrie:
    def test_prefixes_and_complete_values(self):
        trie = IriTrie(ROOMS)
        assert trie.has_prefix("homeont:M") and not trie.has_prefix("Master")
        assert trie.is_complete("homeont:LaundryRoom") and not trie.is_complete("homeont:")

    def test_forced_extension_stops_where_the_trie_branches(self):
        trie = IriTrie(ROOMS)
        assert trie.forced_extension("") == "homeont:"
        assert trie.forced_extension("homeont:M") == "asterBedroom"

    def test_a_token_is_allowed_only_inside_the_trie(self):
        trie = IriTrie(ROOMS)
        assert trie.accept("", "home") == ("home", None)
        assert trie.accept("", "Master") is None
        assert trie.accept("", "") is None

    def test_a_closing_quote_only_on_a_complete_value(self):
        trie = IriTrie(["NA", "NAME"])
        assert trie.accept("N", 'A",') == ("NA", '",')
        assert trie.accept("", '"') is None
        assert not trie.is_leaf("NA") and trie.is_leaf("NAME")

    def test_nearest_ignores_notation(self):
        assert nearest("Master bedroom", ROOMS) == "homeont:MasterBedroom"


# -- the valid paths ----------------------------------------------------------

class TestPaths:
    def setup_method(self):
        self.data = _home_data()
        self.rows, self.vocab = iri_paths(self.data), vocabulary(self.data)

    def test_device_names_follow_the_written_room(self):
        allowed = slot_values("artifact_name", {"location_class": "homeont:LaundryRoom"},
                              self.rows, self.vocab)
        assert allowed.values == ["laundry_washer"] and allowed.conditioned

    def test_actions_follow_the_written_device(self):
        allowed = slot_values("affordance_name", {"artifact_name": "master_bedroom_light"},
                              self.rows, self.vocab)
        assert allowed.values == ["brightness", "onOff"]

    def test_parameters_of_the_action_and_na(self):
        allowed = slot_values("parameter_name", {"affordance_name": "state"},
                              self.rows, self.vocab)
        assert allowed.values == ["operation", "NA"]

    def test_a_home_without_parameters_still_allows_only_na(self):
        data = _home_data()
        data["workspaces"][0]["workspaces"].pop()            # no washer: no parameter
        rows, vocab = iri_paths(data), vocabulary(data)
        assert vocab.parameter_names == []
        assert slot_values("parameter_name", {"affordance_name": "brightness"}, rows,
                           vocab).values == ["NA"]
        text = json.dumps({"affordance_name": "brightness", "parameter_name": "volume"})
        assert first_invalid_slot(text, 0, rows, vocab).trie.values == ["NA"]

    def test_no_fitting_path_falls_back_to_the_whole_vocabulary(self):
        allowed = slot_values("affordance_name", {"artifact_name": "laundry_washer",
                                                  "location_class": "homeont:MasterBedroom"},
                              self.rows, self.vocab)
        assert not allowed.conditioned
        assert set(allowed.values) == {"brightness", "onOff", "state"}

    def test_a_value_the_home_lacks_does_not_narrow(self):
        allowed = slot_values("artifact_name", {"location_class": "Bedroom"},
                              self.rows, self.vocab)
        assert allowed.conditioned and len(allowed.values) == 2

    def test_environment_variables_and_rooms(self):
        assert slot_values("property_class", {"property_name": "illuminance"}, self.rows,
                           self.vocab).values == ["homeont:Illuminance"]
        assert slot_values("space_name", {"space_class": "homeont:LaundryRoom"}, self.rows,
                           self.vocab).values == ["Laundry"]


# -- reading the reply --------------------------------------------------------

class TestReply:
    def setup_method(self):
        self.data = _home_data()
        self.rows, self.vocab = iri_paths(self.data), vocabulary(self.data)

    def test_written_slots_are_those_of_the_open_object(self):
        text = ('{"G1": {"artifact_name": "laundry_washer"}, '
                '"G2": {"location_class": "homeont:MasterBedroom", "artifact_name": "')
        assert written_slots(text, len(text)) == {"location_class": "homeont:MasterBedroom"}

    def test_valid_values_pass_and_the_first_invalid_is_found(self):
        text = json.dumps({"location_class": "homeont:MasterBedroom",
                           "artifact_name": "master_bedroom_light",
                           "affordance_class": "SetAbsoluteLevelCommand"})
        invalid = first_invalid_slot(text, 0, self.rows, self.vocab)
        assert invalid.slot == "affordance_class"
        assert invalid.written == "SetAbsoluteLevelCommand"
        assert invalid.trie.values == ["saref:SetAbsoluteLevelCommand",
                                       "homeont:SetOnOffCommand"]

    def test_null_and_free_slots_are_not_checked(self):
        text = json.dumps({"location_class": None, "intent_text": "x",
                           "parent_class": "anything", "parameter_name": "NA"})
        assert first_invalid_slot(text, 0, self.rows, self.vocab) is None

    def test_a_valid_name_in_the_wrong_room_is_invalid(self):
        text = json.dumps({"location_class": "homeont:LaundryRoom",
                           "artifact_name": "master_bedroom_light"})
        invalid = first_invalid_slot(text, 0, self.rows, self.vocab)
        assert invalid.slot == "artifact_name" and invalid.trie.values == ["laundry_washer"]


# -- decoding -----------------------------------------------------------------

def _top(*tokens):
    """A one-token response whose top logprobs are ``tokens``, most probable first."""
    tops = [SimpleNamespace(token=t, logprob=-i) for i, t in enumerate(tokens)]
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=tokens[0]),
        logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=tops)]))])


def _text(content, reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning=reasoning), logprobs=None)])


class FakeOllama:
    """Answers the free call, then the scripted prefilled calls, recording each."""

    def __init__(self, first, script):
        self.first, self.script, self.calls = first, list(script), []

    async def create(self, model, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if messages[-1]["role"] != "assistant":
            return self.first
        return self.script.pop(0)


MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


class TestDecoding:
    @pytest.mark.asyncio
    async def test_a_reply_inside_its_tries_costs_one_call(self):
        reply = json.dumps({"location_class": "homeont:MasterBedroom",
                            "artifact_name": "master_bedroom_light"})
        fake = FakeOllama(_text(reply), [])
        result = await decode_with_iri_trie(fake.create, "m", MESSAGES,
                                            {"temperature": 0}, _home_data())
        assert result.text == reply and result.calls == 1 and result.interventions == []

    @pytest.mark.asyncio
    async def test_an_unprefixed_class_is_decoded_under_the_trie(self):
        first = _text('\n\n{"location_class": "Master bedroom", "artifact_name": "x"}',
                      reasoning="\nthe master bedroom\n")
        fake = FakeOllama(first, [
            # "homeont:" is forced (both rooms start so); "Bed" leaves the trie.
            _top("Bed", "Master", "Laundry"),
            # "Bedroom" is forced; the closing token is the model's own `",`.
            _top("Bed", '",', '"'),
            _text(' "artifact_name": "master_bedroom_light"}'),
        ])
        result = await decode_with_iri_trie(fake.create, "m", MESSAGES,
                                            {"temperature": 0, "max_tokens": 900},
                                            _home_data())
        assert json.loads(result.text) == {"location_class": "homeont:MasterBedroom",
                                           "artifact_name": "master_bedroom_light"}
        [change] = result.interventions
        assert change["free"] == "Master bedroom" and change["how"] == "trie"
        # The thinking is prefilled back; a step asks for one token and 20 logprobs.
        step = fake.calls[1]
        assert step["messages"][-1]["content"].startswith("<think>\nthe master bedroom\n"
                                                          "</think>")
        assert step["max_tokens"] == 1 and step["top_logprobs"] == 20
        assert fake.calls[-1]["max_tokens"] == 900 and result.calls == 4

    @pytest.mark.asyncio
    async def test_no_allowed_top_token_snaps_to_the_nearest_value(self):
        first = _text('{"location_class": "Laundry room"}')
        fake = FakeOllama(first, [_top("Wash", " Laundry"), _text("Laundry room"),
                                  _top("Room", '"}'), _text("")])
        result = await decode_with_iri_trie(fake.create, "m", MESSAGES, {}, _home_data())
        assert json.loads(result.text) == {"location_class": "homeont:LaundryRoom"}
        assert result.interventions[0]["how"] == "nearest"
        assert fake.calls[2]["stop"] == ['"']


class TestVariant:
    def test_the_axis_defaults_to_free(self):
        assert pv.resolve("structuring").chosen["decoding"] == "free"
        assert "iri_trie" in gsv.DECODING.options

    @pytest.mark.asyncio
    async def test_the_behaviour_records_what_the_trie_changed(self):
        bare = dict(BARE, location_class="homeont:MasterBedroom",
                    artifact_class="homeont:DimmableLight",
                    affordance_class="saref:SetAbsoluteLevelCommand")
        behaviour, call = await _run({"decoding": "iri_trie"}, bare,
                                     GoalContext("HOME TEXT", _home_data()))
        assert behaviour.decoding == {"option": "iri_trie", "calls": 1,
                                      "interventions": []}
        assert behaviour.result.goals["G1"].artifact_name == "master_bedroom_light"

    @pytest.mark.asyncio
    async def test_the_default_records_nothing(self):
        behaviour, _ = await _run({}, BARE, GoalContext("HOME TEXT", _home_data()))
        assert behaviour.decoding is None

    @pytest.mark.asyncio
    async def test_not_with_a_schema(self):
        llm = SimpleNamespace(client=MagicMock(), model="m")
        with pytest.raises(ValueError, match="response_format"):
            await gsv.iri_trie_decoding(llm, MESSAGES, {"response_format": {}},
                                        GoalContext("T", _home_data()))
