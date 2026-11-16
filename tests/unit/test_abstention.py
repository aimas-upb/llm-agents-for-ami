"""Tests for abstention in the constrained decodings (`utils/abstention.py`,
`decoding: iri_trie_v2`): a device kind the room lacks is reported through
`unmatched_device`, not replaced by the room's nearest device.

The home is `_home_data()` of the prompt-variant tests: a Master Bedroom with a
`DimmableLight` and a Laundry room with a `Washer`.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ami_agents.agents.user_assistant.utils import abstention as ab
from ami_agents.agents.user_assistant.utils import goal_structuring_variants as gsv
from ami_agents.agents.user_assistant.utils.goal_context import GoalContext, vocabulary
from ami_agents.agents.user_assistant.utils.iri_trie import decode_with_iri_trie, iri_paths
from tests.unit.test_iri_trie import MESSAGES, FakeOllama, _text
from tests.unit.test_prompt_variants import _home_data


class TestClassNames:
    def test_a_key_ignores_prefix_case_spacing_and_plural(self):
        assert ab.class_key("homeont:AirPurifier") == ab.class_key("air purifiers")
        assert ab.class_key(None) == "" and ab.class_key("") == ""

    def test_the_family_is_the_last_camel_word(self):
        assert {ab.class_family(c) for c in ("homeont:OnOffLight", "DimmableLight",
                                             "homeont:ColorLight")} == {"light"}
        assert ab.class_family("homeont:AirPurifier") == "purifier"
        assert ab.class_family(None) == ""

    def test_words_are_what_a_user_says(self):
        assert ab.class_words("homeont:AirConditioner") == "air conditioner"
        assert ab.class_words("homeont:Fan") == "fan"


class TestOntology:
    def test_device_families_are_read_from_the_ontology(self):
        keys = ab.device_classes()
        assert {"airconditioner", "dimmablelight", "humidifier"} <= keys

    def test_rooms_and_commands_are_not_devices(self):
        keys = ab.device_classes()
        assert "masterbedroom" not in keys and "setonoffcommand" not in keys


class TestDeviceAbsent:
    def setup_method(self):
        self.data = _home_data()
        self.rows, self.vocab = iri_paths(self.data), vocabulary(self.data)

    def absent(self, free_class, room):
        return ab.device_absent(free_class, room, self.rows, self.vocab)

    def test_a_device_kind_the_room_lacks_is_absent(self):
        assert self.absent("homeont:AirConditioner", "homeont:MasterBedroom")

    def test_the_room_is_read_as_the_model_wrote_it(self):
        assert self.absent("AirConditioner", "Master bedroom")

    def test_a_device_the_room_has_is_not_absent(self):
        assert not self.absent("homeont:DimmableLight", "homeont:MasterBedroom")

    def test_a_sibling_of_the_rooms_device_is_not_absent(self):
        # A ColorLight request in a room with a DimmableLight: the same family, so the
        # trie may repair it (hb1_feasible_seed_3528).
        assert not self.absent("homeont:ColorLight", "homeont:MasterBedroom")

    def test_a_value_that_is_not_a_device_kind_is_left_to_the_trie(self):
        assert not self.absent("homeont:Thermostat", "homeont:MasterBedroom")
        assert not self.absent("study_room_ac", "homeont:MasterBedroom")
        assert not self.absent(None, "homeont:MasterBedroom")

    def test_an_unknown_room_cannot_be_judged(self):
        assert not self.absent("homeont:AirConditioner", "Store Room")
        assert not self.absent("homeont:AirConditioner", None)


GOAL = ('{\n  "goal_specificity": "explicit",\n  "location_class": '
        '"homeont:MasterBedroom",\n  "artifact_class": "homeont:AirConditioner",\n'
        '  "artifact_name": "master_bedroom_ac",\n  "affordance_class": '
        '"saref:SetAbsoluteLevelCommand",\n  "affordance_name": "temperature",\n'
        '  "parameter_name": null,\n  "target_value_text": "18",\n'
        '  "unmatched_device": null,\n  "unmatched_action": null\n}')


class TestAbstainObject:
    def test_the_iri_slots_become_null_and_the_device_is_named(self):
        goal = json.loads(ab.abstain_object(GOAL, "air conditioner"))
        assert goal["unmatched_device"] == "air conditioner"
        for slot in ("artifact_class", "artifact_name", "affordance_class",
                     "affordance_name"):
            assert goal[slot] is None
        # The room and the request's own words are not touched.
        assert goal["location_class"] == "homeont:MasterBedroom"
        assert goal["target_value_text"] == "18"

    def test_an_object_without_the_key_gets_it(self):
        bare = '{"location_class": "homeont:MasterBedroom", "artifact_class": "x"}'
        goal = json.loads(ab.abstain_object(bare, "fan"))
        assert goal == {"location_class": "homeont:MasterBedroom",
                        "artifact_class": None, "unmatched_device": "fan"}


class TestOverlay:
    def setup_method(self):
        self.data = _home_data()
        self.rows, self.vocab = iri_paths(self.data), vocabulary(self.data)

    def test_the_decoded_goal_is_rewritten_when_the_free_one_names_a_missing_device(self):
        decoded = GOAL.replace("AirConditioner", "DimmableLight")
        text, done = ab.overlay(GOAL, decoded, self.rows, self.vocab)
        assert json.loads(text)["unmatched_device"] == "air conditioner"
        assert json.loads(text)["artifact_class"] is None
        assert done == [{"index": 0, "free_class": "homeont:AirConditioner",
                         "room": "homeont:MasterBedroom", "words": "air conditioner"}]

    def test_a_goal_the_home_can_serve_is_left_alone(self):
        free = GOAL.replace("AirConditioner", "DimmableLight")
        text, done = ab.overlay(free, free, self.rows, self.vocab)
        assert text == free and done == []

    def test_replies_with_different_numbers_of_goals_are_not_paired(self):
        two = f'{{"G1": {GOAL}, "G2": {GOAL}}}'
        text, done = ab.overlay(two, GOAL, self.rows, self.vocab)
        assert text == GOAL and done == []

    def test_each_goal_of_a_dependency_reply_is_judged_on_its_own(self):
        served = GOAL.replace("AirConditioner", "DimmableLight")
        free = f'{{"G1": {served}, "G2": {GOAL}}}'
        text, done = ab.overlay(free, free, self.rows, self.vocab)
        goals = json.loads(text)
        assert goals["G1"]["artifact_class"] == "homeont:DimmableLight"
        assert goals["G2"]["unmatched_device"] == "air conditioner"
        assert [d["index"] for d in done] == [1]


class TestTrieAbstention:
    @pytest.mark.asyncio
    async def test_a_missing_device_is_reported_not_replaced(self):
        free = _text(GOAL)
        fake = FakeOllama(free, [])
        result = await decode_with_iri_trie(fake.create, "m", MESSAGES, {},
                                            _home_data(), abstain=True)
        goal = json.loads(result.text)
        assert goal["unmatched_device"] == "air conditioner"
        assert goal["artifact_class"] is None and goal["artifact_name"] is None
        [change] = result.interventions
        assert change["how"] == "abstain" and change["decoded"] is None
        assert change["free"] == "homeont:AirConditioner"
        # No call but the free one: nothing was decoded again.
        assert result.calls == 1
        events = [e["event"] for e in result.trace]
        assert events[:2] == ["free", "kept"] and "abstain" in events
        assert result.as_dict()["abstain"] is True

    @pytest.mark.asyncio
    async def test_without_the_option_the_trie_still_replaces_it(self):
        # `iri_trie` as it was: the class is decoded again under the room's trie.
        reply = _text('{"location_class": "homeont:MasterBedroom", '
                      '"artifact_class": "homeont:AirConditioner"}')
        from tests.unit.test_iri_trie import _top
        fake = FakeOllama(reply, [_top('"}', '"'), _text("")])
        result = await decode_with_iri_trie(fake.create, "m", MESSAGES, {}, _home_data())
        assert json.loads(result.text)["artifact_class"] == "homeont:DimmableLight"
        assert "abstain" not in result.as_dict()

    @pytest.mark.asyncio
    async def test_a_misspelt_class_is_still_repaired_by_the_trie(self):
        reply = _text('{"location_class": "homeont:MasterBedroom", '
                      '"artifact_class": "Dimmable light"}')
        from tests.unit.test_iri_trie import _top
        fake = FakeOllama(reply, [_top('"}', '"'), _text("")])
        result = await decode_with_iri_trie(fake.create, "m", MESSAGES, {},
                                            _home_data(), abstain=True)
        assert json.loads(result.text)["artifact_class"] == "homeont:DimmableLight"
        assert result.interventions[0]["how"] == "trie"


class TestVariants:
    def test_the_new_options_are_on_the_decoding_axis(self):
        assert {"iri_trie_v2", "genre_v2"} <= set(gsv.DECODING.options)

    @pytest.mark.asyncio
    async def test_iri_trie_v2_is_not_with_a_schema(self):
        llm = SimpleNamespace(client=MagicMock(), model="m")
        with pytest.raises(ValueError, match="response_format"):
            await gsv.iri_trie_v2_decoding(llm, MESSAGES, {"response_format": {}},
                                           GoalContext("T", _home_data()))
