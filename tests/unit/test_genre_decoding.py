"""Tests for the token-level rules of GENRE decoding (`decoding: genre`).

The rules are torch-free: a fake vocabulary of token strings stands in for the model's,
and `HomeRules.next_tokens` is asked what may follow a reply written so far. The option
itself runs against a fake chat client that answers like `genre_server.py`.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ami_agents.agents.user_assistant.utils import goal_structuring_variants as gsv
from ami_agents.agents.user_assistant.utils.genre_decoding import (
    HomeRules, JsonPrefix, TokenTrie, changed_slots, json_allowed, may_end,
    slot_pairs,
)
from ami_agents.agents.user_assistant.utils.goal_context import GoalContext
from tests.unit.test_prompt_variants import _home_data

TOKENS = [
    "homeont", ":", "Master", "Bed", "room", "Bedroom_", "Laundry", "Room",   # 0-7
    "homeont:", "saref", ' "', '"', '",', '"}', '",\n  "', '"abc', "master",  # 8-16
    "_", "bed", "light", "NA", "A", "N",                                      # 17-22
]
ID = {text: i for i, text in enumerate(TOKENS)}


def _ids(*texts):
    return {ID[t] for t in texts}


class TestTokenTrie:
    def test_prefixes_of_a_text(self):
        trie = TokenTrie(TOKENS)
        assert trie.prefixes_of("homeont:Master") == _ids("homeont", "homeont:")
        assert trie.prefixes_of("xyz") == set()

    def test_tokens_starting_with_a_text(self):
        trie = TokenTrie(TOKENS)
        assert trie.starting_with('"') == _ids('"', '",', '"}', '",\n  "', '"abc')

    def test_an_empty_or_missing_token_is_never_listed(self):
        trie = TokenTrie(["a", None, ""])
        assert trie.prefixes_of("a") == {0}
        assert trie.starting_with("") == {0}


class TestNextTokens:
    def setup_method(self):
        self.rules = HomeRules(_home_data(), TokenTrie(TOKENS))

    def _next(self, text):
        return self.rules.next_tokens(text)

    def test_outside_a_value_nothing_is_masked(self):
        assert self._next('{"goal_kind": "achievement", "intent_text": "set') is None
        assert self._next('{"location_class": "homeont:MasterBedroom", "artifact') is None

    def test_values_no_token_can_write_leave_an_empty_set_for_the_caller(self):
        # the server leaves such a row unmasked (counted as a dead step)
        allowed = self._next('{"property_name": "')
        assert allowed.slot == "property_name" and allowed.ids == set()

    def test_at_the_start_of_a_room_only_tokens_that_prefix_a_room(self):
        allowed = self._next('{"location_class": "')
        assert allowed.slot == "location_class" and allowed.prefix == ""
        assert allowed.ids == _ids("homeont", "homeont:")

    def test_inside_a_name_the_tokens_that_keep_it_in_the_trie(self):
        allowed = self._next('{"location_class": "homeont:')
        assert allowed.ids == _ids("Master", "Laundry")
        allowed = self._next('{"location_class": "homeont:Master')
        assert allowed.ids == _ids("Bed")
        assert not allowed.dead

    def test_every_tokenization_of_a_name_is_accepted(self):
        tokens = TOKENS + ["Bedroom"]
        rules = HomeRules(_home_data(), TokenTrie(tokens))
        allowed = rules.next_tokens('{"location_class": "homeont:Master')
        assert {tokens[i] for i in allowed.ids} == {"Bed", "Bedroom"}

    def test_a_complete_value_may_be_closed_in_the_same_token(self):
        allowed = self._next('{"location_class": "homeont:LaundryRoom')
        assert allowed.ids == _ids('"', '",', '"}', '",\n  "')
        assert ID['"abc'] not in allowed.ids       # a letter cannot follow a quote

    def test_a_token_that_finishes_the_name_and_closes_it(self):
        rules = HomeRules(_home_data(), TokenTrie(["homeont:LaundryRoom\","] + TOKENS))
        allowed = rules.next_tokens('{"location_class": "')
        assert 0 in allowed.ids

    def test_the_trie_is_conditioned_on_the_object(self):
        text = '{"location_class": "homeont:LaundryRoom", "artifact_name": "'
        allowed = self._next(text)
        # the room has one device, laundry_washer: only "laundry..." could be written
        assert allowed.slot == "artifact_name"
        assert ID["master"] not in allowed.ids

    def test_a_value_outside_its_trie_is_a_dead_end_left_unmasked(self):
        allowed = self._next('{"location_class": "Garage')
        assert allowed.dead and allowed.ids == set()

    def test_a_closed_value_is_not_constrained_again(self):
        assert self._next('{"location_class": "homeont:LaundryRoom", "x": 1') is None


class TestSlotReading:
    def test_slot_pairs_in_order(self):
        text = '{"location_class": "A", "intent_text": "x", "artifact_name": "b"}'
        assert slot_pairs(text) == [("location_class", "A"), ("artifact_name", "b")]

    def test_changed_slots_pair_by_position(self):
        free = '{"location_class": "Foyer", "artifact_name": "light"}'
        decoded = '{"location_class": "homeont:Foyer", "artifact_name": "light"}'
        assert changed_slots(free, decoded) == [
            {"slot": "location_class", "free": "Foyer", "decoded": "homeont:Foyer",
             "index": 0}]

    def test_pairing_stops_where_the_shapes_differ(self):
        free = '{"location_class": "A", "artifact_name": "b"}'
        decoded = '{"location_class": "A", "artifact_class": "c", "artifact_name": "d"}'
        assert changed_slots(free, decoded) == []


class TestMayEnd:
    def test_a_complete_object_may_end(self):
        assert may_end('{"location_class": null}')
        assert may_end('```json\n{"a": 1}\n```')
        assert may_end('\n{"a": {"b": 2}}\n')

    def test_nothing_or_a_partial_reply_may_not_end(self):
        assert not may_end("")
        assert not may_end('{"a": 1,')
        assert not may_end('```json\n{"a": 1')

    def test_a_comment_or_a_non_object_is_not_a_reply(self):
        assert not may_end('{"a": null,  // no room\n "b": 1}')
        assert not may_end("[1, 2]")


class TestJsonPrefix:
    DOCS = [
        '{"location_class": null, "artifact_name": "homeont:Lamp", "n": [1, -2.5e+3]}',
        '{"a": {"b": [true, false, null, {}], "c": []}, "d": "x\\"y\\u00e9\\n"}',
        '```json\n{"a": 0.5, "b": [ ], "c": { }}\n```\n',
        '\n  {"a":1,"b":{"c":[{"d":-0}]}}  ',
    ]

    def test_every_prefix_of_a_json_object_is_accepted(self):
        for doc in self.DOCS:
            for end in range(len(doc) + 1):
                assert JsonPrefix.of(doc[:end]).ok, doc[:end]

    def test_the_comment_after_a_null_is_refused(self):
        # hb1_infeasible_seed_4644: `"location_class": null,  // ...`
        state = JsonPrefix.of('{"location_class": null,  ')
        assert state.ok and not state.accepts("//") and not state.accepts("/")
        assert state.accepts('"artifact_class"')

    @pytest.mark.parametrize("text", [
        "Sure: {", '{"a": 1,}', '{"a" 1}', '{"a": 1} extra', '{"a": 1}\n```',
        '{"a": tru}', '{"a": 01}', '{"a": 1.}', '{"a": -}', '{"a": "x\ny"}',
        '{"a": "\\q"}', '{"a": "\\u12g4"}', "[1]", '{"a": [1, 2}', "{1: 2}",
        '{"a": 1}}', '```json\n```json', '{"a": nul1}', '{"a": 1e}',
    ])
    def test_what_cannot_become_a_json_object_is_refused(self, text):
        assert not JsonPrefix.of(text).ok

    def test_a_number_ends_where_a_structural_character_follows(self):
        assert JsonPrefix.of('{"a": 12').accepts(', "b": 1}')
        assert JsonPrefix.of('{"a": 1.5e2').accepts("}")
        assert not JsonPrefix.of('{"a": 1.').accepts("}")

    def test_a_closing_fence_is_accepted_only_after_an_opening_one(self):
        assert JsonPrefix.of('```json\n{"a": 1}\n').accepts("```")
        assert not JsonPrefix.of('{"a": 1}\n').accepts("```")
        assert not JsonPrefix.of('```json\n{"a": 1}\n```').accepts("x")

    def test_a_complete_document_is_one_may_end_accepts(self):
        for doc in self.DOCS:
            assert may_end(doc) and JsonPrefix.of(doc).ok

    def test_accepts_does_not_change_the_state(self):
        state = JsonPrefix.of('{"a": [1, ')
        assert state.accepts('{"b": 2}]}') and not state.accepts("}")
        assert state.stack == ["o", "a"] and state.ok


class TestJsonAllowed:
    TOK = ['"', ",", "//", " ", "}", "x", None, None, "\n"]    # 6: special, 7: bytes

    def _allowed(self, text):
        trie = TokenTrie(self.TOK, byte_pieces=[7])
        ids = range(len(self.TOK) + 2)                          # two padding rows
        return set(json_allowed(JsonPrefix.of(text), trie, ids))

    def test_after_a_null_only_json_continues(self):
        assert self._allowed('{"a": null') == {1, 3, 4, 8}

    def test_a_key_may_start_and_a_comment_may_not(self):
        assert self._allowed('{"a": null,') == {0, 3, 8}

    def test_in_a_string_anything_but_a_control_character_continues(self):
        assert self._allowed('{"a": "') == {0, 1, 2, 3, 4, 5, 7}

    def test_a_piece_of_a_character_is_only_for_strings(self):
        assert 7 not in self._allowed('{"a": null')
        assert 6 not in self._allowed('{"a": "')                # a special token: never



class TestGenreOption:
    def _cfg(self, response):
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        return SimpleNamespace(client=client, model="m"), create

    def _response(self, record):
        message = SimpleNamespace(content='{"location_class": "homeont:Foyer"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], genre=record)

    @pytest.mark.asyncio
    async def test_the_request_carries_the_home_and_the_record_comes_back(self):
        data = _home_data()
        cfg, create = self._cfg(self._response({"option": "genre", "beams": 4}))
        decoded = await gsv.genre_decoding(cfg, [{"role": "user", "content": "x"}],
                                           {"temperature": 0}, GoalContext("", data))
        assert decoded.text == '{"location_class": "homeont:Foyer"}'
        assert decoded.record == {"option": "genre", "beams": 4}
        kwargs = create.call_args.kwargs
        assert kwargs["extra_body"] == {"genre": {"data": data}}
        assert kwargs["temperature"] == 0

    @pytest.mark.asyncio
    async def test_an_endpoint_without_a_genre_record_is_an_error(self):
        cfg, _ = self._cfg(SimpleNamespace(
            choices=[SimpleNamespace(message=MagicMock(content="{}"))]))
        with pytest.raises(ValueError, match="genre_server"):
            await gsv.genre_decoding(cfg, [], {}, GoalContext("", _home_data()))

    @pytest.mark.asyncio
    async def test_not_with_a_response_format(self):
        cfg, _ = self._cfg(self._response({}))
        with pytest.raises(ValueError, match="response_format"):
            await gsv.genre_decoding(cfg, [], {"response_format": {}},
                                     GoalContext("", _home_data()))

    @pytest.mark.asyncio
    async def test_the_home_data_is_required(self):
        cfg, _ = self._cfg(self._response({}))
        with pytest.raises(ValueError):
            await gsv.genre_decoding(cfg, [], {}, GoalContext("text only"))

    def test_the_option_is_registered_and_iri_trie_is_untouched(self):
        assert {"free", "iri_trie", "genre"} <= set(gsv.DECODING.options)


class TestV2Rules:
    """`HomeRules.token_filter`: what decoding v2 holds a token to outside a value."""

    TOK = ["null", ' "', '"', " ", '"Store', "ation", "ation_class", "-class", '":',
           '": "', "{", "_class"]
    ID = {text: i for i, text in enumerate(TOK)}

    def _rules(self, **kwargs):
        return HomeRules(_home_data(), TokenTrie(self.TOK), **kwargs)

    def _filter(self, rules, text):
        return rules.token_filter(text, JsonPrefix.of(text))

    def test_v1_holds_no_token_to_anything(self):
        rules = self._rules()
        assert self._filter(rules, '{"location_class": ') is None
        assert self._filter(rules, '{"loc') is None

    def test_null_is_not_a_token_for_a_room_the_home_has_values_for(self):
        accept = self._filter(self._rules(no_null=True), '{"location_class": ')
        assert accept(' "') and accept('"') and accept(" ")
        assert not accept("null") and not accept("{")

    def test_a_token_that_starts_the_value_itself_is_out(self):
        # `"Store` would write the first characters of the value past the trie.
        accept = self._filter(self._rules(no_null=True), '{"artifact_class":  ')
        assert not accept('"Store')

    def test_a_slot_outside_the_list_may_still_be_null(self):
        assert self._filter(self._rules(no_null=True), '{"parameter_name": ') is None
        assert self._filter(self._rules(no_null=True), '{"unmatched_device": ') is None

    def test_a_slot_the_home_has_no_values_for_may_still_be_null(self):
        home = {"workspaces": [{"title": "home", "artifacts": [], "workspaces": []}]}
        rules = HomeRules(home, TokenTrie(self.TOK), no_null=True)
        assert self._filter(rules, '{"location_class": ') is None

    def test_a_key_stays_a_prefix_of_a_key_the_free_answer_wrote(self):
        rules = self._rules(keys=["location_class", "artifact_class"])
        accept = self._filter(rules, '{"loc')
        assert not accept("-class") and not accept("{")
        accept = self._filter(rules, '{"location')
        assert accept("_class") and not accept("-class")

    def test_a_key_may_be_closed_only_on_a_whole_one(self):
        rules = self._rules(keys=["location_class"])
        assert self._filter(rules, '{"location_class')('":')
        assert self._filter(rules, '{"location_class')('": "')
        assert not self._filter(rules, '{"location')('":')

    def test_the_filter_narrows_what_json_allows(self):
        rules = self._rules(no_null=True)
        state = JsonPrefix.of('{"location_class": ')
        everything = set(json_allowed(state, rules.tokens, range(len(self.TOK))))
        held = set(json_allowed(state, rules.tokens, range(len(self.TOK)),
                                rules.token_filter('{"location_class": ', state)))
        assert self.ID["null"] in everything and self.ID["null"] not in held
        assert held == {self.ID[' "'], self.ID['"'], self.ID[" "]}


class TestFilterIds:
    """`HomeRules.filter_ids` is `token_filter` found from the trie: the same tokens."""

    TOK = TestV2Rules.TOK + ['_class"', '_class":', '_class"}', "location", "loc",
                             'ation_class": "', "ation_class\"x", "", None]
    KEYS = ["location_class", "artifact_class", "unmatched_device"]

    def _rules(self, **kwargs):
        return HomeRules(_home_data(), TokenTrie(self.TOK), **kwargs)

    def _same(self, rules, text):
        state = JsonPrefix.of(text)
        accept = rules.token_filter(text, state)
        held = rules.filter_ids(text, state)
        assert (held is None) == (accept is None)
        if accept is None:
            return None
        every = range(len(self.TOK))
        assert set(json_allowed(state, rules.tokens, held)) == \
            set(json_allowed(state, rules.tokens, every, accept))
        return held

    def test_no_filter_is_none(self):
        assert self._same(self._rules(), '{"location_class": ') is None
        assert self._same(self._rules(no_null=True), '{"parameter_name": ') is None

    def test_no_null_value_opening(self):
        rules = self._rules(no_null=True)
        for text in ('{"location_class": ', '{"artifact_class":  ',
                     '{"a": 1, "affordance_name": '):
            assert self._same(rules, text)

    def test_every_key_prefix(self):
        rules = self._rules(keys=self.KEYS)
        for key in self.KEYS:
            for cut in range(len(key) + 1):
                self._same(rules, '{"a": 1, "' + key[:cut])
        assert self._same(rules, '{"unknown_ke') == set()

    def test_both_rules_together(self):
        rules = self._rules(no_null=True, keys=self.KEYS)
        self._same(rules, '{"location_class": ')
        self._same(rules, '{"loc')


class TestGenreV2Option:
    def _cfg(self, record):
        message = SimpleNamespace(content="{}")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                   genre=record)
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        return SimpleNamespace(client=client, model="m"), create

    @pytest.mark.asyncio
    async def test_the_request_switches_the_four_rules_on(self):
        data = _home_data()
        cfg, create = self._cfg({"option": "genre"})
        decoded = await gsv.genre_v2_decoding(cfg, [], {"temperature": 0},
                                              GoalContext("", data))
        assert create.call_args.kwargs["extra_body"] == {"genre": {
            "data": data, "no_null": True, "keys": True, "abstain": True,
            "close_thinking": True}}
        assert decoded.record == {"option": "genre"}

    @pytest.mark.asyncio
    async def test_an_endpoint_without_a_genre_record_is_an_error(self):
        cfg, create = self._cfg(None)
        cfg.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=MagicMock(content="{}"))])
        with pytest.raises(ValueError, match="genre_server"):
            await gsv.genre_v2_decoding(cfg, [], {}, GoalContext("", _home_data()))

    @pytest.mark.asyncio
    async def test_not_with_a_response_format(self):
        cfg, _ = self._cfg({})
        with pytest.raises(ValueError, match="response_format"):
            await gsv.genre_v2_decoding(cfg, [], {"response_format": {}},
                                        GoalContext("", _home_data()))
