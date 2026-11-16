"""Tests for the logits mask of `tests/simuhome/bench/genre_server.py` (`GenreProcessor`).

The model is not loaded: a fake tokenizer over a small vocabulary decodes the ids, and
the scores of each step are set by the test, so what is checked is which tokens the
mask leaves possible. Needs torch and transformers (as the server itself does).
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ami_agents.agents.user_assistant.utils.genre_decoding import (  # noqa: E402
    HomeRules, TokenTrie,
)
from tests.simuhome.bench import genre_server  # noqa: E402
from tests.simuhome.bench.genre_server import GenreProcessor  # noqa: E402
from tests.unit.test_prompt_variants import _home_data  # noqa: E402

PIECES = ["null", "//", " //", "homeont:", "Laundry", "Room", '",', '"}', "<eos>",
          "location_class", "artifact_class", "â\u0098"]
TOKENS = [chr(c) for c in range(32, 127)] + ["\n"] + PIECES
ID = {text: i for i, text in enumerate(TOKENS)}
EOS = ID["<eos>"]
BYTES = ID["â\u0098"]                # stands for the piece of a multi-byte character


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(TOKENS[i] for i in ids.tolist() if i != EOS)


@pytest.fixture(autouse=True)
def _whole_vocabulary_first(monkeypatch):
    """Equal scores make the best-scored batch arbitrary: look at every token."""
    monkeypatch.setattr(genre_server, "JSON_BATCHES", (len(TOKENS),))


def _processor(beams=2):
    texts = [None if t in ("<eos>", "â\u0098") else t for t in TOKENS]
    rules = HomeRules(_home_data(), TokenTrie(texts, byte_pieces=[BYTES]))
    return GenreProcessor(rules, FakeTokenizer(), start=1, eos_ids=[EOS], beams=beams)


def _step(processor, text, favoured=()):
    """The scores after the mask for a row whose reply so far is ``text``; every token
    scores 0 except the ``favoured`` ones (best)."""
    ids = torch.tensor([[0] + [ID[c] for c in _chars(text)]])
    scores = torch.full((1, len(TOKENS)), -5.0)
    for token in favoured:
        scores[0, ID[token]] = 0.0
    return processor(ids, scores)[0]


def _chars(text):
    """``text`` as token ids' texts: pieces where the vocabulary has them."""
    out, i = [], 0
    while i < len(text):
        piece = next((p for p in sorted(PIECES, key=len, reverse=True)
                      if p not in ("<eos>",) and text.startswith(p, i)), text[i])
        out.append(piece)
        i += len(piece)
    return out


def _possible(scores):
    return {TOKENS[i] for i in torch.nonzero(scores > float("-inf")).flatten().tolist()}


class TestGenreProcessor:
    def test_after_a_null_a_comment_is_impossible_even_when_the_model_prefers_it(self):
        # hb1_infeasible_seed_4644: `"location_class": null,  // ...`
        scores = _step(_processor(), '{"location_class": null, ', favoured=["//", " //"])
        possible = _possible(scores)
        assert "//" not in possible and " //" not in possible
        assert '"' in possible and "\n" in possible

    def test_null_itself_is_still_a_way_out_of_a_value(self):
        scores = _step(_processor(), '{"location_class": ', favoured=["null"])
        assert "null" in _possible(scores)
        assert scores[ID["null"]] == 0.0

    def test_the_end_is_held_until_the_reply_is_a_complete_object(self):
        processor = _processor()
        assert "<eos>" not in _possible(_step(processor, '{"a": 1', favoured=["<eos>"]))
        assert "<eos>" not in _possible(_step(processor, '{"a": 1,', favoured=["<eos>"]))
        assert "<eos>" in _possible(_step(processor, '{"a": 1}', favoured=["<eos>"]))
        assert processor.ends_held == 2

    def test_only_the_best_scored_tokens_are_tried_first(self, monkeypatch):
        monkeypatch.setattr(genre_server, "JSON_BATCHES", (4, 40))
        processor = _processor(beams=2)               # needs 4 candidates to pick from
        ids = torch.tensor([[0] + [ID[c] for c in _chars('{"a": "')]])
        scores = -torch.arange(len(TOKENS), dtype=torch.float) * 0.01
        out = processor(ids, scores.unsqueeze(0))[0]
        assert len(_possible(out)) == 4 and processor.json_widened == 0
        scores[:4] = -9.0                    # the best four are control characters
        scores[ID["\n"]] = 0.0
        out = processor(ids, scores.unsqueeze(0))[0]
        assert "\n" not in _possible(out) and len(_possible(out)) >= 4

    def test_after_the_object_only_whitespace_or_the_end(self):
        possible = _possible(_step(_processor(), '{"a": null}', favoured=["x"]))
        assert possible == {"<eos>", " ", "\n"}

    def test_inside_a_value_only_what_keeps_it_in_the_trie(self):
        processor = _processor()
        possible = _possible(_step(processor, '{"location_class": "homeont:'))
        assert possible == {"L", "M", "Laundry"}
        assert processor.steps == 1 and processor.json_steps == 0

    def test_a_value_that_left_its_trie_is_still_a_json_string(self):
        processor = _processor()
        possible = _possible(_step(processor, '{"location_class": "Garage'))
        assert processor.dead == 1
        assert "\n" not in possible and "x" in possible    # a raw newline is not JSON
        assert '",' in possible

    def test_a_piece_of_a_character_is_allowed_in_a_string_only(self):
        assert "â\u0098" in _possible(_step(_processor(), '{"intent_text": "ab'))
        assert "â\u0098" not in _possible(_step(_processor(), '{"a": null,'))

    def test_the_whole_vocabulary_is_tried_when_the_best_are_all_refused(self):
        # a row whose 64 best tokens are all comments or letters that cannot follow
        processor = _processor(beams=2)
        ids = torch.tensor([[0] + [ID[c] for c in _chars('{"a": 1}')]])
        scores = torch.full((1, len(TOKENS)), -9.0)
        scores[0, ID["x"]] = 0.0                   # refused after the object
        out = processor(ids, scores)[0]
        assert _possible(out) == {"<eos>", " ", "\n"}


class TestGenreModelAnswer:
    """`GenreModel._genre` end to end with the generations faked: it builds the
    processor, runs the three decodings and reports the record."""

    def test_the_answer_path_builds_its_processor_and_records_the_mask(self):
        model = genre_server.GenreModel.__new__(genre_server.GenreModel)
        model.beams, model.answer_max, model.length_penalty = 4, 50, 1.0
        model.tokenizer = FakeTokenizer()
        model.tokens = TokenTrie([None if t in ("<eos>",) else t for t in TOKENS])
        model._eos_ids = lambda: [EOS]
        seen = {}
        thinking, answer = torch.tensor([ID["x"]]), torch.tensor(
            [ID[c] for c in _chars('{"a": null}')])

        def generate(ids, **kwargs):
            if "stopping_criteria" in kwargs:             # the thinking
                return thinking
            if "logits_processor" in kwargs:         # the beam answer: one masked step
                seen["processor"] = kwargs["logits_processor"][0]
                seen["processor"](ids, torch.zeros((1, len(TOKENS))))
                return answer
            return answer                            # the free answer

        model._generate = generate
        model._text = lambda ids: "</think>" if ids is thinking else             FakeTokenizer().decode(ids)
        body = {"model": "m", "messages": []}
        ids = torch.tensor([[0, 1]])
        reply = model._genre(body, ids, 20, {"data": _home_data(), "num_beams": 3})
        record = reply["genre"]
        assert reply["choices"][0]["message"]["content"] == '{"a": null}'
        assert seen["processor"].keep_min == 6          # 2 x the beams asked for
        assert {"json_steps", "json_widened", "masked_steps", "ends_held"} <= set(record)
        assert record["answer"] == '{"a": null}'


class TestGenreModelV2:
    """`GenreModel._genre` with the rules of decoding v2 switched on (`genre_v2`)."""

    GOAL = ('{"location_class": "homeont:MasterBedroom", "artifact_class": '
            '"homeont:AirConditioner", "artifact_name": "ac", "unmatched_device": null}')

    def _model(self, thinking_text, free_text, answer_text):
        model = genre_server.GenreModel.__new__(genre_server.GenreModel)
        model.beams, model.answer_max, model.length_penalty = 4, 50, 1.0
        model.tokenizer = FakeTokenizer()
        model.tokens = TokenTrie([None if t in ("<eos>",) else t for t in TOKENS])
        model._eos_ids = lambda: [EOS]
        thinking, free, answer = (torch.tensor([ID["x"]]), torch.tensor([ID["x"]]),
                                  torch.tensor([ID["x"]]))
        closed = torch.tensor([ID["x"], ID["x"]])
        model._generate = lambda ids, **kw: (
            thinking if "stopping_criteria" in kw
            else answer if "logits_processor" in kw else free)
        model._close_thinking = lambda thought: closed
        texts = {id(thinking): thinking_text, id(closed): "cut</think>",
                 id(free): free_text, id(answer): answer_text}
        model._text = lambda ids: texts[id(ids)]
        return model

    def _run(self, model, **rules):
        genre = {"data": _home_data(), **rules}
        return model._genre({"model": "m", "messages": []}, torch.tensor([[0, 1]]), 20,
                            genre)

    def test_a_missing_device_is_reported_from_the_free_answer(self):
        served = self.GOAL.replace("AirConditioner", "DimmableLight")
        model = self._model("t</think>", self.GOAL, served)
        reply = self._run(model, abstain=True)
        goal = json.loads(reply["choices"][0]["message"]["content"])
        assert goal["unmatched_device"] == "air conditioner"
        assert goal["artifact_class"] is None
        record = reply["genre"]
        assert record["abstained"][0]["words"] == "air conditioner"
        assert record["v2"]["abstain"] and not record["v2"]["no_null"]

    def test_v1_leaves_the_answer_as_the_beam_wrote_it(self):
        served = self.GOAL.replace("AirConditioner", "DimmableLight")
        reply = self._run(self._model("t</think>", self.GOAL, served))
        assert reply["choices"][0]["message"]["content"] == served
        assert reply["genre"]["abstained"] == []

    def test_a_thinking_cut_at_the_cap_is_closed_in_v2_and_an_error_in_v1(self):
        model = self._model("never ends", self.GOAL, self.GOAL)
        assert "genre" not in self._run(model)            # v1: no record, an error
        reply = self._run(model, close_thinking=True)
        assert reply["genre"]["thinking_closed"] is True

    def test_the_keys_of_the_free_answer_bound_the_rules(self):
        model = self._model("t</think>", self.GOAL, self.GOAL)
        seen = {}
        original = genre_server.HomeRules

        def spy(data, tokens, no_null=False, keys=None):
            seen.update(no_null=no_null, keys=sorted(set(keys or [])))
            return original(data, tokens, no_null=no_null, keys=keys)

        genre_server.HomeRules = spy
        try:
            self._run(model, no_null=True, keys=True)
        finally:
            genre_server.HomeRules = original
        assert seen["no_null"] is True
        assert seen["keys"] == ["artifact_class", "artifact_name", "location_class",
                                "unmatched_device"]


class TestGenreModelWorker:
    """Every model call runs on one persistent thread, never on the caller's: a thread
    per request (ThreadingHTTPServer's) leaks host memory per generated token."""

    def _model(self):
        model = genre_server.GenreModel.__new__(genre_server.GenreModel)
        model.worker = genre_server.ThreadPoolExecutor(max_workers=1)
        return model

    def test_requests_from_different_threads_share_one_worker_thread(self):
        import threading
        model = self._model()
        model._complete = lambda body: threading.get_ident()
        callers, workers = [], []

        def request():
            callers.append(threading.get_ident())
            workers.append(model.complete({}))

        for _ in range(3):
            thread = threading.Thread(target=request)
            thread.start()
            thread.join()
        assert len(set(workers)) == 1 and set(workers).isdisjoint(callers)

    def test_an_error_of_the_model_reaches_the_caller(self):
        model = self._model()

        def fail(body):
            raise ValueError("boom")

        model._complete = fail
        with pytest.raises(ValueError, match="boom"):
            model.complete({})


class TestThinkEnd:
    """The stop of the thinking: the text written after the prompt ends with
    `</think>` (replaces `stop_strings`, which leaks host memory on every call)."""

    @staticmethod
    def _stop(prompt, written):
        ids = torch.tensor([[ID[c] for c in prompt + written]])
        criterion = genre_server.ThinkEnd(FakeTokenizer(), start=len(prompt))
        return bool(criterion(ids, torch.zeros((1, len(TOKENS))))[0])

    def test_stops_when_the_written_text_ends_with_the_close_tag(self):
        assert self._stop("ab", "thinking</think>")

    def test_does_not_stop_before_the_tag_is_complete(self):
        assert not self._stop("ab", "thinking</thin")
        assert not self._stop("ab", "thinking")

    def test_a_close_tag_in_the_prompt_does_not_stop(self):
        assert not self._stop("</think>", "")
        assert not self._stop("x</think>", "ab")

    def test_the_mask_has_one_entry_per_row(self):
        ids = torch.tensor([[ID[c] for c in "a</think>"]] * 2)
        mask = genre_server.ThinkEnd(FakeTokenizer(), 1)(ids, None)
        assert mask.shape == (2,) and mask.dtype == torch.bool and bool(mask.all())
