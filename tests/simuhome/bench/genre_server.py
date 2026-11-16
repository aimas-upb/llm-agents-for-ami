"""An OpenAI-compatible chat endpoint that runs a Hugging Face model in-process and can
decode a structuring reply the way GENRE does (`decoding: genre`).

Why not Ollama: constrained decoding needs the logits of the whole vocabulary at every
step. Ollama shows 20 tokens per step, so `decoding: iri_trie` emulates the constraint
(`utils/iri_trie.py`). Here the model is a `transformers` model on the GPU, and:

    1. the thinking of a reasoning model is decoded freely (greedy) up to `</think>`;
    2. the answer is decoded by beam search with a mask on the logits: inside the
       quoted value of a slot that names something of the home, only the tokens that
       keep the value inside the trie of that slot (`utils/genre_decoding.py`).

`genre` may also carry the rules of decoding v2 (`no_null`, `keys`, `abstain`,
`close_thinking`, set by `decoding: genre_v2`; see `utils/genre_decoding.py` and
`utils/abstention.py`): each is off unless the request asks for it.

A request without `genre` is one free greedy completion (the control on this backend).
A request with `genre` (`{"data": <goal context data>}`, from the `genre` option of
`goal_structuring_variants.py`) is decoded as above; the reply also carries the free
greedy answer from the same thinking, and what the constraint changed.

    python -m tests.simuhome.bench.genre_server --model-dir ~/hf_models/DeepSeek-R1-Distill-Qwen-7B

One generation at a time, all on one worker thread: the harness's workers queue on the
GPU. The single thread is not only a lock. `ThreadingHTTPServer` starts a thread per
request, and every thread that runs the model leaves about 1 MB of host memory behind
per generated token (torch / CUDA per-thread state that is never freed: 0.4 GB for a
400-token request, 100 GB after the 117 requests of 34 ECAI-400 episodes, until the
job's memory limit killed the server). Generation on one persistent thread stays flat.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList,
)

from ami_agents.agents.user_assistant.utils.abstention import overlay
from ami_agents.agents.user_assistant.utils.genre_decoding import (
    HomeRules, JsonPrefix, TokenTrie, changed_slots, json_allowed, may_end,
)

logger = logging.getLogger("genre_server")

THINK_END = "</think>"
DEFAULT_PORT = 11600
DEFAULT_BEAMS = 4
DEFAULT_LENGTH_PENALTY = 1.0     # transformers' default: score = sum logprob / length
DEFAULT_MAX_TOKENS = 4096
DEFAULT_ANSWER_MAX = 2048
# How many of a row's best-scored tokens are tried against the JSON check before the
# next, wider batch (the last batch is the whole vocabulary).
JSON_BATCHES = (64, 1024)
# The rules of decoding v2 a request may switch on (`genre_v2`; see genre_decoding.py).
V2_RULES = ("no_null", "keys", "abstain", "close_thinking")


def _rss_gb() -> float:
    """Resident memory of this process in GB (0.0 where /proc does not exist)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


class ThinkEnd(StoppingCriteria):
    """Ends a generation once the text written after the prompt ends with `</think>`.
    It stands in for transformers' `stop_strings`, whose set-up on every call left about
    40 MB of host memory behind per request (11 GB over the first 270 requests of the
    ECAI-400 run); the last few tokens are decoded, so a `</think>` spelled as several
    tokens ends it too."""

    TAIL = 8        # tokens decoded at each step: `</think>` is one token or a few

    def __init__(self, tokenizer: Any, start: int) -> None:
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs: Any) -> torch.BoolTensor:
        written = input_ids[0, max(self.start, input_ids.shape[1] - self.TAIL):]
        done = self.tokenizer.decode(written, skip_special_tokens=False
                                     ).endswith(THINK_END)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool,
                          device=input_ids.device)


class GenreProcessor(LogitsProcessor):
    """Masks the logits of every row. Inside a constrained value only the token ids
    `HomeRules.next_tokens` allows keep their score. Elsewhere (the keys, free text, a
    `null`) only the tokens that can continue a JSON object do (`JsonPrefix`), and the
    end of the sequence only once the reply is a complete one (`may_end`)."""

    def __init__(self, rules: HomeRules, tokenizer: Any, start: int,
                 eos_ids: List[int], beams: int) -> None:
        self.rules, self.tokenizer, self.start = rules, tokenizer, start
        self.eos_ids = eos_ids
        self.keep_min = 2 * beams     # candidates a beam step needs to pick from
        self.steps = 0          # rows masked to a value's trie
        self.dead = 0           # rows whose value had left its trie
        self.ends_held = 0      # rows where the end of the sequence was refused
        self.json_steps = 0     # rows masked to what continues a JSON object
        self.json_widened = 0   # of those, rows that needed more than the first batch
        self.filtered = 0       # of those, rows held to a key or to a quote (v2 rules)

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        for row in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[row, self.start:],
                                         skip_special_tokens=True)
            allowed = self.rules.next_tokens(text)
            if allowed is not None and not allowed.dead and allowed.ids:
                self.steps += 1
                self._keep(scores[row], sorted(allowed.ids))
                continue
            if allowed is not None:
                self.dead += 1
            self._json(scores[row], text)
        return scores

    @staticmethod
    def _keep(row: torch.Tensor, ids: List[int]) -> None:
        """Keep the scores of ``ids``; every other token is impossible."""
        index = torch.tensor(ids, device=row.device)
        kept = row[index].clone()
        row[:] = float("-inf")
        row[index] = kept

    def _json(self, row: torch.Tensor, text: str) -> None:
        """Outside a constrained value: the tokens that continue a JSON object, found
        among the best-scored ones (the whole vocabulary is never walked unless the
        best are all refused), and the end of the sequence once the reply is whole."""
        state = JsonPrefix.of(text)
        ends = may_end(text)
        if not ends:
            self.ends_held += 1
        if not state.ok:                # not a prefix at all: nothing to guide
            if not ends:
                row[self.eos_ids] = float("-inf")
            return
        self.json_steps += 1
        accept = self.rules.token_filter(text, state)
        if accept is not None:
            self.filtered += 1
        size = row.shape[0]
        for batch, width in enumerate((*JSON_BATCHES, size)):
            width = min(width, size)
            top = torch.topk(row, width).indices.tolist()
            keep = json_allowed(state, self.rules.tokens, top, accept)
            if ends:
                keep += self.eos_ids
            if len(keep) >= self.keep_min or width == size:
                break
        if batch:
            self.json_widened += 1
        if keep:
            self._keep(row, keep)


class GenreModel:
    """The model, its tokenizer and the token trie, and the two ways to complete."""

    def __init__(self, model_dir: str, beams: int, answer_max: int,
                 length_penalty: float = DEFAULT_LENGTH_PENALTY) -> None:
        self.beams, self.answer_max = beams, answer_max
        self.length_penalty = length_penalty
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="genre-model")
        started = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16).to("cuda").eval()
        logger.info("model loaded in %.0f s", time.time() - started)
        self.tokens = self._token_trie()

    def _token_trie(self) -> TokenTrie:
        """The text of every token id, for the mask (specials and partial-UTF-8 tokens
        are never allowed inside a name)."""
        started = time.time()
        size = len(self.tokenizer)
        texts = self.tokenizer.batch_decode([[i] for i in range(size)])
        special = set(self.tokenizer.all_special_ids)
        texts = [None if i in special or "\ufffd" in t else t
                 for i, t in enumerate(texts)]
        logger.info("token trie of %d tokens in %.0f s", size, time.time() - started)
        pieces = [i for i, t in enumerate(texts) if t is None and i not in special]
        return TokenTrie(texts, pieces)

    def _generate(self, ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """The new token ids of one generation, greedy unless beams are asked for."""
        out = self.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        return out[0, ids.shape[1]:]

    def _eos_ids(self) -> List[int]:
        """Every token id that ends a generation (the tokenizer's and the model's)."""
        found = self.model.generation_config.eos_token_id
        ids = {self.tokenizer.eos_token_id,
               *(found if isinstance(found, list) else [found])}
        return sorted(i for i in ids if i is not None)

    def _text(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def complete(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """One chat completion, run on the model's one worker thread (see the module
        docstring); with `body["genre"]` the answer is GENRE-decoded."""
        return self.worker.submit(self._complete, body).result()

    def _complete(self, body: Dict[str, Any]) -> Dict[str, Any]:
        genre = body.get("genre")
        max_tokens = (body.get("max_tokens") or body.get("max_completion_tokens")
                      or DEFAULT_MAX_TOKENS)
        prompt = self.tokenizer.apply_chat_template(
            body["messages"], add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=False).input_ids.to("cuda")
        with torch.inference_mode():
            if genre:
                return self._genre(body, ids, int(max_tokens), genre)
            started = time.time()
            new = self._generate(ids, max_new_tokens=int(max_tokens))
            return self._reply(body, ids, self._split(self._text(new)), new.shape[0],
                               {"seconds": round(time.time() - started, 2)}, None)

    @staticmethod
    def _split(text: str) -> Dict[str, str]:
        """The thinking and the answer of a reasoning model's output."""
        thinking, found, answer = text.partition(THINK_END)
        return ({"thinking": thinking, "content": answer} if found
                else {"thinking": "", "content": text})

    def _close_thinking(self, thought: torch.Tensor) -> torch.Tensor:
        """The thinking that ran to the token cap, ended with `</think>`."""
        ending = self.tokenizer("\n" + THINK_END + "\n\n", return_tensors="pt",
                                add_special_tokens=False).input_ids[0]
        return torch.cat([thought, ending.to(thought.device)])

    def _genre(self, body: Dict[str, Any], ids: torch.Tensor, max_tokens: int,
               genre: Dict[str, Any]) -> Dict[str, Any]:
        beams = int(genre.get("num_beams") or self.beams)
        # The v2 rules, each off unless the request (`genre_v2`) asks for it.
        v2 = {rule: bool(genre.get(rule)) for rule in V2_RULES}
        times: Dict[str, float] = {}
        rss = [_rss_gb()]       # resident GB at the start and after each phase (logged)
        # 1. The thinking, free, up to </think> (the stop string ends the generation).
        started = time.time()
        thought = self._generate(
            ids, max_new_tokens=max_tokens, stopping_criteria=StoppingCriteriaList(
                [ThinkEnd(self.tokenizer, ids.shape[1])]))
        times["thinking"] = round(time.time() - started, 2)
        rss.append(_rss_gb())
        text = self._text(thought)
        closed = False
        if THINK_END not in text:
            # Thinking ran to the token cap. v1 gave up (an infrastructure error, 3 of
            # the 400 ECAI episodes); v2 closes it where it stands and answers from it.
            if not v2["close_thinking"]:
                record = {"option": "genre", "error": "no_think_end", "beams": beams,
                          "seconds": times}
                return self._reply(body, ids, self._split(text), thought.shape[0],
                                   record, None)
            thought = self._close_thinking(thought)
            text, closed = self._text(thought), True
        prefix = torch.cat([ids, thought.unsqueeze(0)], dim=1)
        # 2. The answer as the model writes it freely (the comparison).
        started = time.time()
        free = self._generate(prefix, max_new_tokens=self.answer_max)
        times["free"] = round(time.time() - started, 2)
        rss.append(_rss_gb())
        # 3. The answer by beam search under the mask.
        free_text = self._text(free)
        keys = (re.findall(r'"(\w+)"\s*:', free_text) if v2["keys"] else None)
        rules = HomeRules(genre["data"], self.tokens, no_null=v2["no_null"], keys=keys)
        processor = GenreProcessor(rules, self.tokenizer, prefix.shape[1],
                                   self._eos_ids(), beams)
        started = time.time()
        answer = self._generate(
            prefix, max_new_tokens=self.answer_max, num_beams=beams,
            length_penalty=self.length_penalty, early_stopping=True,
            logits_processor=LogitsProcessorList([processor]))
        times["genre"] = round(time.time() - started, 2)
        rss.append(_rss_gb())
        logger.info("genre request: prompt %d tokens, thinking %d, free %d, answer %d; "
                    "resident GB at start / after thinking / free / beam: %s",
                    ids.shape[1], thought.shape[0], free.shape[0], answer.shape[0],
                    " / ".join(f"{gb:.1f}" for gb in rss))
        # 4. A device the room lacks is reported, from what the free answer named.
        answer_text, abstained = self._text(answer), []
        if v2["abstain"]:
            answer_text, abstained = overlay(free_text, answer_text, rules.rows,
                                             rules.vocab)
        record = {
            "option": "genre", "backend": "transformers", "beams": beams,
            "length_penalty": self.length_penalty,
            "thinking_tokens": int(thought.shape[0]),
            "answer_tokens": int(answer.shape[0]),
            "free_answer_tokens": int(free.shape[0]),
            "masked_steps": processor.steps, "dead_steps": processor.dead,
            "ends_held": processor.ends_held, "json_steps": processor.json_steps,
            "json_widened": processor.json_widened, "filtered_steps": processor.filtered,
            "v2": v2, "thinking_closed": closed, "abstained": abstained,
            "free_answer": free_text, "answer": answer_text,
            "changed": changed_slots(free_text, answer_text), "seconds": times}
        parts = {"thinking": text.partition(THINK_END)[0], "content": answer_text}
        return self._reply(body, ids, parts,
                           thought.shape[0] + answer.shape[0], record, record)

    @staticmethod
    def _reply(body: Dict[str, Any], ids: torch.Tensor, parts: Dict[str, str],
               new_tokens: int, timing: Dict[str, Any],
               genre: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The OpenAI chat completion; the thinking in `reasoning`, as Ollama does."""
        out = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": body.get("model", "genre"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": parts["content"],
                "reasoning": parts["thinking"]}}],
            "usage": {"prompt_tokens": int(ids.shape[1]),
                      "completion_tokens": int(new_tokens),
                      "total_tokens": int(ids.shape[1] + new_tokens)},
            "timing": timing}
        if genre is not None:
            out["genre"] = genre
        return out


def make_handler(model: GenreModel) -> type:
    """The request handler class bound to ``model``."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("/health", "/v1/models"):
                self._send(200, {"status": "ok", "data": [{"id": "genre"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                self._send(200, model.complete(body))
            except Exception as exc:  # reported to the client, not swallowed
                logger.exception("request failed")
                self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s " + fmt, self.address_string(), *args)

    return Handler


def main(argv: Optional[List[str]] = None) -> None:
    """Load the model and serve."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--num-beams", type=int, default=DEFAULT_BEAMS)
    parser.add_argument("--answer-max", type=int, default=DEFAULT_ANSWER_MAX)
    parser.add_argument("--length-penalty", type=float,
                        default=DEFAULT_LENGTH_PENALTY)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    model = GenreModel(args.model_dir, args.num_beams, args.answer_max,
                       args.length_penalty)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    logger.info("serving on %s:%d (beams %d)", args.host, args.port, args.num_beams)
    server.serve_forever()


if __name__ == "__main__":
    main()
