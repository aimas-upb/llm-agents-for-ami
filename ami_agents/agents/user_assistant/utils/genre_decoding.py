"""The token-level rules of GENRE decoding of a structuring reply (`decoding: genre`).

`iri_trie.py` emulates GENRE over an API that shows 20 tokens per step (Ollama). This
module is what a real GENRE needs when the model runs in-process (a Hugging Face model
on the cluster, `tests/simuhome/bench/genre_server.py`): at every step, which tokens of
the whole vocabulary may be chosen. No torch here: the server turns the ids this
returns into a mask over the logits.

The rule is `iri_trie`'s, applied online instead of after the fact. The text written so
far is read; when it ends inside the quoted value of a slot that names something of the
home (`SLOT_VOCABULARY`), only tokens that keep the value inside the trie of that slot
are allowed, with the trie conditioned on the slots the same object already holds
(`slot_values`). A value that is complete may be closed (`"`, `",`, `"}`). Outside such
a value (the keys, free-text slots, a `null`) only the tokens that can continue a JSON
object are allowed (`JsonPrefix`), and the end of the sequence only once the reply is a
complete one (`may_end`): `null` stays a way out for a request the home cannot serve,
but what a model writes after it must still be JSON, not a `//` comment.

Tokens, not characters, are what GENRE masks, but the trie stays on characters, so
every tokenization of a valid name is accepted: a character trie over the vocabulary's
token strings (`TokenTrie`) is walked along what remains of each valid value.

Decoding v2 (`decoding: genre_v2`) closes what the fep run of v1 showed (job 300297, 400
episodes, 32.0%; `iri_trie` 34.3%): beam search, which scores whole sequences, took
`null` as the cheap way out of a slot it could only fill with a long IRI, and wrote JSON
keys the slots could not be read from. Two rules are added outside the values:

    no_null   `null` is not a token for a room, device or action slot while the home has
              values for it (`NO_NULL_SLOTS`, `HomeRules.token_filter`); the device the
              home lacks is reported by `abstention.overlay`, which reads it from the
              free answer of the same thinking, not by a `null` the beam happens to pick
    keys      a JSON key stays a prefix of a key the free answer wrote
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .goal_context import Vocabulary
from .iri_trie import (
    SLOT_OPEN, SLOT_VALUE, IriTrie, Row, flat_values, iri_paths, slot_values,
    vocabulary, written_slots,
)

# What may follow the quote that closes a value inside the same token: the JSON that
# comes next (`",`, `"}`, `",\n  "`: a trailing quote opens the next key).
CLOSING_TAIL = re.compile(r'[\s,}\]]*"?')
# What may follow the quote that closes a key inside the same token: the colon, and the
# quote that opens a string value.
KEY_TAIL = re.compile(r'\s*:?\s*"?')

# The slots that, in decoding v2, may not be `null` while the home has values for them
# (`HomeRules.no_null`): a `null` is a cheap exit for beam search, which scores whole
# sequences, and on the fep run it took the room of 67 feasible goals and a device slot
# of 100 more. A device the home lacks is reported by `abstention.overlay` instead.
NO_NULL_SLOTS = ("location_class", "artifact_class", "artifact_name",
                 "affordance_class", "affordance_name")
# The point where a no-null slot's value begins: its key, the colon, and no value yet.
VALUE_OPEN = re.compile(r'"(%s)"\s*:\s*$' % "|".join(NO_NULL_SLOTS))

_LEAF = -1     # the TokenTrie key under which a node lists the ids that end there


class TokenTrie:
    """A character trie over the vocabulary's token strings.

    ``tokens[i]`` is the text token ``i`` adds ("" or None: a token that is never
    allowed inside a name, a special or a partial-UTF-8 one). ``byte_pieces`` lists
    the ids of the partial-UTF-8 ones: text that is only allowed inside a JSON string."""

    def __init__(self, tokens: Sequence[Optional[str]],
                 byte_pieces: Iterable[int] = ()) -> None:
        self._root: Dict[Any, Any] = {}
        self.tokens = list(tokens)
        self.byte_pieces = set(byte_pieces)
        for token_id, text in enumerate(self.tokens):
            if not text:
                continue
            node = self._root
            for char in text:
                node = node.setdefault(char, {})
            node.setdefault(_LEAF, []).append(token_id)

    def prefixes_of(self, text: str) -> Set[int]:
        """Ids of the tokens whose text is a non-empty prefix of ``text``."""
        found: Set[int] = set()
        node = self._root
        for char in text:
            node = node.get(char)
            if node is None:
                break
            found.update(node.get(_LEAF, ()))
        return found

    def starting_with(self, text: str) -> Set[int]:
        """Ids of the tokens whose text starts with ``text``."""
        node = self._root
        for char in text:
            node = node.get(char)
            if node is None:
                return set()
        found: Set[int] = set()
        stack = [node]
        while stack:
            node = stack.pop()
            found.update(node.get(_LEAF, ()))
            stack.extend(child for key, child in node.items() if key != _LEAF)
        return found


@dataclass
class Allowed:
    """What the next token may be inside a constrained value: its ``slot``, the value
    written so far (``prefix``), and the token ids allowed (``ids``; empty when the
    prefix left the trie, a ``dead`` end the caller leaves unconstrained)."""

    slot: str
    prefix: str
    ids: Set[int]
    dead: bool = False


class HomeRules:
    """The rules for one home: its valid paths and vocabulary, and the token trie.

    Two rules are off unless asked for (decoding v2): ``no_null``, the slots of
    `NO_NULL_SLOTS` must name something of the home, and ``keys``, the names a JSON key
    may take (the keys of the free answer of the same thinking). Without the second,
    nothing stopped beam search from writing `"location-class"` for `"location_class"`:
    no slot was then recognised, nothing was masked, and the reply could not be parsed
    (15 of the 400 ECAI episodes)."""

    def __init__(self, data: Dict[str, Any], tokens: TokenTrie, no_null: bool = False,
                 keys: Optional[Iterable[str]] = None) -> None:
        self.rows: List[Row] = iri_paths(data)
        self.vocab: Vocabulary = vocabulary(data)
        self.tokens = tokens
        self.no_null = no_null
        self.keys: Optional[IriTrie] = IriTrie(sorted(set(keys))) if keys else None
        self._tries: Dict[Tuple[str, ...], IriTrie] = {}
        self._spaces: Optional[Set[int]] = None

    def _trie(self, values: List[str]) -> IriTrie:
        key = tuple(values)
        if key not in self._tries:
            self._tries[key] = IriTrie(values)
        return self._tries[key]

    def next_tokens(self, text: str) -> Optional[Allowed]:
        """The token ids allowed after ``text`` (the reply so far), or None when the
        text does not end inside a constrained value: nothing is masked."""
        # 1. Inside a value: the last slot opened and not closed yet.
        opened = None
        for opened in SLOT_OPEN.finditer(text):
            pass
        if opened is None or '"' in text[opened.end():]:
            return None
        slot, prefix = opened.group(1), text[opened.end():]
        # 2. What the slot may hold given the object so far; the home has no values of
        #    this kind: the slot is free.
        allowed = slot_values(slot, written_slots(text, opened.start()), self.rows,
                              self.vocab)
        if not allowed.values:
            return None
        trie = self._trie(allowed.values)
        if not trie.has_prefix(prefix):
            return Allowed(slot, prefix, set(), dead=True)
        # 3. A token continues a value (its text is a prefix of what remains), or
        #    finishes it and closes the string in the same token.
        ids: Set[int] = set()
        for value in trie.completions(prefix):
            rest = value[len(prefix):]
            ids |= self.tokens.prefixes_of(rest)
            for token_id in self.tokens.starting_with(rest + '"'):
                tail = (self.tokens.tokens[token_id] or "")[len(rest) + 1:]
                if CLOSING_TAIL.fullmatch(tail):
                    ids.add(token_id)
        return Allowed(slot, prefix, ids)

    def token_filter(self, text: str,
                     state: "JsonPrefix") -> Optional[Callable[[str], bool]]:
        """For a reply written so far outside a constrained value, the test a token's
        text must pass on top of continuing the JSON, or None when there is none:

        - inside a key (``keys`` known): the key stays a prefix of a known key, and may
          be closed only on a whole one;
        - at the start of a `NO_NULL_SLOTS` value (``no_null``, and the home has values
          for the slot): only a quote, or whitespace before it. The tokens `null`, a
          digit, `{` are out, and so is a token that starts the value itself (`"Store`):
          the value is then written under its trie from its first character."""
        if self.keys is not None and state.in_string and state.is_key:
            prefix = text[text.rfind('"') + 1:]
            return lambda token: self._key_token(prefix, token)
        if self.no_null:
            opened = VALUE_OPEN.search(text)
            if opened and flat_values(opened.group(1), self.vocab):
                return _quote_or_space
        return None

    def filter_ids(self, text: str, state: "JsonPrefix") -> Optional[Set[int]]:
        """`token_filter` as the set of token ids it accepts, or None when there is no
        filter. The same rule, found from the trie instead of by testing a candidate
        token at a time: the filter rejects nearly every token the model scores high
        (a key must stay inside the known keys, a no-null value opens only with a quote),
        so testing candidates widened to the whole vocabulary at almost every step and a
        beam took 60 s instead of 5 (job 300721)."""
        if self.keys is not None and state.in_string and state.is_key:
            return self._key_ids(text[text.rfind('"') + 1:])
        if self.no_null:
            opened = VALUE_OPEN.search(text)
            if opened and flat_values(opened.group(1), self.vocab):
                return self._quote_or_space_ids()
        return None

    def _quote_or_space_ids(self) -> Set[int]:
        """The ids of the tokens `_quote_or_space` accepts (worked out once)."""
        if self._spaces is None:
            self._spaces = {i for i, text in enumerate(self.tokens.tokens)
                            if text and _quote_or_space(text)}
        return self._spaces

    def _key_ids(self, prefix: str) -> Set[int]:
        """The ids of the tokens `_key_token` accepts after the key written so far
        (``prefix``): those that keep it inside a known key, and those that close a
        whole key with what may follow its quote."""
        ids: Set[int] = set()
        if self.keys is None:
            return ids
        for key in self.keys.completions(prefix):
            rest = key[len(prefix):]
            ids |= self.tokens.prefixes_of(rest)
            for token_id in self.tokens.starting_with(rest + '"'):
                tail = (self.tokens.tokens[token_id] or "")[len(rest) + 1:]
                if KEY_TAIL.fullmatch(tail):
                    ids.add(token_id)
        return ids

    def _key_token(self, prefix: str, token: str) -> bool:
        """Whether ``token`` continues the key written so far (``prefix``) inside the
        known keys, or closes it on a whole one."""
        accepted = self.keys.accept(prefix, token) if self.keys else None
        if accepted is None:
            return False
        closing = accepted[1]
        return closing is None or KEY_TAIL.fullmatch(closing[1:]) is not None


def _quote_or_space(token: str) -> bool:
    """Whether ``token`` is only whitespace, or whitespace and then one quote."""
    stripped = token.lstrip()
    return not stripped or stripped == '"'


def may_end(text: str) -> bool:
    """Whether the reply written so far is a complete JSON object, so that the end of
    the sequence may be chosen. Beam search scores whole sequences: without this the
    cheapest sequence is one that stops at once (length penalty 0) or one that keeps
    going with a comment (length penalty 1). A leading code fence is ignored."""
    body = text.strip()
    if body.startswith("```"):
        body = body.partition("\n")[2]
    body = body.rpartition("```")[0] if body.rstrip().endswith("```") else body
    try:
        return isinstance(json.loads(body), dict)
    except ValueError:
        return False


_WS = " \t\n\r"
_DIGITS = "0123456789"
_HEX = "0123456789abcdefABCDEF"
_ESCAPES = '"\\/bfnrt'
_LITERALS = {"t": "true", "f": "false", "n": "null"}
# The states of a number (JSON's grammar), by what was just read.
_NUMBER = {
    ("n_minus", "0"): "n_zero", ("n_zero", "."): "n_dot", ("n_int", "."): "n_dot",
    ("n_e", "+"): "n_esign", ("n_e", "-"): "n_esign",
}
for _e in "eE":
    _NUMBER.update({("n_zero", _e): "n_e", ("n_int", _e): "n_e",
                    ("n_frac", _e): "n_e"})
for _d in _DIGITS[1:]:
    _NUMBER[("n_minus", _d)] = "n_int"
for _d in _DIGITS:
    _NUMBER.update({("n_int", _d): "n_int", ("n_dot", _d): "n_frac",
                    ("n_frac", _d): "n_frac", ("n_e", _d): "n_exp",
                    ("n_esign", _d): "n_exp", ("n_exp", _d): "n_exp"})
_NUMBER_DONE = {"n_zero", "n_int", "n_frac", "n_exp"}    # where a number may end


class JsonPrefix:
    """Whether the text written so far can still become a JSON object reply (an
    optional code fence around it is allowed, as `may_end` does).

    Why: the mask of `GenreProcessor` constrains only the inside of a quoted IRI value.
    A model that leaves the trie by writing `null` for the slot (no device of the home
    fits) was then free to write anything after it, and wrote `,  // no such room`:
    not JSON, so `may_end` never opened and the answer ran to the token cap
    (`hb1_infeasible_seed_4644`). Refusing the tokens that cannot continue a JSON
    object keeps `null` an exit and takes the other exits away (comments, prose after
    the object, a second fence): a character automaton, torch-free, so the server need
    only ask which tokens it accepts (`json_allowed`)."""

    def __init__(self) -> None:
        self.ok = True
        self.mode = "pre"
        self.stack: List[str] = []       # "o" or "a" for each open object or array
        self.is_key = False              # the string being written is an object key
        self.fenced = False
        self.count = 0                   # backticks of a fence, hex digits of \u
        self.rest = ""                   # what a true / false / null still needs

    @classmethod
    def of(cls, text: str) -> "JsonPrefix":
        """The state after ``text`` (``ok`` is False when it is not a valid prefix)."""
        state = cls()
        state.feed_text(text)
        return state

    @property
    def in_string(self) -> bool:
        """Whether the next character is part of a string (any text may follow)."""
        return self.ok and self.mode in ("string", "esc", "uni")

    def copy(self) -> "JsonPrefix":
        """An independent copy of this state."""
        other = JsonPrefix()
        other.__dict__.update(self.__dict__)
        other.stack = list(self.stack)
        return other

    def accepts(self, text: str) -> bool:
        """Whether ``text`` may follow what was written (this state is not changed)."""
        probe = self.copy()
        return probe.feed_text(text)

    def feed_text(self, text: str) -> bool:
        """Read ``text``; False (and ``ok`` False) at the first character that cannot
        continue a JSON object."""
        for char in text:
            if not self.feed(char):
                self.ok = False
                return False
        return True

    def feed(self, char: str) -> bool:
        """Read one character; whether it can continue the reply."""
        mode = self.mode
        if mode.startswith("n_"):
            nxt = _NUMBER.get((mode, char))
            if nxt:
                self.mode = nxt
                return True
            if mode not in _NUMBER_DONE:
                return False
            mode = self.mode = "after"       # the number ended; this char follows it
        if mode == "string":
            return self._string(char)
        if mode == "esc":
            self.mode = "uni" if char == "u" else "string"
            self.count = 4
            return char in _ESCAPES or char == "u"
        if mode == "uni":
            self.count -= 1
            self.mode = "string" if self.count == 0 else "uni"
            return char in _HEX
        if mode == "lit":
            self.mode = "lit" if len(self.rest) > 1 else "after"
            ok, self.rest = char == self.rest[0], self.rest[1:]
            return ok
        if mode in ("fence", "postfence"):
            return self._fence(char)
        if char in _WS:
            return True
        return self._structure(mode, char)

    def _string(self, char: str) -> bool:
        if char == '"':
            self.mode = "colon" if self.is_key else "after"
        elif char == "\\":
            self.mode = "esc"
        return char >= " "

    def _fence(self, char: str) -> bool:
        if self.count < 3:
            self.count += 1
            if self.count == 3 and self.mode == "postfence":
                self.mode = "post_end"        # the closing fence is complete
            return char == "`"
        if self.mode == "fence":              # the language tag, up to the newline
            if char == "\n":
                self.mode = "pre"
            return char == "\n" or char.isalpha()
        return False

    def _open(self, char: str) -> bool:
        """A value starts with ``char`` (the mode says one may)."""
        if char == "{":
            self.stack.append("o")
            self.mode = "key_or_end"
        elif char == "[":
            self.stack.append("a")
            self.mode = "value_or_end"
        elif char == '"':
            self.mode, self.is_key = "string", False
        elif char == "-":
            self.mode = "n_minus"
        elif char in _DIGITS:
            self.mode = "n_zero" if char == "0" else "n_int"
        elif char in _LITERALS:
            self.mode, self.rest = "lit", _LITERALS[char][1:]
        else:
            return False
        return True

    def _close(self, kind: str) -> bool:
        if not self.stack or self.stack[-1] != kind:
            return False
        self.stack.pop()
        self.mode = "after" if self.stack else "post"
        return True

    def _structure(self, mode: str, char: str) -> bool:
        """``char`` is not whitespace, inside no string, number or literal."""
        if mode == "pre":
            if char == "`" and not self.fenced:
                self.mode, self.fenced, self.count = "fence", True, 1
                return True
            if char != "{":
                return False
            return self._open(char)
        if mode == "post_end":
            return False
        if mode == "post":
            if char == "`" and self.fenced:
                self.mode, self.count = "postfence", 1
                return True
            return False
        if mode in ("key_or_end", "key"):
            if char == "}" and mode == "key_or_end":
                return self._close("o")
            if char != '"':
                return False
            self.mode, self.is_key = "string", True
            return True
        if mode == "colon":
            self.mode = "value"
            return char == ":"
        if mode == "after":
            if char == "," and self.stack:
                self.mode = "key" if self.stack[-1] == "o" else "value"
                return True
            return char in "}]" and self._close("o" if char == "}" else "a")
        if char == "]" and mode == "value_or_end":
            return self._close("a")
        return self._open(char)               # "value", "value_or_end"


def json_allowed(state: JsonPrefix, tokens: TokenTrie, candidates: Iterable[int],
                 accept: Optional[Callable[[str], bool]] = None) -> List[int]:
    """Of the ``candidates`` token ids, those whose text can follow what ``state`` has
    read. A token with no text of its own (a piece of a multi-byte character) is
    allowed only inside a string; a special token never (the caller adds the end of
    the sequence when ``may_end``). ``accept`` is a further test on a token's text
    (`HomeRules.token_filter`); with it a piece of a multi-byte character is out."""
    allowed = []
    for token_id in candidates:
        if token_id >= len(tokens.tokens):        # a padding row of the logits
            continue
        text = tokens.tokens[token_id]
        if text:
            if state.accepts(text) and (accept is None or accept(text)):
                allowed.append(token_id)
        elif token_id in tokens.byte_pieces and state.in_string and accept is None:
            allowed.append(token_id)
    return allowed


def slot_pairs(text: str) -> List[Tuple[str, str]]:
    """The constrained slots a reply writes, in order, as (slot, value)."""
    return [(m.group(1), m.group(2)) for m in SLOT_VALUE.finditer(text)]


def changed_slots(free: str, decoded: str) -> List[Dict[str, Any]]:
    """The slot values the decoding changed: the free reply's and the decoded
    reply's slots paired by position while their slot names agree (the object shapes
    differ after the first slot that does not)."""
    changes = []
    for index, (a, b) in enumerate(zip(slot_pairs(free), slot_pairs(decoded))):
        if a[0] != b[0]:
            break
        if a[1] != b[1]:
            changes.append({"slot": a[0], "free": a[1], "decoded": b[1],
                            "index": index})
    return changes
