"""HomeBench's official metrics (EM, Precision, Recall, F1), re-implemented, and the
generic per-part rates.

The official script is `code/eval.py` of the HomeBench repository (read, not run: it has
hard-coded paths and reads a field `output2` that exists in no dataset file, so the
official numbers cannot be reproduced bit for bit; this module re-implements its logic).
That logic, row by row:

* the gold (`'''a,b,'''`) loses the quotes, spaces and newlines, is split on commas,
  empty items dropped; the prediction is read from `{...}` in the model's text (several
  `{}` blocks are joined with a comma), spaces and newlines removed, split the same way;
* both become multisets (`Counter`): order is ignored, a repeated call counts twice,
  `error_input` is just a token;
* EM of a row = the two multisets are equal; Precision and Recall are micro-averaged
  over tokens (`tp` = size of the multiset intersection, summed over the rows);
* the rows fall in five buckets by the dataset's `type` -- normal_single, unexist_single
  (`unexist_device` and `unexist_attribute`), normal_multi, mix_multi, error_multi (every
  other `multiN_unexist_*`) -- plus `all`.

Two edge cases of the original are kept on purpose, because they are part of the
metric: removing spaces turns a predicted `fan only` into `fanonly`, which misses the
gold's `fan_only`; and a prediction is compared token by token, so `set_mode(heat)` and
`set_mode(heating)` are different tokens. Two are changed because they only crash the
original: an empty bucket reports `None` instead of raising `ZeroDivisionError`, and a
bucket where nothing was predicted reports precision 0.0 (the usual zero-division
convention) instead of raising.

What the prediction is, for our agents (caveat, printed next to every table): they split
a request into atomic intents and answer in prose, so there is no `{...}` text. The
prediction of an episode is built from what happened (`prediction`): one gold token per
action the home accepted (`gold_log.py`) plus one `error_input` per goal answered
"impossible" or with a question (a clarification): either way the request was reported
to the user instead of executed, which is what HomeBench's `error_input` stands for. That
approximates the official token stream, it is not the stream itself.

Everything is computed from stored data (the record, or the `homebench` block that
`evaluate.py` stores in each .eval.json), so it re-runs without LLM calls.

Worked example::

    gold = gold_tokens("'''error_input,kitchen.light.turn_on(),'''")
    pred = predicted_tokens("{kitchen.light.turn_on(), error_input}")
    compare(pred, gold)    # {"em": True, "tp": 2, "pred": 2, "gold": 2}
    by_bucket([{"bucket": "mix_multi", **compare(pred, gold)}])["all"]["em"]   # 1.0
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

ERROR_INPUT = "error_input"
#: The official buckets, in the order eval.py prints them (after "all").
BUCKETS = ("normal_single", "unexist_single", "normal_multi", "mix_multi", "error_multi")
ALL = "all"
#: Plan outcomes that refuse a goal (`pipeline._entry_outcome`).
IMPOSSIBLE, CLARIFY = "impossible", "clarify"

#: Printed next to the official numbers (plan 3.7, family 2).
CAVEAT_OFFICIAL = (
    "Official HomeBench metrics are re-implemented from the logic of code/eval.py (multiset "
    "EM, micro precision / recall / F1, the five buckets), not run from it: the shipped "
    "script reads a field `output2` that no dataset file has, so the numbers cannot be "
    "reproduced bit for bit. Our agents answer in prose, so the prediction is built from "
    "what happened: one gold token per accepted action plus one `error_input` per goal "
    "answered impossible or with a question. This approximates the official token stream; "
    "it is not that stream."
)

_BRACES = re.compile(r"\{(.*?)\}")


# -- the official preprocessing --------------------------------------------------------------


def _split(text: str) -> List[str]:
    """Comma-split ``text`` and drop empty items (the last step of both sides)."""
    return [item for item in text.split(",") if item != ""]


def gold_tokens(gold: str) -> List[str]:
    """The tokens of a gold string, as eval.py reads ``expected``: quotes, spaces and
    newlines removed, split on commas, empty items dropped."""
    text = gold.replace("'''", "").replace(" ", "").replace("\n", "")
    return _split(text)


def predicted_tokens(text: str) -> List[str]:
    """The tokens of a model's text, as eval.py reads ``generated``: spaces and
    newlines removed first (so ``fan only`` becomes ``fanonly``), the contents of every
    ``{...}`` block joined with a comma, split on commas, empty items dropped. A text
    with no ``{}`` block has no tokens."""
    # 1. Spaces and newlines go before the braces are looked for, as in the original.
    compact = text.replace(" ", "").replace("\n", "")
    # 2. Every {...} block, joined; none means an empty prediction.
    blocks = _BRACES.findall(compact)
    return _split(",".join(blocks))


def prediction_text(tokens: Sequence[str]) -> str:
    """Tokens written the way a model is asked to answer: ``{a,b}``."""
    return "{" + ",".join(tokens) + "}"


def bucket_of(homebench_type: str) -> str:
    """The official bucket of a dataset ``type``, by eval.py's own rules: ``normal`` is
    normal_single, the two ``unexist_*`` singles are unexist_single, and a ``multiN_<kind>``
    is mix_multi, normal_multi or (any other kind) error_multi."""
    if homebench_type == "normal":
        return "normal_single"
    if homebench_type in ("unexist_device", "unexist_attribute"):
        return "unexist_single"
    parts = homebench_type.split("_", 1)
    if len(parts) != 2 or not parts[0].startswith("multi"):
        raise ValueError(f"unknown HomeBench type {homebench_type!r}")
    kind = parts[1].split("_")[0]
    return {"mix": "mix_multi", "normal": "normal_multi"}.get(kind, "error_multi")


# -- the prediction of one episode --------------------------------------------------------


def refused_goals(record: Dict[str, Any]) -> Dict[str, int]:
    """How many goals the agents refused: ``impossible`` (a plan entry answered impossible)
    and ``clarify`` (a goal stopped at structuring by a question, or a plan entry that asks
    one). Each goal is one structure and at most one plan entry, and a clarified goal has
    no plan entry, so the two never count the same goal twice."""
    structures = record.get("structures") or []
    plans = record.get("plans") or []
    clarified = sum(1 for s in structures if s.get("clarification"))
    return {IMPOSSIBLE: sum(1 for p in plans if p.get("outcome") == IMPOSSIBLE),
            CLARIFY: clarified + sum(1 for p in plans if p.get("outcome") == CLARIFY)}


def prediction(action_tokens: Iterable[str], record: Dict[str, Any]) -> List[str]:
    """The predicted tokens of an episode: the gold token of every accepted action, then
    one ``error_input`` per goal answered impossible or with a question (module
    docstring, caveat)."""
    refused = refused_goals(record)
    return [*action_tokens, *[ERROR_INPUT] * (refused[IMPOSSIBLE] + refused[CLARIFY])]


# -- scoring ------------------------------------------------------------------------------------


def compare(predicted: Sequence[str], gold: Sequence[str]) -> Dict[str, Any]:
    """One row's counts: multiset equality (``em``), the true positives (size of the
    multiset intersection), and the token counts of both sides."""
    predicted_count, gold_count = Counter(predicted), Counter(gold)
    return {"em": predicted_count == gold_count,
            "tp": sum((predicted_count & gold_count).values()),
            "pred": len(predicted), "gold": len(gold)}


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, 0.0 for an empty denominator (the zero-division
    convention: nothing predicted is precision 0)."""
    return numerator / denominator if denominator else 0.0


def score(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """EM, micro Precision, Recall and F1 of ``rows`` (each from `compare`).

    An empty list is an empty bucket: every metric is None (the original raises
    ZeroDivisionError). F1 is 0 when precision and recall are both 0, as in the original.
    """
    n = len(rows)
    # 1. An empty bucket has no rate at all (not 0, and not an error).
    if n == 0:
        return {"n": 0, "em": None, "precision": None, "recall": None, "f1": None,
                "tp": 0, "pred": 0, "gold": 0}
    # 2. Micro averages: the counts are summed over the rows, then divided once.
    tp = sum(r["tp"] for r in rows)
    pred = sum(r["pred"] for r in rows)
    gold = sum(r["gold"] for r in rows)
    precision, recall = _ratio(tp, pred), _ratio(tp, gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"n": n, "em": sum(1 for r in rows if r["em"]) / n, "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "pred": pred, "gold": gold}


def by_bucket(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{bucket: score} for ``all`` and the five official buckets, in that order. Each row
    is a `compare` result with its ``bucket``; a bucket with no row is an empty bucket
    (all None), never an error."""
    rows = list(rows)
    out = {ALL: score(rows)}
    for bucket in BUCKETS:
        out[bucket] = score([r for r in rows if r.get("bucket") == bucket])
    return out


# -- generic per-part rates ----------------------------------------------------------------------


def parts_rates(blocks: Iterable[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """The generic rates, pooled over episodes (our definitions, not HomeBench's).

    Each block holds ``parts`` = ``{"feasible", "feasible_ok", "infeasible",
    "infeasible_ok"}`` (see `tests/homebench/scoring.py`): ``parts_rate`` is the parts
    passed over all parts, ``feasible_rate`` the gold calls met, and ``impossible_rate``
    the `error_input` parts met by a goal answered impossible or with a question. A rate
    over no parts is None.
    """
    totals = Counter()
    for block in blocks:
        for key, value in (block.get("parts") or {}).items():
            totals[key] += value

    def rate(passed: int, total: int) -> Optional[float]:
        """passed / total, None for no parts."""
        return round(passed / total, 3) if total else None

    return {"parts_rate": rate(totals["feasible_ok"] + totals["infeasible_ok"],
                               totals["feasible"] + totals["infeasible"]),
            "feasible_rate": rate(totals["feasible_ok"], totals["feasible"]),
            "impossible_rate": rate(totals["infeasible_ok"], totals["infeasible"]),
            "parts": totals["feasible"] + totals["infeasible"]}
