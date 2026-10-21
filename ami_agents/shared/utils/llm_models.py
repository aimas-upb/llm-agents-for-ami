"""
Which LLM models are reasoning models.

Reasoning models reject `temperature` and `max_tokens`; they take
`reasoning_effort` and `max_completion_tokens` instead. Every client builder
and every call site that shapes the request kwargs asks `is_reasoning_model`,
so a model is never configured as one kind and called as the other.

For example, "gpt-6-luna" and "o3" are reasoning models (no temperature,
reasoning_effort sent), while "gpt-4o-mini" is not (temperature and
max_tokens sent).
"""

import re

# o-series ("o1", "o3-mini", "o4-mini") and the gpt families from 5 upward
# ("gpt-5", "gpt-5-nano", "gpt-6-luna"). "gpt-4o" / "gpt-4.1" stay non-reasoning.
_O_SERIES = re.compile(r"^o\d")
_GPT_MAJOR = re.compile(r"^gpt-(\d+)")
_FIRST_REASONING_GPT = 5


def is_reasoning_model(model: str) -> bool:
    """
    True when `model` is a reasoning model, False otherwise.

    Matches the o-series and every gpt major version from 5 upward, so a new
    family (e.g. gpt-6-luna) needs no code change.
    """
    name = (model or "").strip().lower()
    if _O_SERIES.match(name):
        return True
    gpt = _GPT_MAJOR.match(name)
    return bool(gpt) and int(gpt.group(1)) >= _FIRST_REASONING_GPT
