"""Timing and LLM use per reasoning step.

Every LLM call any client makes goes through `AsyncCompletions.create` (Chat
Completions) or `AsyncResponses.create` (the Responses API, which the BT
planner uses on OpenAI's endpoint), so wrapping those two methods sees them
all, including clients a behaviour builds for itself. A context variable says which step is
running; a call is booked to the innermost step open when it was made. Tasks
created inside a step inherit it, so concurrent structuring calls are booked
correctly.

    meter = Meter()
    with meter.installed():
        async with meter.step("segmentation"):
            ...                                  # LLM calls booked here
    meter.summary()   -> {step: {seconds, calls, llm_seconds, tokens_in, tokens_out}}
    meter.stages()    -> the same per processing stage (STAGES), each counted once

Steps are named with dots for nesting ("planning.dependency.profiling"); each
records its own wall time, so a parent's time includes its children's.

The stages are what `run.py` reports at the end of a run:

    segmentation   the "segmentation" step
    structuring    the "structuring" step (its per-intent children are inside it)
    planning       the "planning" steps, without the profiling and guarding
                   timed inside them (a dependency plan's own steps, added with
                   `within="planning"`)
    profiling      a dependency plan's profiling
    guarding       the condition-guard check and the power-on guard: their own
                   steps for simple plans, the dependency planner's for its plans
    execution      the "execution" step
    other          anything else (state reads, the reply)

A step replayed from the step cache (`cached`) took no time in this run: it is
left out, as are its calls.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Iterator, List, Optional

from openai.resources.chat.completions import AsyncCompletions
from openai.resources.responses import AsyncResponses

_STEP: contextvars.ContextVar[Optional["Meter"]] = contextvars.ContextVar("bench_meter",
                                                                          default=None)
_NAME: contextvars.ContextVar[str] = contextvars.ContextVar("bench_step", default="other")


# The processing stages, in pipeline order; anything else is "other".
STAGES = ("segmentation", "structuring", "planning", "profiling", "guarding", "execution")

# The methods wrapped, and the names each API gives its token counts in `usage`.
WRAPPED = ((AsyncCompletions, "prompt_tokens", "completion_tokens"),
           (AsyncResponses, "input_tokens", "output_tokens"))


def _stage(name: str) -> str:
    """The stage a step or a call's booking belongs to: its first dotted part
    when that is a stage ("planning.achievement" -> planning), else other."""
    head = name.split(".", 1)[0]
    return head if head in STAGES else "other"


def _empty() -> Dict[str, Any]:
    return {"seconds": 0.0, "calls": 0, "llm_seconds": 0.0, "tokens_in": 0, "tokens_out": 0}


class Meter:
    """Wall time per step and every LLM call, for one episode."""

    _installed = 0
    _originals: Dict[type, Any] = {}

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []      # {step, seconds, ...extra}
        self.calls: List[Dict[str, Any]] = []      # {step, model, seconds, tokens_in, ...}

    # -- the wrapper -----------------------------------------------------------

    @staticmethod
    def _wrap(original, tokens_in: str, tokens_out: str):
        """``original`` (a `create` method), booking each call to the meter and
        step open when it was made; the token counts read from ``usage`` under
        the API's own names."""
        async def create(self, *args, **kwargs):
            meter, step = _STEP.get(), _NAME.get()
            started = time.perf_counter()
            ok = False
            try:
                response = await original(self, *args, **kwargs)
                ok = True
                return response
            finally:
                if meter is not None:
                    usage = getattr(response, "usage", None) if ok else None
                    meter.calls.append({
                        "step": step, "model": kwargs.get("model"), "ok": ok,
                        "seconds": time.perf_counter() - started,
                        "tokens_in": getattr(usage, tokens_in, None),
                        "tokens_out": getattr(usage, tokens_out, None)})
        return create

    @classmethod
    def _install(cls) -> None:
        """Wrap both APIs' `create` on the first install; later ones only count."""
        if cls._installed == 0:
            for api, tokens_in, tokens_out in WRAPPED:
                cls._originals[api] = api.create
                api.create = cls._wrap(api.create, tokens_in, tokens_out)
        cls._installed += 1

    @classmethod
    def _uninstall(cls) -> None:
        """Restore the original methods when the last install ends."""
        cls._installed -= 1
        if cls._installed == 0:
            for api, original in cls._originals.items():
                api.create = original
            cls._originals.clear()

    @contextmanager
    def installed(self) -> Iterator["Meter"]:
        """Book this episode's calls to this meter while the block runs."""
        self._install()
        token = _STEP.set(self)
        try:
            yield self
        finally:
            _STEP.reset(token)
            self._uninstall()

    # -- steps -----------------------------------------------------------------

    @asynccontextmanager
    async def step(self, name: str, **extra: Any):
        """Time ``name``; LLM calls inside it are booked to it. ``extra`` is
        recorded with it (and the block may add to the yielded dict)."""
        token = _NAME.set(name)
        record: Dict[str, Any] = {"step": name, **extra}
        started = time.perf_counter()
        try:
            yield record
        finally:
            record["seconds"] = time.perf_counter() - started
            _NAME.reset(token)
            self.steps.append(record)

    @staticmethod
    @contextmanager
    def tag(name: str) -> Iterator[None]:
        """Book LLM calls made inside the block to ``name``, without timing a
        step (the step is timed elsewhere and added with `add`)."""
        token = _NAME.set(name)
        try:
            yield
        finally:
            _NAME.reset(token)

    def add(self, name: str, seconds: float, **extra: Any) -> None:
        """A step timed elsewhere (e.g. by the dependency planner itself)."""
        self.steps.append({"step": name, "seconds": seconds, **extra})

    # -- results ---------------------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Per step name: total seconds, and the LLM calls booked to it."""
        out: Dict[str, Dict[str, Any]] = {}
        for record in self.steps:
            entry = out.setdefault(record["step"], {"seconds": 0.0, "count": 0, "calls": 0,
                                                    "llm_seconds": 0.0, "tokens_in": 0,
                                                    "tokens_out": 0})
            entry["seconds"] += record["seconds"]
            entry["count"] += 1
        for call in self.calls:
            entry = out.setdefault(call["step"], {"seconds": 0.0, "count": 0, "calls": 0,
                                                  "llm_seconds": 0.0, "tokens_in": 0,
                                                  "tokens_out": 0})
            entry["calls"] += 1
            entry["llm_seconds"] += call["seconds"]
            entry["tokens_in"] += call["tokens_in"] or 0
            entry["tokens_out"] += call["tokens_out"] or 0
        return out

    def stages(self) -> Dict[str, Dict[str, Any]]:
        """Per stage (STAGES, then "other"): wall seconds, LLM calls, their
        seconds and tokens -- each second counted once (see the module
        docstring)."""
        out = {name: _empty() for name in (*STAGES, "other")}
        # 1. Wall time: only top-level steps ("structuring", not its
        #    "structuring.intent" children), and a step timed inside another
        #    taken out of that one.
        for record in self.steps:
            if record.get("cached"):
                continue
            name = record["step"]
            stage = _stage(name)
            if "." in name and record.get("within") is None:
                continue                                # inside its parent already
            out[stage]["seconds"] += record["seconds"]
            if record.get("within"):
                out[_stage(record["within"])]["seconds"] -= record["seconds"]
        # 2. The calls, to the stage they were booked to.
        for call in self.calls:
            if call.get("cached"):
                continue
            entry = out[_stage(call["step"])]
            entry["calls"] += 1
            entry["llm_seconds"] += call["seconds"]
            entry["tokens_in"] += call["tokens_in"] or 0
            entry["tokens_out"] += call["tokens_out"] or 0
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "calls": self.calls, "summary": self.summary(),
                "stages": self.stages()}
