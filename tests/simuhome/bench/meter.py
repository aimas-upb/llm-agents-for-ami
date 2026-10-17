"""Timing and LLM use per reasoning step.

Every chat-completion call any client makes goes through
`AsyncCompletions.create`, so wrapping that one method sees them all, including
clients a behaviour builds for itself. A context variable says which step is
running; a call is booked to the innermost step open when it was made. Tasks
created inside a step inherit it, so concurrent structuring calls are booked
correctly.

    meter = Meter()
    with meter.installed():
        async with meter.step("segmentation"):
            ...                                  # LLM calls booked here
    meter.summary()   -> {step: {seconds, calls, llm_seconds, tokens_in, tokens_out}}

Steps are named with dots for nesting ("planning.dependency.profiling"); each
records its own wall time, so a parent's time includes its children's.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Iterator, List, Optional

from openai.resources.chat.completions import AsyncCompletions

_STEP: contextvars.ContextVar[Optional["Meter"]] = contextvars.ContextVar("bench_meter",
                                                                          default=None)
_NAME: contextvars.ContextVar[str] = contextvars.ContextVar("bench_step", default="other")


class Meter:
    """Wall time per step and every LLM call, for one episode."""

    _installed = 0
    _original = None

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []      # {step, seconds, ...extra}
        self.calls: List[Dict[str, Any]] = []      # {step, model, seconds, tokens_in, ...}

    # -- the wrapper -----------------------------------------------------------

    @classmethod
    def _install(cls) -> None:
        if cls._installed == 0:
            original = AsyncCompletions.create
            cls._original = original

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
                            "tokens_in": getattr(usage, "prompt_tokens", None),
                            "tokens_out": getattr(usage, "completion_tokens", None)})

            AsyncCompletions.create = create
        cls._installed += 1

    @classmethod
    def _uninstall(cls) -> None:
        cls._installed -= 1
        if cls._installed == 0 and cls._original is not None:
            AsyncCompletions.create = cls._original

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

    def as_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "calls": self.calls, "summary": self.summary()}
