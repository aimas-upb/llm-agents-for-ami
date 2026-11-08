"""Record every LLM request the agents make, in full, with what came back.

`meter.py` counts the calls; this keeps them. Every LLM call any client makes goes
through `AsyncCompletions.create` (Chat Completions) or `AsyncResponses.create` (the
Responses API), so wrapping those two methods sees all of them -- including the clients
a behaviour builds for itself and the planner's tool-forcing conversation.

    recorder = PromptRecorder()
    with recorder.installed():
        ...                              # run an episode: every call is kept
    recorder.calls  -> [{seq, step, site, model, request, response, ...}, ...]

One call, as kept (JSON-safe, so a record can be written as it is):

    seq        order the call started in (steps run concurrently, so not always the
               order they finished)
    step       the pipeline step open when it was made (`meter.current_step`)
    site       where in `ami_agents/` it was made: {file, line, function}
    api        "chat" | "responses"; model, base_url: who answered
    request    {messages: [{role, content, ...}], params: {tools, temperature, ...}};
               the Responses API's `instructions` is the first message, role "system"
    response   {content, reasoning, tool_calls, finish_reason, usage}; None on an error
    seconds, ok, error

Installing is nestable with `Meter.installed` in either order of use, but the recorder
must be installed *outside* `run_episode` for its wrapper to sit under the meter's: the
meter, installed second, wraps this wrapper and puts it back when it uninstalls.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from openai.resources.chat.completions import AsyncCompletions
from openai.resources.responses import AsyncResponses

from .meter import current_step

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Where a call's origin is looked for: the agents' code, not the bench around it.
_AGENT_DIR = os.path.normcase(str(PROJECT_ROOT / "ami_agents")) + os.sep
# The session that carries the planner's calls is the same for every planning call, so
# the caller worth naming is the one that asked it (bt_planner), not this file.
_TRANSPORT_FILES = ("llm_tool_call.py",)


# -- reading a request / a response ------------------------------------------------

def _plain(value: Any) -> Any:
    """``value`` as JSON-safe plain data: containers recursed, an SDK model dumped,
    anything else (a client object, a callable) as its repr."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain(dump())
    return repr(value)


def _messages_of(api: str, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The conversation sent, as one message list for either API: Chat Completions'
    `messages` as given; the Responses API's `instructions` (the system prompt, resent
    with every call) followed by its `input` items."""
    # 1. Chat Completions: already a message list.
    if api == "chat":
        return _plain(list(kwargs.get("messages") or []))
    # 2. Responses: the instructions are the system message, the input a string or items
    #    (a user message, or the tool's output when feedback answers a tool call).
    messages: List[Dict[str, Any]] = []
    if kwargs.get("instructions"):
        messages.append({"role": "system", "content": kwargs["instructions"]})
    items = kwargs.get("input")
    for item in ([{"role": "user", "content": items}] if isinstance(items, str)
                 else _plain(list(items or []))):
        if "role" in item:
            messages.append(item)
        else:
            messages.append({"role": str(item.get("type", "item")),
                             "content": item.get("output") or item})
    return messages


def _params_of(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Everything sent besides the conversation: temperature, tools, response format,
    ... (the model is kept beside the call, not here)."""
    skip = ("messages", "instructions", "input", "model")
    return {k: _plain(v) for k, v in kwargs.items() if k not in skip}


def _response_of(api: str, response: Any) -> Dict[str, Any]:
    """What came back, in one shape for either API: the text, the model's reasoning
    when the server returns it, the tool calls (name + raw arguments), why it stopped
    and the token counts."""
    # 1. Chat Completions: the first choice's message.
    if api == "chat":
        choice = response.choices[0]
        message = choice.message
        calls = [{"name": c.function.name, "arguments": c.function.arguments}
                 for c in (getattr(message, "tool_calls", None) or [])]
        usage = getattr(response, "usage", None)
        return {"content": message.content,
                "reasoning": (getattr(message, "reasoning", None)
                              or getattr(message, "reasoning_content", None)),
                "tool_calls": calls, "finish_reason": getattr(choice, "finish_reason", None),
                "usage": {"tokens_in": getattr(usage, "prompt_tokens", None),
                          "tokens_out": getattr(usage, "completion_tokens", None)}}
    # 2. Responses: the output items, text and function calls.
    calls = [{"name": item.name, "arguments": item.arguments}
             for item in (response.output or []) if getattr(item, "type", "") == "function_call"]
    usage = getattr(response, "usage", None)
    return {"content": getattr(response, "output_text", None), "reasoning": None,
            "tool_calls": calls, "finish_reason": getattr(response, "status", None),
            "usage": {"tokens_in": getattr(usage, "input_tokens", None),
                      "tokens_out": getattr(usage, "output_tokens", None)}}


def _call_site() -> Optional[Dict[str, Any]]:
    """The agents' code that made the call being recorded: the nearest frame up the
    stack inside ``ami_agents/`` that is not the planner's transport, as {file (from the
    project root), line, function (``Class.method`` for a method)}; None when none is."""
    frame = sys._getframe(1)
    while frame is not None:
        path = os.path.normcase(os.path.abspath(frame.f_code.co_filename))
        if path.startswith(_AGENT_DIR) and os.path.basename(path) not in _TRANSPORT_FILES:
            owner = frame.f_locals.get("self")
            name = frame.f_code.co_name
            return {"file": Path(frame.f_code.co_filename).resolve()
                    .relative_to(PROJECT_ROOT).as_posix(),
                    "line": frame.f_lineno,
                    "function": f"{type(owner).__name__}.{name}" if owner is not None else name}
        frame = frame.f_back
    return None


# -- the recorder ------------------------------------------------------------------

class PromptRecorder:
    """Every LLM call made while it is installed, in the order they started."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._started = time.perf_counter()

    def _wrap(self, original, api: str):
        """``original`` (a `create` method) keeping each call it makes: the request
        before the call goes out, the response (or the error) when it returns. The
        entry is appended first, so `calls` stays in start order under concurrency."""
        async def create(client_api, *args, **kwargs):
            # 1. The request, and who is asking, while the caller is still on the stack.
            entry: Dict[str, Any] = {
                "seq": len(self.calls) + 1, "step": current_step(), "api": api,
                "model": kwargs.get("model"),
                "base_url": str(getattr(getattr(client_api, "_client", None), "base_url", "")),
                "site": _call_site(),
                "started": round(time.perf_counter() - self._started, 3),
                "request": {"messages": _messages_of(api, kwargs), "params": _params_of(kwargs)},
                "response": None, "ok": False, "error": None}
            self.calls.append(entry)
            # 2. The call; an error is kept and still raised, as the agents handle it.
            began = time.perf_counter()
            try:
                response = await original(client_api, *args, **kwargs)
                entry["response"] = _response_of(api, response)
                entry["ok"] = True
                return response
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                entry["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                entry["seconds"] = round(time.perf_counter() - began, 3)
        return create

    @contextmanager
    def installed(self) -> Iterator["PromptRecorder"]:
        """Keep this block's LLM calls: wrap both APIs' `create`, put them back after."""
        originals = {AsyncCompletions: AsyncCompletions.create,
                     AsyncResponses: AsyncResponses.create}
        AsyncCompletions.create = self._wrap(originals[AsyncCompletions], "chat")
        AsyncResponses.create = self._wrap(originals[AsyncResponses], "responses")
        try:
            yield self
        finally:
            for api, original in originals.items():
                api.create = original
