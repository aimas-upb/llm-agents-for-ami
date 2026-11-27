"""One text (or JSON) answer from an LLM, over the API its backend serves.

Every LLM call that is not a forced tool call goes through `complete`; the
forced tool call goes through `llm_tool_call.ToolCallSession`. Both choose the
API the same way (`llm_tool_call.api_for`):

- OpenAI's own endpoint (api.openai.com) gets the Responses API, which OpenAI
  recommends for its models;
- any other OpenAI-compatible endpoint (for example Ollama, vLLM or LM Studio)
  gets Chat Completions, which they all serve -- so small local models are
  called exactly as before;
- a benchmark may force one transport through the client's `_ami_tool_api`.

Callers keep building their arguments in the Chat Completions shape they
always used (`messages`, `temperature`, `max_tokens`, `reasoning_effort`,
`max_completion_tokens`, `response_format`), and read the answer from the
returned `TextReply`. For example, segmenting a request with gpt-6-luna on
api.openai.com:

    reply = await complete(client, "gpt-6-luna",
                           [{"role": "system", "content": PROMPT},
                            {"role": "user", "content": "turn on the light"}],
                           reasoning_effort="low", max_completion_tokens=8192)

sends `responses.create(model=..., instructions=PROMPT, input=[{"role": "user",
...}], reasoning={"effort": "low"}, max_output_tokens=8192)` and returns its
`output_text`; on Ollama it sends the same `chat.completions.create` call as
before and returns `choices[0].message.content`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ami_agents.shared.utils.llm_models import is_reasoning_model
from ami_agents.shared.utils.llm_tool_call import CHAT, RESPONSES, api_for


# -- the reply ----------------------------------------------------------------

@dataclass
class TextReply:
    """What one call returned: the text, and why generation stopped
    ("stop", or "length" when the output token limit cut it short)."""
    content: str = ""
    finish_reason: Optional[str] = None


# -- the call -----------------------------------------------------------------

async def complete(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    api: Optional[str] = None,
) -> TextReply:
    """One call on `messages`, over the client's API (`api` forces one).

    Temperature goes only to non-reasoning models and the reasoning effort
    only to reasoning ones, as the agents' clients configure them. API errors
    are raised to the caller, which decides what a failed call means.
    """
    settings = dict(temperature=temperature, max_tokens=max_tokens,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort, response_format=response_format)
    if (api or api_for(client)) == RESPONSES:
        response = await client.responses.create(**responses_kwargs(model, messages, **settings))
        return _responses_reply(response)
    response = await client.chat.completions.create(**chat_kwargs(model, messages, **settings))
    choice = response.choices[0]
    return TextReply(content=choice.message.content or "",
                     finish_reason=getattr(choice, "finish_reason", None))


# -- Chat Completions -----------------------------------------------------------

def chat_kwargs(model: str, messages: List[Dict[str, Any]], *, temperature=None,
                max_tokens=None, max_completion_tokens=None, reasoning_effort=None,
                response_format=None) -> Dict[str, Any]:
    """The `chat.completions.create` arguments: the caller's, settings a model
    rejects left out (temperature / max_tokens on a reasoning model)."""
    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if is_reasoning_model(model):
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


# -- the Responses API ------------------------------------------------------------

def responses_kwargs(model: str, messages: List[Dict[str, Any]], *, temperature=None,
                     max_tokens=None, max_completion_tokens=None, reasoning_effort=None,
                     response_format=None) -> Dict[str, Any]:
    """The `responses.create` arguments for the same call.

    System messages become the `instructions`; the rest stay the `input`, in
    order. One output token limit (`max_output_tokens`) covers both of Chat
    Completions' (`max_tokens`, `max_completion_tokens`).
    """
    # 1. The prompt: system text as instructions, the conversation as input.
    system = [str(m.get("content") or "") for m in messages if m.get("role") == "system"]
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [{"role": m["role"], "content": m.get("content") or ""}
                  for m in messages if m.get("role") != "system"],
    }
    if system:
        kwargs["instructions"] = "\n\n".join(system)
    # 2. The model's settings.
    if is_reasoning_model(model):
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
    elif temperature is not None:
        kwargs["temperature"] = temperature
    limit = max_completion_tokens if max_completion_tokens is not None else max_tokens
    if limit is not None:
        kwargs["max_output_tokens"] = limit
    # 3. The output format, in the Responses API's shape.
    text_format = _text_format(response_format)
    if text_format is not None:
        kwargs["text"] = {"format": text_format}
    return kwargs


def _text_format(response_format: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Chat Completions' `response_format` as the Responses API's `text.format`:
    json_object stays as it is; json_schema moves its name, schema and strict
    up one level."""
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        spec = response_format.get("json_schema") or {}
        out = {"type": "json_schema", "name": spec.get("name", "response"),
               "schema": spec.get("schema") or {}}
        if "strict" in spec:
            out["strict"] = spec["strict"]
        return out
    return {"type": response_format.get("type", "text")}


def _responses_reply(response: Any) -> TextReply:
    """A Responses API result as a TextReply: its output text, and "length"
    when the output token limit stopped it (as Chat Completions reports it)."""
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details else None
    if getattr(response, "status", None) == "incomplete":
        finish = "length" if reason == "max_output_tokens" else (reason or "incomplete")
    else:
        finish = "stop"
    return TextReply(content=getattr(response, "output_text", None) or "", finish_reason=finish)


__all__ = ["TextReply", "complete", "chat_kwargs", "responses_kwargs", "CHAT", "RESPONSES"]
