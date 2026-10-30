"""Responses transport for the benchmark's existing chat-based agent interface.

The agents still consume ChatCompletion objects. Keep the translation here so
the same prompts, planner validation loop and executors run with either API.
For example, a rejected generate_behavior_tree call becomes a
function_call_output on the retry, preceded by the original reasoning items.
"""
from __future__ import annotations

from copy import deepcopy

from openai.types.chat import ChatCompletion


def _dict(value):
    """Normalize SDK objects and caller-owned dictionaries without mutating either."""
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else dict(value)


class ResponsesCompletions:
    """Translate the harness's text/function calls using one SDK client's Responses API."""

    def __init__(self, client):
        """Keep this phase's client and the output items needed for planner retries."""
        self.client = client
        # Replaying a planner's assistant tool call must also replay its
        # reasoning items. Scoped to this client (one benchmark phase/episode).
        self.outputs = {}

    def _input(self, messages):
        """Return Responses items, restoring reasoning alongside prior function calls."""
        items = []
        for original in messages:
            message = _dict(original)
            # Tool feedback refers to the model's call_id, not its output item id.
            if message["role"] == "tool":
                items.append({"type": "function_call_output",
                              "call_id": message["tool_call_id"],
                              "output": message["content"]})
                continue
            calls = [_dict(call) for call in message.get("tool_calls") or []]
            if calls:
                # Replay exact output items rather than rebuilding away reasoning.
                key = tuple(call["id"] for call in calls)
                if key in self.outputs:
                    items.extend(deepcopy(self.outputs[key]))
                else:
                    if message.get("content"):
                        items.append({"role": "assistant", "content": message["content"]})
                    for call in calls:
                        function = _dict(call["function"])
                        items.append({"type": "function_call", "call_id": call["id"],
                                      "name": function["name"], "arguments": function["arguments"]})
            else:
                items.append({"role": message["role"], "content": message.get("content") or ""})
        return items

    async def create(self, *, messages, **kwargs):
        """Return a chat-shaped completion after a non-streaming Responses request."""
        # 1. Translate chat parameter names, keeping reasoning and output budgets.
        request = dict(kwargs)
        if request.pop("stream", False):
            raise ValueError("The benchmark Responses adapter does not support streaming")
        if request.pop("n", 1) != 1:
            raise ValueError("Responses supports one generation per request")
        request["input"] = self._input(messages)
        request["store"] = False
        request["include"] = ["reasoning.encrypted_content"]
        effort = request.pop("reasoning_effort", None)
        if effort is not None:
            request["reasoning"] = {"effort": effort}
        cap = request.pop("max_completion_tokens", None)
        legacy_cap = request.pop("max_tokens", None)
        if cap is not None or legacy_cap is not None:
            request["max_output_tokens"] = cap if cap is not None else legacy_cap
        if "tools" in request:
            # 2. Flatten function schemas while preserving chat's strictness default.
            tools = []
            for tool in request["tools"]:
                tool = _dict(tool)
                if tool["type"] != "function":
                    raise ValueError("The benchmark Responses adapter supports function tools only")
                function = _dict(tool["function"])
                # Chat schemas are non-strict unless explicitly marked strict.
                # Responses otherwise normalizes optional fields into required ones.
                tools.append({"type": "function", "strict": False, **function})
            request["tools"] = tools
        choice = request.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            request["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
        response_format = request.pop("response_format", None)
        if response_format:
            fmt = dict(response_format)
            if fmt["type"] == "json_schema":
                fmt = {"type": "json_schema", **fmt["json_schema"]}
            request["text"] = {"format": fmt}

        # 3. Use the SDK resource so its retries and the benchmark meter still apply.
        response = await self.client.responses.create(**request)
        if response.status == "failed":
            raise RuntimeError(f"Responses request failed: {response.error}")
        # 4. Preserve output for retries and expose the interface the agents expect.
        output = [_dict(item) for item in response.output]
        calls = [{"id": item["call_id"], "type": "function",
                  "function": {"name": item["name"], "arguments": item["arguments"]}}
                 for item in output if item["type"] == "function_call"]
        if calls:
            self.outputs[tuple(call["id"] for call in calls)] = output
        reason = "tool_calls" if calls else "stop"
        if response.status == "incomplete":
            details = response.incomplete_details
            reason = "length" if details and details.reason == "max_output_tokens" else "content_filter"
        usage = response.usage
        return ChatCompletion(
            id=response.id, model=response.model, created=int(response.created_at),
            object="chat.completion",
            choices=[{"index": 0, "finish_reason": reason,
                      "message": {"role": "assistant", "content": response.output_text or None,
                                  "tool_calls": calls or None}}],
            usage=({"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens} if usage else None))


def use_responses(client):
    """Route this client's chat-shaped calls through /v1/responses."""
    client.chat.completions.create = ResponsesCompletions(client).create
    return client
