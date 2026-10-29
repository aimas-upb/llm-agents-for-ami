"""
One forced tool call to an LLM, over the API its backend serves.

The caller sees the same interface whatever the backend:

    session = ToolCallSession(client, model, system=..., user=..., tool=GENERATE_BT_TOOL,
                              temperature=..., reasoning_effort=..., max_output_tokens=...)
    reply = await session.ask()        # ToolReply: the arguments, or the text it wrote
    session.correct("The previous response was invalid: ...")
    reply = await session.ask()        # the model sees its last answer and the feedback

Which API a call goes over is decided by the client's endpoint (`api_for`):

- OpenAI's own endpoint (api.openai.com) gets the Responses API, which
  OpenAI recommends for its models. It is also the only one of the two that
  combines function tools with `reasoning_effort` for the newer models:
  gpt-6-luna rejects that pair on /v1/chat/completions.
- Any other OpenAI-compatible endpoint (for example Ollama, vLLM or LM Studio)
  gets Chat Completions, which they all serve.

Example: planning "turn on the light" with gpt-6-luna on api.openai.com sends
`responses.create(instructions=<system>, input=[<user>], tools=[{type: function,
name: generate_behavior_tree, ...}], tool_choice={type: function, name: ...},
reasoning={effort: low})`. The `function_call` item of the response carries the
arguments. A retry sends `previous_response_id` and a `function_call_output`
holding the feedback, so the model sees its own last answer. The same planning
on Ollama sends `chat.completions.create(messages=..., tools=..., tool_choice=...)`,
and a retry appends the assistant message and a `tool` message to the messages.

`tool` is given in the Chat Completions shape (`{"type": "function",
"function": {name, description, parameters}}`) and converted for the
Responses API.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

from ami_agents.shared.utils.llm_models import is_reasoning_model

# The two APIs a call can go over.
RESPONSES = "responses"
CHAT = "chat"

# The host whose endpoint serves the Responses API with every OpenAI model.
OPENAI_HOST = "api.openai.com"


# -- choosing the API ---------------------------------------------------------

def api_for(client: Any) -> str:
    """RESPONSES for a client on OpenAI's own endpoint, CHAT for any other.

    Local and third-party servers implement Chat Completions, while their
    Responses support is partial or missing, so only OpenAI's endpoint moves.
    A client with no `base_url` (a test double) is treated as a local one.
    """
    base_url = str(getattr(client, "base_url", "") or "")
    return RESPONSES if OPENAI_HOST in base_url.lower() else CHAT


# -- the reply ----------------------------------------------------------------

@dataclass
class ToolReply:
    """What one call returned.

    `arguments` holds the tool call's arguments when the model made the call
    and they parsed as a JSON object; `error` says why they did not parse.
    `content` is the text the model wrote, which is where small local models
    put the call when they do not make it as a tool call.
    """
    arguments: Optional[dict] = None
    content: Optional[str] = None
    called: bool = False
    error: Optional[str] = None


def _parse_arguments(raw: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """The arguments as a dict and None, or None and why they did not parse."""
    try:
        data = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        return None, f"the tool call arguments were not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "the tool call arguments were not a JSON object"
    return data, None


# -- the session --------------------------------------------------------------

class ToolCallSession:
    """One conversation in which the model must call `tool`, retried with
    feedback until the caller is satisfied (see the module docstring)."""

    def __init__(
        self,
        client: Any,
        model: str,
        system: str,
        user: str,
        tool: dict,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        api: Optional[str] = None,
    ):
        """`api` forces RESPONSES or CHAT; by default it follows the client
        (`api_for`). Temperature goes only to non-reasoning models and the
        reasoning effort only to reasoning ones, as the agents' clients do."""
        # 1. The call's settings.
        self.client = client
        self.model = model
        self.api = api or api_for(client)
        self.system = system
        self.tool = tool
        self.name = tool["function"]["name"]
        self.reasoning = is_reasoning_model(model)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

        # 2. The conversation so far: the full message list for Chat
        #    Completions; for the Responses API, the input still to send and
        #    the previous response, which holds everything before it.
        self.messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        self._input: list[dict] = [{"role": "user", "content": user}]
        self._previous_id: Optional[str] = None
        self._call_id: Optional[str] = None

    # -- one call ---------------------------------------------------------------

    async def ask(self) -> ToolReply:
        """Call the model once, on the conversation so far. API errors are
        raised to the caller, which decides whether the plan failed."""
        if self.api == RESPONSES:
            return await self._ask_responses()
        return await self._ask_chat()

    def correct(self, feedback: str) -> None:
        """Add `feedback` on the last reply, for the next `ask`: as the tool's
        output when the model made the call, else as a user message."""
        # 1. Responses API: only the new input is sent, after the previous response.
        if self.api == RESPONSES:
            self._input = ([{"type": "function_call_output", "call_id": self._call_id,
                             "output": feedback}] if self._call_id
                           else [{"role": "user", "content": feedback}])
            return
        # 2. Chat Completions: appended to the messages, which are sent whole.
        if self._call_id:
            self.messages.append({"role": "tool", "tool_call_id": self._call_id,
                                  "content": feedback})
        else:
            self.messages.append({"role": "user", "content": feedback})

    # -- the Responses API --------------------------------------------------------

    def responses_kwargs(self) -> dict:
        """The `responses.create` arguments for the next call."""
        # 1. The prompt and the forced tool. `strict` is off: the tree schema
        #    is recursive and open, which strict mode does not accept.
        function = self.tool["function"]
        kwargs: dict[str, Any] = {
            "model": self.model,
            # Instructions are not carried over by previous_response_id, so
            # they go with every call.
            "instructions": self.system,
            "input": self._input,
            "tools": [{"type": "function", "name": function["name"],
                       "description": function.get("description", ""),
                       "parameters": function["parameters"], "strict": False}],
            "tool_choice": {"type": "function", "name": self.name},
        }
        if self._previous_id:
            kwargs["previous_response_id"] = self._previous_id
        # 2. The model's settings.
        if self.reasoning and self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if not self.reasoning and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        return kwargs

    async def _ask_responses(self) -> ToolReply:
        """One `responses.create` call: the first function call it holds."""
        response = await self.client.responses.create(**self.responses_kwargs())
        # 1. The next call continues from this response.
        self._previous_id = response.id
        self._call_id = None
        # 2. The function call, if the model made one, and any text.
        reply = ToolReply(content=getattr(response, "output_text", None) or None)
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) == "function_call":
                self._call_id = item.call_id
                reply.called = True
                reply.arguments, reply.error = _parse_arguments(item.arguments)
                break
        return reply

    # -- Chat Completions -----------------------------------------------------------

    def chat_kwargs(self) -> dict:
        """The `chat.completions.create` arguments for the next call."""
        # 1. The prompt and the forced tool.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "tools": [self.tool],
            "tool_choice": {"type": "function", "function": {"name": self.name}},
        }
        # 2. The model's settings.
        if not self.reasoning and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_output_tokens
        return kwargs

    async def _ask_chat(self) -> ToolReply:
        """One `chat.completions.create` call: its first tool call, or its text."""
        response = await self.client.chat.completions.create(**self.chat_kwargs())
        message = response.choices[0].message
        # 1. The assistant's turn joins the conversation, for a retry.
        assistant: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant["tool_calls"] = message.tool_calls
        self.messages.append(assistant)
        # 2. The tool call, if the model made one, and any text.
        self._call_id = None
        reply = ToolReply(content=message.content)
        if message.tool_calls:
            call = message.tool_calls[0]
            self._call_id = call.id
            reply.called = True
            reply.arguments, reply.error = _parse_arguments(call.function.arguments)
        return reply
