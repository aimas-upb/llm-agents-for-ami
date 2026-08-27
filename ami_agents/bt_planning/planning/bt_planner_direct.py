"""
Direct Code BehaviorTree Planner.

Generates executable py_trees Python code (with inline compute nodes) instead
of JSON IR. It is a drop-in for AsyncBTPlanner: the ``generate_bt`` signature is
identical so the harness/interaction-solver cannot tell them apart. The return
envelope differs -- ``tree`` is ``None`` and the payload lives in ``code`` --
because inline compute behaviours are not representable in the JSON IR and must
be executed directly from the py_trees object (see CodeBTExecutor).
"""

import ast
import json
import logging
import re
from typing import Any, Optional

from openai import AsyncOpenAI

from .prompts import (
    BT_CODE_GENERATION_SYSTEM_PROMPT,
    GENERATE_CODE_TOOL,
    build_affordance_index,
    build_url_to_ref,
    format_capability_context,
    format_code_context,
    format_observable_property_hints,
    format_signifier_hints,
)

logger = logging.getLogger(__name__)


def extract_impossible_subgoals(code: str) -> list[str]:
    """Return descriptions from ``# IMPOSSIBLE: <desc>`` comment lines."""
    pattern = r"^#\s*IMPOSSIBLE:\s*(.+)$"
    return [m.strip() for m in re.findall(pattern, code or "", re.MULTILINE)]


def strip_code_fences(code: str) -> str:
    """Remove a leading ```python / ``` fence and trailing ``` from code."""
    code = (code or "").strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```"):
        code = code[len("```"):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


class DirectCodeBTPlanner:
    """Generates py_trees Python code via the OpenAI API (tool call or text)."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    async def generate_bt(
        self,
        intents: list[str],
        affordances: list[dict],
        state: Optional[dict] = None,
        signifier_hints: Optional[dict] = None,
        observable_property_hints: Optional[dict] = None,
        client: Optional[AsyncOpenAI] = None,
        model: str = "gpt-4",
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        use_tool_calling: bool = True,
    ) -> dict:
        """
        Generate a py_trees code plan.

        Returns a dict with keys: ``tree`` (always ``None`` for this mode),
        ``code`` (py_trees source), ``explanation``, ``impossible``,
        ``detected_impossible`` and ``intents``.
        """
        if client is None:
            raise ValueError("AsyncOpenAI client is required")

        affordance_index = build_affordance_index(affordances)
        url_to_ref = build_url_to_ref(affordance_index)
        capability_context = format_capability_context(affordances, state, index=affordance_index)
        sig_hints_text = format_signifier_hints(signifier_hints, url_to_ref=url_to_ref)
        obs_hints_text = format_observable_property_hints(observable_property_hints, url_to_ref=url_to_ref)

        context_block = format_code_context(capability_context, sig_hints_text, obs_hints_text)
        system_prompt = BT_CODE_GENERATION_SYSTEM_PROMPT.replace("{context}", context_block)
        user_message = self._format_user_message(intents)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        api_kwargs = self._build_api_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            use_tool_calling=use_tool_calling,
        )

        last_error = ""
        for attempt in range(self.max_attempts):
            try:
                response = await client.chat.completions.create(**api_kwargs)
            except Exception as e:
                logger.error(
                    "LLM call failed on attempt %d/%d: %s (%s)",
                    attempt + 1, self.max_attempts, e, type(e).__name__,
                )
                return self._failure(intents, f"LLM call failed: {e}")

            message = response.choices[0].message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = message.tool_calls
            messages.append(assistant_msg)

            code, explanation, tool_call_id = self._extract_code(message)

            if not code:
                last_error = "the response contained no parseable Python code"
                logger.warning(
                    "No code in response (attempt %d/%d)", attempt + 1, self.max_attempts
                )
            else:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    last_error = f"the generated code has a syntax error: {e}"
                    logger.warning(
                        "Generated code failed ast.parse (attempt %d/%d): %s",
                        attempt + 1, self.max_attempts, e,
                    )
                else:
                    detected_impossible = extract_impossible_subgoals(code)
                    logger.info(
                        "Generated py_trees code (%d chars, %d impossible sub-goals)",
                        len(code), len(detected_impossible),
                    )
                    return {
                        "tree": None,
                        "code": code,
                        "explanation": explanation,
                        "impossible": bool(detected_impossible),
                        "detected_impossible": detected_impossible,
                        "intents": intents,
                    }

            feedback = (
                f"The previous response was invalid: {last_error}. "
                "Return corrected Python code that defines a top-level 'tree' variable."
            )
            if tool_call_id:
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": feedback})
            else:
                messages.append({"role": "user", "content": feedback})
            api_kwargs["messages"] = messages

        return self._failure(intents, "Generation failed: " + last_error)

    def _extract_code(self, message) -> tuple[str, str, Optional[str]]:
        """Return (code, explanation, tool_call_id) from a chat message."""
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse tool call arguments: %s", e)
                return "", "", tool_call.id
            return (
                strip_code_fences(args.get("code", "")),
                args.get("explanation", ""),
                tool_call.id,
            )
        # Text fallback for SLMs that ignore tool_choice: recover a ```python block
        # or a bare tool-call JSON from the message content.
        content = message.content or ""
        args = self._parse_code_content(content)
        if args is not None:
            return strip_code_fences(args.get("code", "")), args.get("explanation", ""), None
        block = self._first_python_block(content)
        if block:
            return block, "", None
        return "", "", None

    @staticmethod
    def _parse_code_content(content: str) -> Optional[dict]:
        """Recover a {"code": ..., "explanation": ...} object from JSON text."""
        if not content:
            return None
        candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", content, re.S)]
        candidates.append(content)
        for candidate in candidates:
            candidate = candidate.strip()
            data = None
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                start, end = candidate.find("{"), candidate.rfind("}")
                if start != -1 and end > start:
                    try:
                        data = json.loads(candidate[start:end + 1])
                    except json.JSONDecodeError:
                        data = None
            if isinstance(data, dict):
                arguments = data.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = None
                if isinstance(arguments, dict) and "code" in arguments:
                    return arguments
                if "code" in data:
                    return data
        return None

    @staticmethod
    def _first_python_block(content: str) -> str:
        """Return the first fenced code block's body, if any."""
        if not content:
            return ""
        m = re.search(r"```(?:python)?\s*(.*?)```", content, re.S)
        if m:
            return m.group(1).strip()
        return ""

    def _format_user_message(self, intents: list[str]) -> str:
        if len(intents) == 1:
            return intents[0]
        return "Please handle the following requests:\n" + "\n".join(
            f"- {intent}" for intent in intents
        )

    def _build_api_kwargs(
        self,
        model: str,
        messages: list[dict],
        temperature: Optional[float],
        reasoning_effort: Optional[str],
        max_completion_tokens: Optional[int],
        use_tool_calling: bool,
    ) -> dict:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if use_tool_calling:
            kwargs["tools"] = [GENERATE_CODE_TOOL]
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": "generate_behavior_tree_code"},
            }
        is_reasoning = model.startswith("o")
        if not is_reasoning and temperature is not None:
            kwargs["temperature"] = temperature
        if is_reasoning and reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        return kwargs

    @staticmethod
    def _failure(intents: list[str], explanation: str) -> dict:
        return {
            "tree": None,
            "code": "",
            "explanation": explanation,
            "impossible": False,
            "detected_impossible": [],
            "intents": intents,
        }
