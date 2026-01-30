"""
Async BehaviorTree Planner

Generates JSON IR behavior trees using async OpenAI API,
adapted for the SPADE multi-agent context in the AAMAS 2026 demo.
"""

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from .schema import GENERATE_BT_TOOL
from .prompts import (
    BT_PLANNING_SYSTEM_PROMPT,
    format_capability_context,
    format_signifier_hints,
)

logger = logging.getLogger(__name__)


class AsyncBTPlanner:
    """
    Generates JSON IR behavior trees using async OpenAI API.

    This planner:
    1. Formats environment context (affordances + state + signifier hints)
    2. Calls the LLM with the generate_behavior_tree tool
    3. Validates the response tree structure
    4. Returns the JSON IR dict

    Usage:
        planner = AsyncBTPlanner()
        result = await planner.generate_bt(
            intents=["turn on the light", "open the blinds"],
            affordances=[...],
            state={...},
            signifier_hints={...},
            client=async_openai_client,
            model="o3",
        )
    """

    def __init__(self, max_attempts: int = 3):
        """
        Args:
            max_attempts: Maximum LLM call attempts for validation retries
        """
        self.max_attempts = max_attempts

    async def generate_bt(
        self,
        intents: list[str],
        affordances: list[dict],
        state: Optional[dict] = None,
        signifier_hints: Optional[dict] = None,
        client: Optional[AsyncOpenAI] = None,
        model: str = "o3",
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate a JSON IR behavior tree.

        Args:
            intents: List of user intents to satisfy
            affordances: List of available affordance dicts
            state: Optional current state dict
            signifier_hints: Optional signifier match data for context injection
            client: AsyncOpenAI client
            model: Model name (e.g., "o3", "gpt-4o")
            temperature: Optional temperature override
            reasoning_effort: Optional reasoning effort ("low", "medium", "high")
            max_completion_tokens: Optional max completion tokens

        Returns:
            Dict with keys: tree (JSON IR spec), explanation (str),
            impossible (bool), intents (list)
        """
        if client is None:
            raise ValueError("AsyncOpenAI client is required")

        # Build the planning prompt
        capability_context = format_capability_context(affordances, state)
        sig_hints_text = format_signifier_hints(signifier_hints)

        system_prompt = BT_PLANNING_SYSTEM_PROMPT.format(
            capability_context=capability_context,
            signifier_hints=sig_hints_text,
        )

        # Format user message with intents
        user_message = self._format_user_message(intents)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Build API kwargs
        api_kwargs = self._build_api_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
        )

        # Retry loop for validation
        last_explanation = ""
        validation_errors: list[str] = []

        for attempt in range(self.max_attempts):
            try:
                response = await client.chat.completions.create(**api_kwargs)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return {
                    "tree": {},
                    "explanation": f"LLM call failed: {e}",
                    "impossible": False,
                    "intents": intents,
                }

            message = response.choices[0].message

            # Append assistant message for potential retry
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = message.tool_calls
            messages.append(assistant_msg)

            if not message.tool_calls:
                logger.warning("No tool call in response")
                return {
                    "tree": {},
                    "explanation": f"Generation failed: {message.content}",
                    "impossible": False,
                    "intents": intents,
                }

            tool_call = message.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return {
                    "tree": {},
                    "explanation": f"JSON parse error: {e}",
                    "impossible": False,
                    "intents": intents,
                }

            tree_spec = args.get("tree", {})
            last_explanation = args.get("explanation", "")
            impossible = args.get("impossible", False)

            if impossible:
                logger.info(f"Goal marked as impossible: {last_explanation}")
                return {
                    "tree": {},
                    "explanation": last_explanation,
                    "impossible": True,
                    "intents": intents,
                }

            # Normalize tree (fill in missing optional fields like 'name')
            tree_spec = self._normalize_tree(tree_spec)

            # Validate
            validation_errors = self._validate_tree(tree_spec)
            if not validation_errors:
                node_count = self._count_nodes(tree_spec)
                logger.info(f"Generated JSON IR with {node_count} nodes")
                return {
                    "tree": tree_spec,
                    "explanation": last_explanation,
                    "impossible": False,
                    "intents": intents,
                }

            logger.warning(
                "Invalid tree spec (attempt %d/%d): %s",
                attempt + 1,
                self.max_attempts,
                "; ".join(validation_errors),
            )

            # Provide feedback for retry
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    "The previous behavior tree was invalid:\n- "
                    + "\n- ".join(validation_errors)
                    + "\nPlease call generate_behavior_tree again with a corrected, non-empty tree."
                ),
            })

            # Update kwargs with new messages
            api_kwargs["messages"] = messages

        return {
            "tree": {},
            "explanation": "Generation failed: " + "; ".join(validation_errors),
            "impossible": False,
            "intents": intents,
        }

    def _format_user_message(self, intents: list[str]) -> str:
        """Format the user message from intents."""
        if len(intents) == 1:
            return intents[0]
        return (
            "Please handle the following requests:\n"
            + "\n".join(f"- {intent}" for intent in intents)
        )

    def _build_api_kwargs(
        self,
        model: str,
        messages: list[dict],
        temperature: Optional[float],
        reasoning_effort: Optional[str],
        max_completion_tokens: Optional[int],
    ) -> dict:
        """Build kwargs for the OpenAI API call."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": [GENERATE_BT_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "generate_behavior_tree"},
            },
        }

        # Reasoning models (o-series) don't support temperature
        is_reasoning = model.startswith("o")
        if not is_reasoning and temperature is not None:
            kwargs["temperature"] = temperature

        if is_reasoning and reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens

        return kwargs

    def _normalize_tree(self, spec: dict, _counter: list | None = None) -> dict:
        """
        Normalize a tree spec by filling in default values for optional fields.

        LLMs (especially smaller models like gpt-4o-mini) often omit the 'name'
        field. This method generates sensible defaults so validation passes.
        """
        if not isinstance(spec, dict) or not spec:
            return spec

        if _counter is None:
            _counter = [0]

        # Auto-generate name if missing
        if not spec.get("name") or not isinstance(spec.get("name"), str):
            node_type = spec.get("type", "node")
            if node_type == "action":
                url = spec.get("action_url", "")
                action_name = url.rstrip("/").rsplit("/", 1)[-1] if url else "action"
                spec["name"] = action_name.replace("_", " ").title().replace(" ", "")
            elif node_type == "condition":
                prop = spec.get("property_url", "")
                prop_name = prop.rstrip("/").rsplit("/", 1)[-1] if prop else "check"
                expected = spec.get("expected_value", "")
                spec["name"] = f"Check{prop_name.title()}Is{expected}".replace(" ", "")
            else:
                _counter[0] += 1
                spec["name"] = f"{node_type.title()}_{_counter[0]}"

        # Recursively normalize children
        children = spec.get("children")
        if isinstance(children, list):
            spec["children"] = [
                self._normalize_tree(child, _counter) for child in children
            ]

        return spec

    def _count_nodes(self, spec: dict) -> int:
        """Count nodes in a tree spec."""
        if not spec:
            return 0
        count = 1
        for child in spec.get("children", []):
            count += self._count_nodes(child)
        return count

    def _validate_tree(self, spec: dict, path: str = "tree") -> list[str]:
        """Validate that the tree spec is non-empty and structurally sound."""
        errors: list[str] = []

        if not spec:
            errors.append(f"{path}: tree is empty")
            return errors

        if not isinstance(spec, dict):
            errors.append(f"{path}: expected object, got {type(spec).__name__}")
            return errors

        name = spec.get("name")
        if not name or not isinstance(name, str):
            errors.append(f"{path}: missing 'name'")

        node_type = spec.get("type")
        valid_types = {"sequence", "selector", "parallel", "action", "condition"}
        if node_type not in valid_types:
            errors.append(f"{path}: missing or invalid 'type'")

        if node_type in {"sequence", "selector", "parallel"}:
            children = spec.get("children")
            if not isinstance(children, list) or not children:
                errors.append(f"{path}: composite nodes require non-empty 'children'")
            else:
                for idx, child in enumerate(children):
                    errors.extend(
                        self._validate_tree(child, path=f"{path}.children[{idx}]")
                    )
        elif node_type == "action":
            action_url = spec.get("action_url")
            if not action_url or not isinstance(action_url, str):
                errors.append(f"{path}: action nodes require 'action_url'")
        elif node_type == "condition":
            property_url = spec.get("property_url")
            if not property_url or not isinstance(property_url, str):
                errors.append(f"{path}: condition nodes require 'property_url'")
            if "expected_value" not in spec:
                errors.append(f"{path}: condition nodes require 'expected_value'")

        return errors
