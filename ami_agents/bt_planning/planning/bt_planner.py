"""
Async BehaviorTree Planner

Generates JSON IR behavior trees using async OpenAI API,
adapted for the SPADE multi-agent context in the AAMAS 2026 demo.
"""

import difflib
import json
import logging
import re
from typing import Any, Optional

from openai import AsyncOpenAI

from .schema import GENERATE_BT_TOOL
from .prompts import (
    BT_PLANNING_SYSTEM_PROMPT,
    build_affordance_index,
    build_url_to_ref,
    format_capability_context,
    format_observable_property_hints,
    format_signifier_hints,
)

logger = logging.getLogger(__name__)

SETTLING_WAIT_MARGIN_SECONDS = 5.0


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
            model="gpt-4",
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
        observable_property_hints: Optional[dict] = None,
        client: Optional[AsyncOpenAI] = None,
        model: str = "gpt-4",
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

        # Build the planning prompt.  Affordances are shown as short ids and
        # resolved back to target URLs after generation.
        affordance_index = build_affordance_index(affordances)
        url_to_ref = build_url_to_ref(affordance_index)
        capability_context = format_capability_context(affordances, state, index=affordance_index)
        sig_hints_text = format_signifier_hints(signifier_hints, url_to_ref=url_to_ref)
        obs_hints_text = format_observable_property_hints(observable_property_hints, url_to_ref=url_to_ref)

        system_prompt = BT_PLANNING_SYSTEM_PROMPT.format(
            capability_context=capability_context,
            signifier_hints=sig_hints_text,
            observable_property_hints=obs_hints_text,
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
        state_artifact_count = len((state or {}).get("artifacts", {})) if isinstance(state, dict) else 0
        signifier_match_count = len((signifier_hints or {}).get("matches", [])) if isinstance(signifier_hints, dict) else 0
        observable_result_count = len((observable_property_hints or {}).get("results", [])) if isinstance(observable_property_hints, dict) else 0
        prompt_debug = {
            "model": model,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
            "intents": intents,
            "intent_count": len(intents),
            "affordance_count": len(affordances),
            "state_artifact_count": state_artifact_count,
            "signifier_match_count": signifier_match_count,
            "observable_result_count": observable_result_count,
            "system_prompt_chars": len(system_prompt),
            "user_message_chars": len(user_message),
        }

        for attempt in range(self.max_attempts):
            try:
                response = await client.chat.completions.create(**api_kwargs)
            except Exception as e:
                logger.error(
                    "LLM call failed on attempt %d/%d: %s (%s) context=%s",
                    attempt + 1,
                    self.max_attempts,
                    e,
                    type(e).__name__,
                    prompt_debug,
                )
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

            args: Optional[dict] = None
            tool_call_id: Optional[str] = None
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_call_id = tool_call.id
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool call arguments: {e}")
                    validation_errors = [f"the tool call arguments were not valid JSON: {e}"]
            else:
                # Small local models often emit the tool call as fenced JSON
                # text instead of a structured tool call; recover it.
                args = self._parse_tool_call_content(message.content)
                if args is not None:
                    logger.info("Recovered generate_behavior_tree arguments from message content")
                else:
                    logger.warning(
                        "No tool call in response and content fallback failed (attempt %d/%d)",
                        attempt + 1,
                        self.max_attempts,
                    )
                    validation_errors = [
                        "the response contained neither a generate_behavior_tree tool call "
                        "nor parseable JSON arguments"
                    ]

            if args is not None:
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

                # Resolve short affordance ids back to target URLs so the final
                # IR is executable and downstream consumers stay URL-based.
                resolution_errors = self._resolve_affordance_refs(tree_spec, affordance_index)

                # Normalize tree (fill in missing optional fields like 'name')
                tree_spec = self._normalize_tree(tree_spec)
                settling_times = self._collect_action_settling_times(
                    affordances=affordances,
                    observable_property_hints=observable_property_hints,
                )
                if settling_times:
                    self._apply_settling_times(tree_spec, settling_times)

                # Validate
                validation_errors = (
                    resolution_errors
                    + self._validate_tree(tree_spec)
                    + self._detect_equality_gates(tree_spec)
                )
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
            feedback = (
                "The previous response was invalid:\n- "
                + "\n- ".join(validation_errors)
                + "\nPlease call generate_behavior_tree again with a corrected, non-empty tree."
            )
            if tool_call_id:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": feedback,
                })
            else:
                messages.append({"role": "user", "content": feedback})

            # Update kwargs with new messages
            api_kwargs["messages"] = messages

        return {
            "tree": {},
            "explanation": "Generation failed: " + "; ".join(validation_errors),
            "impossible": False,
            "intents": intents,
        }

    @staticmethod
    def _parse_tool_call_content(content: Optional[str]) -> Optional[dict]:
        """
        Recover generate_behavior_tree arguments from plain-text content.

        Ollama does not enforce tool_choice, so small local models often emit
        the tool call as (fenced) JSON text. Accepts either the tool-call
        wrapper ``{"name": ..., "arguments": {...}}`` or the bare arguments
        ``{"tree": ..., "explanation": ..., "impossible": ...}``.
        """
        if not content or not isinstance(content, str):
            return None

        candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", content, re.S)]
        candidates.append(content)

        for candidate in candidates:
            candidate = candidate.strip()
            data = None
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                start = candidate.find("{")
                end = candidate.rfind("}")
                if start != -1 and end > start:
                    try:
                        data = json.loads(candidate[start:end + 1])
                    except json.JSONDecodeError:
                        data = None
            if not isinstance(data, dict):
                continue

            arguments = data.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = None
            if isinstance(arguments, dict):
                return arguments
            if "tree" in data:
                return data
        return None

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

    def _resolve_affordance_refs(
        self, spec: dict, index: dict[str, dict], path: str = "tree"
    ) -> list[str]:
        """
        Resolve short ``affordance_id`` references to target URLs in-place.

        Action nodes get ``action_url``; condition/wait_condition nodes get
        ``property_url``. Unknown ids become validation errors so the LLM
        retry loop can correct them. Literal http(s) URLs are passed through
        for condition nodes fed by state keys or observable-property hints,
        and for legacy trees that already carry URLs.
        """
        errors: list[str] = []
        if not isinstance(spec, dict) or not spec:
            return errors

        node_type = spec.get("type")
        if node_type == "action":
            errors.extend(self._resolve_node_ref(spec, index, "action_url", path))
        elif node_type in {"condition", "wait_condition"}:
            errors.extend(self._resolve_node_ref(spec, index, "property_url", path))

        children = spec.get("children")
        if isinstance(children, list):
            for idx, child in enumerate(children):
                errors.extend(
                    self._resolve_affordance_refs(child, index, path=f"{path}.children[{idx}]")
                )
        return errors

    def _resolve_node_ref(
        self, spec: dict, index: dict[str, dict], url_field: str, path: str
    ) -> list[str]:
        node_type = spec.get("type")
        ref = spec.get("affordance_id")

        if not ref or not isinstance(ref, str):
            legacy_url = spec.get(url_field)
            if isinstance(legacy_url, str) and legacy_url:
                return []
            return [f"{path}: {node_type} nodes require 'affordance_id'"]

        aff = index.get(ref)
        if aff is not None:
            aff_type = str(aff.get("affordance_type") or "action")
            if node_type == "action" and aff_type != "action":
                return [
                    f"{path}: affordance '{ref}' is a {aff_type} affordance; "
                    "action nodes require an [action] affordance id"
                ]
            target = aff.get("target") or aff.get("affordance_uri") or aff.get("href")
            if not isinstance(target, str) or not target:
                return [f"{path}: affordance '{ref}' has no target URL"]
            spec[url_field] = target
            if node_type == "action":
                return self._validate_action_parameters(spec, aff, path)
            return []

        if ref.startswith("http://") or ref.startswith("https://"):
            spec[url_field] = ref
            return []

        # Models often drop long artifact-id prefixes (e.g.
        # 'utility_room_environment/x' for 'qt2_feasible_seed_1_utility_room_environment/x');
        # accept the ref when it is an unambiguous token-boundary suffix.
        suffix_matches = self._suffix_matches(ref, index)
        if len(suffix_matches) == 1:
            resolved = suffix_matches[0]
            logger.info("Resolved affordance_id '%s' via unique suffix match '%s'", ref, resolved)
            spec["affordance_id"] = resolved
            return self._resolve_node_ref(spec, index, url_field, path)
        if len(suffix_matches) > 1:
            return [
                f"{path}: affordance_id '{ref}' is ambiguous -- matches: "
                + ", ".join(suffix_matches[:4])
            ]

        # Models tend to repeat a hallucinated id verbatim across retries when
        # only told it is unknown; suggesting close matches breaks that loop.
        suggestions = difflib.get_close_matches(ref, index.keys(), n=3, cutoff=0.6)
        message = f"{path}: unknown affordance_id '{ref}'"
        if suggestions:
            message += " -- did you mean: " + ", ".join(suggestions) + "?"
        else:
            message += " (use an exact id from the affordances list)"
        return [message]

    @staticmethod
    def _validate_action_parameters(spec: dict, aff: dict, path: str) -> list[str]:
        """
        Validate action parameters against the affordance's input schema, plus
        the Home Assistant climate.set_temperature payload rules that only
        surface as HTTP 400 at execution time otherwise.
        """
        errors: list[str] = []
        params = spec.get("parameters")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return [f"{path}: action 'parameters' must be an object"]

        schema = aff.get("input_schema")
        props = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(props, dict) and props:
            unknown = sorted(k for k in params if k not in props)
            if unknown:
                errors.append(
                    f"{path}: unknown parameter(s) {', '.join(unknown)} -- "
                    f"allowed: {', '.join(sorted(props))}"
                )

        name = str(aff.get("action_name") or aff.get("name") or "")
        target = str(aff.get("target") or "")
        is_set_temperature = (
            target.rstrip("/").endswith("set_temperature")
            or "settemperature" in name.replace("_", "").lower()
        )
        if is_set_temperature:
            has_temp = "temperature" in params
            has_high = "target_temp_high" in params
            has_low = "target_temp_low" in params
            if has_temp and (has_high or has_low):
                errors.append(
                    f"{path}: set_temperature must use either 'temperature' or the pair "
                    "'target_temp_high'+'target_temp_low', never both styles"
                )
            elif has_high != has_low:
                errors.append(
                    f"{path}: set_temperature range mode requires both "
                    "'target_temp_high' and 'target_temp_low'"
                )
            elif not has_temp and not has_high:
                errors.append(
                    f"{path}: set_temperature requires 'temperature' (or "
                    "'target_temp_high'+'target_temp_low'); 'hvac_mode' alone is not valid"
                )
        return errors

    @staticmethod
    def _suffix_matches(ref: str, index: dict[str, dict]) -> list[str]:
        """Index keys for which ``ref`` is a token-boundary suffix."""
        matches = []
        ref_lower = ref.lower()
        for key in index:
            key_lower = key.lower()
            if key_lower == ref_lower:
                matches.append(key)
                continue
            if key_lower.endswith(ref_lower):
                boundary = key_lower[-len(ref_lower) - 1]
                if boundary in "_/.-:":
                    matches.append(key)
        return matches

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
                url = spec.get("affordance_id") or spec.get("action_url", "")
                action_name = url.rstrip("/").rsplit("/", 1)[-1] if url else "action"
                spec["name"] = action_name.replace("_", " ").title().replace(" ", "")
            elif node_type in {"condition", "wait_condition"}:
                prop = spec.get("affordance_id") or spec.get("property_url", "")
                prop_name = prop.rstrip("/").rsplit("/", 1)[-1] if prop else "check"
                expected = spec.get("expected_value", "")
                prefix = "WaitFor" if node_type == "wait_condition" else "Check"
                spec["name"] = f"{prefix}{prop_name.title()}Is{expected}".replace(" ", "")
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

    def _collect_action_settling_times(
        self,
        *,
        affordances: list[dict],
        observable_property_hints: Optional[dict],
    ) -> dict[str, float]:
        """Build action_url -> settling seconds from planner inputs."""
        settling_times: dict[str, float] = {}

        for aff in affordances or []:
            if not isinstance(aff, dict):
                continue
            target = aff.get("target") or aff.get("affordance_uri") or aff.get("href")
            seconds = aff.get("settling_time_seconds")
            if isinstance(target, str) and target and isinstance(seconds, (int, float)):
                target_key = self._normalize_action_url(target)
                settling_times[target_key] = max(settling_times.get(target_key, 0.0), float(seconds))

        results = (
            observable_property_hints.get("results")
            if isinstance(observable_property_hints, dict)
            else None
        )
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                actions = result.get("actions")
                if not isinstance(actions, list):
                    continue
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    target = action.get("action_target")
                    seconds = action.get("settling_time_seconds")
                    if isinstance(target, str) and target and isinstance(seconds, (int, float)):
                        target_key = self._normalize_action_url(target)
                        settling_times[target_key] = max(settling_times.get(target_key, 0.0), float(seconds))

        return settling_times

    @staticmethod
    def _normalize_action_url(action_url: str) -> str:
        return action_url.rstrip("/")

    def _apply_settling_times(self, spec: dict, settling_times: dict[str, float]) -> float:
        """
        Annotate action nodes and expand following wait_condition timeouts.

        Returns the maximum settling time that this subtree may introduce before
        a parent sequence can verify its effects.
        """
        if not isinstance(spec, dict) or not spec:
            return 0.0

        node_type = spec.get("type")
        if node_type == "action":
            action_url = spec.get("action_url")
            seconds = (
                settling_times.get(self._normalize_action_url(action_url))
                if isinstance(action_url, str)
                else None
            )
            if seconds is not None and seconds > 0:
                spec["settling_time_seconds"] = float(seconds)
                return float(seconds)
            return 0.0

        if node_type == "wait_condition":
            return 0.0

        children = spec.get("children")
        if not isinstance(children, list):
            return 0.0

        if node_type == "sequence":
            pending_settling = 0.0
            subtree_settling = 0.0
            for child in children:
                if not isinstance(child, dict):
                    continue
                if child.get("type") == "wait_condition" and pending_settling > 0:
                    minimum_timeout = pending_settling + SETTLING_WAIT_MARGIN_SECONDS
                    current_timeout = child.get("timeout_seconds", 30.0)
                    if not isinstance(current_timeout, (int, float)) or float(current_timeout) < minimum_timeout:
                        child["timeout_seconds"] = minimum_timeout
                    pending_settling = 0.0
                    self._apply_settling_times(child, settling_times)
                    continue

                child_settling = self._apply_settling_times(child, settling_times)
                pending_settling = max(pending_settling, child_settling)
                subtree_settling = max(subtree_settling, child_settling)
            return subtree_settling

        return max(
            (
                self._apply_settling_times(child, settling_times)
                for child in children
                if isinstance(child, dict)
            ),
            default=0.0,
        )

    def _detect_equality_gates(self, spec: dict, path: str = "tree") -> list[str]:
        """
        Flag the equality-gate anti-pattern: a sequence whose first child is a
        numeric ``==`` condition followed by an action. When the current value
        does not match exactly, the sequence fails before the action ever
        runs, so the plan silently does nothing. The correct forms are a
        selector (idempotent check) or a range operator.
        """
        errors: list[str] = []
        if not isinstance(spec, dict):
            return errors

        children = spec.get("children")
        if isinstance(children, list) and children:
            if spec.get("type") == "sequence":
                first = children[0] if isinstance(children[0], dict) else {}
                expected = first.get("expected_value")
                has_action_after = any(
                    isinstance(child, dict) and child.get("type") == "action"
                    for child in children[1:]
                )
                if (
                    first.get("type") == "condition"
                    and first.get("operator", "==") == "=="
                    and isinstance(expected, (int, float))
                    and not isinstance(expected, bool)
                    and has_action_after
                ):
                    errors.append(
                        f"{path}.children[0]: a sequence must not start with an equality "
                        f"condition on a numeric value (== {expected}) before an action -- "
                        "if the current value differs, the action never runs. Use a "
                        "selector (condition 'already in desired state' first, action "
                        "second) or a range operator (<=, >=)."
                    )
            for idx, child in enumerate(children):
                errors.extend(
                    self._detect_equality_gates(child, path=f"{path}.children[{idx}]")
                )
        return errors

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
        valid_types = {"sequence", "selector", "parallel", "action", "condition", "wait_condition"}
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
                errors.append(f"{path}: action nodes require a resolvable 'affordance_id'")
        elif node_type in {"condition", "wait_condition"}:
            property_url = spec.get("property_url")
            if not property_url or not isinstance(property_url, str):
                errors.append(f"{path}: {node_type} nodes require a resolvable 'affordance_id'")
            if "expected_value" not in spec:
                errors.append(f"{path}: {node_type} nodes require 'expected_value'")
            if node_type == "wait_condition":
                timeout_seconds = spec.get("timeout_seconds")
                if timeout_seconds is not None and not isinstance(timeout_seconds, (int, float)):
                    errors.append(f"{path}: wait_condition timeout_seconds must be numeric")
                poll_interval = spec.get("poll_interval_seconds")
                if poll_interval is not None and (
                    not isinstance(poll_interval, (int, float)) or poll_interval < 0
                ):
                    errors.append(f"{path}: wait_condition poll_interval_seconds must be >= 0")

        return errors
