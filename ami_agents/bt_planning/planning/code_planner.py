"""
Async direct-code BT planner (``python_code`` mode).

Subclass of :class:`AsyncBTPlanner` that swaps the generation format — a
plain chat completion returning a fenced Python script over the builder DSL
(``bt_dsl.py``) — while inheriting the retry loop and the entire
post-generation pipeline (affordance-ref resolution, normalization,
settling-time annotation, validation). Keeping that pipeline byte-identical
across modes is what makes the JSON-IR vs direct-code comparison fair.

Plain completion (no forced tool call) is deliberate: the target SLMs are
code-completion-tuned, and forcing them to emit Python source as a
JSON-escaped tool-argument string is a known failure mode. The reference
models use the identical call shape.
"""

import concurrent.futures
import logging
import re
from typing import Any, Optional

from .bt_planner import AsyncBTPlanner
from .bt_dsl import DslResult, build_env_snapshot, run_bt_code
from .prompts import BT_CODE_PLANNING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Generated scripts are straight-line tree construction; anything running
# longer is a runaway loop. The worker thread may outlive the timeout (there
# is no safe way to kill an exec thread) — acceptable for the benchmark
# harness, and why run_bt_code must never move to the execution side.
CODE_EXEC_TIMEOUT_SECONDS = 5.0

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.S)


class AsyncCodeBTPlanner(AsyncBTPlanner):
    """Generates JSON IR behaviour trees from LLM-written builder-DSL code."""

    def __init__(self, max_attempts: int = 3):
        super().__init__(max_attempts=max_attempts)
        self._env_snapshot: dict = {}
        self._last_code: Optional[str] = None

    def _system_prompt(
        self,
        capability_context: str,
        signifier_hints: str,
        observable_property_hints: str,
        state: Optional[dict],
    ) -> str:
        self._env_snapshot = build_env_snapshot(state)
        self._last_code = None
        return BT_CODE_PLANNING_SYSTEM_PROMPT.format(
            capability_context=capability_context,
            signifier_hints=signifier_hints,
            observable_property_hints=observable_property_hints,
        )

    def _build_api_kwargs(
        self,
        model: str,
        messages: list[dict],
        temperature: Optional[float],
        reasoning_effort: Optional[str],
        max_completion_tokens: Optional[int],
    ) -> dict:
        kwargs = super()._build_api_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
        )
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        return kwargs

    def _extract_args(
        self, message: Any, attempt: int
    ) -> tuple[Optional[dict], Optional[str], list[str]]:
        code = self._parse_code_content(message.content)
        if code is None:
            logger.warning(
                "No Python code found in response (attempt %d/%d)",
                attempt + 1,
                self.max_attempts,
            )
            return None, None, [
                "the response did not contain a ```python code block building the tree"
            ]

        self._last_code = code
        result = self._run_code_guarded(code)
        if result.errors:
            logger.warning(
                "Generated code failed (attempt %d/%d): %s",
                attempt + 1,
                self.max_attempts,
                "; ".join(result.errors),
            )
            return None, None, result.errors

        return {
            "tree": result.tree or {},
            "explanation": result.explanation,
            "impossible": result.impossible,
        }, None, []

    def _run_code_guarded(self, code: str):
        """Run the generated script with a wall-clock cap (see module note)."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(run_bt_code, code, self._env_snapshot)
        try:
            return future.result(timeout=CODE_EXEC_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            return DslResult(errors=[
                f"the code did not finish within {CODE_EXEC_TIMEOUT_SECONDS:g} seconds "
                "(possible infinite loop)"
            ])
        finally:
            # wait=False so a runaway script's thread cannot block the retry
            # loop; the abandoned thread is the documented leak.
            pool.shutdown(wait=False)

    @staticmethod
    def _parse_code_content(content: Optional[str]) -> Optional[str]:
        """Extract the generated script from the completion text."""
        if not content or not isinstance(content, str):
            return None

        fences = [m.group(1).strip() for m in _CODE_FENCE_RE.finditer(content)]
        fences = [f for f in fences if f]
        if fences:
            # SLMs often restate the API before the actual plan — take the
            # last block that assigns `tree` or marks the goal impossible,
            # falling back to the last block.
            for block in reversed(fences):
                if "tree" in block or "mark_impossible" in block:
                    return block
            return fences[-1]

        stripped = content.strip()
        if "tree" in stripped and (
            "sequence(" in stripped
            or "selector(" in stripped
            or "parallel(" in stripped
            or "action(" in stripped
            or "condition(" in stripped
            or "=" in stripped
        ):
            return stripped
        if "mark_impossible" in stripped:
            return stripped
        return None

    def _retry_feedback(self, validation_errors: list[str]) -> str:
        return (
            "The previous response was invalid:\n- "
            + "\n- ".join(validation_errors)
            + "\nReply again with a single corrected ```python code block that "
            "assigns the behaviour tree to `tree` (or calls mark_impossible)."
        )

    def _extra_result_fields(self) -> dict:
        fields: dict = {"plan_mode": "python_code"}
        if self._last_code:
            fields["generated_code"] = self._last_code
        return fields
