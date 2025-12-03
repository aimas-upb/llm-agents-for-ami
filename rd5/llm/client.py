"""OpenAI LLM client for RD5 plan generation.

This module provides a wrapper around LangChain's OpenAI integration
for intent extraction and code generation.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI

from rd5.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI LLM client for RD5 operations.

    Provides methods for intent extraction, code generation,
    and feedback summarization.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize LLM client.

        Args:
            model: OpenAI model name. Defaults to settings.
            temperature: Temperature for responses. Defaults to settings.
            api_key: OpenAI API key. Defaults to settings.
        """
        settings = get_settings()
        self.model = model or settings.openai_model
        self.temperature = temperature if temperature is not None else settings.openai_temperature
        self.api_key = api_key or settings.openai_api_key
        self.max_tokens = settings.openai_max_tokens

        if not self.api_key:
            logger.warning("No OpenAI API key configured")

        self._llm: Optional[ChatOpenAI] = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create LLM instance.

        Returns:
            Configured ChatOpenAI instance.
        """
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            logger.info(f"Initialized LLM: {self.model}")
        return self._llm

    async def extract_intent(
        self,
        user_request: str,
        available_device_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract structured intent from natural language request.

        Args:
            user_request: Natural language user request.
            available_device_types: Optional list of available device types.

        Returns:
            Structured intent dictionary with:
            - intent: Cleaned intent text
            - action_verb: Primary action verb
            - target_objects: List of target objects/devices
            - parameters: Extracted parameters
        """
        from rd5.llm.prompts.intent_extraction import INTENT_EXTRACTION_PROMPT

        device_info = ""
        if available_device_types:
            device_info = f"\nAvailable device types: {', '.join(available_device_types)}"

        messages = [
            SystemMessage(content=INTENT_EXTRACTION_PROMPT),
            HumanMessage(content=f"{user_request}{device_info}"),
        ]

        parser = JsonOutputParser()
        chain = self.llm | parser

        try:
            result = await chain.ainvoke(messages)
            logger.info(f"Extracted intent: {result.get('intent', 'unknown')}")
            return result
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            # Return basic fallback
            return {
                "intent": user_request,
                "action_verb": "unknown",
                "target_objects": [],
                "parameters": {},
            }

    async def generate_plan_code(
        self,
        intent: str,
        affordances: List[Dict[str, Any]],
        similar_plans: Optional[List[Dict[str, Any]]] = None,
        retry_feedback: Optional[str] = None,
    ) -> str:
        """Generate Python code for executing a plan.

        Args:
            intent: Extracted intent description.
            affordances: List of available affordances with their schemas.
            similar_plans: Optional list of similar past plans for reference.
            retry_feedback: Optional feedback from failed validation for retry.

        Returns:
            Generated Python code string.
        """
        from rd5.llm.prompts.code_generation import CODE_GENERATION_PROMPT

        # Format affordances for prompt
        affordance_info = self._format_affordances(affordances)

        # Format similar plans if available
        similar_plans_info = ""
        if similar_plans:
            similar_plans_info = self._format_similar_plans(similar_plans)

        # Add retry context if this is a retry
        retry_info = ""
        if retry_feedback:
            retry_info = f"\n\nPREVIOUS ATTEMPT FAILED:\n{retry_feedback}\nPlease fix the issues and try again."

        prompt = CODE_GENERATION_PROMPT.format(
            intent=intent,
            affordances=affordance_info,
            similar_plans=similar_plans_info,
            retry_feedback=retry_info,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Generate code to: {intent}"),
        ]

        parser = StrOutputParser()
        chain = self.llm | parser

        try:
            result = await chain.ainvoke(messages)
            # Extract code block if wrapped in markdown
            code = self._extract_code_block(result)
            logger.info(f"Generated plan code ({len(code)} chars)")
            return code
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    async def summarize_execution(
        self,
        intent: str,
        code: str,
        execution_result: Dict[str, Any],
    ) -> str:
        """Summarize execution result for feedback/logging.

        Args:
            intent: Original intent.
            code: Executed code.
            execution_result: Result dictionary with stdout, stderr, success.

        Returns:
            Human-readable summary of execution.
        """
        from rd5.llm.prompts.feedback_summary import FEEDBACK_SUMMARY_PROMPT

        success = execution_result.get("success", False)
        stdout = execution_result.get("stdout", "")
        stderr = execution_result.get("stderr", "")

        prompt = FEEDBACK_SUMMARY_PROMPT.format(
            intent=intent,
            success="SUCCESS" if success else "FAILED",
            stdout=stdout[:500] if stdout else "(no output)",
            stderr=stderr[:500] if stderr else "(no errors)",
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Summarize the execution result."),
        ]

        parser = StrOutputParser()
        chain = self.llm | parser

        try:
            result = await chain.ainvoke(messages)
            return result.strip()
        except Exception as e:
            logger.error(f"Execution summary failed: {e}")
            return f"Execution {'succeeded' if success else 'failed'}: {intent}"

    def _format_affordances(self, affordances: List[Dict[str, Any]]) -> str:
        """Format affordances for LLM prompt.

        Args:
            affordances: List of affordance dictionaries.

        Returns:
            Formatted string describing affordances.
        """
        if not affordances:
            return "No affordances available."

        lines = []
        for aff in affordances:
            uri = aff.get("uri", aff.get("affordance_uri", "unknown"))
            name = aff.get("name", "unnamed")
            aff_type = aff.get("type", aff.get("affordance_type", "unknown"))
            form = aff.get("form", {})

            lines.append(f"- {name} ({aff_type})")
            lines.append(f"  URI: {uri}")

            if form:
                href = form.get("href", "")
                method = form.get("method", "GET")
                lines.append(f"  Endpoint: {method} {href}")

            if "input_schema" in aff:
                lines.append(f"  Input: {aff['input_schema']}")

        return "\n".join(lines)

    def _format_similar_plans(self, plans: List[Dict[str, Any]]) -> str:
        """Format similar plans for reference in prompt.

        Args:
            plans: List of similar plan dictionaries.

        Returns:
            Formatted string with plan examples.
        """
        if not plans:
            return ""

        lines = ["\n\nSIMILAR PAST PLANS FOR REFERENCE:"]
        for i, plan in enumerate(plans[:3], 1):  # Limit to 3 examples
            intent = plan.get("intent_text", plan.get("extracted_intent", ""))
            code = plan.get("plan_code", "")[:300]  # Truncate
            success = plan.get("success_count", 0)

            lines.append(f"\nExample {i} (used {success} times):")
            lines.append(f"Intent: {intent}")
            lines.append(f"Code snippet:\n```python\n{code}\n```")

        return "\n".join(lines)

    def _extract_code_block(self, text: str) -> str:
        """Extract Python code from markdown code block.

        Args:
            text: Text potentially containing markdown code block.

        Returns:
            Extracted code or original text if no block found.
        """
        import re

        # Try to find Python code block
        pattern = r"```(?:python)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # No code block found, return as-is
        return text.strip()


# Module-level client instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client.

    Returns:
        Configured LLMClient instance.
    """
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
