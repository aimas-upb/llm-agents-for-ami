"""Behaviour: infer affected environment variables from implicit goal intents.

For implicit intents with subtype == "implicit_intent", calls the LLM-backed
INFER_IMPLICIT_INTENT_DESIRE_PROMPT to extract environment variables and their
desired directions (increase/decrease).

Result is used for environment-variable-based signifier matching (v3).
"""

from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.logger import LoggerFactory


class ImplicitGoalDesireInferenceBehaviour(OneShotBehaviour):
    """Infer affected environment variables from an implicit intent via LLM."""

    def __init__(
        self,
        intent_text: str,
        logger=None,
    ) -> None:
        super().__init__()
        self.intent_text = intent_text
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Populated by ``run``: {"intent_text": str, "affected_env_vars": [...]}
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None

    async def run(self) -> None:
        agent = self.agent
        try:
            from ..prompts import INFER_IMPLICIT_INTENT_DESIRE_PROMPT

            # Build the LLM prompt
            messages = [
                {"role": "system", "content": INFER_IMPLICIT_INTENT_DESIRE_PROMPT},
                {"role": "user", "content": f"Intent: {self.intent_text}"},
            ]

            # Call LLM
            response = await agent.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )

            # Parse response
            if not response or not response.choices:
                self.error = "LLM returned empty response"
                return

            response_text = response.choices[0].message.content.strip()

            # Try to parse as JSON
            import json

            from ....shared.utils.logger import LoggerFactory as LF

            stripped = response_text.strip()
            if stripped.startswith("```"):
                # Remove markdown fences if present
                stripped = stripped.split("```")[1]
                if stripped.startswith("json"):
                    stripped = stripped[4:]
                stripped = stripped.strip()

            parsed = json.loads(stripped)

            # Validate structure
            if "intent_text" not in parsed or "affected_env_vars" not in parsed:
                self.error = "Response missing required fields"
                return

            if not isinstance(parsed.get("affected_env_vars"), list):
                self.error = "affected_env_vars is not a list"
                return

            if not parsed["affected_env_vars"]:
                self.error = "affected_env_vars is empty"
                return

            self.result = {
                "intent_text": parsed["intent_text"],
                "affected_env_vars": parsed["affected_env_vars"],
            }

        except Exception as e:
            self.logger.error("Implicit goal desire inference failed: %s", e)
            self.error = str(e)
