"""Behaviour: invoke the LLM-backed AsyncBTPlanner to generate a BT JSON IR.

Per Alex's chat: "generating a plan is a behavior". Wraps the
``AsyncBTPlanner.generate_bt`` call so the orchestration behaviour
doesn't reach into the LLM client directly.
"""

from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.logger import LoggerFactory


class BTPlanGenerationBehaviour(OneShotBehaviour):
    """Run AsyncBTPlanner against the gathered context to produce a BT IR."""

    def __init__(
        self,
        intent_strings: List[str],
        affordances: List[Dict[str, Any]],
        state: Optional[Dict[str, Any]],
        signifier_hints: Dict[str, Any],
        affected_env_vars: Optional[Dict[str, List[Dict[str, str]]]] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.intent_strings = intent_strings
        self.affordances = affordances
        self.state = state
        self.signifier_hints = signifier_hints
        self.affected_env_vars = affected_env_vars  # Maps intent_str -> list of {variable, direction}
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Populated by ``run`` — either an LLM result dict or {"error": ...}.
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None

    async def run(self) -> None:
        agent = self.agent
        try:
            self.result = await agent.bt_planner.generate_bt(
                intents=self.intent_strings,
                affordances=self.affordances,
                state=self.state,
                signifier_hints=self.signifier_hints,
                affected_env_vars=self.affected_env_vars,
                client=agent.llm_client,
                model=agent.model,
                temperature=agent.temperature if not (agent.model.startswith("o") or agent.model.startswith("gpt-5")) else None,
                reasoning_effort=agent.reasoning_effort,
                max_completion_tokens=agent.max_completion_tokens,
            )
        except Exception as e:
            self.error = str(e)
            self.logger.warning(f"BT generation failed: {e}")
            self.result = {}
