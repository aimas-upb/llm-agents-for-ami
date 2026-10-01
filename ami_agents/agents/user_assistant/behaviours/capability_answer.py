"""Behaviour: phrase an ENV_CAPABILITY_QUERY answer that lists what was found.

A yes/no, a "cannot" and a "none" are built in code (see
`utils/capability_answers.py`). A `query` that found something is not: the
answer has to be put in terms of what was asked -- the fan modes rather than the
fan-mode property, "off, low, high" rather than `[0, 1, 3]` -- which needs the
user's sentence read. One behaviour per answer, per CLAUDE.md 3.2.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..prompts.intent_response_prompts import CAPABILITY_ANSWER_PROMPT
from ..utils import loose_json_loads
from ..utils.capability_answers import entries_for_prompt
from ..utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs


class CapabilityAnswerBehaviour(OneShotBehaviour):
    """Ask a model to phrase one capability answer."""

    def __init__(self, text_intent: str, entries: List[Dict[str, Any]],
                 logger=None) -> None:
        super().__init__()
        self.text_intent = text_intent
        self.entries = entries
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        # Populated by ``run``: the sentence, or None with ``error`` set.
        self.result: Optional[str] = None
        self.error: Optional[str] = None

    async def run(self) -> None:
        payload = {
            "question": self.text_intent,
            "found": entries_for_prompt(self.entries),
        }
        self.logger.info(demo(
            f"[LLM CALL] Phrasing capabilities for {self.text_intent[:60]!r}"))

        try:
            llm_cfg = build_behaviour_llm_client(self.agent.config,
                                                 "capabilities_analysis")
            response = await llm_cfg.client.chat.completions.create(
                model=llm_cfg.model,
                messages=[
                    {"role": "system", "content": CAPABILITY_ANSWER_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                **build_llm_call_kwargs(llm_cfg),
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = loose_json_loads(raw)
            answer = parsed.get("answer") if isinstance(parsed, dict) else None
            if not answer:
                raise ValueError(f"no answer in response: {raw[:200]!r}")
            self.result = str(answer).strip()
        except Exception as exc:
            # The caller falls back to a plainer sentence built from the same
            # entries: worse phrasing, never a missing answer.
            self.error = str(exc)
            self.logger.warning("Capability answer phrasing failed: %s", exc)
