"""Behaviour: phrase an ENV_STATE answer that needs the user's intent read.

Most state answers are built in code -- a value is a value. Two are not:

- a *mismatch*, where several unrelated properties matched and only the
  phrasing says whether the user asked about a collection ("is anything on in
  the kitchen") or about one device ("is the fan on");
- a reading whose value is a *dictionary*, where the question decides which of
  its fields is the answer.

Both need the sentence the user wrote, so both come here. One behaviour per
answer, per CLAUDE.md 3.2.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.class_labels import label_for
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..prompts.intent_response_prompts import STATE_ANSWER_PROMPT
from ..utils import loose_json_loads
from ..utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs


def readings_for_prompt(affordances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The affordances as the model should see them.

    Classes are rendered as their labels and the artifact URI and read target
    are dropped: an identifier or a URL in the input is an identifier or a URL
    that can end up in the answer. The value is passed through untouched --
    a dictionary stays a dictionary, because choosing one of its fields is the
    whole reason this call exists.
    """
    readings = []
    for affordance in affordances:
        reading = {
            "device": label_for(affordance.get("artifact_type"))
                      or affordance.get("artifact_name"),
            "property": label_for(affordance.get("affordance_type")),
            "value": affordance.get("value"),
        }
        room = affordance.get("workspace_name")
        if room:
            reading["room"] = room
        if affordance.get("detail"):
            reading["note"] = affordance["detail"]
        readings.append(reading)
    return readings


class StateAnswerBehaviour(OneShotBehaviour):
    """Ask a small model to phrase one state answer."""

    def __init__(self, text_intent: str, outcome: str,
                 affordances: List[Dict[str, Any]], logger=None) -> None:
        super().__init__()
        self.text_intent = text_intent
        self.outcome = outcome
        self.affordances = affordances
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        # Populated by ``run``: the sentence, or None with ``error`` set.
        self.result: Optional[str] = None
        self.error: Optional[str] = None

    async def run(self) -> None:
        payload = {
            "question": self.text_intent,
            "outcome": self.outcome,
            "readings": readings_for_prompt(self.affordances),
        }
        self.logger.info(demo(
            f"[LLM CALL] Phrasing {self.outcome} for {self.text_intent[:60]!r}"))

        try:
            llm_cfg = build_behaviour_llm_client(self.agent.config, "state_answer")
            response = await llm_cfg.client.chat.completions.create(
                model=llm_cfg.model,
                messages=[
                    {"role": "system", "content": STATE_ANSWER_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                **build_llm_call_kwargs(llm_cfg),
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = loose_json_loads(raw)
            answer = (parsed.get("answer") if isinstance(parsed, dict) else None)
            if not answer:
                raise ValueError(f"no answer in response: {raw[:200]!r}")
            self.result = str(answer).strip()
        except Exception as exc:
            # The caller falls back to a plainer sentence built from the same
            # readings: worse phrasing, never a missing answer.
            self.error = str(exc)
            self.logger.warning("State answer phrasing failed: %s", exc)
