"""Demo-only request classifier behaviour.

Logs a compact EXPLICIT / IMPLICIT / QUERY classification of user input
so demos have a readable trace. Does not influence planning.
"""

import logging
import re

from spade.behaviour import CyclicBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory


class DemoRequestClassifierBehaviour(CyclicBehaviour):
    """Demo-only logging helper: classify user requests as EXPLICIT vs IMPLICIT."""

    def __init__(self, logger=None):
        """
        Initialize with optional logger.

        Args:
            logger: Logger instance. If None, creates a basic logger.
        """
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("message_type") != "llm":
            return

        text = (msg.body or "").strip()
        low = text.lower()
        if low in ("yes", "no", "ok", "okay", "proceed", "continue"):
            return

        kind = "IMPLICIT"
        reason_parts = []

        # Check for action patterns (EXPLICIT/IMPLICIT GOAL_REQUEST)
        has_action = any(
            kw in low
            for kw in (
                "turn ", "toggle", "open", "close",
                "set ", "raise", "lower", "increase", "decrease",
            )
        )
        mentions_device = (
            any(tok in low for tok in ("light", "blinds"))
            or re.search(r"\b\w+\d{3}\b", low) is not None
            or "%" in low
        )

        if has_action and mentions_device:
            kind = "EXPLICIT"
            reason_parts.append("has_action+mentions_device")
        elif has_action or any(kw in low for kw in ("make ", "it's ", "too ", "more ", "less ")):
            kind = "IMPLICIT_GOAL"
            reason_parts.append("implied_action_or_complaint")
        elif any(kw in low for kw in ("list", "show", "what", "which", "can you", "able to", "do you")):
            # Query about capabilities or state
            if any(kw in low for kw in ("device", "devices", "command", "action", "can you", "able to")):
                kind = "ENV_CAPABILITIES"
                reason_parts.append("asks_about_capabilities")
            elif any(kw in low for kw in ("state", "on", "off", "intensity", "brightness", "temperature", "humidity")):
                kind = "ENV_STATE"
                reason_parts.append("asks_about_state")
            else:
                kind = "QUERY"
                reason_parts.append("generic_query")

        reason = "[" + ", ".join(reason_parts) + "]" if reason_parts else "[default]"
        self.logger.info(
            demo("[HEURISTIC CLASSIFIER] Request classified as %s %s: %r"),
            kind,
            reason,
            text,
        )
