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
        if low.startswith(("what", "show", "list", "which", "is", "are")) and (
            "workspace" in low or "workspaces" in low or "device" in low
            or "devices" in low or "state" in low
        ):
            kind = "QUERY"
        else:
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

        self.logger.info(demo("Request classified as %s: %r"), kind, text)
