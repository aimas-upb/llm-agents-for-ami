"""
Custom behaviors for the User Assistant.
"""

import asyncio
import logging
from typing import Optional
from spade.behaviour import OneShotBehaviour

logger = logging.getLogger("UserAssistant")


class ResponseListenerBehaviour(OneShotBehaviour):
    """
    Waits for a single message from a specific sender and sets the result on a Future.
    Used to bridge synchronous LLM tool calls with asynchronous XMPP messaging.
    """

    def __init__(
        self,
        future: asyncio.Future,
        expected_sender_prefix: str,
        expected_correlation_id: Optional[str] = None,
        expected_type: Optional[str] = None,
        timeout: int = 15,
    ):
        super().__init__()
        self.future = future
        self.expected_sender_prefix = expected_sender_prefix
        self.expected_correlation_id = expected_correlation_id
        self.expected_type = expected_type
        self.timeout = timeout

    async def run(self):
        deadline = asyncio.get_running_loop().time() + self.timeout

        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break

            msg = await self.receive(timeout=remaining)
            if not msg:
                break

            sender_str = str(msg.sender)
            if not sender_str.startswith(self.expected_sender_prefix):
                logger.debug("Ignoring message from %s (expected prefix %s)", sender_str, self.expected_sender_prefix)
                continue

            if self.expected_correlation_id and msg.get_metadata("correlation_id") != self.expected_correlation_id:
                logger.debug(
                    "Ignoring message with correlation_id=%s (expected %s)",
                    msg.get_metadata("correlation_id"),
                    self.expected_correlation_id,
                )
                continue

            if self.expected_type and msg.get_metadata("type") != self.expected_type:
                logger.debug(
                    "Ignoring message with type=%s (expected %s)",
                    msg.get_metadata("type"),
                    self.expected_type,
                )
                continue

            if not self.future.done():
                self.future.set_result(msg.body)
            return

        if not self.future.done():
            self.future.set_result("Error: Timeout waiting for reply.")
