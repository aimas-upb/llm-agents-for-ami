"""
The UA's front door.

This used to be the whole agent: receive, segment, parse, plan, summarise,
confirm, execute, record signifiers -- about 1400 lines in one `run()`. While
any of that was in flight the UA answered nothing else, because the work was
awaited inline.

Now it only receives. Each utterance gets a `UserRequestBehaviour` FSM of its
own and this returns immediately, so a second utterance is picked up on the very
next tick and the two proceed side by side under SPADE's scheduler.

Messages belonging to a request in progress (a yes/no on a proposed plan) carry
`active_request_id` and are matched by that FSM's own template, so they never
reach this behaviour.
"""

from __future__ import annotations

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from ....shared.models.messages import META_REQUEST_ID, new_request_id
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..models import ConversationPhase
from .user_request import UserRequestBehaviour


class UserMessageBehaviour(CyclicBehaviour):
    """Accepts user utterances and starts a request FSM for each."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")

    async def run(self) -> None:
        msg = await self.receive(timeout=1)
        if msg is None:
            return

        text = (msg.body or "").strip()
        if not text:
            return

        thread = str(getattr(msg, "thread", None) or "__default__")
        request_id = new_request_id()

        conv = self.agent.get_conversation(thread)
        conv.request_id = request_id
        conv.user_message = text
        conv.phase = ConversationPhase.SEGMENTING

        self.agent.requests.open(request_id, thread, text)
        self.logger.info(
            demo("Received user message: request=%s thread=%s text=%r"),
            request_id, thread, text)

        request = UserRequestBehaviour(
            request_id=request_id,
            thread=thread,
            user_text=text,
            original_msg=msg,
            logger=self.logger,
        )
        # The FSM answers only to messages tagged with its own request id, so
        # two concurrent requests never read each other's replies.
        template = Template()
        template.set_metadata(META_REQUEST_ID, request_id)
        self.agent.add_behaviour(request, template)
