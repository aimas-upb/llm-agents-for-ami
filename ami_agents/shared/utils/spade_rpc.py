"""
SPADE request/reply helper.

Implements a small, reliable RPC-style primitive on top of SPADE messages:
- a single always-on receiver behaviour per agent
- routes replies into pending[correlation_id] -> Future dict

This avoids common pitfalls:
- behaviour mailboxes filtered by templates (request handler doesn't receive replies)
- race where reply arrives before a listener is attached
- brittle sender matching with XMPP resources
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message as SpadeMessage

from ..models.messages import (
    META_TYPE,
    META_CORRELATION_ID,
    META_CONVERSATION_ID,
    ensure_correlation_id,
    serialize_body,
)


class RpcTimeoutError(TimeoutError):
    """Raised when an RPC call times out."""


@dataclass(frozen=True)
class RpcResult:
    correlation_id: str
    body: str
    sender: str
    thread: Optional[str]


@dataclass
class _Pending:
    future: asyncio.Future
    expect_type: Optional[str] = None
    thread: Optional[str] = None

class RpcRouterBehaviour(CyclicBehaviour):
    """
    Always-on router behaviour that receives all messages (template=None) and
    resolves pending Futures for correlated replies.
    """

    def __init__(self):
        super().__init__()
        self.pending: Dict[str, _Pending] = {}

    def register(self, correlation_id: str, future: asyncio.Future, expect_type: Optional[str], thread: Optional[str]):
        self.pending[correlation_id] = _Pending(future=future, expect_type=expect_type, thread=thread)

    def unregister(self, correlation_id: str):
        self.pending.pop(correlation_id, None)

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        corr_id = msg.get_metadata(META_CORRELATION_ID)
        if not corr_id:
            return
        corr_id = str(corr_id)

        pending = self.pending.get(corr_id)
        if not pending:
            return

        # Optional guards for type/thread
        if pending.expect_type:
            msg_type = msg.get_metadata(META_TYPE)
            if str(msg_type) != str(pending.expect_type):
                return

        if pending.thread:
            if not getattr(msg, "thread", None) or str(msg.thread) != str(pending.thread):
                return

        if not pending.future.done():
            pending.future.set_result(
                RpcResult(
                    correlation_id=corr_id,
                    body=msg.body or "",
                    sender=str(msg.sender),
                    thread=str(msg.thread) if getattr(msg, "thread", None) else None,
                )
            )
        self.unregister(corr_id)


def _get_or_create_router(agent: Agent) -> RpcRouterBehaviour:
    """
    Attach (or retrieve) the per-agent RpcRouterBehaviour.

    SPADE dispatches each received message to *every* behaviour whose template matches.
    Since this router is added with template=None, it receives a copy of all messages
    without "stealing" them from other behaviours.
    """
    router = agent.get("_rpc_router") if hasattr(agent, "get") else None
    if isinstance(router, RpcRouterBehaviour):
        return router

    router = RpcRouterBehaviour()
    agent.add_behaviour(router, template=None)
    if hasattr(agent, "set"):
        agent.set("_rpc_router", router)
    return router


def get_rpc_router(agent: Agent) -> RpcRouterBehaviour:
    """Public accessor for the per-agent RPC router behaviour."""
    return _get_or_create_router(agent)


async def send_via_router(agent: Agent, msg: SpadeMessage) -> None:
    """
    Send a SPADE message using a behaviour (required by SPADE's container/XMPP send path).
    """
    router = _get_or_create_router(agent)
    await router.send(msg)


async def rpc_call(
    agent: Agent,
    *,
    to_jid: str,
    request_type: str,
    body: Any,
    expect_type: str,
    timeout: float = 30.0,
    thread: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RpcResult:
    """
    Send a request message and await a single correlated response.

    Args:
        agent: SPADE Agent instance.
        to_jid: Receiver JID.
        request_type: metadata["type"] for the request.
        body: message body payload; dict/list/etc will be JSON-serialized.
        expect_type: metadata["type"] expected in the reply.
        timeout: max seconds to wait.
        thread: optional XMPP thread to set on request and require on reply template.
        metadata: optional extra metadata to add to the request.
    """
    if metadata is None:
        metadata = {}

    correlation_id = ensure_correlation_id(metadata)
    router = _get_or_create_router(agent)

    # 1) Register pending future BEFORE send (prevents races)
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    router.register(correlation_id=correlation_id, future=future, expect_type=expect_type, thread=thread)

    # 2) Send request
    msg = SpadeMessage(to=to_jid)
    msg.set_metadata(META_TYPE, request_type)
    msg.set_metadata(META_CORRELATION_ID, correlation_id)
    if thread:
        msg.thread = thread
        msg.set_metadata(META_CONVERSATION_ID, thread)

    for k, v in (metadata or {}).items():
        if v is None:
            continue
        # avoid overwriting the standard keys we already set
        if k in (META_TYPE, META_CORRELATION_ID):
            continue
        msg.set_metadata(str(k), str(v))

    msg.body = serialize_body(body)
    
    await send_via_router(agent, msg)

    try:
        result = await asyncio.wait_for(future, timeout=timeout)
        return result
    except asyncio.TimeoutError as e:
        router.unregister(correlation_id)
        raise RpcTimeoutError("Timeout waiting for RPC reply") from e
    finally:
        # ensure we don't leak pending entries if the awaiter is cancelled
        if future.cancelled():
            router.unregister(correlation_id)
