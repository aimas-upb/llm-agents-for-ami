"""
WebSocket utility functions and connection management.

Provides utilities for managing WebSocket connections, generating IDs,
and formatting timestamps for the manual mode dialog interface.
"""

import asyncio
import uuid
import weakref
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Any, Callable
from fastapi import WebSocket
import logging

from .protocol import WebSocketMessage, serialize_message, deserialize_message


def generate_thread_id() -> str:
    """
    Generate a unique thread ID for conversation tracking.

    Returns:
        UUID string suitable for thread identification
    """
    return str(uuid.uuid4())


def generate_plan_id() -> str:
    """
    Generate a unique plan ID for plan tracking.

    Returns:
        UUID string suitable for plan identification
    """
    return str(uuid.uuid4())


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format a timestamp for WebSocket messages.

    Args:
        dt: Datetime object to format. If None, uses current UTC time.

    Returns:
        ISO format timestamp string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.isoformat()


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for the manual mode dialog interface.

    Handles connection registration, message broadcasting, and cleanup.
    """

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.thread_connections: Dict[str, str] = {}  # thread_id -> connection_id
        self.connection_threads: Dict[str, Set[str]] = {}  # connection_id -> set of thread_ids
        self.message_handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)

    async def register_connection(self, websocket: WebSocket, connection_id: str = None) -> str:
        """
        Register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
            connection_id: Optional connection ID. If None, generates a new UUID.

        Returns:
            Connection ID for the registered connection
        """
        if connection_id is None:
            connection_id = str(uuid.uuid4())

        self.connections[connection_id] = websocket
        self.connection_threads[connection_id] = set()

        self.logger.info(f"Registered WebSocket connection: {connection_id}")
        return connection_id

    async def unregister_connection(self, connection_id: str):
        """
        Unregister a WebSocket connection and clean up associated data.

        Args:
            connection_id: ID of the connection to unregister
        """
        if connection_id in self.connections:
            # Clean up thread mappings
            threads = self.connection_threads.get(connection_id, set())
            for thread_id in threads:
                self.thread_connections.pop(thread_id, None)

            # Remove connection data
            del self.connections[connection_id]
            self.connection_threads.pop(connection_id, None)

            self.logger.info(f"Unregistered WebSocket connection: {connection_id}")

    def associate_thread(self, connection_id: str, thread_id: str):
        """
        Associate a conversation thread with a WebSocket connection.

        Args:
            connection_id: WebSocket connection ID
            thread_id: Conversation thread ID
        """
        if connection_id in self.connections:
            self.thread_connections[thread_id] = connection_id
            self.connection_threads[connection_id].add(thread_id)
            self.logger.debug(f"Associated thread {thread_id} with connection {connection_id}")

    def get_connection_for_thread(self, thread_id: str) -> Optional[WebSocket]:
        """
        Get the WebSocket connection associated with a thread.

        Args:
            thread_id: Conversation thread ID

        Returns:
            WebSocket connection if found, None otherwise
        """
        connection_id = self.thread_connections.get(thread_id)
        if connection_id:
            return self.connections.get(connection_id)
        return None

    async def send_to_thread(self, thread_id: str, message: WebSocketMessage) -> bool:
        """
        Send a message to a specific conversation thread.

        Args:
            thread_id: Target thread ID
            message: Message to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        connection = self.get_connection_for_thread(thread_id)
        if connection:
            try:
                serialized = serialize_message(message)
                await connection.send_text(serialized)
                self.logger.debug(f"Sent message to thread {thread_id}: {message.type}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to send message to thread {thread_id}: {e}")
                return False
        else:
            self.logger.warning(f"No connection found for thread {thread_id}")
            return False

    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """
        Send a message to a specific WebSocket connection.

        Args:
            connection_id: Target connection ID
            message: Message to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        connection = self.connections.get(connection_id)
        if connection:
            try:
                serialized = serialize_message(message)
                await connection.send_text(serialized)
                self.logger.debug(f"Sent message to connection {connection_id}: {message.type}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to send message to connection {connection_id}: {e}")
                return False
        else:
            self.logger.warning(f"Connection not found: {connection_id}")
            return False

    async def broadcast_to_all(self, message: WebSocketMessage):
        """
        Broadcast a message to all connected WebSocket clients.

        Args:
            message: Message to broadcast
        """
        if not self.connections:
            return

        serialized = serialize_message(message)
        disconnected = []

        for connection_id, websocket in self.connections.items():
            try:
                await websocket.send_text(serialized)
            except Exception as e:
                self.logger.error(f"Failed to send to connection {connection_id}: {e}")
                disconnected.append(connection_id)

        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.unregister_connection(connection_id)

    def register_message_handler(self, message_type: str, handler: Callable):
        """
        Register a handler function for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")

    async def handle_incoming_message(self, connection_id: str, raw_message: str) -> bool:
        """
        Process an incoming WebSocket message.

        Args:
            connection_id: ID of the connection that sent the message
            raw_message: Raw message string

        Returns:
            True if message was handled successfully, False otherwise
        """
        try:
            message = deserialize_message(raw_message)
            handler = self.message_handlers.get(message.type)

            if handler:
                await handler(connection_id, message)
                return True
            else:
                self.logger.warning(f"No handler for message type: {message.type}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to handle message from {connection_id}: {e}")
            return False

    def get_connection_count(self) -> int:
        """
        Get the number of active WebSocket connections.

        Returns:
            Number of active connections
        """
        return len(self.connections)

    def get_thread_count(self) -> int:
        """
        Get the number of active conversation threads.

        Returns:
            Number of active threads
        """
        return len(self.thread_connections)

    async def cleanup_stale_connections(self):
        """
        Clean up connections that are no longer active.

        This should be called periodically to remove connections
        that have been closed without proper cleanup.
        """
        stale_connections = []

        for connection_id, websocket in self.connections.items():
            try:
                # Try to ping the connection
                await websocket.ping()
            except Exception:
                stale_connections.append(connection_id)

        for connection_id in stale_connections:
            await self.unregister_connection(connection_id)

        if stale_connections:
            self.logger.info(f"Cleaned up {len(stale_connections)} stale connections")


class AsyncWebSocketQueue:
    """
    Async queue for managing WebSocket message processing.

    Provides a thread-safe way to queue and process WebSocket messages
    with proper error handling and backpressure control.
    """

    def __init__(self, maxsize: int = 1000):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.processing = False
        self.logger = logging.getLogger(__name__)

    async def put(self, item: WebSocketMessage):
        """
        Add a message to the processing queue.

        Args:
            item: Message to queue for processing
        """
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.logger.warning("WebSocket message queue is full, dropping message")

    async def get(self) -> WebSocketMessage:
        """
        Get the next message from the queue.

        Returns:
            Next message to process
        """
        return await self.queue.get()

    def task_done(self):
        """Mark a queued message as processed."""
        self.queue.task_done()

    async def join(self):
        """Wait for all messages in the queue to be processed."""
        await self.queue.join()

    def qsize(self) -> int:
        """Get the current queue size."""
        return self.queue.qsize()

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self.queue.empty()

    def full(self) -> bool:
        """Check if the queue is full."""
        return self.queue.full()