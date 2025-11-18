"""
Protocol interfaces for agent communication and behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models.messages import Message, MessageType
from ..models.plan import Plan
from ..models.environment import ChangeEvent


class IAgent(ABC):
    """Base interface for all agents in the system."""

    @abstractmethod
    async def start(self) -> None:
        """
        Start the agent and initialize all behaviors.

        TODO: Implementation steps:
        1. Connect to SPADE server
        2. Register all behaviors
        3. Subscribe to relevant message topics
        4. Perform initial setup (load configurations, etc.)
        5. Notify system that agent is ready
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the agent and cleanup resources.

        TODO: Implementation steps:
        1. Stop all active behaviors
        2. Save current state (if needed)
        3. Unsubscribe from message topics
        4. Disconnect from SPADE server
        5. Cleanup resources (close database connections, etc.)
        """
        pass

    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """
        Send a message to another agent.

        Args:
            message: The message to send.

        Returns:
            True if sent successfully, False otherwise.

        TODO: Implementation steps:
        1. Validate message structure
        2. Serialize message
        3. Send via SPADE messaging
        4. Handle send errors
        5. Log message sent
        """
        pass

    @abstractmethod
    async def receive_message(self, message: Message) -> None:
        """
        Receive and process a message from another agent.

        Args:
            message: The received message.

        TODO: Implementation steps:
        1. Deserialize message
        2. Validate message type
        3. Route to appropriate handler based on message type
        4. Log message received
        """
        pass


class IMessageRouter(ABC):
    """Interface for routing messages based on type and content."""

    @abstractmethod
    async def route_message(self, message: Message) -> None:
        """
        Route a message to the appropriate handler.

        Args:
            message: The message to route.

        TODO: Implementation steps:
        1. Identify message type
        2. Determine appropriate handler
        3. Invoke handler with message
        4. Handle routing errors
        """
        pass

    @abstractmethod
    def register_handler(self, message_type: MessageType, handler: callable) -> None:
        """
        Register a handler for a specific message type.

        Args:
            message_type: The type of message to handle.
            handler: The handler function.

        TODO: Implementation steps:
        1. Validate handler signature
        2. Store handler in routing table
        3. Handle duplicate registrations
        """
        pass


class IMemoryManager(ABC):
    """Interface for managing conversation and agent memory."""

    @abstractmethod
    async def store_conversation(self, conversation_id: str, message: str,
                                 role: str, metadata: Dict[str, Any]) -> None:
        """
        Store a conversation message.

        Args:
            conversation_id: Unique identifier for the conversation.
            message: The message content.
            role: Role (user/assistant).
            metadata: Additional metadata.

        TODO: Implementation steps:
        1. Validate conversation_id
        2. Create message record
        3. Store in database
        4. Update conversation index
        5. Check if summarization is needed
        """
        pass

    @abstractmethod
    async def retrieve_conversation_history(self, conversation_id: str,
                                            limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history.

        Args:
            conversation_id: Unique identifier for the conversation.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of conversation messages.

        TODO: Implementation steps:
        1. Query database for conversation
        2. Apply limit
        3. Format messages
        4. Return in chronological order
        """
        pass

    @abstractmethod
    async def store_memory(self, key: str, value: Any, metadata: Dict[str, Any]) -> None:
        """
        Store a memory item.

        Args:
            key: Memory identifier.
            value: Memory content.
            metadata: Additional metadata.

        TODO: Implementation steps:
        1. Serialize value
        2. Store in vector database (if enabled)
        3. Store in primary database
        4. Update indices
        """
        pass

    @abstractmethod
    async def retrieve_memory(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory item.

        Args:
            key: Memory identifier.

        Returns:
            Memory content if found, None otherwise.

        TODO: Implementation steps:
        1. Query database
        2. Deserialize value
        3. Return memory
        """
        pass

    @abstractmethod
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search over memories.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of relevant memories.

        TODO: Implementation steps:
        1. Generate query embedding
        2. Search vector store
        3. Rank results by similarity
        4. Return top-k results
        """
        pass
