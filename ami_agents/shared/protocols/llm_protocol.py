"""
Protocol interfaces for LLM integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ILLMProvider(ABC):
    """Interface for LLM provider implementations."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for a prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            The generated completion.

        TODO: Implementation steps:
        1. Validate prompt
        2. Apply rate limiting
        3. Call LLM API
        4. Handle API errors and retries
        5. Return completion
        """
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        TODO: Implementation steps:
        1. Validate text
        2. Call embedding API
        3. Handle API errors
        4. Return embedding vector
        """
        pass

    @abstractmethod
    async def complete_with_functions(self, prompt: str, functions: List[Dict[str, Any]],
                                     **kwargs) -> Dict[str, Any]:
        """
        Generate a completion with function calling support.

        Args:
            prompt: The input prompt.
            functions: Available function definitions.
            **kwargs: Additional parameters.

        Returns:
            Completion with potential function calls.

        TODO: Implementation steps:
        1. Validate prompt and functions
        2. Format function definitions for LLM
        3. Call LLM API with function support
        4. Parse function calls from response
        5. Return structured response
        """
        pass


class IIntentExtractor(ABC):
    """Interface for extracting intents from user messages."""

    @abstractmethod
    async def extract_intent(self, message: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Extract a logical, concise intent from a user message.

        Args:
            message: The user message.
            context: Conversation context.

        Returns:
            The extracted intent, or None if extraction fails.

        TODO: Implementation steps:
        1. Prepare prompt with message and context
        2. Call LLM to extract intent
        3. Validate that intent is environment-related
        4. Validate that intent is achievable
        5. Return concise intent or None
        """
        pass

    @abstractmethod
    async def validate_intent(self, intent: str, environment_capabilities: List[str]) -> bool:
        """
        Validate that an intent can be achieved with available capabilities.

        Args:
            intent: The extracted intent.
            environment_capabilities: List of available environment capabilities.

        Returns:
            True if intent is achievable, False otherwise.

        TODO: Implementation steps:
        1. Prepare prompt with intent and capabilities
        2. Call LLM to assess achievability
        3. Parse response
        4. Return validation result
        """
        pass


class IMessageClassifier(ABC):
    """Interface for classifying user messages."""

    @abstractmethod
    async def classify(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a user message into one of the defined categories.

        Args:
            message: The user message.
            context: Conversation context.

        Returns:
            Classification result with category and confidence.

        TODO: Implementation steps:
        1. Prepare classification prompt
        2. Call LLM with message and categories
        3. Parse classification result
        4. Validate confidence threshold
        5. Return classification with confidence score
        """
        pass


class IPlanGenerator(ABC):
    """Interface for generating execution plans."""

    @abstractmethod
    async def generate_plan(self, goal: str, context: Dict[str, Any],
                          affordances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a behavior tree plan for a goal.

        Args:
            goal: The goal to achieve.
            context: Planning context.
            affordances: Available affordances.

        Returns:
            A behavior tree plan structure.

        TODO: Implementation steps:
        1. Prepare planning prompt with goal and affordances
        2. Call LLM to generate plan
        3. Parse response into behavior tree structure
        4. Validate plan structure
        5. Add node templates with metadata
        6. Return complete plan
        """
        pass

    @abstractmethod
    async def explain_plan(self, plan: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of a plan.

        Args:
            plan: The behavior tree plan.

        Returns:
            Human-readable plan summary.

        TODO: Implementation steps:
        1. Extract key actions from plan
        2. Prepare explanation prompt
        3. Call LLM to generate summary
        4. Return natural language explanation
        """
        pass
