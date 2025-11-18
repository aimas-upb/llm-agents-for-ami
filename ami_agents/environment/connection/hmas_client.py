"""
Client for interacting with Hypermedia MAS (HMAS) environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...shared.models.environment import (
    Workspace, Artifact, ThingDescription, ChangeEvent
)


class IHMASClient(ABC):
    """Interface for HMAS environment client."""

    @abstractmethod
    async def connect(self, endpoint_url: str) -> bool:
        """
        Connect to the HMAS environment.

        Args:
            endpoint_url: The URL of the HMAS endpoint.

        Returns:
            True if connection successful, False otherwise.

        TODO: Implementation steps:
        1. Validate endpoint URL
        2. Establish HTTP/WebSocket connection
        3. Authenticate if required
        4. Verify HMAS environment
        5. Return connection status
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the HMAS environment.

        TODO: Implementation steps:
        1. Unsubscribe from all notifications
        2. Close WebSocket connections
        3. Cleanup resources
        """
        pass

    @abstractmethod
    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Retrieve a workspace by ID.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            The workspace if found, None otherwise.

        TODO: Implementation steps:
        1. Send GET request to workspace endpoint
        2. Parse response
        3. Create Workspace object
        4. Return workspace
        """
        pass

    @abstractmethod
    async def list_workspaces(self, parent_id: Optional[str] = None) -> List[Workspace]:
        """
        List all workspaces or sub-workspaces of a parent.

        Args:
            parent_id: Optional parent workspace ID.

        Returns:
            List of workspaces.

        TODO: Implementation steps:
        1. Construct query with parent_id filter if provided
        2. Send GET request
        3. Parse response
        4. Create Workspace objects
        5. Return list of workspaces
        """
        pass

    @abstractmethod
    async def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: The artifact identifier.

        Returns:
            The artifact if found, None otherwise.

        TODO: Implementation steps:
        1. Send GET request to artifact endpoint
        2. Parse response
        3. Create Artifact object with ThingDescription
        4. Return artifact
        """
        pass

    @abstractmethod
    async def list_artifacts(self, workspace_id: str) -> List[Artifact]:
        """
        List all artifacts in a workspace.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            List of artifacts.

        TODO: Implementation steps:
        1. Send GET request to workspace artifacts endpoint
        2. Parse response
        3. Create Artifact objects
        4. Return list of artifacts
        """
        pass

    @abstractmethod
    async def get_thing_description(self, artifact_id: str) -> Optional[ThingDescription]:
        """
        Retrieve the Thing Description for an artifact.

        Args:
            artifact_id: The artifact identifier.

        Returns:
            The Thing Description if found, None otherwise.

        TODO: Implementation steps:
        1. Send GET request to TD endpoint
        2. Parse JSON-LD response
        3. Create ThingDescription object
        4. Extract properties, actions, events
        5. Return ThingDescription
        """
        pass

    @abstractmethod
    async def invoke_action(self, artifact_id: str, action_name: str,
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke an action on an artifact.

        Args:
            artifact_id: The artifact identifier.
            action_name: The action name.
            params: Action parameters.

        Returns:
            Action result.

        TODO: Implementation steps:
        1. Get ThingDescription for artifact
        2. Find action affordance
        3. Validate params against input schema
        4. Send POST request to action endpoint
        5. Parse and return result
        """
        pass

    @abstractmethod
    async def read_property(self, artifact_id: str, property_name: str) -> Any:
        """
        Read a property from an artifact.

        Args:
            artifact_id: The artifact identifier.
            property_name: The property name.

        Returns:
            Property value.

        TODO: Implementation steps:
        1. Get ThingDescription for artifact
        2. Find property affordance
        3. Send GET request to property endpoint
        4. Parse and return value
        """
        pass

    @abstractmethod
    async def write_property(self, artifact_id: str, property_name: str,
                            value: Any) -> bool:
        """
        Write a value to a property.

        Args:
            artifact_id: The artifact identifier.
            property_name: The property name.
            value: The value to write.

        Returns:
            True if successful, False otherwise.

        TODO: Implementation steps:
        1. Get ThingDescription for artifact
        2. Find property affordance
        3. Validate value against schema
        4. Send PUT request to property endpoint
        5. Return success status
        """
        pass

    @abstractmethod
    async def subscribe_to_event(self, artifact_id: str, event_name: str,
                                 callback: callable) -> str:
        """
        Subscribe to an event from an artifact.

        Args:
            artifact_id: The artifact identifier.
            event_name: The event name.
            callback: Callback function to handle events.

        Returns:
            Subscription ID.

        TODO: Implementation steps:
        1. Get ThingDescription for artifact
        2. Find event affordance
        3. Establish WebSocket or SSE connection
        4. Register callback
        5. Return subscription ID
        """
        pass

    @abstractmethod
    async def unsubscribe_from_event(self, subscription_id: str) -> bool:
        """
        Unsubscribe from an event.

        Args:
            subscription_id: The subscription ID.

        Returns:
            True if successful, False otherwise.

        TODO: Implementation steps:
        1. Find subscription
        2. Close connection
        3. Remove callback
        4. Return success status
        """
        pass

    @abstractmethod
    async def subscribe_to_workspace_changes(self, workspace_id: str,
                                            callback: callable) -> str:
        """
        Subscribe to all changes in a workspace.

        Args:
            workspace_id: The workspace identifier.
            callback: Callback function to handle change events.

        Returns:
            Subscription ID.

        TODO: Implementation steps:
        1. Establish WebSocket connection to workspace
        2. Subscribe to change events (add/remove artifacts, etc.)
        3. Register callback
        4. Return subscription ID
        """
        pass

    @abstractmethod
    async def crawl_environment(self, root_workspace_id: str,
                               depth: int = -1) -> Dict[str, Any]:
        """
        Crawl the entire environment starting from root workspace.

        Args:
            root_workspace_id: The root workspace to start from.
            depth: Maximum depth to crawl (-1 for unlimited).

        Returns:
            Complete environment structure.

        TODO: Implementation steps:
        1. Start from root workspace
        2. Recursively traverse sub-workspaces
        3. Collect all artifacts and their TDs
        4. Build complete environment map
        5. Return structured representation
        """
        pass
