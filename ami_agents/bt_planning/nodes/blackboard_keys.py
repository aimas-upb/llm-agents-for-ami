"""
Blackboard Keys for Behavior Tree Affordance Nodes

Defines standardized keys for sharing data between nodes via the py-trees blackboard.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlackboardKeys:
    """
    Standardized blackboard key definitions for affordance nodes.
    
    The blackboard is used to share state between behavior tree nodes.
    These keys provide a consistent interface for accessing:
    - Last action results
    - Property values
    - Error information
    - Request/response metadata
    
    Usage:
        # Register keys on blackboard
        blackboard = py_trees.blackboard.Client(name="MyNode")
        blackboard.register_key(key=BlackboardKeys.LAST_ACTION_RESULT, access=py_trees.common.Access.WRITE)
        
        # Write data
        blackboard.set(BlackboardKeys.LAST_ACTION_RESULT, {"status": "success"})
    """
    
    # Action-related keys
    LAST_ACTION_RESULT: str = "affordance/last_action_result"
    LAST_ACTION_URL: str = "affordance/last_action_url"
    LAST_ACTION_STATUS_CODE: str = "affordance/last_action_status_code"
    LAST_ACTION_ERROR: str = "affordance/last_action_error"
    
    # Property-related keys
    LAST_PROPERTY_VALUE: str = "affordance/last_property_value"
    LAST_PROPERTY_URL: str = "affordance/last_property_url"
    LAST_PROPERTY_STATUS_CODE: str = "affordance/last_property_status_code"
    LAST_PROPERTY_ERROR: str = "affordance/last_property_error"
    
    # General metadata
    LAST_REQUEST_TIMESTAMP: str = "affordance/last_request_timestamp"
    LAST_RESPONSE_HEADERS: str = "affordance/last_response_headers"
    
    @classmethod
    def property_value_key(cls, property_name: str) -> str:
        """
        Generate a namespaced key for a specific property value.
        
        Args:
            property_name: The name of the property (e.g., "state", "brightness")
            
        Returns:
            A namespaced blackboard key for the property value
            
        Example:
            key = BlackboardKeys.property_value_key("brightness")
            # Returns: "affordance/properties/brightness"
        """
        return f"affordance/properties/{property_name}"
    
    @classmethod
    def artifact_property_key(cls, artifact_id: str, property_name: str) -> str:
        """
        Generate a fully namespaced key for an artifact's property.
        
        Args:
            artifact_id: Unique identifier for the artifact
            property_name: The name of the property
            
        Returns:
            A fully namespaced blackboard key
            
        Example:
            key = BlackboardKeys.artifact_property_key("balconyLight", "state")
            # Returns: "affordance/artifacts/balconyLight/state"
        """
        return f"affordance/artifacts/{artifact_id}/{property_name}"
    
    @classmethod
    def action_result_key(cls, action_name: str) -> str:
        """
        Generate a namespaced key for a specific action's result.
        
        Args:
            action_name: The name of the action (e.g., "turnOn", "setColor")
            
        Returns:
            A namespaced blackboard key for the action result
            
        Example:
            key = BlackboardKeys.action_result_key("turnOn")
            # Returns: "affordance/actions/turnOn/result"
        """
        return f"affordance/actions/{action_name}/result"
