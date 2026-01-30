"""
Behavior Tree Affordance Nodes

This module provides py-trees behavior tree node templates for working with
HMAS (Hypermedia Multi-Agent Systems) Thing Description affordances.

Node Types:
- ActionAffordanceNode: Execute action affordances (POST requests)
- PropertyAffordanceNode: Read property affordances (GET requests)
- PropertyConditionNode: Check if property matches expected value
- ComparisonPropertyConditionNode: Compare property values using operators

All nodes follow the py-trees Status convention:
- SUCCESS: Operation completed successfully
- FAILURE: Operation failed or condition not met
- RUNNING: Operation in progress (for async operations)
"""

import py_trees
from py_trees.common import Status
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import logging

from .http_client import HTTPClient, HTTPClientConfig, HTTPError
from .blackboard_keys import BlackboardKeys

logger = logging.getLogger(__name__)


class ComparisonOperator(Enum):
    """
    Comparison operators for property condition checks.
    
    Used by ComparisonPropertyConditionNode to compare property values
    against expected values using different comparison semantics.
    """
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    IN = "in"  # Check if value is in a list/set
    NOT_IN = "not_in"  # Check if value is not in a list/set
    CONTAINS = "contains"  # Check if value contains substring/element
    MATCHES = "matches"  # Regex match (for strings)


@dataclass
class ActionResult:
    """
    Encapsulates the result of an action affordance invocation.
    
    Attributes:
        success: Whether the action completed successfully
        status_code: HTTP status code from the response
        response_body: Parsed response body (JSON or string)
        error_message: Error description if action failed
        elapsed_time: Time taken for the action in seconds
        url: The action URL that was invoked
    """
    success: bool
    status_code: Optional[int] = None
    response_body: Any = None
    error_message: Optional[str] = None
    elapsed_time: float = 0.0
    url: str = ""


@dataclass
class PropertyValue:
    """
    Encapsulates a property affordance value reading.
    
    Attributes:
        success: Whether the property was read successfully
        value: The property value (can be any JSON-compatible type)
        status_code: HTTP status code from the response
        error_message: Error description if read failed
        elapsed_time: Time taken for the read in seconds
        url: The property URL that was queried
    """
    success: bool
    value: Any = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    elapsed_time: float = 0.0
    url: str = ""


class ActionAffordanceNode(py_trees.behaviour.Behaviour):
    """
    Behavior tree node for invoking action affordances.
    
    This node makes an HTTP POST request to the action URL with the provided
    parameters. It can be used for any action affordance defined in a Thing
    Description, such as turnOn, setColor, pickup, stack, etc.
    
    The node follows py-trees conventions:
    - setup(): Initialize HTTP client and validate configuration
    - initialise(): Reset state for new tick cycle
    - update(): Execute the action and return status
    - terminate(): Clean up resources
    
    Example:
        # Simple action without parameters
        turn_on = ActionAffordanceNode(
            name="TurnOnLight",
            action_url="http://localhost:8080/workspaces/home0/balcony/artifacts/balconyLight/turn_on"
        )
        
        # Action with parameters
        set_brightness = ActionAffordanceNode(
            name="SetBrightness",
            action_url="http://localhost:8080/workspaces/home0/corridor/artifacts/corridorLight/set_brightness",
            parameters={"brightness": 75}
        )
        
        # Action with dynamic parameters from blackboard
        dynamic_action = ActionAffordanceNode(
            name="DynamicSetColor",
            action_url="http://localhost:8080/artifacts/light/set_color",
            parameter_keys={"color": "user/selected_color"}
        )
        
        # Blocksworld stack action
        stack_action = ActionAffordanceNode(
            name="StackBlocks",
            action_url="http://localhost:8080/workspaces/blocksworld/artifacts/world-1/stack",
            parameters={"target_block": "a", "to_block": "b"}
        )
    
    Attributes:
        action_url: The HTTP endpoint for the action affordance
        parameters: Static parameters to send with the action
        parameter_keys: Blackboard keys to read dynamic parameters from
        store_result: Whether to store the result on the blackboard
        result_key: Blackboard key for storing the result
    """
    
    def __init__(
        self,
        name: str,
        action_url: str,
        parameters: Optional[Dict[str, Any]] = None,
        parameter_keys: Optional[Dict[str, str]] = None,
        store_result: bool = True,
        result_key: Optional[str] = None,
    ):
        """
        Initialize the action affordance node.
        
        Args:
            name: The name of this behavior tree node
            action_url: HTTP endpoint URL for the action affordance
            parameters: Static parameters to include in the request body
            parameter_keys: Map of parameter names to blackboard keys for dynamic values
            store_result: Whether to store the action result on the blackboard
            result_key: Custom blackboard key for the result (defaults to standard key)
        """
        super().__init__(name)
        
        self.action_url = action_url
        self.parameters = parameters or {}
        self.parameter_keys = parameter_keys or {}
        self.store_result = store_result
        self.result_key = result_key or BlackboardKeys.LAST_ACTION_RESULT

        # HTTP client
        self._http_client: Optional[HTTPClient] = None
        
        # Runtime state
        self._last_result: Optional[ActionResult] = None
        
        # Blackboard setup
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            key=self.result_key,
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=BlackboardKeys.LAST_ACTION_URL,
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=BlackboardKeys.LAST_ACTION_STATUS_CODE,
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=BlackboardKeys.LAST_ACTION_ERROR,
            access=py_trees.common.Access.WRITE
        )
        
        # Register read access for dynamic parameter keys
        for bb_key in self.parameter_keys.values():
            self.blackboard.register_key(
                key=bb_key,
                access=py_trees.common.Access.READ
            )
    
    def setup(self, **kwargs) -> None:
        """
        Setup the node before first tick.
        
        Creates the HTTP client if not provided during initialization.
        """
        if self._http_client is None:
            self._http_client = HTTPClient(config=HTTPClientConfig())
        
        logger.debug(f"[{self.name}] Setup complete, action URL: {self.action_url}")
    
    def initialise(self) -> None:
        """
        Reset state at the start of a new tick cycle.
        """
        self._last_result = None
        logger.debug(f"[{self.name}] Initializing for new tick")
    
    def _build_parameters(self) -> Dict[str, Any]:
        """
        Build the final parameter dictionary by combining static and dynamic params.
        
        Returns:
            Combined parameter dictionary
        """
        params = dict(self.parameters)
        
        # Resolve dynamic parameters from blackboard
        for param_name, bb_key in self.parameter_keys.items():
            try:
                value = self.blackboard.get(bb_key)

                if isinstance(value, PropertyValue):
                    value = value.value  # Extract actual value

                params[param_name] = value
            except KeyError:
                logger.warning(
                    f"[{self.name}] Blackboard key '{bb_key}' not found for parameter '{param_name}'"
                )
        
        return params
    
    def update(self) -> Status:
        """
        Execute the action affordance.
        
        Makes an HTTP POST request to the action URL with the configured parameters.
        
        Returns:
            Status.SUCCESS if the action completed successfully
            Status.FAILURE if the action failed
        """
        params = self._build_parameters()
        
        logger.info(f"[{self.name}] Invoking action: {self.action_url}")
        logger.debug(f"[{self.name}] Parameters: {params}")
        
        try:
            response = self._http_client.post(self.action_url, payload=params)
            
            self._last_result = ActionResult(
                success=response.is_success,
                status_code=response.status_code,
                response_body=response.body,
                elapsed_time=response.elapsed_time,
                url=self.action_url
            )
            
            if response.is_success:
                logger.info(
                    f"[{self.name}] Action succeeded (status: {response.status_code}, "
                    f"time: {response.elapsed_time:.3f}s)"
                )
                
                self._store_result()
                return Status.SUCCESS
            else:
                self._last_result.error_message = f"HTTP {response.status_code}"
                logger.warning(
                    f"[{self.name}] Action failed with status {response.status_code}"
                )
                
                self._store_result()
                return Status.FAILURE

        except HTTPError as e:
            self._last_result = ActionResult(
                success=False,
                status_code=e.status_code,
                response_body=e.response_body,
                error_message=e.message,
                url=self.action_url
            )
            
            logger.error(f"[{self.name}] Action failed: {e.message}")
            
            self._store_result()
            return Status.FAILURE
    
    def _store_result(self) -> None:
        """Store the action result on the blackboard."""
        if self.store_result and self._last_result:
            self.blackboard.set(self.result_key, self._last_result)
            self.blackboard.set(BlackboardKeys.LAST_ACTION_URL, self._last_result.url)
            self.blackboard.set(
                BlackboardKeys.LAST_ACTION_STATUS_CODE,
                self._last_result.status_code
            )
            if self._last_result.error_message:
                self.blackboard.set(
                    BlackboardKeys.LAST_ACTION_ERROR,
                    self._last_result.error_message
                )
    
    def terminate(self, new_status: Status) -> None:
        """
        Cleanup when the node is terminated.
        
        Args:
            new_status: The status that caused termination
        """
        logger.debug(f"[{self.name}] Terminating with status {new_status}")
    
    @property
    def last_result(self) -> Optional[ActionResult]:
        """Get the result of the last action execution."""
        return self._last_result


class PropertyAffordanceNode(py_trees.behaviour.Behaviour):
    """
    Behavior tree node for reading property affordances.
    
    This node makes an HTTP GET request to the property URL and stores
    the result on the blackboard. It can be used to read any property
    defined in a Thing Description, such as state, temperature, brightness, etc.
    
    Example:
        # Read a simple property
        read_state = PropertyAffordanceNode(
            name="ReadLightState",
            property_url="http://localhost:8080/workspaces/home0/balcony/artifacts/balconyLight/properties/state"
        )
        
        # Read with custom result key
        read_temp = PropertyAffordanceNode(
            name="ReadTemperature",
            property_url="http://localhost:8080/artifacts/thermostat/properties/temperature",
            result_key="sensors/current_temperature"
        )
        
        # Read blocksworld state
        read_world = PropertyAffordanceNode(
            name="ReadWorldState",
            property_url="http://localhost:8080/workspaces/blocksworld/artifacts/world-1/properties/state"
        )
    
    Attributes:
        property_url: The HTTP endpoint for the property affordance
        store_result: Whether to store the result on the blackboard
        result_key: Blackboard key for storing the value
        property_name: Name of the property (extracted from URL or specified)
    """
    
    def __init__(
        self,
        name: str,
        property_url: str,
        store_result: bool = True,
        result_key: Optional[str] = None,
        property_name: Optional[str] = None,
    ):
        """
        Initialize the property affordance node.
        
        Args:
            name: The name of this behavior tree node
            property_url: HTTP endpoint URL for the property affordance
            store_result: Whether to store the property value on the blackboard
            result_key: Custom blackboard key for the value (defaults to standard key)
            property_name: Optional name of the property (auto-extracted if not provided)
        """
        super().__init__(name)
        
        self.property_url = property_url
        self.store_result = store_result
        self.result_key = result_key or BlackboardKeys.LAST_PROPERTY_VALUE
        self.property_name = property_name or self._extract_property_name(property_url)

        # HTTP client
        self._http_client: Optional[HTTPClient] = None
        
        # Runtime state
        self._last_value: Optional[PropertyValue] = None
        
        # Blackboard setup
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            key=self.result_key,
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=BlackboardKeys.LAST_PROPERTY_URL,
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=BlackboardKeys.LAST_PROPERTY_STATUS_CODE,
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=BlackboardKeys.LAST_PROPERTY_ERROR,
            access=py_trees.common.Access.WRITE
        )
        
        # Also register a property-specific key
        if self.property_name:
            self.property_key = BlackboardKeys.property_value_key(self.property_name)
            self.blackboard.register_key(
                key=self.property_key,
                access=py_trees.common.Access.WRITE
            )
    
    @staticmethod
    def _extract_property_name(url: str) -> Optional[str]:
        """Extract the property name from the URL."""
        # URLs typically end with /properties/<name>
        if "/properties/" in url:
            return url.split("/properties/")[-1].split("?")[0].split("#")[0]
        return None
    
    def setup(self, **kwargs) -> None:
        """Setup the node before first tick."""
        if self._http_client is None:
            self._http_client = HTTPClient(config=HTTPClientConfig())
        
        logger.debug(f"[{self.name}] Setup complete, property URL: {self.property_url}")
    
    def initialise(self) -> None:
        """Reset state at the start of a new tick cycle."""
        self._last_value = None
        logger.debug(f"[{self.name}] Initializing for new tick")
    
    def update(self) -> Status:
        """
        Read the property affordance.
        
        Makes an HTTP GET request to the property URL and stores the result.
        
        Returns:
            Status.SUCCESS if the property was read successfully
            Status.FAILURE if the read failed
        """
        logger.info(f"[{self.name}] Reading property: {self.property_url}")
        
        try:
            response = self._http_client.get(self.property_url)
            
            self._last_value = PropertyValue(
                success=response.is_success,
                value=response.body,
                status_code=response.status_code,
                elapsed_time=response.elapsed_time,
                url=self.property_url
            )
            
            if response.is_success:
                logger.info(
                    f"[{self.name}] Property read succeeded: {response.body} "
                    f"(time: {response.elapsed_time:.3f}s)"
                )
                
                self._store_result()
                return Status.SUCCESS
            else:
                self._last_value.error_message = f"HTTP {response.status_code}"
                logger.warning(
                    f"[{self.name}] Property read failed with status {response.status_code}"
                )
                
                self._store_result()
                return Status.FAILURE
                
        except HTTPError as e:
            self._last_value = PropertyValue(
                success=False,
                status_code=e.status_code,
                error_message=e.message,
                url=self.property_url
            )
            
            logger.error(f"[{self.name}] Property read failed: {e.message}")
            
            self._store_result()
            return Status.FAILURE
    
    def _store_result(self) -> None:
        """Store the property value on the blackboard."""
        if self.store_result and self._last_value:
            self.blackboard.set(self.result_key, self._last_value)
            self.blackboard.set(BlackboardKeys.LAST_PROPERTY_URL, self._last_value.url)
            self.blackboard.set(
                BlackboardKeys.LAST_PROPERTY_STATUS_CODE,
                self._last_value.status_code
            )
            
            # Store the actual value with property-specific key
            if self.property_name and self._last_value.value is not None:
                self.blackboard.set(self.property_key, self._last_value.value)
            
            if self._last_value.error_message:
                self.blackboard.set(
                    BlackboardKeys.LAST_PROPERTY_ERROR,
                    self._last_value.error_message
                )
    
    def terminate(self, new_status: Status) -> None:
        """Cleanup when the node is terminated."""
        logger.debug(f"[{self.name}] Terminating with status {new_status}")
    
    @property
    def last_value(self) -> Optional[PropertyValue]:
        """Get the last property value reading."""
        return self._last_value


class PropertyConditionNode(py_trees.behaviour.Behaviour):
    """
    Behavior tree condition node for checking property values.
    
    This node reads a property and compares it against an expected value.
    It's useful for implementing guards and preconditions in behavior trees.
    
    The node always returns immediately (never RUNNING), making it suitable
    for use in conditional branches.
    
    Example:
        # Check if light is on
        is_light_on = PropertyConditionNode(
            name="IsLightOn",
            property_url="http://localhost:8080/artifacts/light/properties/state",
            expected_value="on"
        )
        
        # Check with comparison from blackboard value
        brightness_check = PropertyConditionNode(
            name="IsBrightEnough",
            property_url="http://localhost:8080/artifacts/light/properties/brightness",
            expected_value_key="target/brightness"
        )
        
        # Check nested property (e.g., blocksworld hand)
        hand_empty = PropertyConditionNode(
            name="IsHandEmpty",
            property_url="http://localhost:8080/workspaces/blocksworld/artifacts/world-1/properties/state",
            expected_value="empty",
            value_path=["hand"]  # Navigate to state.hand
        )
    
    Attributes:
        property_url: The HTTP endpoint for the property affordance
        expected_value: The value to compare against
        expected_value_key: Blackboard key for dynamic expected value
        value_path: Path to navigate in nested response objects
        negate: If True, succeed when values DON'T match
    """
    
    def __init__(
        self,
        name: str,
        property_url: str,
        expected_value: Any = None,
        expected_value_key: Optional[str] = None,
        value_path: Optional[List[str]] = None,
        negate: bool = False,
    ):
        """
        Initialize the property condition node.
        
        Args:
            name: The name of this behavior tree node
            property_url: HTTP endpoint URL for the property affordance
            expected_value: The value to compare against (static)
            expected_value_key: Blackboard key for dynamic expected value
            value_path: List of keys to navigate nested response objects
            negate: If True, return SUCCESS when values don't match
        """
        super().__init__(name)
        
        self.property_url = property_url
        self.expected_value = expected_value
        self.expected_value_key = expected_value_key
        self.value_path = value_path or []
        self.negate = negate

        # HTTP client
        self._http_client: Optional[HTTPClient] = None
        
        # Runtime state
        self._actual_value: Any = None
        self._comparison_result: Optional[bool] = None
        
        # Blackboard setup
        self.blackboard = self.attach_blackboard_client(name=self.name)
        
        if expected_value_key:
            self.blackboard.register_key(
                key=expected_value_key,
                access=py_trees.common.Access.READ
            )
    
    def setup(self, **kwargs) -> None:
        """Setup the node before first tick."""
        if self._http_client is None:
            self._http_client = HTTPClient(config=HTTPClientConfig())
    
    def initialise(self) -> None:
        """Reset state at the start of a new tick cycle."""
        self._actual_value = None
        self._comparison_result = None
    
    def _get_expected_value(self) -> Any:
        """Get the expected value from static config or blackboard."""
        if self.expected_value_key:
            try:
                return self.blackboard.get(self.expected_value_key)
            except KeyError:
                logger.warning(
                    f"[{self.name}] Expected value key '{self.expected_value_key}' not found"
                )
                return self.expected_value
        return self.expected_value
    
    def _navigate_value(self, value: Any) -> Any:
        """Navigate to a nested value using the value_path."""
        result = value
        for key in self.value_path:
            if isinstance(result, dict):
                result = result.get(key)
            elif isinstance(result, list) and key.isdigit():
                idx = int(key)
                result = result[idx] if 0 <= idx < len(result) else None
            else:
                return None
        return result
    
    def update(self) -> Status:
        """
        Read the property and compare against expected value.
        
        Returns:
            Status.SUCCESS if the comparison matches (or doesn't match if negate=True)
            Status.FAILURE if the comparison fails or property cannot be read
        """
        logger.debug(f"[{self.name}] Checking property: {self.property_url}")
        
        try:
            response = self._http_client.get(self.property_url)
            
            if not response.is_success:
                logger.warning(
                    f"[{self.name}] Failed to read property: HTTP {response.status_code}"
                )
                return Status.FAILURE
            
            # Navigate to the target value
            self._actual_value = self._navigate_value(response.body)
            expected = self._get_expected_value()
            
            # Perform comparison
            self._comparison_result = (self._actual_value == expected)
            
            # Apply negation if configured
            final_result = not self._comparison_result if self.negate else self._comparison_result
            
            logger.debug(
                f"[{self.name}] Comparison: {self._actual_value} == {expected} -> "
                f"{self._comparison_result} (negate={self.negate}, final={final_result})"
            )
            
            return Status.SUCCESS if final_result else Status.FAILURE
            
        except HTTPError as e:
            logger.error(f"[{self.name}] HTTP error: {e.message}")
            return Status.FAILURE
    
    def terminate(self, new_status: Status) -> None:
        """Cleanup when the node is terminated."""
        pass
    
    @property
    def actual_value(self) -> Any:
        """Get the actual value that was read."""
        return self._actual_value
    
    @property
    def comparison_result(self) -> Optional[bool]:
        """Get the raw comparison result (before negation)."""
        return self._comparison_result


class ComparisonPropertyConditionNode(PropertyConditionNode):
    """
    Extended property condition node supporting various comparison operators.
    
    This node extends PropertyConditionNode with support for different
    comparison operators beyond simple equality.
    
    Example:
        # Check if temperature is above threshold
        temp_check = ComparisonPropertyConditionNode(
            name="IsTooHot",
            property_url="http://localhost:8080/artifacts/thermostat/properties/temperature",
            expected_value=25,
            operator=ComparisonOperator.GREATER_THAN
        )
        
        # Check if mode is one of several values
        mode_check = ComparisonPropertyConditionNode(
            name="IsCoolingMode",
            property_url="http://localhost:8080/artifacts/ac/properties/mode",
            expected_value=["cool", "auto"],
            operator=ComparisonOperator.IN
        )
        
        # Check if brightness is in acceptable range
        brightness_min = ComparisonPropertyConditionNode(
            name="BrightnessAboveMin",
            property_url="http://localhost:8080/artifacts/light/properties/brightness",
            expected_value=20,
            operator=ComparisonOperator.GREATER_THAN_OR_EQUAL
        )
    """
    
    def __init__(
        self,
        name: str,
        property_url: str,
        expected_value: Any = None,
        operator: ComparisonOperator = ComparisonOperator.EQUAL,
        expected_value_key: Optional[str] = None,
        value_path: Optional[List[str]] = None,
        negate: bool = False,
    ):
        """
        Initialize the comparison property condition node.
        
        Args:
            name: The name of this behavior tree node
            property_url: HTTP endpoint URL for the property affordance
            expected_value: The value to compare against
            operator: The comparison operator to use
            expected_value_key: Blackboard key for dynamic expected value
            value_path: List of keys to navigate nested response objects
            negate: If True, return SUCCESS when comparison is False
        """
        super().__init__(
            name=name,
            property_url=property_url,
            expected_value=expected_value,
            expected_value_key=expected_value_key,
            value_path=value_path,
            negate=negate,
        )

        self.operator = operator
    
    def _compare(self, actual: Any, expected: Any) -> bool:
        """
        Compare two values using the configured operator.
        
        Args:
            actual: The actual value from the property
            expected: The expected value to compare against
            
        Returns:
            True if the comparison succeeds, False otherwise
        """
        try:
            if self.operator == ComparisonOperator.EQUAL:
                return actual == expected
            elif self.operator == ComparisonOperator.NOT_EQUAL:
                return actual != expected
            elif self.operator == ComparisonOperator.GREATER_THAN:
                return actual > expected
            elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
                return actual >= expected
            elif self.operator == ComparisonOperator.LESS_THAN:
                return actual < expected
            elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
                return actual <= expected
            elif self.operator == ComparisonOperator.IN:
                return actual in expected
            elif self.operator == ComparisonOperator.NOT_IN:
                return actual not in expected
            elif self.operator == ComparisonOperator.CONTAINS:
                if isinstance(actual, str):
                    return expected in actual
                elif isinstance(actual, (list, tuple, set)):
                    return expected in actual
                elif isinstance(actual, dict):
                    return expected in actual
                return False
            elif self.operator == ComparisonOperator.MATCHES:
                import re
                if isinstance(actual, str) and isinstance(expected, str):
                    return bool(re.match(expected, actual))
                return False
            else:
                logger.warning(f"[{self.name}] Unknown operator: {self.operator}")
                return False
        except TypeError as e:
            logger.warning(
                f"[{self.name}] Type error in comparison: {e} "
                f"(actual={type(actual)}, expected={type(expected)})"
            )
            return False
    
    def update(self) -> Status:
        """
        Read the property and compare using the configured operator.
        
        Returns:
            Status.SUCCESS if the comparison matches (or doesn't match if negate=True)
            Status.FAILURE if the comparison fails or property cannot be read
        """
        logger.debug(
            f"[{self.name}] Checking property with operator {self.operator.value}: "
            f"{self.property_url}"
        )
        
        try:
            response = self._http_client.get(self.property_url)
            
            if not response.is_success:
                logger.warning(
                    f"[{self.name}] Failed to read property: HTTP {response.status_code}"
                )
                return Status.FAILURE
            
            # Navigate to the target value
            self._actual_value = self._navigate_value(response.body)
            expected = self._get_expected_value()
            
            # Perform comparison with operator
            self._comparison_result = self._compare(self._actual_value, expected)
            
            # Apply negation if configured
            final_result = not self._comparison_result if self.negate else self._comparison_result
            
            logger.debug(
                f"[{self.name}] Comparison: {self._actual_value} {self.operator.value} "
                f"{expected} -> {self._comparison_result} (negate={self.negate}, final={final_result})"
            )
            
            return Status.SUCCESS if final_result else Status.FAILURE
            
        except HTTPError as e:
            logger.error(f"[{self.name}] HTTP error: {e.message}")
            return Status.FAILURE
