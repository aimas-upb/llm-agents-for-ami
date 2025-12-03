"""Synthetic data generator for smart home test scenarios.

This module generates clean test data for benchmarking the RD5 workflow.
No voice assistant prefixes (Alexa, OK Google) - just pure commands.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum


class DeviceType(Enum):
    """Supported device types in the smart home."""
    LIGHTS = "lights"
    THERMOSTAT = "thermostat"
    BLINDS = "blinds"
    SPEAKER = "speaker"
    TV = "tv"
    LOCK = "lock"
    FAN = "fan"
    CAMERA = "camera"


class ActionVerb(Enum):
    """Action verbs for smart home commands."""
    TURN_ON = "turn on"
    TURN_OFF = "turn off"
    SET = "set"
    ADJUST = "adjust"
    DIM = "dim"
    BRIGHTEN = "brighten"
    OPEN = "open"
    CLOSE = "close"
    LOCK = "lock"
    UNLOCK = "unlock"
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    INCREASE = "increase"
    DECREASE = "decrease"


@dataclass
class TestScenario:
    """A test scenario for smart home plan generation."""

    # Input
    user_request: str

    # Expected intent extraction results
    expected_action_verb: str
    expected_target_device: str
    expected_location: Optional[str] = None
    expected_parameters: Dict[str, Any] = field(default_factory=dict)

    # Expected workflow behavior
    should_succeed: bool = True
    expected_affordance_count: int = 1

    # Metadata
    category: str = "general"
    difficulty: str = "easy"
    scenario_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Location options
LOCATIONS = [
    "living room",
    "bedroom",
    "kitchen",
    "bathroom",
    "garage",
    "office",
    "hallway",
    "basement",
    "backyard",
    "front door",
]

# Device-specific actions and parameters
DEVICE_ACTIONS = {
    DeviceType.LIGHTS: {
        "actions": [
            ActionVerb.TURN_ON,
            ActionVerb.TURN_OFF,
            ActionVerb.DIM,
            ActionVerb.BRIGHTEN,
            ActionVerb.SET,
        ],
        "parameters": {
            "brightness": list(range(10, 101, 10)),  # 10-100%
            "color": ["warm", "cool", "daylight", "red", "blue", "green"],
        },
    },
    DeviceType.THERMOSTAT: {
        "actions": [
            ActionVerb.SET,
            ActionVerb.INCREASE,
            ActionVerb.DECREASE,
            ActionVerb.TURN_ON,
            ActionVerb.TURN_OFF,
        ],
        "parameters": {
            "temperature": list(range(60, 81)),  # 60-80°F
            "mode": ["heat", "cool", "auto", "off"],
        },
    },
    DeviceType.BLINDS: {
        "actions": [
            ActionVerb.OPEN,
            ActionVerb.CLOSE,
            ActionVerb.SET,
            ActionVerb.ADJUST,
        ],
        "parameters": {
            "position": list(range(0, 101, 10)),  # 0-100%
        },
    },
    DeviceType.SPEAKER: {
        "actions": [
            ActionVerb.PLAY,
            ActionVerb.PAUSE,
            ActionVerb.STOP,
            ActionVerb.SET,
            ActionVerb.INCREASE,
            ActionVerb.DECREASE,
        ],
        "parameters": {
            "volume": list(range(0, 101, 10)),
            "playlist": ["jazz", "classical", "rock", "ambient", "news"],
        },
    },
    DeviceType.TV: {
        "actions": [
            ActionVerb.TURN_ON,
            ActionVerb.TURN_OFF,
            ActionVerb.SET,
        ],
        "parameters": {
            "channel": list(range(1, 100)),
            "input": ["hdmi1", "hdmi2", "streaming", "cable"],
        },
    },
    DeviceType.LOCK: {
        "actions": [
            ActionVerb.LOCK,
            ActionVerb.UNLOCK,
        ],
        "parameters": {},
    },
    DeviceType.FAN: {
        "actions": [
            ActionVerb.TURN_ON,
            ActionVerb.TURN_OFF,
            ActionVerb.SET,
            ActionVerb.INCREASE,
            ActionVerb.DECREASE,
        ],
        "parameters": {
            "speed": ["low", "medium", "high", "auto"],
        },
    },
    DeviceType.CAMERA: {
        "actions": [
            ActionVerb.TURN_ON,
            ActionVerb.TURN_OFF,
        ],
        "parameters": {
            "mode": ["recording", "streaming", "motion-detect"],
        },
    },
}

# Request templates by action type
REQUEST_TEMPLATES = {
    ActionVerb.TURN_ON: [
        "Turn on the {device}",
        "Turn on the {device} in the {location}",
        "Switch on the {location} {device}",
        "Please turn the {device} on",
        "Can you turn on the {device}?",
    ],
    ActionVerb.TURN_OFF: [
        "Turn off the {device}",
        "Turn off the {device} in the {location}",
        "Switch off the {location} {device}",
        "Please turn the {device} off",
        "Shut off the {device}",
    ],
    ActionVerb.SET: [
        "Set the {device} to {value}",
        "Set the {location} {device} to {value}",
        "Change the {device} to {value}",
        "Put the {device} at {value}",
    ],
    ActionVerb.DIM: [
        "Dim the {device}",
        "Dim the {device} to {value}%",
        "Lower the {device} brightness",
        "Make the {location} {device} dimmer",
    ],
    ActionVerb.BRIGHTEN: [
        "Brighten the {device}",
        "Increase the {device} brightness",
        "Make the {location} {device} brighter",
    ],
    ActionVerb.OPEN: [
        "Open the {device}",
        "Open the {location} {device}",
        "Raise the {device}",
    ],
    ActionVerb.CLOSE: [
        "Close the {device}",
        "Close the {location} {device}",
        "Lower the {device}",
        "Shut the {device}",
    ],
    ActionVerb.LOCK: [
        "Lock the {device}",
        "Lock the {location} {device}",
        "Secure the {device}",
    ],
    ActionVerb.UNLOCK: [
        "Unlock the {device}",
        "Unlock the {location} {device}",
    ],
    ActionVerb.PLAY: [
        "Play music on the {device}",
        "Start playing on the {location} {device}",
        "Play {value} on the {device}",
    ],
    ActionVerb.PAUSE: [
        "Pause the {device}",
        "Pause the {location} {device}",
    ],
    ActionVerb.STOP: [
        "Stop the {device}",
        "Stop the {location} {device}",
    ],
    ActionVerb.INCREASE: [
        "Increase the {device}",
        "Turn up the {device}",
        "Raise the {device} {param}",
    ],
    ActionVerb.DECREASE: [
        "Decrease the {device}",
        "Turn down the {device}",
        "Lower the {device} {param}",
    ],
    ActionVerb.ADJUST: [
        "Adjust the {device}",
        "Adjust the {location} {device} to {value}",
    ],
}


def generate_smart_home_request(
    device_type: Optional[DeviceType] = None,
    action: Optional[ActionVerb] = None,
    location: Optional[str] = None,
    include_parameter: bool = False,
) -> TestScenario:
    """Generate a single smart home request scenario.

    Args:
        device_type: Specific device type or random.
        action: Specific action or random for device.
        location: Specific location or random/none.
        include_parameter: Whether to include a parameter value.

    Returns:
        TestScenario with generated request.
    """
    # Select device type
    if device_type is None:
        device_type = random.choice(list(DeviceType))

    device_config = DEVICE_ACTIONS[device_type]

    # Select action (must be valid for device)
    if action is None or action not in device_config["actions"]:
        action = random.choice(device_config["actions"])

    # Select location (optional)
    use_location = location is not None or random.random() > 0.5
    if use_location and location is None:
        location = random.choice(LOCATIONS)

    # Select parameter value if applicable
    param_value = None
    param_name = None
    if include_parameter and device_config["parameters"]:
        param_name = random.choice(list(device_config["parameters"].keys()))
        param_value = random.choice(device_config["parameters"][param_name])

    # Build request from template
    templates = REQUEST_TEMPLATES.get(action, ["{action} the {device}"])
    template = random.choice(templates)

    # Format template
    request = template.format(
        device=device_type.value,
        location=location or "",
        action=action.value,
        value=param_value if param_value else "",
        param=param_name or "",
    )

    # Clean up extra spaces
    request = " ".join(request.split())

    # Build expected parameters
    expected_params = {}
    if param_value is not None and param_name:
        expected_params[param_name] = param_value

    # Create scenario
    scenario = TestScenario(
        user_request=request,
        expected_action_verb=action.value,
        expected_target_device=device_type.value,
        expected_location=location,
        expected_parameters=expected_params,
        should_succeed=True,
        expected_affordance_count=1,
        category=device_type.value,
        difficulty="easy" if not include_parameter else "medium",
        scenario_id=f"{device_type.value}-{action.value}-{random.randint(1000, 9999)}",
    )

    return scenario


def generate_complex_scenario() -> TestScenario:
    """Generate a more complex multi-step scenario."""
    complex_templates = [
        {
            "request": "Turn on the living room lights and set them to 50%",
            "action_verb": "turn on",
            "target_device": "lights",
            "location": "living room",
            "parameters": {"brightness": 50},
            "difficulty": "hard",
        },
        {
            "request": "When I get home, turn on the lights and set the thermostat to 72",
            "action_verb": "turn on",
            "target_device": "lights",
            "location": None,
            "parameters": {"temperature": 72},
            "difficulty": "hard",
        },
        {
            "request": "Dim the bedroom lights and close the blinds",
            "action_verb": "dim",
            "target_device": "lights",
            "location": "bedroom",
            "parameters": {},
            "difficulty": "hard",
        },
        {
            "request": "Set a movie scene: dim lights to 20%, close blinds, and turn on the TV",
            "action_verb": "set",
            "target_device": "lights",
            "location": None,
            "parameters": {"brightness": 20},
            "difficulty": "hard",
        },
        {
            "request": "Good night mode: turn off all lights and lock the front door",
            "action_verb": "turn off",
            "target_device": "lights",
            "location": None,
            "parameters": {},
            "difficulty": "hard",
        },
    ]

    template = random.choice(complex_templates)

    return TestScenario(
        user_request=template["request"],
        expected_action_verb=template["action_verb"],
        expected_target_device=template["target_device"],
        expected_location=template["location"],
        expected_parameters=template["parameters"],
        should_succeed=True,
        expected_affordance_count=2,  # Complex scenarios involve multiple affordances
        category="complex",
        difficulty=template["difficulty"],
        scenario_id=f"complex-{random.randint(1000, 9999)}",
    )


def generate_edge_case_scenario() -> TestScenario:
    """Generate edge case scenarios that might fail or need special handling."""
    edge_cases = [
        {
            "request": "",  # Empty request
            "action_verb": "unknown",
            "target_device": "unknown",
            "should_succeed": False,
            "category": "edge_case",
            "difficulty": "edge",
        },
        {
            "request": "Do something",  # Vague request
            "action_verb": "unknown",
            "target_device": "unknown",
            "should_succeed": False,
            "category": "edge_case",
            "difficulty": "edge",
        },
        {
            "request": "Turn on the quantum flux capacitor",  # Non-existent device
            "action_verb": "turn on",
            "target_device": "unknown",
            "should_succeed": False,
            "category": "edge_case",
            "difficulty": "edge",
        },
        {
            "request": "Set the temperature to -100 degrees",  # Invalid parameter
            "action_verb": "set",
            "target_device": "thermostat",
            "should_succeed": False,
            "category": "edge_case",
            "difficulty": "edge",
        },
        {
            "request": "Alexa, turn on the lights",  # Voice prefix (should be cleaned)
            "action_verb": "turn on",
            "target_device": "lights",
            "should_succeed": True,  # Should still work after cleaning
            "category": "voice_prefix",
            "difficulty": "medium",
        },
    ]

    template = random.choice(edge_cases)

    return TestScenario(
        user_request=template["request"],
        expected_action_verb=template["action_verb"],
        expected_target_device=template["target_device"],
        expected_location=None,
        expected_parameters={},
        should_succeed=template["should_succeed"],
        expected_affordance_count=0 if not template["should_succeed"] else 1,
        category=template["category"],
        difficulty=template["difficulty"],
        scenario_id=f"edge-{random.randint(1000, 9999)}",
    )


def generate_test_scenarios(
    count: int = 100,
    include_complex: bool = True,
    include_edge_cases: bool = True,
    seed: Optional[int] = None,
) -> List[TestScenario]:
    """Generate a set of test scenarios.

    Args:
        count: Total number of scenarios to generate.
        include_complex: Whether to include complex multi-device scenarios.
        include_edge_cases: Whether to include edge cases.
        seed: Random seed for reproducibility.

    Returns:
        List of TestScenario objects.
    """
    if seed is not None:
        random.seed(seed)

    scenarios = []

    # Distribution of scenario types
    simple_count = int(count * 0.6)  # 60% simple
    with_params_count = int(count * 0.2)  # 20% with parameters
    complex_count = int(count * 0.1) if include_complex else 0  # 10% complex
    edge_count = int(count * 0.1) if include_edge_cases else 0  # 10% edge cases

    # Adjust for rounding
    remaining = count - simple_count - with_params_count - complex_count - edge_count
    simple_count += remaining

    # Generate simple scenarios
    for _ in range(simple_count):
        scenarios.append(generate_smart_home_request(include_parameter=False))

    # Generate scenarios with parameters
    for _ in range(with_params_count):
        scenarios.append(generate_smart_home_request(include_parameter=True))

    # Generate complex scenarios
    for _ in range(complex_count):
        scenarios.append(generate_complex_scenario())

    # Generate edge cases
    for _ in range(edge_count):
        scenarios.append(generate_edge_case_scenario())

    # Shuffle to mix types
    random.shuffle(scenarios)

    return scenarios


def save_scenarios_to_json(
    scenarios: List[TestScenario],
    filepath: str,
) -> None:
    """Save scenarios to a JSON file.

    Args:
        scenarios: List of test scenarios.
        filepath: Output file path.
    """
    data = [s.to_dict() for s in scenarios]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_scenarios_from_json(filepath: str) -> List[TestScenario]:
    """Load scenarios from a JSON file.

    Args:
        filepath: Input file path.

    Returns:
        List of TestScenario objects.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return [TestScenario(**item) for item in data]


if __name__ == "__main__":
    # Generate sample scenarios
    scenarios = generate_test_scenarios(count=20, seed=42)

    print("Generated Test Scenarios:")
    print("=" * 60)

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. [{scenario.category}] {scenario.user_request}")
        print(f"   Action: {scenario.expected_action_verb}")
        print(f"   Device: {scenario.expected_target_device}")
        if scenario.expected_location:
            print(f"   Location: {scenario.expected_location}")
        if scenario.expected_parameters:
            print(f"   Params: {scenario.expected_parameters}")
        print(f"   Should succeed: {scenario.should_succeed}")
