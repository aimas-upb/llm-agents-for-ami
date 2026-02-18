"""
Shared test fixtures for BT planning integration tests.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Sample Data Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_action_spec():
    """A valid single-action BT JSON IR spec."""
    return {
        "name": "TurnOnLight",
        "type": "action",
        "action_url": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
        "parameters": {},
    }


@pytest.fixture
def sample_condition_spec():
    """A valid condition BT JSON IR spec."""
    return {
        "name": "IsLightOn",
        "type": "condition",
        "property_url": "http://localhost:8080/workspaces/lab308/artifacts/light308/properties/state",
        "expected_value": "on",
    }


@pytest.fixture
def sample_sequence_spec():
    """A valid sequence BT JSON IR spec with multiple actions."""
    return {
        "name": "IncreaseLightSequence",
        "type": "sequence",
        "children": [
            {
                "name": "TurnOnLight",
                "type": "action",
                "action_url": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
                "parameters": {},
            },
            {
                "name": "SetBrightness100",
                "type": "action",
                "action_url": "http://localhost:8080/workspaces/lab308/artifacts/light308/set_brightness",
                "parameters": {"brightness": 100},
            },
            {
                "name": "OpenBlinds",
                "type": "action",
                "action_url": "http://localhost:8080/workspaces/lab308/artifacts/blinds308/set_position",
                "parameters": {"position": 100},
            },
        ],
    }


@pytest.fixture
def sample_selector_spec():
    """A valid selector BT JSON IR spec (idempotent pattern)."""
    return {
        "name": "EnsureLightOn",
        "type": "selector",
        "children": [
            {
                "name": "IsLightOn",
                "type": "condition",
                "property_url": "http://localhost:8080/workspaces/lab308/artifacts/light308/properties/state",
                "expected_value": "on",
            },
            {
                "name": "TurnOnLight",
                "type": "action",
                "action_url": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
                "parameters": {},
            },
        ],
    }


@pytest.fixture
def sample_parallel_spec():
    """A valid parallel BT JSON IR spec."""
    return {
        "name": "ParallelActions",
        "type": "parallel",
        "policy": "success_on_all",
        "children": [
            {
                "name": "TurnOnLight",
                "type": "action",
                "action_url": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
            },
            {
                "name": "OpenBlinds",
                "type": "action",
                "action_url": "http://localhost:8080/workspaces/lab308/artifacts/blinds308/set_position",
                "parameters": {"position": 100},
            },
        ],
    }


@pytest.fixture
def sample_nested_spec():
    """A deeply nested BT JSON IR spec."""
    return {
        "name": "Root",
        "type": "sequence",
        "children": [
            {
                "name": "HandleLight",
                "type": "selector",
                "children": [
                    {
                        "name": "IsLightOn",
                        "type": "condition",
                        "property_url": "http://localhost:8080/artifacts/light/properties/state",
                        "expected_value": "on",
                    },
                    {
                        "name": "TurnOnLight",
                        "type": "action",
                        "action_url": "http://localhost:8080/artifacts/light/turn_on",
                    },
                ],
            },
            {
                "name": "SetBrightness",
                "type": "action",
                "action_url": "http://localhost:8080/artifacts/light/set_brightness",
                "parameters": {"brightness": 100},
            },
        ],
    }


@pytest.fixture
def sample_affordances():
    """Standard affordance list for Lab308."""
    return [
        {
            "action_name": "turn_on",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308",
            "method": "POST",
            "input_schema": None,
        },
        {
            "action_name": "turn_off",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_off",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308",
            "method": "POST",
            "input_schema": None,
        },
        {
            "action_name": "set_brightness",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/set_brightness",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308",
            "method": "POST",
            "input_schema": {"type": "object", "properties": {"brightness": {"type": "integer", "minimum": 0, "maximum": 100}}},
        },
        {
            "action_name": "set_position",
            "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/blinds308/set_position",
            "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/blinds308",
            "method": "POST",
            "input_schema": {"type": "object", "properties": {"position": {"type": "integer", "minimum": 0, "maximum": 100}}},
        },
    ]


@pytest.fixture
def sample_signifier_matches():
    """Pre-built signifier match results."""
    return {
        "increase light level": {
            "matches": [
                {
                    "signifier_id": "sig-001",
                    "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/set_brightness",
                    "payload_hint": {"brightness": 100},
                    "intent_similarity": 0.92,
                    "source": "lab308",
                },
            ],
            "final_matches": ["sig-001"],
        },
    }


@pytest.fixture
def sample_multi_action_signifier_matches():
    """Signifier matches where one intent maps to multiple actions (multi-action reuse)."""
    return {
        "increase light level": {
            "matches": [
                {
                    "signifier_id": "sig-001",
                    "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/turn_on",
                    "payload_hint": {},
                    "intent_similarity": 0.92,
                    "source": "lab308",
                },
                {
                    "signifier_id": "sig-002",
                    "affordance_uri": "http://localhost:8080/workspaces/lab308/artifacts/light308/set_brightness",
                    "payload_hint": {"brightness": 100},
                    "intent_similarity": 0.88,
                    "source": "lab308",
                },
            ],
            "final_matches": ["sig-001", "sig-002"],
        },
    }


@pytest.fixture
def sample_state():
    """Sample environment state."""
    return {
        "http://localhost:8080/workspaces/lab308/artifacts/light308": {
            "state": "off",
            "brightness": 0,
        },
        "http://localhost:8080/workspaces/lab308/artifacts/blinds308": {
            "position": 0,
        },
    }


# ── Mock Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_http_response_success():
    """Mock a successful HTTP response."""
    from ami_agents.bt_planning.nodes.http_client import HTTPResponse

    return HTTPResponse(
        status_code=200,
        body={"status": "success", "message": "Action completed"},
        headers={"content-type": "application/json"},
        url="http://localhost:8080/test",
        elapsed_time=0.05,
    )


@pytest.fixture
def mock_http_response_property():
    """Mock a property read HTTP response."""
    from ami_agents.bt_planning.nodes.http_client import HTTPResponse

    return HTTPResponse(
        status_code=200,
        body="on",
        headers={"content-type": "application/json"},
        url="http://localhost:8080/test/properties/state",
        elapsed_time=0.02,
    )
