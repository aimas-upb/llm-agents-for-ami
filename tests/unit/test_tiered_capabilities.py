"""
Unit tests for tiered capabilities formatting (summary vs detailed RDF).

Tests the new format_capabilities_summary_hierarchical() and
format_capabilities_detailed_rdf() functions without a live simulator.
"""

import json
from unittest.mock import Mock, MagicMock

import pytest

from ami_agents.agents.env_explorer.utils.data_formatting import (
    format_capabilities_summary_hierarchical,
    format_capabilities_detailed_rdf,
)
from ami_agents.shared.models.environment import (
    Workspace,
    Artifact,
    ThingDescription,
    Affordance,
    AffordanceForm,
    AffordanceType,
    ArtifactCategory,
    WorkspaceCategory,
)


@pytest.fixture
def mock_agent():
    """Create a mock agent with minimal environment setup."""
    agent = Mock()
    agent.discovery_complete = False
    agent.environment_map = {}
    agent.artifacts = {}
    agent.integration_engine = Mock()
    return agent


@pytest.fixture
def agent_with_minimal_env(mock_agent):
    """Create a mock agent with a minimal environment."""
    # Create a root workspace
    root_ws = Workspace(
        workspace_id="lab308",
        workspace_type=WorkspaceCategory.AREA,
        name="Lab 308",
        parent_workspace_id=None,
        rdf='@prefix td: <https://www.w3.org/2019/wot/td#> .\n<http://example.org/ws/lab308> a td:Thing .',
        artifacts=["light308"],
        sub_workspaces=[],
        metadata={},
    )

    # Create an artifact
    light_td = ThingDescription(
        id="light308",
        title="Lab 308 Light",
        description="Overhead light",
        rdf='@prefix td: <https://www.w3.org/2019/wot/td#> .\n<http://example.org/light308> a td:Thing .',
        properties=[],
        actions=[],
        events=[],
        metadata={},
    )
    light_artifact = Artifact(
        artifact_id="light308",
        artifact_type=ArtifactCategory.PHYSICAL_DEVICE,
        name="Light 308",
        workspace_id="lab308",
        thing_description=light_td,
        current_state={},
        metadata={},
    )

    # Setup agent
    mock_agent.discovery_complete = True
    mock_agent.environment_map = {"lab308": root_ws}
    mock_agent.artifacts = {"light308": light_artifact}

    # Mock affordances for light308
    turn_on_affordance = Affordance(
        affordance_id="aff_light308_turn_on",
        affordance_type=AffordanceType.ACTION,
        name="turn_on",
        description="Turn on the light",
        artifact_id="light308",
        rdf="@prefix td: <https://www.w3.org/2019/wot/td#> .",
        form=AffordanceForm(href="http://example.org/light308/turn_on", method="POST"),
        semantic_types=[],
        input_schema={"type": "https://www.w3.org/2019/wot/json-schema#ObjectSchema", "properties": {}},
        output_schema=None,
        metadata={},
    )

    brightness_affordance = Affordance(
        affordance_id="aff_light308_brightness",
        affordance_type=AffordanceType.PROPERTY,
        name="brightness",
        description="Light brightness level",
        artifact_id="light308",
        rdf="@prefix td: <https://www.w3.org/2019/wot/td#> .",
        form=AffordanceForm(href="http://example.org/light308/brightness", method="GET"),
        semantic_types=[],
        input_schema=None,
        output_schema={
            "type": "https://www.w3.org/2019/wot/json-schema#ObjectSchema",
            "properties": {"level": {"type": "integer"}, "unit": {"type": "string"}},
        },
        metadata={},
    )

    mock_agent.integration_engine.get_affordances_for_artifact.return_value = [
        turn_on_affordance,
        brightness_affordance,
    ]

    return mock_agent


class TestCapabilitiesSummaryHierarchical:
    """Tests for format_capabilities_summary_hierarchical()."""

    def test_not_discovery_complete(self, mock_agent):
        """Should return empty structure when discovery is incomplete."""
        result = format_capabilities_summary_hierarchical(mock_agent)
        assert result["type"] == "environment"
        assert result["discovery_complete"] is False
        assert result["workspaces"] == []

    def test_discovery_complete_minimal(self, agent_with_minimal_env):
        """Should return hierarchical structure with one workspace and one artifact."""
        result = format_capabilities_summary_hierarchical(agent_with_minimal_env)
        assert result["type"] == "environment"
        assert result["discovery_complete"] is True
        assert len(result["workspaces"]) == 1

        ws = result["workspaces"][0]
        assert ws["type"] == "workspace"
        assert ws["id"] == "lab308"
        assert ws["name"] == "Lab 308"
        assert len(ws["artifacts"]) == 1

    def test_artifact_affordances_in_summary(self, agent_with_minimal_env):
        """Should include both ACTION and PROPERTY affordances with parameters."""
        result = format_capabilities_summary_hierarchical(agent_with_minimal_env)
        artifact = result["workspaces"][0]["artifacts"][0]

        assert artifact["type"] == "artifact"
        assert artifact["id"] == "light308"
        assert len(artifact["affordances"]) == 2

        # Check action affordance
        action_aff = next((a for a in artifact["affordances"] if a["type"] == "action_affordance"), None)
        assert action_aff is not None
        assert action_aff["name"] == "turn_on"
        assert action_aff["parameters"] == []

        # Check property affordance
        property_aff = next((a for a in artifact["affordances"] if a["type"] == "property_affordance"), None)
        assert property_aff is not None
        assert property_aff["name"] == "brightness"
        assert "level" in property_aff["parameters"]
        assert "unit" in property_aff["parameters"]

    def test_synthetic_descriptions_filtered(self, agent_with_minimal_env):
        """Should filter out synthetic descriptions like 'Action affordance: ...'."""
        agent = agent_with_minimal_env
        # Replace affordances with synthetic descriptions
        affordances = agent.integration_engine.get_affordances_for_artifact.return_value
        for aff in affordances:
            if aff.affordance_type == AffordanceType.ACTION:
                aff.description = "Action affordance: turnOn"
            else:
                aff.description = "Property affordance: brightness"

        result = format_capabilities_summary_hierarchical(agent)
        artifact = result["workspaces"][0]["artifacts"][0]

        for aff in artifact["affordances"]:
            assert aff["description"] == ""

    def test_hierarchical_sub_workspaces(self, mock_agent):
        """Should handle nested sub-workspaces recursively."""
        # Create root workspace
        root = Workspace(
            workspace_id="home",
            workspace_type=WorkspaceCategory.ROOT,
            name="Home",
            parent_workspace_id=None,
            rdf="",
            artifacts=[],
            sub_workspaces=["lab308"],
            metadata={},
        )

        # Create sub-workspace
        sub = Workspace(
            workspace_id="lab308",
            workspace_type=WorkspaceCategory.AREA,
            name="Lab 308",
            parent_workspace_id="home",
            rdf="",
            artifacts=[],
            sub_workspaces=[],
            metadata={},
        )

        mock_agent.discovery_complete = True
        mock_agent.environment_map = {"home": root, "lab308": sub}
        mock_agent.artifacts = {}
        mock_agent.integration_engine.get_affordances_for_artifact.return_value = []

        result = format_capabilities_summary_hierarchical(mock_agent)
        assert len(result["workspaces"]) == 1
        root_node = result["workspaces"][0]
        assert root_node["id"] == "home"
        assert len(root_node["sub_workspaces"]) == 1
        assert root_node["sub_workspaces"][0]["id"] == "lab308"


class TestCapabilitiesDetailedRDF:
    """Tests for format_capabilities_detailed_rdf()."""

    def test_not_discovery_complete(self, mock_agent):
        """Should return empty string when discovery is incomplete."""
        result = format_capabilities_detailed_rdf(mock_agent)
        assert result == ""

    def test_discovery_complete_returns_turtle(self, agent_with_minimal_env):
        """Should return Turtle string with canonicalized prefixes."""
        result = format_capabilities_detailed_rdf(agent_with_minimal_env)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should include canonical prefixes
        assert "@prefix hmas:" in result
        assert "@prefix td:" in result
        assert "@prefix hctl:" in result
        assert "@prefix jsonschema:" in result

    def test_no_duplicate_prefixes(self, agent_with_minimal_env):
        """Should not have duplicate @prefix declarations."""
        result = format_capabilities_detailed_rdf(agent_with_minimal_env)
        lines = result.split("\n")
        prefix_lines = [l for l in lines if l.strip().startswith("@prefix")]
        # Count unique prefixes
        unique_prefixes = len(set(prefix_lines))
        # All prefixes should be unique (no duplicates)
        assert unique_prefixes == len(prefix_lines)

    def test_includes_workspace_rdf(self, agent_with_minimal_env):
        """Should include workspace RDF in output."""
        result = format_capabilities_detailed_rdf(agent_with_minimal_env)
        # The workspace RDF contains a reference to lab308
        assert "lab308" in result or "Thing" in result

    def test_malformed_rdf_gracefully_skipped(self, agent_with_minimal_env):
        """Should gracefully skip malformed RDF and continue."""
        # Replace workspace RDF with invalid Turtle
        workspace = agent_with_minimal_env.environment_map["lab308"]
        workspace.rdf = "this is not valid turtle @#$%"

        # Should not raise an error
        result = format_capabilities_detailed_rdf(agent_with_minimal_env)
        assert isinstance(result, str)
        # Should still have prefix block and artifact RDF
        assert "@prefix" in result
