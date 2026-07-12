"""
Unit tests for the EnvExplorer capabilities payload and compact summary.
"""

from types import SimpleNamespace

import pytest

from ami_agents.agents.env_explorer.utils.data_formatting import (
    format_capabilities_payload,
    format_capabilities_summary,
)
from ami_agents.shared.models.environment import AffordanceType


def _affordance(aff_type, name, href, method, description="", input_schema=None, output_schema=None):
    return SimpleNamespace(
        affordance_id=href,
        affordance_type=aff_type,
        name=name,
        description=description,
        artifact_id="http://localhost:8080/workspaces/lab308/artifacts/light308#artifact",
        form=SimpleNamespace(
            href=href,
            method=method,
            content_type="application/json",
            operation_type=None,
            additional_fields={},
        ),
        semantic_types=[],
        input_schema=input_schema,
        output_schema=output_schema,
    )


@pytest.fixture
def agent():
    set_brightness = _affordance(
        AffordanceType.ACTION,
        "setBrightness",
        "http://localhost:8080/workspaces/lab308/artifacts/light308/setBrightness",
        "POST",
        description="Set the light brightness",
        input_schema={
            "type": "object",
            "properties": {"brightness": {"type": "integer"}},
            "required": ["brightness"],
        },
    )
    brightness_prop = _affordance(
        AffordanceType.PROPERTY,
        "brightness",
        "http://localhost:8080/workspaces/lab308/artifacts/light308/properties/brightness",
        "GET",
        description="Property affordance: brightness",
    )
    artifact = SimpleNamespace(
        artifact_id="http://localhost:8080/workspaces/lab308/artifacts/light308#artifact",
        name="light308",
        workspace_id="lab308",
    )
    workspace = SimpleNamespace(
        workspace_id="lab308",
        name="Lab 308",
        workspace_type=SimpleNamespace(value="lab"),
        parent_workspace_id=None,
        sub_workspaces=[],
        artifacts=[artifact.artifact_id],
    )
    engine = SimpleNamespace(
        get_affordances_for_artifact=lambda artifact_id: [set_brightness, brightness_prop]
    )
    return SimpleNamespace(
        discovery_complete=True,
        artifacts={artifact.artifact_id: artifact},
        environment_map={"lab308": workspace},
        integration_engine=engine,
        semantic_capabilities={"td_sosa_supported": True},
    )


class TestCompactSummary:
    def test_lists_names_params_and_descriptions(self, agent):
        summary = format_capabilities_summary(agent)
        assert "Workspaces: lab308 (Lab 308)" in summary
        assert "Artifact: light308" in summary
        assert "setBrightness (params: brightness:integer*)" in summary
        assert "Set the light brightness" in summary
        assert "Properties: brightness" in summary

    def test_omits_urls_and_json_schemas(self, agent):
        summary = format_capabilities_summary(agent)
        assert "/setBrightness" not in summary
        assert '"properties"' not in summary
        # Short artifact id is kept (needed for explicit-intent extraction),
        # but the full URI is not. Name == id here, so it appears once.
        assert "Artifact: light308\n" in summary
        assert "http://localhost:8080" not in summary

    def test_iri_schema_types_are_shortened(self, agent):
        action = agent.integration_engine.get_affordances_for_artifact("x")[0]
        action.input_schema = {
            "type": "object",
            "properties": {
                "brightness_pct": {"type": "https://www.w3.org/2019/wot/json-schema#NumberSchema"}
            },
        }
        summary = format_capabilities_summary(agent)
        assert "brightness_pct:number" in summary
        assert "json-schema#" not in summary

    def test_generic_action_description_is_skipped(self, agent):
        action = agent.integration_engine.get_affordances_for_artifact("x")[0]
        action.description = "Action affordance: setBrightness"
        summary = format_capabilities_summary(agent)
        assert "Action affordance:" not in summary

    def test_common_properties_are_factored_out(self, agent):
        base = agent.artifacts[next(iter(agent.artifacts))]
        other = SimpleNamespace(
            artifact_id="http://localhost:8080/workspaces/lab308/artifacts/light309#artifact",
            name="light309",
            workspace_id="lab308",
        )
        agent.artifacts[other.artifact_id] = other

        def affs_for(artifact_id):
            shared = [
                _affordance(
                    AffordanceType.PROPERTY,
                    name,
                    f"{artifact_id.split('#')[0]}/properties/{name}",
                    "GET",
                )
                for name in ("state", "available")
            ]
            if "light308" in artifact_id:
                shared.append(
                    _affordance(
                        AffordanceType.PROPERTY,
                        "brightness",
                        f"{artifact_id.split('#')[0]}/properties/brightness",
                        "GET",
                    )
                )
            return shared

        agent.integration_engine = SimpleNamespace(get_affordances_for_artifact=affs_for)
        summary = format_capabilities_summary(agent)
        assert "Common properties" in summary
        assert summary.count("state") == 1  # only in the common line
        # Both artifacts are property-only: one line each, extras after ':'
        assert "- light308: brightness" in summary
        assert "- light309" in summary

    def test_discovery_in_progress(self, agent):
        agent.discovery_complete = False
        assert "in progress" in format_capabilities_summary(agent)

    def test_no_artifacts(self, agent):
        agent.artifacts = {}
        assert format_capabilities_summary(agent) == "No artifacts found in the environment."


class TestCapabilitiesPayload:
    def test_includes_action_and_property_affordances(self, agent):
        payload = format_capabilities_payload(agent)
        affs = payload["affordances"]
        types = {a["affordance_type"] for a in affs}
        assert types == {"action", "property"}

        action = next(a for a in affs if a["affordance_type"] == "action")
        assert action["action_name"] == "setBrightness"
        assert action["target"].endswith("/setBrightness")
        assert action["input_schema"]["required"] == ["brightness"]

        prop = next(a for a in affs if a["affordance_type"] == "property")
        assert prop["name"] == "brightness"
        assert prop["method"] == "GET"
        assert prop["target"].endswith("/properties/brightness")

    def test_artifacts_are_slim(self, agent):
        payload = format_capabilities_payload(agent)
        artifact = payload["artifacts"][0]
        assert artifact["actions"] == ["setBrightness"]
        assert artifact["properties"] == ["brightness"]

    def test_summary_is_compact(self, agent):
        payload = format_capabilities_payload(agent)
        assert payload["summary"] == format_capabilities_summary(agent)

    def test_semantic_capabilities_preserved(self, agent):
        payload = format_capabilities_payload(agent)
        assert payload["semantic_capabilities"] == {"td_sosa_supported": True}

    def test_workspaces_preserved(self, agent):
        payload = format_capabilities_payload(agent)
        assert payload["workspaces"][0]["workspace_id"] == "lab308"
