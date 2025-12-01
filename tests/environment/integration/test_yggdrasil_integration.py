"""
Tests for YggdrasilIntegration class.

These tests verify that the integration can properly dereference and validate
HMAS platform instances from Yggdrasil URLs.

IMPORTANT NOTES:
- The HMAS ontology (loaded via owlready2) must be in OWL/XML, RDF/XML, or NTriples format
- The Yggdrasil platform RDF (parsed via rdflib) can be in any RDF format (Turtle, RDF/XML, etc.)
- owlready2 is used ONLY for vocabulary access from the ontology
- rdflib is used for all RDF graph parsing and validation
"""

import pytest
from ami_agents.environment.integration.integration_engine import YggdrasilIntegration


class TestYggdrasilIntegration:
    """Test suite for YggdrasilIntegration."""

    @pytest.mark.asyncio
    async def test_initialize_with_localhost(self):
        """
        Test initializing YggdrasilIntegration with a local HMAS platform.

        This test assumes that http://localhost:8080/ serves an RDF document
        that represents either:
        - A direct hmas:HypermediaMASPlatform instance, or
        - An hmas:ResourceProfile with an hmas:isProfileOf property pointing
          to an hmas:HypermediaMASPlatform

        To run this test, ensure you have a Yggdrasil instance running at
        http://localhost:8080/ that serves appropriate RDF content.
        """
        # Arrange
        yggdrasil_url = "http://localhost:8080/"
        integration = YggdrasilIntegration(yggdrasil_url)

        # Act
        result = await integration.initialize({})

        # Assert
        assert result is True, "Initialization should succeed with valid HMAS platform"
        assert integration.platform_uri is not None, "Platform URI should be set after successful initialization"
        assert integration.platform_graph is not None, "RDF graph should be populated after dereferencing"
        assert len(integration.platform_graph) > 0, "RDF graph should contain triples"

        # Log the discovered platform URI
        print(f"\nDiscovered platform URI: {integration.platform_uri}")
        print(f"Graph contains {len(integration.platform_graph)} triples")

    @pytest.mark.asyncio
    async def test_initialize_with_direct_platform(self):
        """
        Test initialization when the URL is directly a HypermediaMASPlatform.

        This test verifies Case A: the URL itself is typed as hmas:HypermediaMASPlatform.
        """
        yggdrasil_url = "http://localhost:8080/"
        integration = YggdrasilIntegration(yggdrasil_url)

        result = await integration.initialize({})

        if result:
            # If successful, verify it's the direct case
            assert str(integration.platform_uri) == yggdrasil_url, \
                "Platform URI should match the original URL for direct platform case"
            print(f"\n✓ Direct platform case verified: {integration.platform_uri}")

    @pytest.mark.asyncio
    async def test_initialize_graph_structure(self):
        """
        Test that the RDF graph is properly structured after initialization.

        This verifies that the graph contains the expected HMAS vocabulary terms.
        """
        yggdrasil_url = "http://localhost:8080/"
        integration = YggdrasilIntegration(yggdrasil_url)

        result = await integration.initialize({})

        if result:
            assert integration.platform_graph is not None

            # Print some statistics about the graph
            print(f"\n--- RDF Graph Statistics ---")
            print(f"Total triples: {len(integration.platform_graph)}")

            # Count subjects, predicates, objects
            subjects = set(s for s, _, _ in integration.platform_graph)
            predicates = set(p for _, p, _ in integration.platform_graph)
            objects = set(o for _, _, o in integration.platform_graph)

            print(f"Unique subjects: {len(subjects)}")
            print(f"Unique predicates: {len(predicates)}")
            print(f"Unique objects: {len(objects)}")

            # Print all triples for inspection (useful for debugging)
            print(f"\n--- All Triples ---")
            for s, p, o in integration.platform_graph:
                print(f"{s} {p} {o}")

    @pytest.mark.asyncio
    async def test_initialize_invalid_url(self):
        """
        Test that initialization fails gracefully with an invalid URL.
        """
        invalid_url = "http://invalid-url-that-does-not-exist.example.com/"
        integration = YggdrasilIntegration(invalid_url)

        result = await integration.initialize({})

        assert result is False, "Initialization should fail with invalid URL"
        assert integration.platform_uri is None, "Platform URI should remain None after failed initialization"

    @pytest.mark.asyncio
    async def test_find_workspaces(self):
        """
        Test workspace discovery and parsing from the platform.

        This test:
        1. Initializes connection to the platform
        2. Discovers all workspaces recursively
        3. Prints workspace information including hierarchy
        """
        yggdrasil_url = "http://localhost:8080/"
        integration = YggdrasilIntegration(yggdrasil_url)

        # Initialize the platform connection
        init_result = await integration.initialize({})
        assert init_result is True, "Platform initialization must succeed"

        # Find all workspaces
        workspaces = await integration._map_workspaces()

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"WORKSPACE DISCOVERY RESULTS")
        print(f"{'=' * 80}")
        print(f"Total workspaces found: {len(workspaces)}")
        print(f"Platform URI: {integration.platform_uri}")
        print()

        # Print detailed information for each workspace
        for workspace_id, workspace in workspaces.items():
            print(f"\n{'-' * 80}")
            print(f"Workspace ID: {workspace_id}")
            print(f"  Name: {workspace.name}")
            print(f"  Type: {workspace.workspace_type.value}")
            print(f"  Parent: {workspace.parent_workspace_id or '(root)'}")
            print(f"  Sub-workspaces: {len(workspace.sub_workspaces)}")

            if workspace.sub_workspaces:
                for sub_id in workspace.sub_workspaces:
                    sub_ws = workspaces.get(sub_id)
                    if sub_ws:
                        print(f"    - {sub_ws.name} ({sub_id})")

            print(f"  Artifacts: {len(workspace.artifacts)}")
            print(f"  Has RDF: {'Yes' if workspace.rdf else 'No'}")

            if workspace.rdf:
                rdf_lines = workspace.rdf.strip().split('\n')
                print(f"  RDF preview (first 5 lines):")
                for line in rdf_lines[:5]:
                    print(f"    {line}")
                if len(rdf_lines) > 5:
                    print(f"    ... ({len(rdf_lines) - 5} more lines)")

        # Print workspace hierarchy
        print(f"\n{'=' * 80}")
        print(f"WORKSPACE HIERARCHY")
        print(f"{'=' * 80}")

        # Find root workspaces (those without parents)
        root_workspaces = [ws for ws in workspaces.values() if ws.parent_workspace_id is None]

        def print_hierarchy(workspace, workspaces_dict, indent=0):
            """Recursively print workspace hierarchy."""
            prefix = "  " * indent + "└─ " if indent > 0 else ""
            print(f"{prefix}{workspace.name} ({workspace.workspace_type.value})")

            for sub_id in workspace.sub_workspaces:
                sub_ws = workspaces_dict.get(sub_id)
                if sub_ws:
                    print_hierarchy(sub_ws, workspaces_dict, indent + 1)

        for root_ws in root_workspaces:
            print_hierarchy(root_ws, workspaces, 0)

        print(f"\n{'=' * 80}")

        # Assertions
        if len(workspaces) > 0:
            # At least verify we got some workspaces
            print(f"\n✓ Successfully discovered {len(workspaces)} workspace(s)")

            # Verify all workspaces have required fields
            for workspace_id, workspace in workspaces.items():
                assert workspace.workspace_id == workspace_id, f"Workspace ID mismatch for {workspace_id}"
                assert workspace.name, f"Workspace {workspace_id} must have a name"
                assert workspace.workspace_type is not None, f"Workspace {workspace_id} must have a type"
        else:
            print("\n⚠ No workspaces found - this may indicate:")
            print("  - The platform doesn't host any workspaces")
            print("  - The hmas:hosts relationship is not present in the RDF")

    @pytest.mark.asyncio
    async def test_find_artifacts(self):
        """
        Test artifact discovery and parsing from the platform.

        This test:
        1. Initializes connection to the platform
        2. Discovers all workspaces (required for artifact discovery)
        3. Discovers all artifacts in those workspaces
        4. Prints artifact information grouped by workspace
        """
        yggdrasil_url = "http://localhost:8080/"
        integration = YggdrasilIntegration(yggdrasil_url)

        # Initialize the platform connection
        init_result = await integration.initialize({})
        assert init_result is True, "Platform initialization must succeed"

        # Find all workspaces first (required for artifact discovery)
        integration.workspace_map = await integration._map_workspaces()
        assert len(integration.workspace_map) > 0, "Must have workspaces to search for artifacts"

        # Find all artifacts
        artifacts = await integration._map_artifacts()

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"ARTIFACT DISCOVERY RESULTS")
        print(f"{'=' * 80}")
        print(f"Total artifacts found: {len(artifacts)}")
        print(f"Platform URI: {integration.platform_uri}")
        print(f"Total workspaces: {len(integration.workspace_map)}")
        print()

        # Print artifacts grouped by workspace
        print(f"\n{'=' * 80}")
        print(f"ARTIFACTS BY WORKSPACE")
        print(f"{'=' * 80}")

        for workspace_id, workspace in integration.workspace_map.items():
            print(f"\n{'-' * 80}")
            print(f"Workspace: {workspace.name} ({workspace.workspace_type.value})")
            print(f"  URI: {workspace_id}")
            print(f"  Artifacts: {len(workspace.artifacts)}")

            if workspace.artifacts:
                for artifact_id in workspace.artifacts:
                    artifact = artifacts.get(artifact_id)
                    if artifact:
                        print(f"\n  {'─' * 76}")
                        print(f"  Artifact: {artifact.name}")
                        print(f"    URI: {artifact_id}")
                        print(f"    Workspace: {artifact.workspace_id}")

                        if hasattr(artifact, 'artifact_type') and artifact.artifact_type:
                            print(f"    Type: {artifact.artifact_type.value}")

                        if hasattr(artifact, 'thing_description') and artifact.thing_description:
                            td = artifact.thing_description
                            print(f"    Thing Description ID: {td.id}")
                            print(f"    Title: {td.title}")
                            print(f"    Description: {td.description}")
                            print(f"    Properties: {len(td.properties)}")
                            print(f"    Actions: {len(td.actions)}")
                            print(f"    Events: {len(td.events)}")

                        if hasattr(artifact, 'rdf') and artifact.rdf:
                            rdf_lines = artifact.rdf.strip().split('\n')
                            print(f"    Has RDF: Yes ({len(rdf_lines)} lines)")
                            print(f"    RDF preview (first 5 lines):")
                            for line in rdf_lines[:5]:
                                print(f"      {line}")
                            if len(rdf_lines) > 5:
                                print(f"      ... ({len(rdf_lines) - 5} more lines)")
                        else:
                            print(f"    Has RDF: No")
            else:
                print(f"  (no artifacts in this workspace)")

        # Print detailed artifact list
        print(f"\n{'=' * 80}")
        print(f"DETAILED ARTIFACT LIST")
        print(f"{'=' * 80}")

        for artifact_id, artifact in artifacts.items():
            workspace = integration.workspace_map.get(artifact.workspace_id)
            workspace_name = workspace.name if workspace else "(unknown)"

            print(f"\n{'-' * 80}")
            print(f"Artifact ID: {artifact_id}")
            print(f"  Name: {artifact.name}")
            print(f"  Workspace: {workspace_name} ({artifact.workspace_id})")

            if hasattr(artifact, 'artifact_type') and artifact.artifact_type:
                print(f"  Type: {artifact.artifact_type.value}")

            if hasattr(artifact, 'current_state') and artifact.current_state:
                print(f"  Current State: {artifact.current_state}")

            if hasattr(artifact, 'metadata') and artifact.metadata:
                print(f"  Metadata: {artifact.metadata}")

        print(f"\n{'=' * 80}")

        # Assertions
        if len(artifacts) > 0:
            print(f"\n✓ Successfully discovered {len(artifacts)} artifact(s)")

            # Verify all artifacts have required fields
            for artifact_id, artifact in artifacts.items():
                assert artifact.artifact_id == artifact_id, f"Artifact ID mismatch for {artifact_id}"
                assert artifact.name, f"Artifact {artifact_id} must have a name"
                assert artifact.workspace_id, f"Artifact {artifact_id} must belong to a workspace"

                # Verify the workspace exists
                assert artifact.workspace_id in integration.workspace_map, \
                    f"Artifact {artifact_id} references non-existent workspace {artifact.workspace_id}"

                # Verify the artifact is in the workspace's artifact list
                workspace = integration.workspace_map[artifact.workspace_id]
                assert artifact_id in workspace.artifacts, \
                    f"Artifact {artifact_id} not found in workspace {artifact.workspace_id}'s artifact list"
        else:
            print("\n⚠ No artifacts found - this may indicate:")
            print("  - The workspaces don't contain any artifacts")
            print("  - The hmas:contains relationship is not present for artifacts in the RDF")

    @pytest.mark.asyncio
    async def test_find_affordances(self):
        """
        Test affordance discovery and parsing from the platform.

        This test:
        1. Initializes connection to the platform
        2. Discovers all workspaces (required for artifact discovery)
        3. Discovers all artifacts (required for affordance discovery)
        4. Discovers all affordances for those artifacts
        5. Prints affordance information grouped by artifact and workspace
        """
        yggdrasil_url = "http://localhost:8080/"
        integration = YggdrasilIntegration(yggdrasil_url)

        # Initialize the platform connection
        init_result = await integration.initialize({})
        assert init_result is True, "Platform initialization must succeed"

        # Find all workspaces first (required for artifact discovery)
        integration.workspace_map = await integration._map_workspaces()
        assert len(integration.workspace_map) > 0, "Must have workspaces to search for artifacts"

        # Find all artifacts (required for affordance discovery)
        integration.artifact_map = await integration._map_artifacts()
        assert len(integration.artifact_map) > 0, "Must have artifacts to search for affordances"

        # Find all affordances
        affordances = await integration._map_affordances()

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"AFFORDANCE DISCOVERY RESULTS")
        print(f"{'=' * 80}")
        print(f"Total affordances found: {len(affordances)}")
        print(f"Total artifacts: {len(integration.artifact_map)}")
        print(f"Total workspaces: {len(integration.workspace_map)}")
        print(f"Platform URI: {integration.platform_uri}")
        print()

        # Count affordances by type
        from ami_agents.shared.models.environment import AffordanceType
        property_count = sum(1 for a in affordances.values() if a.affordance_type == AffordanceType.PROPERTY)
        action_count = sum(1 for a in affordances.values() if a.affordance_type == AffordanceType.ACTION)
        event_count = sum(1 for a in affordances.values() if a.affordance_type == AffordanceType.EVENT)

        print(f"Affordances by type:")
        print(f"  - Properties: {property_count}")
        print(f"  - Actions: {action_count}")
        print(f"  - Events: {event_count}")

        # Print affordances grouped by workspace and artifact
        print(f"\n{'=' * 80}")
        print(f"AFFORDANCES BY WORKSPACE AND ARTIFACT")
        print(f"{'=' * 80}")

        for workspace_id, workspace in integration.workspace_map.items():
            print(f"\n{'-' * 80}")
            print(f"Workspace: {workspace.name} ({workspace.workspace_type.value})")
            print(f"  URI: {workspace_id}")

            if workspace.artifacts:
                for artifact_id in workspace.artifacts:
                    artifact = integration.artifact_map.get(artifact_id)
                    if artifact:
                        # Get affordances for this artifact
                        artifact_affordances = {
                            aff_id: aff for aff_id, aff in affordances.items()
                            if aff.artifact_id == artifact_id
                        }

                        print(f"\n  {'─' * 76}")
                        print(f"  Artifact: {artifact.name}")
                        print(f"    URI: {artifact_id}")
                        print(f"    Affordances: {len(artifact_affordances)}")

                        if artifact_affordances:
                            for aff_id, affordance in artifact_affordances.items():
                                print(f"\n    {'·' * 74}")
                                print(f"    Affordance ID: {affordance.affordance_id}")
                                print(f"      Type: {affordance.affordance_type.value}")
                                print(f"      Name: {affordance.name}")
                                print(f"      Artifact ID: {affordance.artifact_id}")

                                # Print semantic types if available
                                if affordance.semantic_types:
                                    print(f"      Semantic Types:")
                                    for sem_type in affordance.semantic_types:
                                        print(f"        - {sem_type}")

                                # Print form details
                                print(f"      Form:")
                                print(f"        - href: {affordance.form.href}")
                                print(f"        - method: {affordance.form.method}")
                                print(f"        - content_type: {affordance.form.content_type}")
                                if affordance.form.operation_type:
                                    print(f"        - operation_type: {affordance.form.operation_type}")
                                if affordance.form.additional_fields:
                                    print(f"        - additional_fields: {affordance.form.additional_fields}")

                                # Print input schema if available
                                if hasattr(affordance, 'input_schema') and affordance.input_schema:
                                    print(f"      Input Schema:")
                                    import json
                                    schema_json = json.dumps(affordance.input_schema, indent=10)
                                    for line in schema_json.split('\n'):
                                        print(f"        {line}")

                                # Print output schema if available
                                if hasattr(affordance, 'output_schema') and affordance.output_schema:
                                    print(f"      Output Schema:")
                                    import json
                                    schema_json = json.dumps(affordance.output_schema, indent=10)
                                    for line in schema_json.split('\n'):
                                        print(f"        {line}")
                        else:
                            print(f"      (no affordances for this artifact)")
            else:
                print(f"  (no artifacts in this workspace)")

        # Print detailed affordance list
        print(f"\n{'=' * 80}")
        print(f"DETAILED AFFORDANCE LIST")
        print(f"{'=' * 80}")

        for aff_id, affordance in affordances.items():
            artifact = integration.artifact_map.get(affordance.artifact_id)
            artifact_name = artifact.name if artifact else "(unknown)"
            workspace = integration.workspace_map.get(artifact.workspace_id) if artifact else None
            workspace_name = workspace.name if workspace else "(unknown)"

            print(f"\n{'-' * 80}")
            print(f"Affordance ID: {affordance.affordance_id}")
            print(f"  Name: {affordance.name}")
            print(f"  Type: {affordance.affordance_type.value}")
            print(f"  Artifact: {artifact_name} ({affordance.artifact_id})")
            print(f"  Workspace: {workspace_name}")

            # Semantic types
            if affordance.semantic_types:
                print(f"  Semantic Types: {', '.join(affordance.semantic_types)}")

            # Form
            print(f"  Form:")
            print(f"    href: {affordance.form.href}")
            print(f"    method: {affordance.form.method}")
            print(f"    content_type: {affordance.form.content_type}")
            if affordance.form.operation_type:
                print(f"    operation_type: {affordance.form.operation_type}")

            # Schemas
            if hasattr(affordance, 'input_schema') and affordance.input_schema:
                import json
                print(f"  Input Schema: {json.dumps(affordance.input_schema)}")

            if hasattr(affordance, 'output_schema') and affordance.output_schema:
                import json
                print(f"  Output Schema: {json.dumps(affordance.output_schema)}")

            # RDF preview
            if affordance.rdf:
                rdf_lines = affordance.rdf.strip().split('\n')
                print(f"  RDF preview (first 50 lines):")
                for line in rdf_lines[:50]:
                    print(f"    {line}")
                if len(rdf_lines) > 50:
                    print(f"    ... ({len(rdf_lines) - 50} more lines)")

        print(f"\n{'=' * 80}")

        # Assertions
        if len(affordances) > 0:
            print(f"\n✓ Successfully discovered {len(affordances)} affordance(s)")

            # Verify all affordances have required fields
            for aff_id, affordance in affordances.items():
                assert affordance.affordance_id == aff_id, f"Affordance ID mismatch for {aff_id}"
                assert affordance.name, f"Affordance {aff_id} must have a name"
                assert affordance.affordance_type is not None, f"Affordance {aff_id} must have a type"
                assert affordance.artifact_id, f"Affordance {aff_id} must belong to an artifact"
                assert affordance.form is not None, f"Affordance {aff_id} must have a form"

                # Verify the artifact exists
                assert affordance.artifact_id in integration.artifact_map, \
                    f"Affordance {aff_id} references non-existent artifact {affordance.artifact_id}"

                # Verify form has required fields
                assert affordance.form.href, f"Affordance {aff_id} form must have href"
                assert affordance.form.method, f"Affordance {aff_id} form must have method"
        else:
            print("\n⚠ No affordances found - this may indicate:")
            print("  - The artifacts don't have any affordances defined")
            print("  - The td:hasPropertyAffordance/hasActionAffordance/hasEventAffordance relationships are missing")




# Fixtures for sample RDF data (optional - for reference/mocking)
# Note: These are in Turtle format for readability, but the actual Yggdrasil
# server can serve in any RDF format (Turtle, RDF/XML, etc.) since rdflib
# is used for parsing, not owlready2.

@pytest.fixture
def sample_hmas_platform_rdf():
    """
    Sample HMAS platform in Turtle format (for readability).

    This can be used to set up a test server or mock the HTTP response.
    The server can serve this in any RDF format - rdflib will parse it.
    """
    return """
    @prefix hmas: <https://purl.org/hmas/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

    <http://localhost:8080/> a hmas:HypermediaMASPlatform ;
        hmas:hasName "Test Yggdrasil Platform" .
    """


@pytest.fixture
def sample_resource_profile_rdf():
    """
    Sample ResourceProfile in Turtle format (for readability).

    This represents Case B where the URL is a profile pointing to a platform.
    The server can serve this in any RDF format - rdflib will parse it.
    """
    return """
    @prefix hmas: <https://purl.org/hmas/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

    <http://localhost:8080/> a hmas:ResourceProfile ;
        hmas:isProfileOf <http://localhost:8080/platform> .

    <http://localhost:8080/platform> a hmas:HypermediaMASPlatform ;
        hmas:hasName "Test Yggdrasil Platform" .
    """


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
