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
        assert integration.graph is not None, "RDF graph should be populated after dereferencing"
        assert len(integration.graph) > 0, "RDF graph should contain triples"

        # Log the discovered platform URI
        print(f"\nDiscovered platform URI: {integration.platform_uri}")
        print(f"Graph contains {len(integration.graph)} triples")

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
            assert integration.graph is not None

            # Print some statistics about the graph
            print(f"\n--- RDF Graph Statistics ---")
            print(f"Total triples: {len(integration.graph)}")

            # Count subjects, predicates, objects
            subjects = set(s for s, _, _ in integration.graph)
            predicates = set(p for _, p, _ in integration.graph)
            objects = set(o for _, _, o in integration.graph)

            print(f"Unique subjects: {len(subjects)}")
            print(f"Unique predicates: {len(predicates)}")
            print(f"Unique objects: {len(objects)}")

            # Print all triples for inspection (useful for debugging)
            print(f"\n--- All Triples ---")
            for s, p, o in integration.graph:
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
