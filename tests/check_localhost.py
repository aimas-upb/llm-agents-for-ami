#!/usr/bin/env python3
"""
Quick script to check if http://localhost:8080/ is serving valid HMAS RDF content.

Run this before running the actual tests to verify your Yggdrasil instance is working.
"""

import sys
from rdflib import Graph, URIRef
from rdflib.namespace import RDF


def check_localhost():
    """Check if localhost:8080 serves valid HMAS content."""
    url = "http://localhost:8080/"

    print(f"Checking {url}...")
    print("-" * 60)

    try:
        # Try to parse the RDF
        graph = Graph()
        graph.parse(url)

        print(f"✓ Successfully parsed RDF from {url}")
        print(f"  Total triples: {len(graph)}")
        print()

        # Check for HMAS vocabulary
        hmas_namespace = "https://purl.org/hmas/"
        platform_type = URIRef(f"{hmas_namespace}HypermediaMASPlatform")
        profile_type = URIRef(f"{hmas_namespace}ResourceProfile")
        is_profile_of = URIRef(f"{hmas_namespace}isProfileOf")

        # Check Case A: Direct platform
        url_ref = URIRef(url)
        if (url_ref, RDF.type, platform_type) in graph:
            print(f"✓ Case A: URL is directly typed as hmas:HypermediaMASPlatform")
            print(f"  Platform URI: {url}")
            return True

        # Check Case B: ResourceProfile
        if (url_ref, RDF.type, profile_type) in graph:
            print(f"✓ URL is typed as hmas:ResourceProfile")

            # Find the platform it points to
            for _, _, platform in graph.triples((url_ref, is_profile_of, None)):
                if (platform, RDF.type, platform_type) in graph:
                    print(f"✓ Case B: Profile points to hmas:HypermediaMASPlatform")
                    print(f"  Profile URI: {url}")
                    print(f"  Platform URI: {platform}")
                    return True
                else:
                    print(f"✗ Profile points to {platform}, but it's not a HypermediaMASPlatform")
                    return False

            print(f"✗ ResourceProfile found, but no valid isProfileOf property")
            return False

        # Neither case matched
        print(f"✗ URL is neither a HypermediaMASPlatform nor a ResourceProfile")
        print()
        print("RDF content preview:")
        print("-" * 60)
        for s, p, o in graph:
            print(f"{s}")
            print(f"  {p} {o}")
        return False

    except Exception as e:
        print(f"✗ Failed to fetch or parse RDF from {url}")
        print(f"  Error: {e}")
        return False


if __name__ == "__main__":
    success = check_localhost()
    sys.exit(0 if success else 1)
