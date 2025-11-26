#!/usr/bin/env python3
"""
Script to inspect ontology concepts and their IRIs.

This script loads the HMAS and CASHMERE ontologies and displays
all classes, properties, and their IRIs.
"""

import sys
from ami_agents.shared.ontologies import get_hmas_ontology, get_cashmere_ontology


def inspect_ontology(ontology, name):
    """Inspect and display ontology concepts."""
    print(f"\n{'=' * 80}")
    print(f"{name} Ontology Inspection")
    print(f"{'=' * 80}")

    if ontology is None:
        print(f"ERROR: Failed to load {name} ontology")
        return

    print(f"\nBase IRI: {ontology.base_iri}")
    print(f"IRI: {ontology.iri}")

    # Get all classes
    classes = list(ontology.classes())
    print(f"\n--- Classes ({len(classes)}) ---")
    for cls in sorted(classes, key=lambda c: c.name):
        print(f"  {cls.name:40} → {cls.iri}")

    # Get all object properties
    obj_props = list(ontology.object_properties())
    print(f"\n--- Object Properties ({len(obj_props)}) ---")
    for prop in sorted(obj_props, key=lambda p: p.name):
        print(f"  {prop.name:40} → {prop.iri}")

    # Get all data properties
    data_props = list(ontology.data_properties())
    print(f"\n--- Data Properties ({len(data_props)}) ---")
    for prop in sorted(data_props, key=lambda p: p.name):
        print(f"  {prop.name:40} → {prop.iri}")

    # Get all annotation properties
    annot_props = list(ontology.annotation_properties())
    print(f"\n--- Annotation Properties ({len(annot_props)}) ---")
    for prop in sorted(annot_props, key=lambda p: p.name):
        print(f"  {prop.name:40} → {prop.iri}")

    # Get all individuals
    individuals = list(ontology.individuals())
    if individuals:
        print(f"\n--- Individuals ({len(individuals)}) ---")
        for ind in sorted(individuals, key=lambda i: i.name):
            print(f"  {ind.name:40} → {ind.iri}")


def test_access_methods(ontology, name):
    """Test different ways to access ontology concepts."""
    print(f"\n{'=' * 80}")
    print(f"Testing Access Methods for {name}")
    print(f"{'=' * 80}")

    if ontology is None:
        print(f"ERROR: Failed to load {name} ontology")
        return

    # Test accessing specific classes (for HMAS)
    if name == "HMAS":
        print("\nTrying to access HypermediaMASPlatform:")

        # Method 1: Direct attribute access
        try:
            cls1 = ontology.HypermediaMASPlatform
            print(f"  ✓ Direct access: {cls1}")
            print(f"    IRI: {cls1.iri}")
        except AttributeError as e:
            print(f"  ✗ Direct access failed: {e}")

        # Method 2: Using search
        try:
            cls2 = ontology.search_one(iri=f"{ontology.base_iri}HypermediaMASPlatform")
            print(f"  ✓ Search by IRI: {cls2}")
            if cls2:
                print(f"    IRI: {cls2.iri}")
        except Exception as e:
            print(f"  ✗ Search failed: {e}")

        # Method 3: Using getattr
        try:
            cls3 = getattr(ontology, "HypermediaMASPlatform", None)
            print(f"  ✓ getattr: {cls3}")
            if cls3:
                print(f"    IRI: {cls3.iri}")
        except Exception as e:
            print(f"  ✗ getattr failed: {e}")

        # Method 4: Search by name
        try:
            results = ontology.search(type=ontology.world.Class)
            hmas_platform = [c for c in results if "HypermediaMASPlatform" in c.name]
            if hmas_platform:
                print(f"  ✓ Search by name: {hmas_platform[0]}")
                print(f"    IRI: {hmas_platform[0].iri}")
            else:
                print(f"  ✗ Search by name: No results")
        except Exception as e:
            print(f"  ✗ Search by name failed: {e}")


def main():
    """Main function."""
    print("Loading ontologies...")

    # Load HMAS ontology
    hmas_onto = get_hmas_ontology()
    inspect_ontology(hmas_onto, "HMAS")
    test_access_methods(hmas_onto, "HMAS")

    # Load CASHMERE ontology
    cashmere_onto = get_cashmere_ontology()
    inspect_ontology(cashmere_onto, "CASHMERE")
    test_access_methods(cashmere_onto, "CASHMERE")

    print(f"\n{'=' * 80}")
    print("Inspection Complete")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
