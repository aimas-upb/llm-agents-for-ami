#!/usr/bin/env python3
"""
Example usage of schema utility functions.

This demonstrates how to:
1. Parse JSON Schema from RDF
2. Validate parameters against schemas
3. Generate JSON-LD and RDF payloads
"""

from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD

from ami_agents.shared.utils import (
    parse_jsonschema_from_rdf,
    validate_params,
    generate_jsonld_payload,
    generate_rdf_payload,
    extract_schema_summary
)


# Define namespaces
JSONSCHEMA = Namespace("http://www.w3.org/2019/wot/json-schema#")
EX = Namespace("http://example.org/")


def example_1_parse_schema():
    """Example 1: Parse a JSON Schema from RDF."""
    print("=" * 80)
    print("Example 1: Parsing JSON Schema from RDF")
    print("=" * 80)

    # Create an RDF graph with a schema for a thermostat control action
    graph = Graph()
    schema_uri = EX.ThermostatControlSchema

    # Define schema type
    graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))
    graph.add((schema_uri, RDFS.comment, Literal("Schema for thermostat control")))

    # Define temperature property
    temp_prop = BNode()
    graph.add((schema_uri, JSONSCHEMA.properties, temp_prop))
    graph.add((temp_prop, JSONSCHEMA.propertyName, Literal("targetTemperature")))
    graph.add((temp_prop, JSONSCHEMA.type, Literal("number")))
    graph.add((temp_prop, RDFS.comment, Literal("Target temperature in degrees")))
    graph.add((temp_prop, JSONSCHEMA.minimum, Literal(10)))
    graph.add((temp_prop, JSONSCHEMA.maximum, Literal(30)))

    # Define unit property
    unit_prop = BNode()
    graph.add((schema_uri, JSONSCHEMA.properties, unit_prop))
    graph.add((unit_prop, JSONSCHEMA.propertyName, Literal("unit")))
    graph.add((unit_prop, JSONSCHEMA.type, Literal("string")))
    graph.add((unit_prop, JSONSCHEMA.enum, Literal("celsius")))
    graph.add((unit_prop, JSONSCHEMA.enum, Literal("fahrenheit")))

    # Define mode property
    mode_prop = BNode()
    graph.add((schema_uri, JSONSCHEMA.properties, mode_prop))
    graph.add((mode_prop, JSONSCHEMA.propertyName, Literal("mode")))
    graph.add((mode_prop, JSONSCHEMA.type, Literal("string")))
    graph.add((mode_prop, JSONSCHEMA.enum, Literal("heat")))
    graph.add((mode_prop, JSONSCHEMA.enum, Literal("cool")))
    graph.add((mode_prop, JSONSCHEMA.enum, Literal("auto")))

    # Define required fields
    graph.add((schema_uri, JSONSCHEMA.required, Literal("targetTemperature")))
    graph.add((schema_uri, JSONSCHEMA.required, Literal("mode")))

    # Parse the schema
    schema = parse_jsonschema_from_rdf(graph, schema_uri)

    print("\nParsed Schema:")
    import json
    print(json.dumps(schema, indent=2))

    # Extract and display summary
    summary = extract_schema_summary(schema)
    print("\nSchema Summary:")
    for param_name, param_info in summary.items():
        print(f"\n  {param_name}:")
        for key, value in param_info.items():
            print(f"    {key}: {value}")

    return schema


def example_2_validate_params(schema):
    """Example 2: Validate parameters against schema."""
    print("\n" + "=" * 80)
    print("Example 2: Validating Parameters")
    print("=" * 80)

    # Test Case 1: Valid parameters
    print("\nTest Case 1: Valid parameters")
    params = {
        "targetTemperature": 22,
        "unit": "celsius",
        "mode": "heat"
    }
    print(f"Parameters: {params}")

    is_valid, errors = validate_params(params, schema)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")

    # Test Case 2: Missing required field
    print("\nTest Case 2: Missing required field")
    params = {
        "targetTemperature": 22,
        "unit": "celsius"
        # Missing 'mode'
    }
    print(f"Parameters: {params}")

    is_valid, errors = validate_params(params, schema)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors:")
        for error in errors:
            print(f"  - {error}")

    # Test Case 3: Value out of range
    print("\nTest Case 3: Value out of range")
    params = {
        "targetTemperature": 35,  # Above maximum
        "mode": "heat"
    }
    print(f"Parameters: {params}")

    is_valid, errors = validate_params(params, schema)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors:")
        for error in errors:
            print(f"  - {error}")

    # Test Case 4: Invalid enum value
    print("\nTest Case 4: Invalid enum value")
    params = {
        "targetTemperature": 22,
        "mode": "fan_only"  # Not in enum
    }
    print(f"Parameters: {params}")

    is_valid, errors = validate_params(params, schema)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors:")
        for error in errors:
            print(f"  - {error}")


def example_3_generate_jsonld(schema):
    """Example 3: Generate JSON-LD payload."""
    print("\n" + "=" * 80)
    print("Example 3: Generating JSON-LD Payload")
    print("=" * 80)

    params = {
        "targetTemperature": 22,
        "unit": "celsius",
        "mode": "heat"
    }

    # Define custom context mapping parameter names to IRIs
    context = {
        "targetTemperature": "http://example.org/thermostat#targetTemperature",
        "unit": "http://example.org/thermostat#unit",
        "mode": "http://example.org/thermostat#mode"
    }

    print("\nParameters:")
    import json
    print(json.dumps(params, indent=2))

    print("\nContext:")
    print(json.dumps(context, indent=2))

    # Generate JSON-LD payload
    jsonld = generate_jsonld_payload(params, schema, context)

    print("\nGenerated JSON-LD:")
    print(jsonld)


def example_4_generate_rdf(schema):
    """Example 4: Generate RDF payload."""
    print("\n" + "=" * 80)
    print("Example 4: Generating RDF Payload")
    print("=" * 80)

    params = {
        "targetTemperature": 22,
        "unit": "celsius",
        "mode": "heat"
    }

    print("\nParameters:")
    import json
    print(json.dumps(params, indent=2))

    # Generate RDF in Turtle format
    print("\n--- RDF in Turtle format ---")
    rdf_turtle = generate_rdf_payload(
        params,
        schema,
        subject_uri="http://example.org/thermostat/action1",
        property_namespace="http://example.org/thermostat#",
        format="turtle"
    )
    print(rdf_turtle)

    # Generate RDF in RDF/XML format
    print("\n--- RDF in RDF/XML format ---")
    rdf_xml = generate_rdf_payload(
        params,
        schema,
        subject_uri="http://example.org/thermostat/action1",
        property_namespace="http://example.org/thermostat#",
        format="xml"
    )
    print(rdf_xml)

    # Generate RDF in JSON-LD format (via RDFLib)
    print("\n--- RDF in JSON-LD format ---")
    rdf_jsonld = generate_rdf_payload(
        params,
        schema,
        subject_uri="http://example.org/thermostat/action1",
        property_namespace="http://example.org/thermostat#",
        format="json-ld"
    )
    print(rdf_jsonld)


def example_5_real_world_scenario():
    """Example 5: Real-world scenario with Thing Description affordance."""
    print("\n" + "=" * 80)
    print("Example 5: Real-World Scenario - Smart Light Control")
    print("=" * 80)

    # Create schema for a smart light control action
    graph = Graph()
    schema_uri = EX.SetBrightnessSchema

    graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))

    # Brightness parameter
    brightness_prop = BNode()
    graph.add((schema_uri, JSONSCHEMA.properties, brightness_prop))
    graph.add((brightness_prop, JSONSCHEMA.propertyName, Literal("brightness")))
    graph.add((brightness_prop, JSONSCHEMA.type, Literal("integer")))
    graph.add((brightness_prop, RDFS.comment, Literal("Brightness level (0-100)")))
    graph.add((brightness_prop, JSONSCHEMA.minimum, Literal(0)))
    graph.add((brightness_prop, JSONSCHEMA.maximum, Literal(100)))
    graph.add((schema_uri, JSONSCHEMA.required, Literal("brightness")))

    # Transition time parameter (optional)
    transition_prop = BNode()
    graph.add((schema_uri, JSONSCHEMA.properties, transition_prop))
    graph.add((transition_prop, JSONSCHEMA.propertyName, Literal("transitionTime")))
    graph.add((transition_prop, JSONSCHEMA.type, Literal("integer")))
    graph.add((transition_prop, RDFS.comment, Literal("Transition time in milliseconds")))
    graph.add((transition_prop, JSONSCHEMA.minimum, Literal(0)))

    # Parse schema
    schema = parse_jsonschema_from_rdf(graph, schema_uri)

    print("\nSchema Summary:")
    summary = extract_schema_summary(schema)
    for param_name, param_info in summary.items():
        print(f"\n  {param_name}:")
        for key, value in param_info.items():
            print(f"    {key}: {value}")

    # Validate and generate payloads for different scenarios
    scenarios = [
        {
            "name": "Immediate brightness change",
            "params": {"brightness": 75}
        },
        {
            "name": "Gradual brightness change",
            "params": {"brightness": 50, "transitionTime": 2000}
        },
        {
            "name": "Turn off (brightness 0)",
            "params": {"brightness": 0, "transitionTime": 500}
        }
    ]

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        params = scenario['params']

        # Validate
        is_valid, errors = validate_params(params, schema)
        print(f"Parameters: {params}")
        print(f"Valid: {is_valid}")

        if is_valid:
            # Generate RDF payload
            rdf = generate_rdf_payload(
                params,
                schema,
                subject_uri=f"http://example.org/light/action/{scenario['name'].replace(' ', '_')}",
                property_namespace="http://example.org/light#",
                format="turtle"
            )
            print("RDF Payload:")
            print(rdf)
        else:
            print(f"Errors: {errors}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("JSON Schema Utilities - Examples")
    print("=" * 80)

    # Example 1: Parse schema
    schema = example_1_parse_schema()

    # Example 2: Validate parameters
    example_2_validate_params(schema)

    # Example 3: Generate JSON-LD
    example_3_generate_jsonld(schema)

    # Example 4: Generate RDF
    example_4_generate_rdf(schema)

    # Example 5: Real-world scenario
    example_5_real_world_scenario()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
