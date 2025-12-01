#!/usr/bin/env python3
"""
Example demonstrating nested object handling in schema utilities.

This shows how the schema utilities handle complex nested structures
including:
- Nested objects (multiple levels)
- Arrays of objects
- Mixed nested structures
"""

from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS

from ami_agents.shared.utils import (
    parse_jsonschema_from_rdf,
    validate_params,
    generate_rdf_payload
)

# Define namespaces
JSONSCHEMA = Namespace("http://www.w3.org/2019/wot/json-schema#")
EX = Namespace("http://example.org/")


def example_1_simple_nested_object():
    """Example 1: Simple nested object (location with coordinates)."""
    print("=" * 80)
    print("Example 1: Simple Nested Object - Device Location")
    print("=" * 80)

    # Create schema
    graph = Graph()
    schema_uri = EX.DeviceSchema

    # Root object
    graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))

    # Device ID property
    device_id_prop = BNode()
    graph.add((schema_uri, JSONSCHEMA.properties, device_id_prop))
    graph.add((device_id_prop, JSONSCHEMA.propertyName, Literal("deviceId")))
    graph.add((device_id_prop, JSONSCHEMA.type, Literal("string")))

    # Location property (nested object)
    location_prop = BNode()
    graph.add((schema_uri, JSONSCHEMA.properties, location_prop))
    graph.add((location_prop, JSONSCHEMA.propertyName, Literal("location")))
    graph.add((location_prop, JSONSCHEMA.type, Literal("object")))

    # Nested: latitude
    lat_prop = BNode()
    graph.add((location_prop, JSONSCHEMA.properties, lat_prop))
    graph.add((lat_prop, JSONSCHEMA.propertyName, Literal("latitude")))
    graph.add((lat_prop, JSONSCHEMA.type, Literal("number")))
    graph.add((lat_prop, JSONSCHEMA.minimum, Literal(-90)))
    graph.add((lat_prop, JSONSCHEMA.maximum, Literal(90)))

    # Nested: longitude
    lon_prop = BNode()
    graph.add((location_prop, JSONSCHEMA.properties, lon_prop))
    graph.add((lon_prop, JSONSCHEMA.propertyName, Literal("longitude")))
    graph.add((lon_prop, JSONSCHEMA.type, Literal("number")))
    graph.add((lon_prop, JSONSCHEMA.minimum, Literal(-180)))
    graph.add((lon_prop, JSONSCHEMA.maximum, Literal(180)))

    # Required fields
    graph.add((schema_uri, JSONSCHEMA.required, Literal("deviceId")))
    graph.add((location_prop, JSONSCHEMA.required, Literal("latitude")))
    graph.add((location_prop, JSONSCHEMA.required, Literal("longitude")))

    # Parse schema
    schema = parse_jsonschema_from_rdf(graph, schema_uri)

    print("\nParsed Schema:")
    import json
    print(json.dumps(schema, indent=2))

    # Test valid parameters
    params = {
        "deviceId": "sensor-001",
        "location": {
            "latitude": 45.5017,
            "longitude": -73.5673
        }
    }

    print("\nValidating parameters:")
    print(json.dumps(params, indent=2))

    is_valid, errors = validate_params(params, schema)
    print(f"\nValid: {is_valid}")
    if errors:
        print("Errors:", errors)

    # Generate RDF
    print("\nGenerated RDF (Turtle):")
    rdf = generate_rdf_payload(
        params,
        schema,
        "http://example.org/devices/sensor-001",
        "http://example.org/vocab#"
    )
    print(rdf)


def example_2_deeply_nested_objects():
    """Example 2: Deeply nested objects (3 levels)."""
    print("\n" + "=" * 80)
    print("Example 2: Deeply Nested Objects - Device Configuration")
    print("=" * 80)

    # Create a schema with 3 levels of nesting
    schema = {
        "type": "object",
        "properties": {
            "device": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "network": {
                        "type": "object",
                        "properties": {
                            "wifi": {
                                "type": "object",
                                "properties": {
                                    "ssid": {"type": "string"},
                                    "password": {"type": "string"},
                                    "encryption": {
                                        "type": "string",
                                        "enum": ["WPA2", "WPA3", "Open"]
                                    }
                                },
                                "required": ["ssid"]
                            }
                        }
                    }
                },
                "required": ["name", "network"]
            }
        }
    }

    print("\nSchema Structure:")
    import json
    print(json.dumps(schema, indent=2))

    # Valid nested parameters
    params = {
        "device": {
            "name": "SmartThermostat",
            "network": {
                "wifi": {
                    "ssid": "HomeNetwork",
                    "password": "secret123",
                    "encryption": "WPA2"
                }
            }
        }
    }

    print("\nParameters:")
    print(json.dumps(params, indent=2))

    # Validate
    is_valid, errors = validate_params(params, schema)
    print(f"\nValidation Result: {is_valid}")

    # Generate RDF
    print("\nGenerated RDF:")
    rdf = generate_rdf_payload(
        params,
        schema,
        "http://example.org/devices/thermostat-1",
        "http://example.org/vocab#"
    )
    print(rdf)

    # Test validation error with missing nested field
    print("\n" + "-" * 80)
    print("Testing validation error (missing nested required field)...")
    params_invalid = {
        "device": {
            "name": "SmartThermostat",
            "network": {
                "wifi": {
                    "password": "secret123"
                    # Missing required 'ssid'
                }
            }
        }
    }

    is_valid, errors = validate_params(params_invalid, schema)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")


def example_3_array_of_nested_objects():
    """Example 3: Array containing nested objects."""
    print("\n" + "=" * 80)
    print("Example 3: Array of Nested Objects - Sensor Readings")
    print("=" * 80)

    schema = {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string"},
            "readings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sensor": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"}
                            },
                            "required": ["id", "type"]
                        },
                        "value": {"type": "number"},
                        "unit": {"type": "string"}
                    },
                    "required": ["sensor", "value"]
                }
            }
        }
    }

    params = {
        "timestamp": "2024-01-15T10:30:00Z",
        "readings": [
            {
                "sensor": {
                    "id": "temp-01",
                    "type": "temperature"
                },
                "value": 22.5,
                "unit": "celsius"
            },
            {
                "sensor": {
                    "id": "humid-01",
                    "type": "humidity"
                },
                "value": 65,
                "unit": "percent"
            }
        ]
    }

    print("\nParameters:")
    import json
    print(json.dumps(params, indent=2))

    # Validate
    is_valid, errors = validate_params(params, schema)
    print(f"\nValidation Result: {is_valid}")

    # Generate RDF
    print("\nGenerated RDF:")
    rdf = generate_rdf_payload(
        params,
        schema,
        "http://example.org/readings/batch-001",
        "http://example.org/vocab#"
    )
    print(rdf)

    # Test validation with error in array item
    print("\n" + "-" * 80)
    print("Testing validation error in array item...")
    params_invalid = {
        "timestamp": "2024-01-15T10:30:00Z",
        "readings": [
            {
                "sensor": {
                    "id": "temp-01",
                    "type": "temperature"
                },
                "value": 22.5,
                "unit": "celsius"
            },
            {
                "sensor": {
                    "id": "humid-01"
                    # Missing required 'type'
                },
                "value": 65,
                "unit": "percent"
            }
        ]
    }

    is_valid, errors = validate_params(params_invalid, schema)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")


def example_4_complex_real_world():
    """Example 4: Complex real-world scenario - Smart Home Action."""
    print("\n" + "=" * 80)
    print("Example 4: Complex Real-World - Smart Home Scene Configuration")
    print("=" * 80)

    schema = {
        "type": "object",
        "properties": {
            "sceneName": {"type": "string"},
            "enabled": {"type": "boolean"},
            "triggers": {
                "type": "object",
                "properties": {
                    "timeOfDay": {"type": "string"},
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sensor": {"type": "string"},
                                "operator": {
                                    "type": "string",
                                    "enum": ["eq", "gt", "lt", "gte", "lte"]
                                },
                                "value": {"type": "number"}
                            },
                            "required": ["sensor", "operator", "value"]
                        }
                    }
                }
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "device": {"type": "string"},
                        "command": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "power": {"type": "boolean"},
                                "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                                "color": {
                                    "type": "object",
                                    "properties": {
                                        "r": {"type": "integer", "minimum": 0, "maximum": 255},
                                        "g": {"type": "integer", "minimum": 0, "maximum": 255},
                                        "b": {"type": "integer", "minimum": 0, "maximum": 255}
                                    }
                                }
                            }
                        }
                    },
                    "required": ["device", "command"]
                }
            }
        },
        "required": ["sceneName", "enabled", "actions"]
    }

    params = {
        "sceneName": "Evening Relaxation",
        "enabled": True,
        "triggers": {
            "timeOfDay": "19:00",
            "conditions": [
                {
                    "sensor": "outdoor_brightness",
                    "operator": "lt",
                    "value": 100
                }
            ]
        },
        "actions": [
            {
                "device": "living_room_light",
                "command": "set_state",
                "parameters": {
                    "power": True,
                    "brightness": 60,
                    "color": {
                        "r": 255,
                        "g": 200,
                        "b": 150
                    }
                }
            },
            {
                "device": "curtains",
                "command": "close",
                "parameters": {}
            }
        ]
    }

    print("\nComplex Smart Home Scene Configuration:")
    import json
    print(json.dumps(params, indent=2))

    # Validate
    is_valid, errors = validate_params(params, schema)
    print(f"\nValidation Result: {is_valid}")
    if errors:
        print(f"Errors: {errors}")

    # Generate RDF
    print("\nGenerated RDF (showing complex nested structure):")
    rdf = generate_rdf_payload(
        params,
        schema,
        "http://example.org/scenes/evening-relaxation",
        "http://example.org/smarthome#"
    )
    print(rdf)

    print("\n" + "=" * 80)
    print("Key observations:")
    print("- Nested objects are represented as blank nodes in RDF")
    print("- Array items are also blank nodes when they are objects")
    print("- Path-based error messages help locate issues in deep structures")
    print("- Full validation cascades through all nesting levels")
    print("=" * 80)


def main():
    """Run all examples."""
    example_1_simple_nested_object()
    example_2_deeply_nested_objects()
    example_3_array_of_nested_objects()
    example_4_complex_real_world()

    print("\n" + "=" * 80)
    print("All nested object examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
