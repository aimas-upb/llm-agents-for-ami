"""
Tests for schema utility functions.

These tests verify the functionality of JSON Schema parsing, validation,
and payload generation.
"""

import pytest
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD

from ami_agents.shared.utils import (
    parse_jsonschema_from_rdf,
    validate_params,
    generate_jsonld_payload,
    generate_rdf_payload,
    extract_schema_summary,
    SchemaParsingError,
    SchemaValidationError
)


# Define namespaces
JSONSCHEMA = Namespace("http://www.w3.org/2019/wot/json-schema#")
EX = Namespace("http://example.org/")


class TestParseJsonSchemaFromRDF:
    """Test JSON Schema parsing from RDF."""

    def test_parse_simple_object_schema(self):
        """Test parsing a simple object schema with basic properties."""
        # Create RDF graph with schema
        graph = Graph()
        schema_uri = EX.TemperatureSchema

        # Define schema type
        graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))

        # Define properties
        temp_prop = BNode()
        graph.add((schema_uri, JSONSCHEMA.properties, temp_prop))
        graph.add((temp_prop, JSONSCHEMA.propertyName, Literal("temperature")))
        graph.add((temp_prop, JSONSCHEMA.type, Literal("number")))
        graph.add((temp_prop, RDFS.comment, Literal("Temperature value")))
        graph.add((temp_prop, JSONSCHEMA.minimum, Literal(0)))
        graph.add((temp_prop, JSONSCHEMA.maximum, Literal(100)))

        unit_prop = BNode()
        graph.add((schema_uri, JSONSCHEMA.properties, unit_prop))
        graph.add((unit_prop, JSONSCHEMA.propertyName, Literal("unit")))
        graph.add((unit_prop, JSONSCHEMA.type, Literal("string")))
        graph.add((unit_prop, JSONSCHEMA.enum, Literal("celsius")))
        graph.add((unit_prop, JSONSCHEMA.enum, Literal("fahrenheit")))

        # Define required fields
        graph.add((schema_uri, JSONSCHEMA.required, Literal("temperature")))

        # Parse schema
        schema = parse_jsonschema_from_rdf(graph, schema_uri)

        # Assertions
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "temperature" in schema["properties"]
        assert "unit" in schema["properties"]

        temp_schema = schema["properties"]["temperature"]
        assert temp_schema["type"] == "number"
        assert temp_schema["description"] == "Temperature value"
        assert temp_schema["minimum"] == 0
        assert temp_schema["maximum"] == 100

        unit_schema = schema["properties"]["unit"]
        assert unit_schema["type"] == "string"
        assert "celsius" in unit_schema["enum"]
        assert "fahrenheit" in unit_schema["enum"]

        assert "temperature" in schema["required"]

    def test_parse_schema_with_string_constraints(self):
        """Test parsing schema with string length constraints."""
        graph = Graph()
        schema_uri = EX.StringSchema

        graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))

        name_prop = BNode()
        graph.add((schema_uri, JSONSCHEMA.properties, name_prop))
        graph.add((name_prop, JSONSCHEMA.propertyName, Literal("name")))
        graph.add((name_prop, JSONSCHEMA.type, Literal("string")))
        graph.add((name_prop, JSONSCHEMA.minLength, Literal(1)))
        graph.add((name_prop, JSONSCHEMA.maxLength, Literal(50)))
        graph.add((name_prop, JSONSCHEMA.pattern, Literal("^[a-zA-Z]+$")))

        schema = parse_jsonschema_from_rdf(graph, schema_uri)

        name_schema = schema["properties"]["name"]
        assert name_schema["minLength"] == 1
        assert name_schema["maxLength"] == 50
        assert name_schema["pattern"] == "^[a-zA-Z]+$"

    def test_parse_schema_with_array_type(self):
        """Test parsing schema with array type and items."""
        graph = Graph()
        schema_uri = EX.ArraySchema

        graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))

        tags_prop = BNode()
        graph.add((schema_uri, JSONSCHEMA.properties, tags_prop))
        graph.add((tags_prop, JSONSCHEMA.propertyName, Literal("tags")))
        graph.add((tags_prop, JSONSCHEMA.type, Literal("array")))
        graph.add((tags_prop, JSONSCHEMA.minItems, Literal(1)))
        graph.add((tags_prop, JSONSCHEMA.maxItems, Literal(10)))

        # Define items schema
        items_schema = BNode()
        graph.add((tags_prop, JSONSCHEMA.items, items_schema))
        graph.add((items_schema, JSONSCHEMA.type, Literal("string")))

        schema = parse_jsonschema_from_rdf(graph, schema_uri)

        tags_schema = schema["properties"]["tags"]
        assert tags_schema["type"] == "array"
        assert tags_schema["minItems"] == 1
        assert tags_schema["maxItems"] == 10
        assert tags_schema["items"]["type"] == "string"


class TestValidateParams:
    """Test parameter validation against schemas."""

    def test_validate_valid_params(self):
        """Test validation with valid parameters."""
        schema = {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "minimum": 0, "maximum": 100},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["temperature"]
        }

        params = {"temperature": 25, "unit": "celsius"}

        is_valid, errors = validate_params(params, schema)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_required_field(self):
        """Test validation fails when required field is missing."""
        schema = {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "unit": {"type": "string"}
            },
            "required": ["temperature"]
        }

        params = {"unit": "celsius"}  # Missing temperature

        is_valid, errors = validate_params(params, schema)

        assert is_valid is False
        assert len(errors) == 1
        assert "temperature" in errors[0]
        assert "missing" in errors[0].lower()

    def test_validate_type_mismatch(self):
        """Test validation fails on type mismatch."""
        schema = {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "name": {"type": "string"}
            }
        }

        params = {"temperature": "hot", "name": 123}  # Wrong types

        is_valid, errors = validate_params(params, schema)

        assert is_valid is False
        assert len(errors) == 2

    def test_validate_numeric_constraints(self):
        """Test validation of numeric min/max constraints."""
        schema = {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "minimum": 0, "maximum": 100}
            }
        }

        # Test below minimum
        params = {"temperature": -10}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False
        assert any("minimum" in err.lower() for err in errors)

        # Test above maximum
        params = {"temperature": 150}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False
        assert any("maximum" in err.lower() for err in errors)

    def test_validate_enum_constraint(self):
        """Test validation of enum values."""
        schema = {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]}
            }
        }

        # Valid enum value
        params = {"unit": "celsius"}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is True

        # Invalid enum value
        params = {"unit": "rankine"}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False
        assert any("allowed values" in err.lower() for err in errors)

    def test_validate_string_constraints(self):
        """Test validation of string length constraints."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 10,
                    "pattern": "^[a-zA-Z]+$"
                }
            }
        }

        # Too short
        params = {"name": "ab"}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False

        # Too long
        params = {"name": "verylongname"}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False

        # Invalid pattern
        params = {"name": "name123"}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False

        # Valid
        params = {"name": "Alice"}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is True

    def test_validate_array_constraints(self):
        """Test validation of array item constraints."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "minItems": 1, "maxItems": 5}
            }
        }

        # Too few items
        params = {"tags": []}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False

        # Too many items
        params = {"tags": [1, 2, 3, 4, 5, 6]}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False

        # Valid
        params = {"tags": [1, 2, 3]}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is True


class TestGenerateJsonLDPayload:
    """Test JSON-LD payload generation."""

    def test_generate_simple_jsonld(self):
        """Test generating a simple JSON-LD payload."""
        schema = {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "unit": {"type": "string"}
            }
        }

        params = {"temperature": 25, "unit": "celsius"}
        context = {
            "temperature": "http://example.org/vocab#temperature",
            "unit": "http://example.org/vocab#unit"
        }

        payload = generate_jsonld_payload(params, schema, context)

        # Parse JSON-LD
        import json
        doc = json.loads(payload)

        assert "@context" in doc
        assert doc["@context"] == context
        assert doc["temperature"] == 25
        assert doc["unit"] == "celsius"

    def test_generate_jsonld_with_default_context(self):
        """Test generating JSON-LD with default context."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            }
        }

        params = {"value": 42}

        payload = generate_jsonld_payload(params, schema)

        import json
        doc = json.loads(payload)

        assert "@context" in doc
        assert "value" in doc["@context"]
        assert doc["value"] == 42

    def test_generate_jsonld_validation_failure(self):
        """Test that JSON-LD generation fails with invalid params."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            },
            "required": ["value"]
        }

        params = {}  # Missing required field

        with pytest.raises(SchemaValidationError):
            generate_jsonld_payload(params, schema)


class TestGenerateRDFPayload:
    """Test RDF payload generation."""

    def test_generate_turtle_payload(self):
        """Test generating RDF in Turtle format."""
        schema = {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "unit": {"type": "string"}
            }
        }

        params = {"temperature": 25, "unit": "celsius"}

        rdf = generate_rdf_payload(
            params,
            schema,
            "http://example.org/sensor1",
            "http://example.org/vocab#",
            format="turtle"
        )

        # Parse the RDF to verify
        graph = Graph()
        graph.parse(data=rdf, format="turtle")

        # Check that triples exist
        assert len(graph) == 2

        # Check specific values
        vocab_ns = Namespace("http://example.org/vocab#")
        subject = URIRef("http://example.org/sensor1")

        temp_value = graph.value(subject, vocab_ns.temperature)
        assert temp_value is not None
        assert float(temp_value) == 25

        unit_value = graph.value(subject, vocab_ns.unit)
        assert unit_value is not None
        assert str(unit_value) == "celsius"

    def test_generate_rdf_with_typed_literals(self):
        """Test RDF generation preserves datatypes."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "value": {"type": "number"},
                "active": {"type": "boolean"},
                "label": {"type": "string"}
            }
        }

        params = {
            "count": 10,
            "value": 3.14,
            "active": True,
            "label": "test"
        }

        rdf = generate_rdf_payload(
            params,
            schema,
            "http://example.org/resource1",
            "http://example.org/vocab#"
        )

        # Parse and verify datatypes
        graph = Graph()
        graph.parse(data=rdf, format="turtle")

        vocab_ns = Namespace("http://example.org/vocab#")
        subject = URIRef("http://example.org/resource1")

        count = graph.value(subject, vocab_ns.count)
        assert count.datatype == XSD.integer

        value = graph.value(subject, vocab_ns.value)
        assert value.datatype == XSD.double

        active = graph.value(subject, vocab_ns.active)
        assert active.datatype == XSD.boolean

        label = graph.value(subject, vocab_ns.label)
        assert label.datatype == XSD.string

    def test_generate_rdf_with_array(self):
        """Test RDF generation with array values."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array"}
            }
        }

        params = {"tags": ["tag1", "tag2", "tag3"]}

        rdf = generate_rdf_payload(
            params,
            schema,
            "http://example.org/resource1",
            "http://example.org/vocab#"
        )

        # Parse and verify
        graph = Graph()
        graph.parse(data=rdf, format="turtle")

        vocab_ns = Namespace("http://example.org/vocab#")
        subject = URIRef("http://example.org/resource1")

        # Check all tags are present
        tags = list(graph.objects(subject, vocab_ns.tags))
        assert len(tags) == 3
        tag_values = [str(tag) for tag in tags]
        assert "tag1" in tag_values
        assert "tag2" in tag_values
        assert "tag3" in tag_values

    def test_generate_rdf_validation_failure(self):
        """Test that RDF generation fails with invalid params."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            },
            "required": ["value"]
        }

        params = {}  # Missing required field

        with pytest.raises(SchemaValidationError):
            generate_rdf_payload(
                params,
                schema,
                "http://example.org/resource1"
            )


class TestExtractSchemaSummary:
    """Test schema summary extraction."""

    def test_extract_basic_summary(self):
        """Test extracting a basic schema summary."""
        schema = {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "description": "Temperature in degrees"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                },
                "location": {
                    "type": "string"
                }
            },
            "required": ["temperature", "unit"]
        }

        summary = extract_schema_summary(schema)

        assert "temperature" in summary
        assert summary["temperature"]["type"] == "number"
        assert summary["temperature"]["required"] == "yes"
        assert summary["temperature"]["description"] == "Temperature in degrees"

        assert "unit" in summary
        assert summary["unit"]["type"] == "string"
        assert summary["unit"]["required"] == "yes"
        assert summary["unit"]["allowed_values"] == ["celsius", "fahrenheit"]

        assert "location" in summary
        assert summary["location"]["required"] == "no"

    def test_extract_summary_with_constraints(self):
        """Test summary includes constraint information."""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 120
                }
            }
        }

        summary = extract_schema_summary(schema)

        assert "age" in summary
        assert "constraints" in summary["age"]
        assert "min: 0" in summary["age"]["constraints"]
        assert "max: 120" in summary["age"]["constraints"]


class TestNestedObjects:
    """Test nested object handling."""

    def test_parse_nested_object_schema(self):
        """Test parsing a schema with nested objects."""
        graph = Graph()
        schema_uri = EX.NestedSchema

        # Top-level schema
        graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))

        # Add a nested object property
        location_prop = BNode()
        graph.add((schema_uri, JSONSCHEMA.properties, location_prop))
        graph.add((location_prop, JSONSCHEMA.propertyName, Literal("location")))
        graph.add((location_prop, JSONSCHEMA.type, Literal("object")))

        # Add properties to the nested object
        lat_prop = BNode()
        graph.add((location_prop, JSONSCHEMA.properties, lat_prop))
        graph.add((lat_prop, JSONSCHEMA.propertyName, Literal("latitude")))
        graph.add((lat_prop, JSONSCHEMA.type, Literal("number")))
        graph.add((lat_prop, JSONSCHEMA.minimum, Literal(-90)))
        graph.add((lat_prop, JSONSCHEMA.maximum, Literal(90)))

        lon_prop = BNode()
        graph.add((location_prop, JSONSCHEMA.properties, lon_prop))
        graph.add((lon_prop, JSONSCHEMA.propertyName, Literal("longitude")))
        graph.add((lon_prop, JSONSCHEMA.type, Literal("number")))
        graph.add((lon_prop, JSONSCHEMA.minimum, Literal(-180)))
        graph.add((lon_prop, JSONSCHEMA.maximum, Literal(180)))

        # Required fields in nested object
        graph.add((location_prop, JSONSCHEMA.required, Literal("latitude")))
        graph.add((location_prop, JSONSCHEMA.required, Literal("longitude")))

        # Parse schema
        schema = parse_jsonschema_from_rdf(graph, schema_uri)

        # Verify nested structure
        assert "location" in schema["properties"]
        location_schema = schema["properties"]["location"]
        assert location_schema["type"] == "object"
        assert "properties" in location_schema
        assert "latitude" in location_schema["properties"]
        assert "longitude" in location_schema["properties"]
        assert location_schema["properties"]["latitude"]["minimum"] == -90
        assert location_schema["properties"]["longitude"]["maximum"] == 180
        assert "latitude" in location_schema["required"]
        assert "longitude" in location_schema["required"]

    def test_validate_nested_object_params(self):
        """Test validation of nested object parameters."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "location": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                        "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                    },
                    "required": ["latitude", "longitude"]
                }
            },
            "required": ["name", "location"]
        }

        # Valid nested params
        params = {
            "name": "Sensor1",
            "location": {
                "latitude": 45.5,
                "longitude": -73.5
            }
        }
        is_valid, errors = validate_params(params, schema)
        assert is_valid is True
        assert len(errors) == 0

        # Missing nested required field
        params = {
            "name": "Sensor1",
            "location": {
                "latitude": 45.5
                # Missing longitude
            }
        }
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False
        assert any("location.longitude" in err for err in errors)

        # Nested value out of range
        params = {
            "name": "Sensor1",
            "location": {
                "latitude": 100,  # Out of range
                "longitude": -73.5
            }
        }
        is_valid, errors = validate_params(params, schema)
        assert is_valid is False
        assert any("location.latitude" in err and "maximum" in err for err in errors)

    def test_generate_rdf_with_nested_objects(self):
        """Test RDF generation with nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "deviceId": {"type": "string"},
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"},
                        "retries": {"type": "integer"}
                    }
                }
            }
        }

        params = {
            "deviceId": "device123",
            "config": {
                "timeout": 5000,
                "retries": 3
            }
        }

        rdf = generate_rdf_payload(
            params,
            schema,
            "http://example.org/device1",
            "http://example.org/vocab#"
        )

        # Parse and verify
        graph = Graph()
        graph.parse(data=rdf, format="turtle")

        vocab_ns = Namespace("http://example.org/vocab#")
        subject = URIRef("http://example.org/device1")

        # Check device ID
        device_id = graph.value(subject, vocab_ns.deviceId)
        assert str(device_id) == "device123"

        # Check nested config object (should be a blank node)
        config_node = graph.value(subject, vocab_ns.config)
        assert config_node is not None
        assert isinstance(config_node, BNode)

        # Check nested properties
        timeout = graph.value(config_node, vocab_ns.timeout)
        assert int(timeout) == 5000

        retries = graph.value(config_node, vocab_ns.retries)
        assert int(retries) == 3

    def test_deeply_nested_objects(self):
        """Test handling of deeply nested objects (3+ levels)."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "string"}
                                    },
                                    "required": ["value"]
                                }
                            }
                        }
                    }
                }
            }
        }

        params = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }

        # Validate
        is_valid, errors = validate_params(params, schema)
        assert is_valid is True

        # Generate RDF
        rdf = generate_rdf_payload(
            params,
            schema,
            "http://example.org/test",
            "http://example.org/vocab#"
        )
        assert rdf is not None

        # Verify RDF structure
        graph = Graph()
        graph.parse(data=rdf, format="turtle")
        assert len(graph) >= 4  # At least 4 triples for the nested structure

    def test_array_of_objects(self):
        """Test handling of arrays containing objects."""
        schema = {
            "type": "object",
            "properties": {
                "sensors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "value": {"type": "number"}
                        },
                        "required": ["id", "value"]
                    }
                }
            }
        }

        params = {
            "sensors": [
                {"id": "sensor1", "value": 23.5},
                {"id": "sensor2", "value": 19.2}
            ]
        }

        # Validate
        is_valid, errors = validate_params(params, schema)
        assert is_valid is True

        # Validate with missing required field in array item
        params_invalid = {
            "sensors": [
                {"id": "sensor1", "value": 23.5},
                {"id": "sensor2"}  # Missing value
            ]
        }
        is_valid, errors = validate_params(params_invalid, schema)
        assert is_valid is False
        assert any("sensors[1].value" in err for err in errors)

        # Generate RDF
        rdf = generate_rdf_payload(
            params,
            schema,
            "http://example.org/test",
            "http://example.org/vocab#"
        )

        # Verify RDF
        graph = Graph()
        graph.parse(data=rdf, format="turtle")

        vocab_ns = Namespace("http://example.org/vocab#")
        subject = URIRef("http://example.org/test")

        # Should have two blank nodes for the two sensors
        sensor_nodes = list(graph.objects(subject, vocab_ns.sensors))
        assert len(sensor_nodes) == 2


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self):
        """Test complete workflow: parse -> validate -> generate."""
        # Step 1: Create RDF schema
        graph = Graph()
        schema_uri = EX.ActionSchema

        graph.add((schema_uri, JSONSCHEMA.type, Literal("object")))

        # Power property
        power_prop = BNode()
        graph.add((schema_uri, JSONSCHEMA.properties, power_prop))
        graph.add((power_prop, JSONSCHEMA.propertyName, Literal("power")))
        graph.add((power_prop, JSONSCHEMA.type, Literal("boolean")))
        graph.add((schema_uri, JSONSCHEMA.required, Literal("power")))

        # Brightness property
        brightness_prop = BNode()
        graph.add((schema_uri, JSONSCHEMA.properties, brightness_prop))
        graph.add((brightness_prop, JSONSCHEMA.propertyName, Literal("brightness")))
        graph.add((brightness_prop, JSONSCHEMA.type, Literal("integer")))
        graph.add((brightness_prop, JSONSCHEMA.minimum, Literal(0)))
        graph.add((brightness_prop, JSONSCHEMA.maximum, Literal(100)))

        # Step 2: Parse schema
        schema = parse_jsonschema_from_rdf(graph, schema_uri)

        # Step 3: Create and validate parameters
        params = {"power": True, "brightness": 75}
        is_valid, errors = validate_params(params, schema)
        assert is_valid is True

        # Step 4: Generate JSON-LD payload
        jsonld = generate_jsonld_payload(params, schema)
        assert jsonld is not None

        # Step 5: Generate RDF payload
        rdf = generate_rdf_payload(
            params,
            schema,
            "http://example.org/light1",
            "http://example.org/action#"
        )
        assert rdf is not None

        # Step 6: Extract summary
        summary = extract_schema_summary(schema)
        assert "power" in summary
        assert "brightness" in summary
        assert summary["power"]["required"] == "yes"
        assert summary["brightness"]["required"] == "no"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
