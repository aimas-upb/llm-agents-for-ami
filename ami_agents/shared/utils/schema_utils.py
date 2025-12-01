"""
Utility functions for working with JSON Schema specifications in RDF format.

This module provides functionality to:
1. Parse JSON Schema from RDF graphs into a dictionary format
2. Validate parameter values against schemas
3. Generate JSON-LD and RDF payloads from parameters
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD
import json

logger = logging.getLogger(__name__)

# JSON Schema namespace (commonly used in RDF representations)
JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")


class SchemaParsingError(Exception):
    """Raised when schema parsing fails."""
    pass


class SchemaValidationError(Exception):
    """Raised when parameter validation fails."""
    pass


def parse_jsonschema_from_rdf(rdf_graph: Graph, schema_uri: URIRef) -> Dict[str, Any]:
    """
    Parse a JSON Schema specification from RDF into a dictionary format.

    This extracts the schema structure including property names, types, descriptions,
    required fields, and other constraints from an RDF representation of JSON Schema.

    Args:
        rdf_graph: The RDF graph containing the schema
        schema_uri: The URI of the schema resource to parse

    Returns:
        Dictionary representation of the schema with structure:
        {
            "type": "object",
            "properties": {
                "paramName": {
                    "type": "string",
                    "description": "...",
                    "required": True/False,
                    ...
                }
            },
            "required": ["param1", "param2"]
        }

    Raises:
        SchemaParsingError: If schema parsing fails
    """
    try:
        schema_dict = {}

        # Extract schema type (object, array, string, number, etc.)
        schema_type = rdf_graph.value(schema_uri, RDF.type)
        if schema_type:
            schema_dict["type"] = str(schema_type)

        # Extract description
        description = rdf_graph.value(schema_uri, RDFS.comment)
        if description:
            schema_dict["description"] = str(description)

        # Extract properties (for object types)
        properties = {}
        for prop_uri in rdf_graph.objects(schema_uri, JSONSCHEMA.properties):
            # Properties can be defined as blank nodes or named resources
            prop_name = _extract_property_name(rdf_graph, prop_uri)
            if prop_name:
                prop_schema = _parse_property_schema(rdf_graph, prop_uri)
                properties[prop_name] = prop_schema

        if properties:
            schema_dict["properties"] = properties

        # Extract required fields
        required_fields = []
        for required_uri in rdf_graph.objects(schema_uri, JSONSCHEMA.required):
            required_fields.append(str(required_uri))

        if required_fields:
            schema_dict["required"] = required_fields

        # Extract items (for array types)
        items = rdf_graph.value(schema_uri, JSONSCHEMA.items)
        if items:
            schema_dict["items"] = _parse_property_schema(rdf_graph, items)

        # Extract enum values
        enum_values = list(rdf_graph.objects(schema_uri, JSONSCHEMA.enum))
        if enum_values:
            schema_dict["enum"] = [str(val) for val in enum_values]

        # Extract constraints
        _extract_constraints(rdf_graph, schema_uri, schema_dict)

        logger.debug(f"Parsed schema: {schema_dict}")
        return schema_dict

    except Exception as e:
        logger.error(f"Failed to parse JSON Schema from RDF: {e}", exc_info=True)
        raise SchemaParsingError(f"Failed to parse schema: {e}")


def _extract_property_name(rdf_graph: Graph, prop_uri: URIRef) -> Optional[str]:
    """
    Extract the property name from an RDF property definition.

    Args:
        rdf_graph: The RDF graph
        prop_uri: The URI of the property

    Returns:
        Property name as string, or None if not found
    """
    # Try different predicates that might contain the property name
    name = rdf_graph.value(prop_uri, JSONSCHEMA.propertyName) or \
           rdf_graph.value(prop_uri, RDFS.label)

    if name:
        return str(name)

    # If it's not a blank node, use the local name from URI
    if not isinstance(prop_uri, BNode):
        if "#" in str(prop_uri):
            return str(prop_uri).split("#")[-1]
        else:
            return str(prop_uri).split("/")[-1]

    return None


def _parse_property_schema(rdf_graph: Graph, prop_uri: URIRef) -> Dict[str, Any]:
    """
    Parse the schema definition for a single property.

    This function recursively handles nested object schemas.

    Args:
        rdf_graph: The RDF graph
        prop_uri: The URI of the property

    Returns:
        Dictionary with property schema details
    """
    prop_schema = {}

    # Type
    prop_type = rdf_graph.value(prop_uri, RDF.type)
    if prop_type:
        prop_schema["type"] = str(prop_type)

    # Description
    description = rdf_graph.value(prop_uri, RDFS.comment)
    if description:
        prop_schema["description"] = str(description)

    # Default value
    default = rdf_graph.value(prop_uri, JSONSCHEMA.default)
    if default:
        prop_schema["default"] = _convert_rdf_literal(default)

    # Enum values
    enum_values = list(rdf_graph.objects(prop_uri, JSONSCHEMA.enum))
    if enum_values:
        prop_schema["enum"] = [_convert_rdf_literal(val) for val in enum_values]

    # Nested properties (for object types)
    # Recursively parse nested object properties
    nested_properties = {}
    for nested_prop_uri in rdf_graph.objects(prop_uri, JSONSCHEMA.properties):
        nested_prop_name = _extract_property_name(rdf_graph, nested_prop_uri)
        if nested_prop_name:
            # Recursive call to handle nested schemas
            nested_prop_schema = _parse_property_schema(rdf_graph, nested_prop_uri)
            nested_properties[nested_prop_name] = nested_prop_schema

    if nested_properties:
        prop_schema["properties"] = nested_properties

    # Required fields for nested objects
    nested_required = []
    for required_uri in rdf_graph.objects(prop_uri, JSONSCHEMA.required):
        nested_required.append(str(required_uri))

    if nested_required:
        prop_schema["required"] = nested_required

    # Items schema (for array types)
    items = rdf_graph.value(prop_uri, JSONSCHEMA.items)
    if items:
        # Recursively parse items schema
        prop_schema["items"] = _parse_property_schema(rdf_graph, items)

    # Constraints
    _extract_constraints(rdf_graph, prop_uri, prop_schema)

    return prop_schema


def _extract_constraints(rdf_graph: Graph, uri: URIRef, schema_dict: Dict[str, Any]) -> None:
    """
    Extract numeric and string constraints from schema.

    Args:
        rdf_graph: The RDF graph
        uri: The URI of the schema/property
        schema_dict: Dictionary to populate with constraints
    """
    # Numeric constraints
    minimum = rdf_graph.value(uri, JSONSCHEMA.minimum)
    if minimum is not None:
        schema_dict["minimum"] = float(minimum)

    maximum = rdf_graph.value(uri, JSONSCHEMA.maximum)
    if maximum is not None:
        schema_dict["maximum"] = float(maximum)

    # String constraints
    min_length = rdf_graph.value(uri, JSONSCHEMA.minLength)
    if min_length is not None:
        schema_dict["minLength"] = int(min_length)

    max_length = rdf_graph.value(uri, JSONSCHEMA.maxLength)
    if max_length is not None:
        schema_dict["maxLength"] = int(max_length)

    pattern = rdf_graph.value(uri, JSONSCHEMA.pattern)
    if pattern is not None:
        schema_dict["pattern"] = str(pattern)

    # Array constraints
    min_items = rdf_graph.value(uri, JSONSCHEMA.minItems)
    if min_items is not None:
        schema_dict["minItems"] = int(min_items)

    max_items = rdf_graph.value(uri, JSONSCHEMA.maxItems)
    if max_items is not None:
        schema_dict["maxItems"] = int(max_items)


def _convert_rdf_literal(value: Any) -> Any:
    """
    Convert an RDF literal to its Python native type.

    Args:
        value: RDF literal or value

    Returns:
        Python native value
    """
    if isinstance(value, Literal):
        # Try to convert based on datatype
        if value.datatype == XSD.integer or value.datatype == XSD.int:
            return int(value)
        elif value.datatype == XSD.float or value.datatype == XSD.double:
            return float(value)
        elif value.datatype == XSD.boolean:
            return bool(value)
        else:
            return str(value)

    return str(value)


def validate_params(params: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> Tuple[bool, List[str]]:
    """
    Validate parameter values against a schema.

    Supports nested object validation through recursion.

    Args:
        params: Dictionary of parameter name -> value pairs
        schema: Schema dictionary (as returned by parse_jsonschema_from_rdf)
        path: Current path in nested structure (for error messages)

    Returns:
        Tuple of (is_valid, list_of_errors)
        - is_valid: True if all validations pass
        - list_of_errors: List of error messages (empty if valid)

    Example:
        >>> params = {"temperature": 25, "unit": "celsius"}
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "temperature": {"type": "number", "minimum": 0, "maximum": 100},
        ...         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        ...     },
        ...     "required": ["temperature"]
        ... }
        >>> is_valid, errors = validate_params(params, schema)
    """
    errors = []

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in params:
            field_path = f"{path}.{field}" if path else field
            errors.append(f"Required field '{field_path}' is missing")

    # Validate each parameter
    properties = schema.get("properties", {})
    for param_name, param_value in params.items():
        # Build the path for nested error messages
        param_path = f"{path}.{param_name}" if path else param_name

        # Check if parameter is defined in schema
        if param_name not in properties:
            logger.warning(f"Parameter '{param_path}' not defined in schema")
            continue

        prop_schema = properties[param_name]

        # Validate type
        expected_type = prop_schema.get("type")
        if expected_type:
            type_errors = _validate_type(param_path, param_value, expected_type)
            errors.extend(type_errors)

            # For nested objects, recursively validate
            if expected_type == "object" and isinstance(param_value, dict):
                # Recursively validate nested object
                nested_valid, nested_errors = validate_params(param_value, prop_schema, param_path)
                errors.extend(nested_errors)

        # Validate enum
        if "enum" in prop_schema:
            if param_value not in prop_schema["enum"]:
                errors.append(
                    f"Parameter '{param_path}' value '{param_value}' not in allowed values: {prop_schema['enum']}"
                )

        # Validate numeric constraints
        if expected_type in ["number", "integer"]:
            if "minimum" in prop_schema and param_value < prop_schema["minimum"]:
                errors.append(
                    f"Parameter '{param_path}' value {param_value} is below minimum {prop_schema['minimum']}"
                )
            if "maximum" in prop_schema and param_value > prop_schema["maximum"]:
                errors.append(
                    f"Parameter '{param_path}' value {param_value} exceeds maximum {prop_schema['maximum']}"
                )

        # Validate string constraints
        if expected_type == "string":
            if "minLength" in prop_schema and len(param_value) < prop_schema["minLength"]:
                errors.append(
                    f"Parameter '{param_path}' length {len(param_value)} is below minimum {prop_schema['minLength']}"
                )
            if "maxLength" in prop_schema and len(param_value) > prop_schema["maxLength"]:
                errors.append(
                    f"Parameter '{param_path}' length {len(param_value)} exceeds maximum {prop_schema['maxLength']}"
                )
            if "pattern" in prop_schema:
                import re
                if not re.match(prop_schema["pattern"], param_value):
                    errors.append(
                        f"Parameter '{param_path}' does not match pattern {prop_schema['pattern']}"
                    )

        # Validate array constraints
        if expected_type == "array":
            if "minItems" in prop_schema and len(param_value) < prop_schema["minItems"]:
                errors.append(
                    f"Parameter '{param_path}' has {len(param_value)} items, minimum is {prop_schema['minItems']}"
                )
            if "maxItems" in prop_schema and len(param_value) > prop_schema["maxItems"]:
                errors.append(
                    f"Parameter '{param_path}' has {len(param_value)} items, maximum is {prop_schema['maxItems']}"
                )

            # Validate array items if schema is provided
            if "items" in prop_schema and isinstance(param_value, list):
                items_schema = prop_schema["items"]
                for idx, item in enumerate(param_value):
                    item_path = f"{param_path}[{idx}]"
                    # If items schema is an object, validate it
                    if items_schema.get("type") == "object" and isinstance(item, dict):
                        nested_valid, nested_errors = validate_params(item, items_schema, item_path)
                        errors.extend(nested_errors)
                    else:
                        # Validate item type
                        item_type = items_schema.get("type")
                        if item_type:
                            type_errors = _validate_type(item_path, item, item_type)
                            errors.extend(type_errors)

    is_valid = len(errors) == 0
    return is_valid, errors


def _validate_type(param_name: str, value: Any, expected_type: str) -> List[str]:
    """
    Validate that a value matches the expected type.

    Args:
        param_name: Name of the parameter
        value: Value to validate
        expected_type: Expected JSON Schema type

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    type_mapping = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None)
    }

    expected_python_type = type_mapping.get(expected_type)
    if expected_python_type and not isinstance(value, expected_python_type):
        errors.append(
            f"Parameter '{param_name}' has type {type(value).__name__}, expected {expected_type}"
        )

    return errors


def generate_jsonld_payload(
    params: Dict[str, Any],
    schema: Dict[str, Any],
    context: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate a JSON-LD payload from parameters and schema.

    Args:
        params: Dictionary of parameter name -> value pairs
        schema: Schema dictionary (for type information and validation)
        context: Optional JSON-LD context mapping (defaults to simple context)

    Returns:
        JSON-LD string representation

    Example:
        >>> params = {"temperature": 25, "unit": "celsius"}
        >>> schema = {"type": "object", "properties": {...}}
        >>> context = {"temperature": "http://example.org/vocab#temperature"}
        >>> payload = generate_jsonld_payload(params, schema, context)
    """
    # Validate parameters first
    is_valid, errors = validate_params(params, schema)
    if not is_valid:
        raise SchemaValidationError(f"Parameter validation failed: {errors}")

    # Build JSON-LD document
    jsonld = {}

    # Add context
    if context:
        jsonld["@context"] = context
    else:
        # Default simple context
        jsonld["@context"] = {
            prop: f"http://example.org/vocab#{prop}"
            for prop in params.keys()
        }

    # Add parameters
    jsonld.update(params)

    # Serialize to JSON string
    return json.dumps(jsonld, indent=2)


def generate_rdf_payload(
    params: Dict[str, Any],
    schema: Dict[str, Any],
    subject_uri: str,
    property_namespace: str = "http://example.org/vocab#",
    format: str = "turtle"
) -> str:
    """
    Generate an RDF payload from parameters and schema.

    Supports nested objects by creating blank nodes for nested structures.

    Args:
        params: Dictionary of parameter name -> value pairs
        schema: Schema dictionary (for type information and validation)
        subject_uri: URI of the subject resource
        property_namespace: Namespace URI for properties
        format: RDF serialization format (turtle, xml, n3, nt, json-ld)

    Returns:
        RDF string representation in the specified format

    Example:
        >>> params = {"temperature": 25, "unit": "celsius"}
        >>> schema = {"type": "object", "properties": {...}}
        >>> rdf = generate_rdf_payload(
        ...     params, schema,
        ...     "http://example.org/sensor1",
        ...     "http://example.org/vocab#"
        ... )
    """
    # Validate parameters first
    is_valid, errors = validate_params(params, schema)
    if not is_valid:
        raise SchemaValidationError(f"Parameter validation failed: {errors}")

    # Create RDF graph
    graph = Graph()

    # Create namespace for properties
    NS = Namespace(property_namespace)
    graph.bind("vocab", NS)

    # Subject URI
    subject = URIRef(subject_uri)

    # Add triples for each parameter
    _add_params_to_graph(graph, subject, params, schema, NS)

    # Serialize to specified format
    return graph.serialize(format=format)


def _add_params_to_graph(
    graph: Graph,
    subject: URIRef,
    params: Dict[str, Any],
    schema: Dict[str, Any],
    namespace: Namespace
) -> None:
    """
    Recursively add parameters to RDF graph, handling nested objects.

    Args:
        graph: RDF graph to add triples to
        subject: Subject URI or BNode
        params: Parameters to add
        schema: Schema for the parameters
        namespace: Namespace for properties
    """
    properties = schema.get("properties", {})

    for param_name, param_value in params.items():
        prop_uri = namespace[param_name]

        # Determine datatype from schema
        prop_schema = properties.get(param_name, {})
        param_type = prop_schema.get("type", "string")

        # Handle nested objects
        if param_type == "object" and isinstance(param_value, dict):
            # Create a blank node for the nested object
            nested_node = BNode()
            graph.add((subject, prop_uri, nested_node))

            # Recursively add nested properties
            _add_params_to_graph(graph, nested_node, param_value, prop_schema, namespace)
            continue

        # Handle arrays
        if param_type == "array" and isinstance(param_value, list):
            items_schema = prop_schema.get("items", {})
            item_type = items_schema.get("type", "string")

            for item in param_value:
                # If array items are objects, create blank nodes
                if item_type == "object" and isinstance(item, dict):
                    item_node = BNode()
                    graph.add((subject, prop_uri, item_node))
                    _add_params_to_graph(graph, item_node, item, items_schema, namespace)
                else:
                    # Simple array items
                    graph.add((subject, prop_uri, Literal(item)))
            continue

        # Handle primitive types
        if param_type == "integer":
            literal_value = Literal(param_value, datatype=XSD.integer)
        elif param_type == "number":
            literal_value = Literal(param_value, datatype=XSD.double)
        elif param_type == "boolean":
            literal_value = Literal(param_value, datatype=XSD.boolean)
        else:
            literal_value = Literal(param_value, datatype=XSD.string)

        graph.add((subject, prop_uri, literal_value))


def extract_schema_summary(schema: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Extract a simplified summary of parameter names and types from schema.

    This is useful for quick inspection of what parameters are expected.

    Args:
        schema: Schema dictionary

    Returns:
        Dictionary mapping parameter names to their type and requirement info:
        {
            "paramName": {
                "type": "string",
                "required": "yes/no",
                "description": "..."
            }
        }
    """
    summary = {}

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for param_name, prop_schema in properties.items():
        summary[param_name] = {
            "type": prop_schema.get("type", "unknown"),
            "required": "yes" if param_name in required else "no"
        }

        if "description" in prop_schema:
            summary[param_name]["description"] = prop_schema["description"]

        if "enum" in prop_schema:
            summary[param_name]["allowed_values"] = prop_schema["enum"]

        if "minimum" in prop_schema or "maximum" in prop_schema:
            constraints = []
            if "minimum" in prop_schema:
                constraints.append(f"min: {prop_schema['minimum']}")
            if "maximum" in prop_schema:
                constraints.append(f"max: {prop_schema['maximum']}")
            summary[param_name]["constraints"] = ", ".join(constraints)

    return summary
